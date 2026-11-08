"""轻量 Web 界面（tornado 单页）。

为什么不是 Gradio：本机未安装 gradio，且 gradio 4.44.1 要求 ``pillow<11``，
会把现有环境中的 pillow 降级并影响同一环境下的其它项目。需求说明书允许
"若某个库与当前 Python 环境不兼容，应选择简单、稳定的替代方案，并在文档中说明"。
tornado 已随 Jupyter 装好，因此用它提供 HTTP 服务 + 一个原生 JS 单页。

界面只做三件事：上传图片、展示结构化结果、如实展示失败状态。
**不存在任何"演示模式"或离线假结果**：没有密钥就无法调用，界面会明确报阻塞原因。
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import tornado.web
from tornado.ioloop import IOLoop

from config import Settings, get_settings
from src.agent.executor import AgentExecutor
from src.agent.llm_client import build_chat_client, tool_model_name
from src.agent.report import build_report, render_markdown
from src.image_utils import ImageValidationError, estimate_visual_tokens, preprocess_image
from src.output_parser import parse_model_output
from src.prompts import VARIANT_ORDER, PromptError, load_prompt, variant_title
from src.schema import POSTURE_CN, validate_result
from src.tools.builder import build_default_registry
from src.vision_client import VisionClient

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: 工具调用日志（每次提问追加，作为报告里的过程证据）
AGENT_LOG = Path(__file__).resolve().parent.parent / "results" / "agent" / "tool_calls.jsonl"

#: 同一时刻只允许一次助手提问：并发跑会重复消耗额度，也容易触发限流
_AGENT_LOCK = threading.Lock()

#: 单次提问的默认/上限步数
DEFAULT_AGENT_STEPS = 6
MAX_AGENT_STEPS = 8

#: tools 参数 -> 装配哪些工具（对应消融条件）
TOOL_PRESETS: Dict[str, Dict[str, bool]] = {
    "all": {"include_data": True, "include_vision": True, "include_rag": True},
    "data": {"include_data": True, "include_vision": False, "include_rag": False},
    "rag": {"include_data": False, "include_vision": False, "include_rag": True},
    "none": {"include_data": False, "include_vision": False, "include_rag": False},
}


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def run_single(settings: Settings, raw: bytes, variant: str) -> Dict[str, Any]:
    """执行一次完整推理，返回可直接 JSON 化的字典。"""
    record: Dict[str, Any] = {
        "ok": False,
        "timestamp": _now(),
        "variant": {"key": variant},
        "request_success": False,
        "error_message": None,
    }

    try:
        prompt = load_prompt(variant)
    except PromptError as exc:
        record["error_message"] = str(exc)
        return record
    record["variant"] = {"key": prompt.key, "title": prompt.title, "sha256": prompt.sha256}

    try:
        processed = preprocess_image(
            raw, max_side=settings.image_max_side, jpeg_quality=settings.jpeg_quality
        )
    except ImageValidationError as exc:
        record["error_message"] = str(exc)
        return record

    record["image_meta"] = {
        "original": "{0}×{1}".format(processed.original_width, processed.original_height),
        "sent": "{0}×{1}".format(processed.width, processed.height),
        "bytes_sent": len(processed.jpeg_bytes),
        "source_format": processed.source_format,
        "estimated_visual_tokens": estimate_visual_tokens(processed.width, processed.height),
    }

    client = VisionClient(settings)
    response = client.call(processed.data_url, prompt.text)

    record["status"] = {
        "request_success": response.ok,
        "http_status": response.http_status,
        "latency_seconds": round(response.latency_seconds, 3),
        "attempt": response.attempt,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "notes": response.notes,
    }
    record["raw_text"] = response.raw_text

    if not response.ok:
        record["error_message"] = response.error
        return record

    parsed = parse_model_output(response.raw_text)
    record["parse"] = {
        "json_valid": parsed.json_valid,
        "method": parsed.method,
        "error": parsed.error,
        "notes": parsed.notes,
    }
    if parsed.data is None:
        record["error_message"] = parsed.error or "模型返回无法解析"
        record["request_success"] = True
        return record

    validation = validate_result(parsed.data)
    record["parse"]["schema_valid"] = validation.valid
    record["parse"]["errors"] = validation.errors
    record["parse"]["warnings"] = validation.warnings
    record["posture"] = validation.posture
    record["posture_cn"] = POSTURE_CN.get(validation.posture or "", "—")
    record["description"] = validation.description
    record["uncertain"] = validation.uncertain
    record["uncertainty_reason"] = validation.uncertainty_reason
    record["request_success"] = True
    record["ok"] = validation.valid

    if not validation.valid:
        record["error_message"] = "模型返回未通过 Schema 校验：" + "；".join(validation.errors)
    return record


def run_agent(question: str, tools: str = "all",
              max_steps: int = DEFAULT_AGENT_STEPS,
              history: Optional[list] = None) -> Dict[str, Any]:
    """执行一次助手问答，返回可直接 JSON 化的字典。

    这是同步阻塞函数，由 handler 放进线程池执行——tornado 是单线程事件循环，
    直接调用会把整个界面卡住。
    """
    preset = TOOL_PRESETS.get(tools or "all")
    if preset is None:
        raise ValueError("未知的工具组合：{0}".format(tools))

    registry = build_default_registry(
        log_path=AGENT_LOG,
        max_calls=MAX_AGENT_STEPS * 2,
        **preset
    )
    client = build_chat_client()
    executor = AgentExecutor(registry, client, max_steps=max_steps)
    result = executor.run(question, history=history)
    report = build_report(result)

    return {
        "ok": report.ok,
        "report": report.to_dict(),
        "markdown": render_markdown(report),
        "tool_stats": registry.stats(),
        "tools": registry.names(),
        "model": client.model,
        "timestamp": _now(),
        "error_message": None if report.ok else (report.error or "未能生成回答"),
    }


class BaseHandler(tornado.web.RequestHandler):
    """统一 JSON 响应与错误格式。"""

    def initialize(self, settings: Settings = None):  # noqa: D401 - tornado hook
        self.settings_obj = settings

    def write_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        self.set_status(status)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        self.write(json.dumps(payload, ensure_ascii=False))

    def write_error(self, status_code: int, **kwargs) -> None:
        message = "服务器内部错误"
        if status_code == 400:
            message = "请求格式不正确"
        elif status_code == 413:
            message = "上传文件过大（上限 {0} MB）".format(MAX_UPLOAD_BYTES // (1024 * 1024))
        elif status_code == 404:
            message = "接口不存在"
        self.write_json({"ok": False, "error_message": message}, status=status_code)


class IndexHandler(BaseHandler):
    def get(self):
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write(INDEX_HTML)


class NotFoundHandler(BaseHandler):
    """兜底 404：返回 JSON，避免前端拿到 HTML 而解析失败。"""

    def prepare(self):
        self.write_json({"ok": False, "error_message": "接口不存在：{0}".format(self.request.path)}, 404)
        self.finish()


def _agent_tool_summary() -> list:
    """给界面用的工具清单（名称 + 说明），装配失败时返回空表而不是报错。"""
    try:
        registry = build_default_registry()
    except Exception:  # noqa: BLE001 —— 界面自检不应因工具装配失败而整体不可用
        return []
    return [
        {"name": spec.name, "description": spec.description}
        for spec in (registry.get(name) for name in registry.names())
    ]


class HealthHandler(BaseHandler):
    """自检：解释器、服务商、密钥、提示词与助手工具是否就绪。"""

    def get(self):
        settings = self.settings_obj or get_settings()
        prompts = []
        for key in VARIANT_ORDER:
            try:
                variant = load_prompt(key)
                prompts.append({"key": key, "title": variant.title, "sha256": variant.sha256})
            except PromptError as exc:
                prompts.append({"key": key, "title": variant_title(key), "error": str(exc)})
        self.write_json({
            "ok": True,
            "provider": settings.provider,
            "provider_label": settings.provider_label,
            "model": settings.model,
            "has_api_key": settings.has_api_key,
            "qps": settings.qps,
            "concurrency": settings.concurrency,
            "prompts": prompts,
            "agent": {
                "model": tool_model_name(),
                "tools": _agent_tool_summary(),
                "presets": list(TOOL_PRESETS),
                "max_steps": MAX_AGENT_STEPS,
            },
            "hint": None if settings.has_api_key else (
                "未配置 {0}，无法调用真实模型。请在项目根目录 .env 中填写密钥。".format(settings.provider)
            ),
        })


class PromptsHandler(BaseHandler):
    def get(self):
        payload = []
        for key in VARIANT_ORDER:
            try:
                variant = load_prompt(key)
                payload.append({
                    "key": key,
                    "title": variant.title,
                    "sha256": variant.sha256,
                    "text": variant.text,
                })
            except PromptError as exc:
                payload.append({"key": key, "error": str(exc)})
        self.write_json({"ok": True, "prompts": payload})


class PredictHandler(BaseHandler):
    """单图识别。"""

    async def post(self):
        settings = self.settings_obj or get_settings()
        variant = (self.get_body_argument("variant", "C") or "C").strip().upper()

        files = self.request.files.get("image") or []
        if not files:
            self.write_json({"ok": False, "error_message": "没有收到图片文件。请选择一张猪只图片后再试。"}, 400)
            return
        upload = files[0]
        raw = upload["body"]
        if len(raw) > MAX_UPLOAD_BYTES:
            self.write_json(
                {"ok": False, "error_message": "图片超过 {0} MB，请压缩后再上传。".format(
                    MAX_UPLOAD_BYTES // (1024 * 1024))}, 413
            )
            return

        if not settings.has_api_key:
            self.write_json({
                "ok": False,
                "error_message": "未配置 API 密钥，无法调用真实模型。"
                                 "请在项目根目录创建 .env 并填写密钥后重试。",
                "need_key": True,
            })
            return

        try:
            result = await IOLoop.current().run_in_executor(
                None, run_single, settings, raw, variant
            )
        except Exception:  # 未预期异常也要如实回报，不能伪造结果
            self.write_json(
                {"ok": False, "error_message": "处理失败：" + traceback.format_exc(limit=2)}, 500
            )
            return

        result["filename"] = upload.get("filename") or ""
        result["settings"] = {
            "provider": settings.provider,
            "model": settings.model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }
        self.write_json(result)


class AgentHandler(BaseHandler):
    """智能助手：自然语言提问 → 工具调用 → 带来源的回答。

    一次提问可能耗时 30–120 秒，因此放进线程池；同一时刻只允许一个提问在跑，
    避免并发重复消耗额度。
    """

    async def post(self):
        settings = self.settings_obj or get_settings()

        try:
            payload = json.loads(self.request.body or b"{}")
        except ValueError:
            self.write_json({"ok": False, "error_message": "请求体不是合法 JSON"}, 400)
            return
        if not isinstance(payload, dict):
            self.write_json({"ok": False, "error_message": "请求体必须是 JSON 对象"}, 400)
            return

        question = str(payload.get("question") or "").strip()
        if not question:
            self.write_json({"ok": False, "error_message": "请先输入问题"}, 400)
            return
        if len(question) > 500:
            self.write_json({"ok": False, "error_message": "问题过长（上限 500 字）"}, 400)
            return

        tools = str(payload.get("tools") or "all").strip().lower()
        if tools not in TOOL_PRESETS:
            self.write_json({
                "ok": False,
                "error_message": "tools 只能是 {0}".format("、".join(TOOL_PRESETS)),
            }, 400)
            return

        raw_steps = payload.get("max_steps")
        if raw_steps is None or raw_steps == "":
            max_steps = DEFAULT_AGENT_STEPS
        else:
            try:
                max_steps = int(raw_steps)
            except (TypeError, ValueError):
                self.write_json({"ok": False, "error_message": "max_steps 必须是整数"}, 400)
                return
        # 显式夹取而不是"0 当作没传"，否则用户给的非法值会被静默放大成默认步数
        max_steps = max(1, min(MAX_AGENT_STEPS, max_steps))

        if not settings.has_api_key:
            self.write_json({
                "ok": False,
                "error_message": "未配置 API 密钥，无法调用助手模型。"
                                 "请在项目根目录创建 .env 并填写密钥后重试。",
                "need_key": True,
            })
            return

        if not _AGENT_LOCK.acquire(blocking=False):
            self.write_json({
                "ok": False,
                "error_message": "已有一次提问正在处理，请等它完成后再试。",
                "busy": True,
            }, 429)
            return

        try:
            result = await IOLoop.current().run_in_executor(
                None, run_agent, question, tools, max_steps,
                payload.get("history") or [],
            )
        except ValueError as exc:
            self.write_json({"ok": False, "error_message": str(exc)}, 400)
            return
        except Exception:  # 未预期异常也要如实回报，不能伪造回答
            self.write_json({
                "ok": False,
                "error_message": "处理失败：" + traceback.format_exc(limit=2),
            }, 500)
            return
        finally:
            _AGENT_LOCK.release()

        self.write_json(result)


def make_app(settings: Optional[Settings] = None) -> tornado.web.Application:
    """构造 tornado 应用。"""
    settings = settings or get_settings()
    return tornado.web.Application(
        [
            (r"/", IndexHandler, {"settings": settings}),
            (r"/api/health", HealthHandler, {"settings": settings}),
            (r"/api/prompts", PromptsHandler, {"settings": settings}),
            (r"/api/predict", PredictHandler, {"settings": settings}),
            (r"/api/agent", AgentHandler, {"settings": settings}),
            (r"/(.*)", NotFoundHandler, {"settings": settings}),
        ],
        max_buffer_size=MAX_UPLOAD_BYTES,
        max_body_size=MAX_UPLOAD_BYTES,
        default_handler_class=None,
    )


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>猪只姿态识别与可见体态描述</title>
<style>
  :root { --bg:#0f1419; --panel:#1a2129; --line:#2b3540; --fg:#e6edf3; --muted:#8b98a5;
          --accent:#3b82f6; --ok:#22c55e; --warn:#f59e0b; --err:#ef4444; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
         background:var(--bg); color:var(--fg); font-size:14px; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex;
           align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; font-weight:600; }
  header .sub { color:var(--muted); font-size:12px; }
  .wrap { display:grid; grid-template-columns:minmax(320px,1fr) minmax(360px,1fr);
          gap:18px; padding:18px; max-width:1400px; margin:0 auto; }
  @media (max-width:900px){ .wrap { grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }
  .card h2 { font-size:14px; margin:0 0 12px; color:var(--muted); font-weight:600;
             letter-spacing:.04em; }
  #drop { border:2px dashed var(--line); border-radius:10px; padding:26px; text-align:center;
          color:var(--muted); cursor:pointer; transition:.15s; }
  #drop.hover { border-color:var(--accent); color:var(--fg); background:#16202b; }
  #preview { max-width:100%; border-radius:8px; display:none; margin-top:12px; }
  select, button { font-family:inherit; font-size:14px; }
  select { background:#0d1117; color:var(--fg); border:1px solid var(--line);
           border-radius:8px; padding:8px 10px; width:100%; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px;
           padding:10px 18px; cursor:pointer; font-weight:600; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; }
  .kv { display:grid; grid-template-columns:110px 1fr; gap:6px 12px; font-size:13px; }
  .kv dt { color:var(--muted); }
  .kv dd { margin:0; word-break:break-word; }
  .pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
          background:#243040; color:var(--fg); }
  .pill.ok { background:rgba(34,197,94,.15); color:#4ade80; }
  .pill.warn { background:rgba(245,158,11,.15); color:#fbbf24; }
  .pill.err { background:rgba(239,68,68,.15); color:#f87171; }
  .posture { font-size:26px; font-weight:700; }
  .desc { line-height:1.7; }
  pre { background:#0d1117; border:1px solid var(--line); border-radius:8px; padding:10px;
        overflow:auto; font-size:12px; max-height:260px; white-space:pre-wrap; }
  details summary { cursor:pointer; color:var(--muted); }
  .banner { margin:0 18px; padding:10px 14px; border-radius:8px; display:none; }
  .banner.show { display:block; }
  .banner.err { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35); }
  .banner.warn { background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.35); }
  .muted { color:var(--muted); font-size:12px; }
  textarea { width:100%; background:#0d1117; color:var(--fg); border:1px solid var(--line);
             border-radius:8px; padding:10px; font-family:inherit; font-size:14px; resize:vertical; }
  textarea:focus { outline:none; border-color:var(--accent); }
  .chip { display:inline-block; padding:5px 10px; margin:0 6px 6px 0; border-radius:999px;
          background:#243040; color:var(--fg); font-size:12px; cursor:pointer;
          border:1px solid var(--line); }
  .chip:hover { border-color:var(--accent); }
  .answer { line-height:1.8; }
  .answer h3 { font-size:14px; margin:14px 0 6px; color:#93c5fd; }
  .answer .li { margin:2px 0; }
  .cite { font-size:12px; color:var(--muted); margin:5px 0; word-break:break-all; }
  .cite a { color:#7dd3fc; }
  .trace td { padding:4px 8px; border-bottom:1px solid var(--line); font-size:12px; }
  .trace th { text-align:left; padding:4px 8px; color:var(--muted); font-size:12px;
              border-bottom:1px solid var(--line); }
</style>
</head>
<body>
<header>
  <h1>猪只姿态识别与可见体态描述</h1>
  <span class="sub">多模态大模型 API · 三组提示词对照实验</span>
</header>
<div id="banner" class="banner"></div>
<div class="wrap">
  <section class="card">
    <h2>1 · 上传图片</h2>
    <div id="drop">点击选择，或把猪只图片拖到这里<br><span class="muted">支持 JPG / PNG / BMP</span></div>
    <input id="file" type="file" accept="image/*" hidden>
    <img id="preview" alt="待识别图片">
    <h2 style="margin-top:18px">2 · 提示词方案</h2>
    <select id="variant"></select>
    <div class="row">
      <button id="run" disabled>开始识别</button>
      <span id="hint" class="muted"></span>
    </div>
    <p class="muted">提示词内容可在下方"查看提示词"中确认；实验记录会保存提示词版本（SHA256）。</p>
  </section>
  <section class="card">
    <h2>3 · 识别结果</h2>
    <div id="result"><span class="muted">尚未运行。上传图片并点击"开始识别"。</span></div>
  </section>
</div>
<div class="wrap" style="padding-top:0">
  <section class="card" style="grid-column:1/-1">
    <h2>4 · 智能助手（检索增强 + 工具调用）</h2>
    <p class="muted">可以直接问养殖数据、查文献出处、看图片姿态。回答会标注来源与工具轨迹；
      养殖数据为模拟数据，不可用于生产决策。</p>
    <div id="agentExamples" style="margin:10px 0"></div>
    <textarea id="agentQ" rows="3" placeholder="例如：最近一周哪个栏位的猪采食量偏低？P2-007 要不要重点关注？"></textarea>
    <div class="row">
      <select id="agentTools" style="width:auto"></select>
      <select id="agentSteps" style="width:auto"></select>
      <button id="agentRun">提问</button>
      <span id="agentHint" class="muted"></span>
    </div>
    <div id="agentResult" style="margin-top:14px"></div>
  </section>
</div>
<div class="wrap" style="padding-top:0">
  <section class="card" style="grid-column:1/-1">
    <details><summary>查看提示词内容</summary><div id="prompts" style="margin-top:10px"></div></details>
  </section>
</div>
<script>
const $ = (s) => document.querySelector(s);
let selectedFile = null;

function pill(text, cls) { return `<span class="pill ${cls || ''}">${text}</span>`; }

async function loadHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    const banner = $('#banner');
    if (!d.has_api_key) {
      banner.className = 'banner err show';
      banner.textContent = d.hint || '未配置 API 密钥，无法调用真实模型。';
    } else {
      banner.className = 'banner warn show';
      banner.textContent = `当前服务商：${d.provider_label}（${d.model}）。图片将发送至该服务商进行处理。`;
    }
    $('#hint').textContent = `QPS ${d.qps} · 模型 ${d.model}`;
  } catch (e) { /* 健康检查失败不阻塞界面 */ }
}

async function loadPrompts() {
  const r = await fetch('/api/prompts');
  const d = await r.json();
  const select = $('#variant');
  const box = $('#prompts');
  box.innerHTML = '';
  d.prompts.forEach((p) => {
    const opt = document.createElement('option');
    opt.value = p.key;
    opt.textContent = p.title;
    if (p.key === 'C') opt.selected = true;
    select.appendChild(opt);
    const el = document.createElement('details');
    el.innerHTML = `<summary>${p.title} · SHA256 ${p.sha256.slice(0, 12)}…</summary><pre>${p.text || p.error}</pre>`;
    box.appendChild(el);
  });
}

$('#drop').addEventListener('click', () => $('#file').click());
$('#drop').addEventListener('dragover', (e) => { e.preventDefault(); $('#drop').classList.add('hover'); });
$('#drop').addEventListener('dragleave', () => $('#drop').classList.remove('hover'));
$('#drop').addEventListener('drop', (e) => {
  e.preventDefault(); $('#drop').classList.remove('hover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
$('#file').addEventListener('change', (e) => { if (e.target.files.length) setFile(e.target.files[0]); });

function setFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  $('#preview').src = url;
  $('#preview').style.display = 'block';
  $('#run').disabled = false;
  $('#result').innerHTML = `<span class="muted">已选择 ${file.name}（${(file.size/1024).toFixed(0)} KB），点击"开始识别"。</span>`;
}

$('#run').addEventListener('click', async () => {
  if (!selectedFile) return;
  const btn = $('#run');
  btn.disabled = true; btn.textContent = '识别中…';
  $('#result').innerHTML = '<span class="muted">正在调用模型，请稍候…</span>';
  const form = new FormData();
  form.append('image', selectedFile);
  form.append('variant', $('#variant').value);
  const started = performance.now();
  try {
    const r = await fetch('/api/predict', { method: 'POST', body: form });
    const d = await r.json();
    render(d, (performance.now() - started) / 1000);
  } catch (e) {
    $('#result').innerHTML = `<div class="pill err">请求失败</div><p class="desc">${e}</p>`;
  } finally {
    btn.disabled = false; btn.textContent = '开始识别';
  }
});

function render(d, wallSeconds) {
  const st = d.status || {};
  const parse = d.parse || {};
  const okBadge = d.request_success ? pill('调用成功', 'ok') : pill('调用失败', 'err');
  const schemaBadge = parse.schema_valid === true ? pill('Schema 通过', 'ok')
                     : parse.schema_valid === false ? pill('Schema 未通过', 'err') : '';
  const jsonBadge = parse.json_valid === true ? pill('JSON 有效', 'ok')
                    : parse.json_valid === false ? pill('JSON 无效', 'warn') : '';
  const uncertainBadge = d.uncertain === true ? pill('存在不确定性', 'warn') : '';
  const meta = d.image_meta || {};
  const variant = d.variant || {};

  let html = `<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
      ${okBadge}${schemaBadge}${jsonBadge}${uncertainBadge}</div>`;

  if (d.error_message) {
    html += `<div class="banner err show" style="margin:0 0 12px">${d.error_message}</div>`;
  }

  if (d.posture_cn) {
    html += `<div class="posture">${d.posture_cn} <span class="muted">(${d.posture})</span></div>`;
  } else {
    html += `<div class="posture muted">无有效类别</div>`;
  }
  html += `<p class="desc">${d.description || '<span class="muted">无描述</span>'}</p>`;
  if (d.uncertainty_reason) {
    html += `<p class="desc muted">不确定性原因：${d.uncertainty_reason}</p>`;
  }

  html += `<dl class="kv">
    <dt>提示词方案</dt><dd>${variant.title || variant.key || '-'} · ${(variant.sha256 || '').slice(0, 12)}</dd>
    <dt>调用状态</dt><dd>${st.request_success ? '成功' : '失败'} · HTTP ${st.http_status ?? '-'} · 尝试 ${st.attempt ?? '-'} 次</dd>
    <dt>响应耗时</dt><dd>${st.latency_seconds ?? '-'} 秒（模型） / ${wallSeconds.toFixed(2)} 秒（含上传）</dd>
    <dt>Token</dt><dd>输入 ${st.input_tokens ?? '-'} · 输出 ${st.output_tokens ?? '-'}</dd>
    <dt>图片</dt><dd>原始 ${meta.original || '-'} → 送入 ${meta.sent || '-'}（${meta.bytes_sent ? (meta.bytes_sent/1024).toFixed(0)+' KB' : '-'}，视觉 Token 约 ${meta.estimated_visual_tokens ?? '-'}）</dd>
    <dt>模型</dt><dd>${st.model || '-'}</dd>
    <dt>时间</dt><dd>${d.timestamp || '-'}</dd>
  </dl>`;

  if (parse.warnings && parse.warnings.length) {
    html += `<p class="muted">提示：${parse.warnings.join('；')}</p>`;
  }
  if (d.raw_text) {
    html += `<details style="margin-top:12px"><summary>模型原始返回</summary><pre>${escapeHtml(d.raw_text)}</pre></details>`;
  }
  $('#result').innerHTML = html;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

loadHealth();
loadPrompts();

// ---------------------------------------------------------------- 智能助手
const AGENT_EXAMPLES = [
  '最近一周哪个栏位的猪采食量偏低？P2-007 要不要重点关注？',
  '猪只姿态识别常用的深度学习方法有哪些？请给出出处。',
  '这张图片里的猪是什么姿态？data/samples/dev/lying_lateral/test_pen1_tur_cam2_20250921_051726_0005.jpg'
];

const TOOL_LABELS = {
  all: '全部工具（数据 + 视觉 + 知识库）',
  data: '仅养殖数据',
  rag: '仅知识库检索',
  none: '不用工具（直接回答）'
};

function mdToHtml(text) {
  const lines = escapeHtml(text || '').split('\n');
  let out = '';
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (/^\s*#{1,6}\s+/.test(line)) {
      out += `<h3>${bold(line.replace(/^\s*#{1,6}\s+/, ''))}</h3>`;
    } else if (/^\s*[-*]\s+/.test(line)) {
      out += `<div class="li">• ${bold(line.replace(/^\s*[-*]\s+/, ''))}</div>`;
    } else if (!line.trim()) {
      out += '<div style="height:6px"></div>';
    } else {
      out += `<div>${bold(line)}</div>`;
    }
  }
  return `<div class="answer">${out}</div>`;
}

function bold(s) { return s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>'); }

async function loadAgentInfo() {
  const box = $('#agentExamples');
  AGENT_EXAMPLES.forEach((q) => {
    const el = document.createElement('span');
    el.className = 'chip';
    el.textContent = q.length > 30 ? q.slice(0, 30) + '…' : q;
    el.title = q;
    el.addEventListener('click', () => { $('#agentQ').value = q; });
    box.appendChild(el);
  });

  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    const agent = d.agent || {};
    const select = $('#agentTools');
    (agent.presets || ['all', 'data', 'rag', 'none']).forEach((key) => {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = TOOL_LABELS[key] || key;
      select.appendChild(opt);
    });
    const steps = $('#agentSteps');
    const maxSteps = agent.max_steps || 8;
    [2, 4, 6, 8].filter((n) => n <= maxSteps).forEach((n) => {
      const opt = document.createElement('option');
      opt.value = n;
      opt.textContent = `最多 ${n} 步`;
      if (n === 6) opt.selected = true;
      steps.appendChild(opt);
    });
    $('#agentHint').textContent = `${agent.model || '未配置'} · ${(agent.tools || []).length} 个工具`;
  } catch (e) { /* 自检失败不阻塞界面 */ }
}

$('#agentRun').addEventListener('click', askAgent);
$('#agentQ').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) askAgent();
});

async function askAgent() {
  const question = $('#agentQ').value.trim();
  if (!question) { $('#agentQ').focus(); return; }
  const btn = $('#agentRun');
  btn.disabled = true; btn.textContent = '思考中…';
  $('#agentResult').innerHTML = '<span class="muted">正在规划并调用工具，通常需要 30–120 秒，请勿关闭页面…</span>';
  const started = performance.now();
  try {
    const r = await fetch('/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        tools: $('#agentTools').value,
        max_steps: parseInt($('#agentSteps').value || '6', 10)
      })
    });
    const d = await r.json();
    renderAgent(d, (performance.now() - started) / 1000);
  } catch (e) {
    $('#agentResult').innerHTML =
      `<div class="banner err show" style="margin:0">请求失败：${escapeHtml(String(e))}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = '提问';
  }
}

function renderAgent(d, wallSeconds) {
  const report = d.report || {};
  const flags = report.flags || {};
  const metrics = report.metrics || {};

  let html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">';
  html += d.ok ? pill('已回答', 'ok') : pill('未完成', 'err');
  if (flags.uses_simulated_data) html += pill('含模拟数据', 'warn');
  if (flags.uses_knowledge) html += pill('已检索知识库', '');
  if (flags.uses_vision) html += pill('含视觉判断', '');
  if (flags.degraded) html += pill('存在降级', 'warn');
  html += '</div>';

  if (d.error_message) {
    html += `<div class="banner err show" style="margin:0 0 12px">${escapeHtml(d.error_message)}</div>`;
  }

  html += mdToHtml(report.answer || '');

  const notes = [];
  if (flags.uses_simulated_data) notes.push('本回答使用了<b>模拟养殖数据</b>，不代表任何真实养殖场。');
  if (flags.uses_vision) notes.push('姿态识别为多模态模型判断，非人工标注。');
  (report.degraded || []).forEach((item) => notes.push(escapeHtml(item)));
  if (notes.length) {
    html += `<div class="banner warn show" style="margin:12px 0 0">${notes.join('<br>')}</div>`;
  }

  if ((report.citations || []).length) {
    html += '<h3 style="font-size:13px;color:#93c5fd;margin:14px 0 6px">引用来源</h3>';
    report.citations.forEach((c) => {
      const title = escapeHtml(c.title || c.chunk_id || '');
      const section = c.section ? ` · ${escapeHtml(c.section)}` : '';
      const lic = c.license ? `（${escapeHtml(c.license)}）` : '';
      const link = c.url ? ` <a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">原文</a>` : '';
      html += `<div class="cite">[${c.index}] ${title}${section}${lic}${link}</div>`;
    });
  }

  if ((report.tool_trace || []).length) {
    html += '<details style="margin-top:12px"><summary>工具调用轨迹（' +
            report.tool_trace.length + ' 次）</summary>' +
            '<table class="trace" style="margin-top:8px;width:100%;border-collapse:collapse">' +
            '<tr><th>工具</th><th>参数</th><th>结果</th><th>耗时</th></tr>';
    report.tool_trace.forEach((t) => {
      const args = escapeHtml(JSON.stringify(t.args || {}));
      const mark = t.ok ? '成功' : `<span style="color:#f87171">失败：${escapeHtml(t.error || '')}</span>`;
      html += `<tr><td>${escapeHtml(t.tool)}</td><td style="word-break:break-all">${args}</td>` +
              `<td>${mark}</td><td>${Math.round(t.ms || 0)} ms</td></tr>`;
    });
    html += '</table></details>';
  }

  html += `<p class="muted" style="margin-top:12px">` +
          `结束原因：${escapeHtml(report.stop_reason_cn || '')} · ` +
          `模型 ${metrics.model_calls || 0} 次 / 工具 ${metrics.tool_calls || 0} 次 · ` +
          `tokens ${metrics.input_tokens || 0}+${metrics.output_tokens || 0} · ` +
          `${(metrics.elapsed_seconds || 0)} 秒（含网络请求，页面等待 ${wallSeconds.toFixed(1)} 秒）· ` +
          `模型 ${escapeHtml(d.model || '-')}</p>`;

  if (report.prompt_meta && report.prompt_meta.sha256) {
    html += `<p class="muted">提示词 v${escapeHtml(report.prompt_meta.version || '')} · ` +
            `SHA256 ${escapeHtml(String(report.prompt_meta.sha256).slice(0, 12))}…</p>`;
  }

  $('#agentResult').innerHTML = html;
}

loadAgentInfo();
</script>
</body>
</html>
"""
