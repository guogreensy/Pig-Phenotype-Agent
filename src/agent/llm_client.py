"""支持工具调用的对话客户端（T11）。

为什么不能复用 :class:`src.vision_client.VisionClient`
------------------------------------------------------
视觉客户端是"一张图 + 一段提示词 → 一段文本"，它丢掉了 ``tool_calls``
字段。工具调用需要保留完整的 assistant 消息（含 ``tool_calls``），并且要
把 ``role="tool"`` 的结果按 ``tool_call_id`` 回填。硬塞进视觉客户端只会
让两边都变形，所以这里单独实现，但**复用**了视觉客户端的限流、退避与错误
提取，保证两种调用的失败行为一致。

模型选择
--------
视觉模型 ``glm-4.6v-flash`` 不支持 ``tool_calls``；工具调用改用
``glm-4.5-flash``（原生 ``tool_calls`` 已实测可用），并显式关闭思考模式以
降低延迟与费用。模型名可用 ``PIG_TOOL_MODEL`` 覆盖。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from config import get_env, get_settings
from src.vision_client import (
    RETRYABLE_STATUS,
    RateLimiter,
    _error_message,
    _extract_text,
    _extract_usage,
)

#: 默认工具调用模型
DEFAULT_TOOL_MODEL = "glm-4.5-flash"

#: 覆盖模型名的环境变量
TOOL_MODEL_ENV = "PIG_TOOL_MODEL"


@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""

    id: str
    name: str
    arguments_raw: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "parse_error": self.parse_error,
        }


@dataclass
class ChatResponse:
    """一次对话调用的完整记录（失败也返回，不抛异常）。"""

    ok: bool
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    message: Dict[str, Any] = field(default_factory=dict)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_seconds: float = 0.0
    http_status: Optional[int] = None
    error: Optional[str] = None
    attempts: int = 1
    model: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def parse_tool_calls(message: Any) -> List[ToolCall]:
    """从 assistant 消息里解析 ``tool_calls``。

    参数是模型生成的字符串，可能不是合法 JSON。这里**不抛异常**，
    而是把错误记在 ``parse_error`` 上，交给执行器回传给模型修正——
    这比直接让整轮失败更有用，也更接近真实使用。
    """
    if not isinstance(message, dict):
        return []
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []

    calls: List[ToolCall] = []
    for index, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        name = str(function.get("name") or "").strip()
        raw_args = function.get("arguments")
        if isinstance(raw_args, dict):
            # 少数实现直接给对象而非字符串
            calls.append(ToolCall(id=str(item.get("id") or "call_{0}".format(index)),
                                  name=name, arguments_raw=json.dumps(raw_args, ensure_ascii=False),
                                  arguments=raw_args))
            continue
        raw_text = raw_args if isinstance(raw_args, str) else ""
        call = ToolCall(id=str(item.get("id") or "call_{0}".format(index)),
                        name=name, arguments_raw=raw_text)
        if not raw_text.strip():
            call.arguments = {}
        else:
            try:
                parsed = json.loads(raw_text)
            except ValueError as exc:
                call.parse_error = "参数不是合法 JSON：{0}".format(exc)
                call.arguments = {}
            else:
                if isinstance(parsed, dict):
                    call.arguments = parsed
                else:
                    call.parse_error = "参数应为 JSON 对象，实际为 {0}".format(
                        type(parsed).__name__)
                    call.arguments = {}
        calls.append(call)
    return calls


class ChatClient:
    """OpenAI 兼容的对话客户端，支持 function calling。"""

    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout_seconds: float = 90.0, max_retries: int = 3,
                 qps: float = 1.0, temperature: float = 0.0,
                 max_tokens: int = 1024,
                 session: Optional[requests.Session] = None,
                 limiter: Optional[RateLimiter] = None,
                 sleep=time.sleep):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._session = session or requests.Session()
        self._limiter = limiter if limiter is not None else RateLimiter(qps)
        self._sleep = sleep
        # 强制思考模型不接受 thinking 关闭参数
        self._extra_body: Dict[str, Any] = {}
        if "thinking" not in model:
            self._extra_body = {"thinking": {"type": "disabled"}}

    # ---------------------------------------------------------------- payload
    def build_payload(self, messages: List[Dict[str, Any]],
                      tools: Optional[List[Dict[str, Any]]] = None,
                      tool_choice: Optional[str] = None,
                      include_extra: bool = True,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if include_extra and self._extra_body:
            payload.update(self._extra_body)
        return payload

    # ------------------------------------------------------------------- call
    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             tool_choice: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> ChatResponse:
        """发起一次对话。失败返回 ``ok=False``，不抛异常。"""
        if not self.api_key:
            return ChatResponse(
                ok=False, model=self.model,
                error="未配置 API 密钥，无法调用对话模型。")

        notes: List[str] = []
        include_extra = True
        last_error = "未知错误"

        for attempt in range(1, self.max_retries + 1):
            self._limiter.acquire()
            payload = self.build_payload(messages, tools, tool_choice,
                                         include_extra, temperature, max_tokens)
            started = time.monotonic()
            try:
                response = self._session.post(
                    self.base_url,
                    headers={"Authorization": "Bearer {0}".format(self.api_key),
                             "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.Timeout:
                last_error = "请求超时（>{0:g}s）".format(self.timeout_seconds)
                if attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                return ChatResponse(ok=False, error=last_error, attempts=attempt,
                                    model=self.model, notes=notes)
            except requests.RequestException as exc:
                last_error = "网络请求失败：{0}".format(exc.__class__.__name__)
                if attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                return ChatResponse(ok=False, error=last_error, attempts=attempt,
                                    model=self.model, notes=notes)

            elapsed = time.monotonic() - started
            status = response.status_code

            if status == 200:
                try:
                    body = response.json()
                except ValueError:
                    return ChatResponse(ok=False, http_status=status,
                                        latency_seconds=elapsed,
                                        error="服务端返回的不是合法 JSON",
                                        attempts=attempt, model=self.model, notes=notes)
                choices = body.get("choices") or []
                message = choices[0].get("message") if choices else {}
                if not isinstance(message, dict):
                    message = {}
                text = _extract_text(message)
                calls = parse_tool_calls(message)
                input_tokens, output_tokens = _extract_usage(body)

                if not text.strip() and not calls:
                    return ChatResponse(ok=False, http_status=status,
                                        latency_seconds=elapsed,
                                        error="模型既没有返回文本也没有请求工具",
                                        attempts=attempt, model=self.model, notes=notes,
                                        input_tokens=input_tokens,
                                        output_tokens=output_tokens)
                return ChatResponse(
                    ok=True, text=text, tool_calls=calls, message=message,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    latency_seconds=elapsed, http_status=status,
                    attempts=attempt, model=self.model, notes=notes,
                )

            # 服务端拒绝 thinking 参数：去掉后重试
            if status == 400 and include_extra and self._extra_body:
                detail = _error_message(response).lower()
                if "thinking" in detail or "unknown" in detail or "invalid" in detail:
                    notes.append("服务端不接受 thinking 参数，已自动去掉后重试")
                    include_extra = False
                    continue

            last_error = "HTTP {0}：{1}".format(status, _error_message(response))
            if status in RETRYABLE_STATUS and attempt < self.max_retries:
                self._backoff(attempt, response.headers.get("Retry-After"), status)
                continue
            hint = ""
            if status in (401, 403):
                hint = "（密钥无效或无调用权限）"
            elif status == 429:
                hint = "（触发限流，可降低 PIG_QPS）"
            return ChatResponse(ok=False, http_status=status, latency_seconds=elapsed,
                                error=last_error + hint, attempts=attempt,
                                model=self.model, notes=notes)

        return ChatResponse(ok=False, error=last_error, attempts=self.max_retries,
                            model=self.model, notes=notes)

    # ---------------------------------------------------------------- helpers
    def _backoff(self, attempt: int, retry_after: Optional[str] = None,
                 status: Optional[int] = None) -> None:
        base = 5.0 * attempt if status == 429 else float(2 ** (attempt - 1))
        delay = min(45.0, base) + random.uniform(0.0, 1.5)
        if retry_after:
            try:
                delay = max(delay, min(60.0, float(retry_after)))
            except (TypeError, ValueError):
                pass
        self._sleep(delay)


def tool_model_name() -> str:
    """当前使用的工具调用模型名。"""
    return get_env(TOOL_MODEL_ENV, DEFAULT_TOOL_MODEL) or DEFAULT_TOOL_MODEL


def build_chat_client(model: Optional[str] = None) -> ChatClient:
    """按项目配置构造对话客户端（默认用工具调用模型）。"""
    settings = get_settings()
    return ChatClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=model or tool_model_name(),
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        qps=settings.qps,
        temperature=settings.temperature,
        max_tokens=max(settings.max_tokens, 1024),
    )