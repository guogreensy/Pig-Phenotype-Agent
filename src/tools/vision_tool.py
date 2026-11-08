"""姿态识别工具（T9）。

把已有的「多模态视觉模型 + 提示词方案 + 输出解析/校验」封装成一个可被
agent 调用的工具，而不是在工具层另写一套调用逻辑——否则两套代码迟早不一致。

安全与失败处理
--------------
* **路径限制**：只允许读取 ``data/`` 下的图片，避免模型（或用户）诱导工具去
  读项目外的任意文件；
* **失败即失败**：模型调用失败、解析失败、校验失败都返回真实原因，
  绝不返回一个"看起来成功"的兜底姿态；
* **可追溯**：返回值带模型名、提示词变体与其 SHA256、耗时、token 用量，
  报告里引用的数字可以直接对回调用记录。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from config import DATA_DIR, get_settings
from config import ROOT as PROJECT_ROOT
from src.image_utils import ImageValidationError, preprocess_image
from src.output_parser import parse_model_output
from src.prompts import PromptError, load_prompt, variant_title
from src.schema import POSTURE_CN, validate_result
from src.tools.registry import ToolError, ToolRegistry, ToolSpec
from src.vision_client import VisionClient

#: 允许读取图片的根目录（防止越权读取任意文件）
ALLOWED_ROOT = DATA_DIR

#: 支持的图片扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_image(image_path: str) -> Path:
    """把工具参数解析成受信任的本地路径。"""
    if not image_path or not str(image_path).strip():
        raise ToolError("image_path 不能为空")

    raw = Path(str(image_path).strip())
    candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        raise ToolError("图片不存在：{0}".format(image_path))

    # 先判越界再判存在：否则会泄露项目外路径是否存在
    root = ALLOWED_ROOT.resolve()
    if root != resolved and root not in resolved.parents:
        raise ToolError(
            "只允许读取 {0} 目录下的图片，拒绝访问：{1}".format(
                ALLOWED_ROOT.name + "/", image_path))
    if not resolved.is_file():
        raise ToolError("图片不存在：{0}".format(image_path))
    if resolved.suffix.lower() not in IMAGE_EXTS:
        raise ToolError("不支持的图片格式 {0}，可用：{1}".format(
            resolved.suffix, "、".join(sorted(IMAGE_EXTS))))
    return resolved


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def analyze_pig_posture(image_path: str, variant: str = "C",
                        client: Optional[VisionClient] = None) -> Dict[str, Any]:
    """对单张图片做姿态识别，返回结构化结果。

    Args:
        image_path: ``data/`` 下的图片路径（相对项目根或绝对路径）。
        variant: 提示词方案 A/B/C。
        client: 可注入的视觉客户端（测试用）。
    """
    path = _resolve_image(image_path)

    try:
        prompt = load_prompt(variant)
    except PromptError as exc:
        raise ToolError(str(exc))

    try:
        processed = preprocess_image(
            path.read_bytes(),
            max_side=get_settings().image_max_side,
            jpeg_quality=get_settings().jpeg_quality,
        )
    except ImageValidationError as exc:
        raise ToolError("图片不可用：{0}".format(exc))

    active_client = client or VisionClient(get_settings())
    response = active_client.call(processed.data_url, prompt.text)
    if not response.ok:
        raise ToolError("模型调用失败：{0}".format(response.error or "未知错误"))

    parsed = parse_model_output(response.raw_text)
    payload: Dict[str, Any] = {
        "source": "model",
        "image": {
            "path": _relative(path),
            "width": processed.width,
            "height": processed.height,
            "original_width": processed.original_width,
            "original_height": processed.original_height,
        },
        "model": response.model or get_settings().model,
        "prompt": {
            "variant": prompt.key,
            "title": variant_title(prompt.key),
            "sha256": prompt.sha256,
        },
        "parse": {"ok": parsed.ok, "method": parsed.method},
        "latency_seconds": round(response.latency_seconds, 2),
        "tokens": {"input": response.input_tokens, "output": response.output_tokens},
    }

    if not parsed.ok or parsed.data is None:
        payload.update({
            "valid": False,
            "errors": [parsed.error or "模型输出无法解析为 JSON"],
            "raw_excerpt": response.raw_text[:300],
        })
        return payload

    validation = validate_result(parsed.data)
    payload.update({
        "valid": validation.valid,
        "posture": validation.posture,
        "posture_cn": POSTURE_CN.get(validation.posture or "", None),
        "description": validation.description,
        "uncertain": validation.uncertain,
        "uncertainty_reason": validation.uncertainty_reason,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
    })
    return payload


def list_sample_images(label: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """列出可分析的示例图片。

    目录结构为 ``data/samples/<split>/<label>/*.jpg``，因此按**父目录名**
    匹配类别，而不是假定标签就在 ``samples/`` 下一层。
    """
    root = DATA_DIR / "samples"
    if not root.is_dir():
        raise ToolError("示例图片目录不存在：{0}".format(root))

    all_images = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    labels = sorted({path.parent.name for path in all_images})

    if label and label not in labels:
        raise ToolError("未知类别 {0!r}，可选：{1}".format(label, "、".join(labels)))

    selected = [path for path in all_images if not label or path.parent.name == label]
    images = [{
        "path": _relative(path),
        "category": path.parent.name,
        "split": path.parent.parent.name,
        "name": path.name,
    } for path in selected[:limit]]

    return {
        "source": "local_dataset",
        "root": _relative(root),
        "labels": labels,
        "matched": len(selected),
        "count": len(images),
        "images": images,
        "note": "示例图片来自公开数据集，用于演示与测试，不代表真实生产图像。",
    }


def build_vision_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register_all([
        ToolSpec(
            name="analyze_pig_posture",
            description="对 data/ 下的一张猪只图片做姿态识别，返回姿态类别、描述与不确定性。",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string",
                                   "description": "图片路径，须位于 data/ 目录内"},
                    "variant": {"type": "string", "enum": ["A", "B", "C"],
                                "default": "C", "description": "提示词方案"},
                },
                "required": ["image_path"],
            },
            handler=lambda args: analyze_pig_posture(**args),
            returns="posture, posture_cn, description, uncertain, valid, model, prompt",
        ),
        ToolSpec(
            name="list_sample_images",
            description="列出可用于分析的示例图片，可按类别筛选。",
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string",
                              "description": "类别目录名，如 standing / lying_lateral"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "required": [],
            },
            handler=lambda args: list_sample_images(**args),
            returns="images[]: path, category, name",
        ),
    ])
    return registry