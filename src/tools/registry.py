"""工具注册表（T6）。

把"模型能调用什么工具"收敛成一份可校验、可审计的注册表。

为什么需要注册表而不是直接 ``if/elif`` 分发
--------------------------------------------
1. **schema 就是契约**：同一份 JSON Schema 既用于校验入参，又原样交给大模型做
   function calling，两边不会漂移；
2. **失败不外抛**：任何工具异常都收敛成 ``ToolResult(ok=False, error=...)``，
   调度器据此继续对话，而不是整轮崩掉；
3. **可审计**：每次调用（含失败）都进调用日志，报告里的"工具调用次数/成功率"
   来自日志而不是事后回忆。

参数校验是**白名单式**的：未声明的参数直接拒绝（除非显式 ``additionalProperties``），
避免模型臆造参数却被静默忽略。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: JSON Schema 类型 -> Python 类型（bool 需先于 int 判断，因为 bool 是 int 子类）
_TYPE_CHECKS: Dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


class ToolError(ValueError):
    """工具参数非法或调用失败，消息为面向用户的中文说明。"""


@dataclass(frozen=True)
class ToolSpec:
    """一个可注册工具。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]
    #: 返回结构的简述，写进系统提示词帮助模型判断是否调用
    returns: str = ""

    def to_schema(self) -> Dict[str, Any]:
        """转成 OpenAI function calling 的工具描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """一次工具调用的结果。"""

    ok: bool
    tool: str
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
        }

    def to_message(self, max_chars: int = 4000) -> str:
        """转成给模型看的工具返回文本（超长截断并显式说明）。"""
        payload = self.to_dict()
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > max_chars:
            return text[:max_chars] + "…（结果过长已截断）"
        return text


# ------------------------------------------------------------------ 参数校验
def _validate_value(name: str, value: Any, spec: Dict[str, Any]) -> Optional[str]:
    """校验单个参数，返回错误描述或 None。"""
    expected = spec.get("type")
    if expected:
        check = _TYPE_CHECKS.get(expected)
        if check is None:
            return "参数 {0} 的 schema 类型 {1!r} 不受支持".format(name, expected)
        if not check(value):
            return "参数 {0} 应为 {1}，实际为 {2}".format(
                name, expected, type(value).__name__)

    if "enum" in spec and value not in spec["enum"]:
        return "参数 {0} 只能是 {1}，实际为 {2!r}".format(
            name, "、".join(map(str, spec["enum"])), value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and value < minimum:
            return "参数 {0} 不能小于 {1}（实际 {2}）".format(name, minimum, value)
        if maximum is not None and value > maximum:
            return "参数 {0} 不能大于 {1}（实际 {2}）".format(name, maximum, value)

    if isinstance(value, list):
        if "minItems" in spec and len(value) < spec["minItems"]:
            return "参数 {0} 至少需要 {1} 项".format(name, spec["minItems"])
        if "maxItems" in spec and len(value) > spec["maxItems"]:
            return "参数 {0} 最多 {1} 项".format(name, spec["maxItems"])
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for index, item in enumerate(value):
                error = _validate_value("{0}[{1}]".format(name, index), item, item_spec)
                if error:
                    return error

    return None


def validate_args(parameters: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """校验并补全默认值，返回新字典；不合法时抛 :class:`ToolError`。"""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("参数必须是 JSON 对象，实际为 {0}".format(type(args).__name__))

    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    allow_extra = bool(parameters.get("additionalProperties", False))

    errors: List[str] = []
    cleaned: Dict[str, Any] = {}

    for key in required:
        if key not in args or args[key] is None:
            errors.append("缺少必填参数 {0}".format(key))

    for key, value in args.items():
        spec = properties.get(key)
        if spec is None:
            if allow_extra:
                cleaned[key] = value
            else:
                errors.append("不支持的参数 {0}".format(key))
            continue
        error = _validate_value(key, value, spec)
        if error:
            errors.append(error)
        else:
            cleaned[key] = value

    for key, spec in properties.items():
        if key not in cleaned and "default" in spec:
            cleaned[key] = spec["default"]

    if errors:
        raise ToolError("；".join(errors))
    return cleaned


# ------------------------------------------------------------------ 注册表
class ToolRegistry:
    """工具集合 + 调用日志。"""

    def __init__(self, log_path: Optional[Path] = None, max_calls: int = 0):
        self._tools: Dict[str, ToolSpec] = {}
        self._log: List[Dict[str, Any]] = []
        self.log_path = Path(log_path) if log_path else None
        #: >0 时限制总调用次数，用于兜住 agent 循环的成本
        self.max_calls = max_calls
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 注册
    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("工具名不能为空")
        if spec.name in self._tools:
            raise ValueError("工具 {0} 已注册".format(spec.name))
        if not isinstance(spec.parameters, dict):
            raise ValueError("工具 {0} 的 parameters 必须是 JSON Schema 对象".format(spec.name))
        self._tools[spec.name] = spec

    def register_all(self, specs: List[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> ToolSpec:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError("未注册的工具：{0}（可用：{1}）".format(
                name, "、".join(self.names()) or "无"))
        return spec

    def names(self) -> List[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> List[Dict[str, Any]]:
        """全部工具的 function calling 描述。"""
        return [spec.to_schema() for spec in self._tools.values()]

    def describe(self) -> str:
        """给系统提示词用的一行式工具清单。"""
        lines = []
        for spec in self._tools.values():
            params = ", ".join((spec.parameters.get("properties") or {}).keys()) or "无参数"
            lines.append("- {0}({1})：{2}".format(spec.name, params, spec.description))
        return "\n".join(lines)

    # ---------------------------------------------------------------- 调用
    def call(self, name: str, args: Optional[Dict[str, Any]] = None,
             persist: bool = True) -> ToolResult:
        """执行工具。任何异常都收敛为 ``ok=False``，不向外抛。"""
        started = time.monotonic()
        raw_args = args if isinstance(args, dict) else {}

        if self.max_calls and len(self._log) >= self.max_calls:
            result = ToolResult(
                ok=False, tool=name, args=raw_args,
                error="工具调用次数已达上限 {0}，本轮不再执行".format(self.max_calls))
            self._record(result)
            return result

        try:
            spec = self.get(name)
            cleaned = validate_args(spec.parameters, args)
        except ToolError as exc:
            result = ToolResult(ok=False, tool=name, args=raw_args, error=str(exc),
                                duration_ms=(time.monotonic() - started) * 1000)
            self._record(result)
            return result

        try:
            data = spec.handler(cleaned)
            result = ToolResult(ok=True, tool=name, data=data, args=cleaned,
                                duration_ms=(time.monotonic() - started) * 1000)
        except Exception as exc:  # noqa: BLE001 —— 工具失败不能拖垮整轮对话
            result = ToolResult(ok=False, tool=name, args=cleaned,
                                error="{0}: {1}".format(exc.__class__.__name__, exc),
                                duration_ms=(time.monotonic() - started) * 1000)
        self._record(result)
        return result

    # ---------------------------------------------------------------- 日志
    def _record(self, result: ToolResult) -> None:
        entry = {
            "seq": len(self._log) + 1,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "tool": result.tool,
            "args": result.args,
            "ok": result.ok,
            "error": result.error,
            "duration_ms": round(result.duration_ms, 1),
        }
        self._log.append(entry)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    @property
    def log(self) -> List[Dict[str, Any]]:
        return list(self._log)

    def stats(self) -> Dict[str, Any]:
        """调用统计（报告用）。"""
        total = len(self._log)
        failed = sum(1 for entry in self._log if not entry["ok"])
        by_tool: Dict[str, int] = {}
        for entry in self._log:
            by_tool[entry["tool"]] = by_tool.get(entry["tool"], 0) + 1
        return {
            "calls": total,
            "failed": failed,
            "success_rate": round((total - failed) / total, 4) if total else None,
            "by_tool": by_tool,
        }

    def reset_log(self) -> None:
        self._log.clear()