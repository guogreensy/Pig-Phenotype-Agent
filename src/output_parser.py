"""从模型原始文本中提取结构化结果。

大模型不保证只输出 JSON，常见情况与对应策略：

===========================  ==========================================
情况                          策略
===========================  ==========================================
纯 JSON                       ``json.loads`` 直接解析
```json 围栏包裹              剥掉围栏后解析
前后夹杂解释文字               用括号配对扫描截取第一个完整对象
单引号 / 尾随逗号 / 智能引号    轻量修补后解析
字段被写散在自然语言里          正则逐字段抽取（标记 json_valid=False）
完全无法解析                   返回 failed，**不编造任何结果**
===========================  ==========================================

``json_valid`` 只表示"成功解析出 JSON 对象"，Schema 合法性由
:func:`src.schema.validate_result` 单独判定，两者分开统计。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.DOTALL | re.IGNORECASE)
_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\uff02": '"', "\u3002": ".",  # 全角双引号（句号仅用于兜底，不参与 JSON）
}
#: 自由文本里出现的规范类别（用于 A/B 组这类非 JSON 输出）。
#: 模型常把 lying_lateral 写成 "lying lateral"/"lying-lateral"，因此允许空格与连字符。
_POSTURE_TOKEN_RE = re.compile(
    r"\b(standing|sitting|unknown|lying[\s_\-]?lateral|lying[\s_\-]?other)\b",
    re.IGNORECASE,
)


def _canon_posture(token: str) -> str:
    """把匹配到的类别词统一成规范写法。"""
    return re.sub(r"[\s\-]+", "_", token.strip().lower())
_FIELD_PATTERNS = {
    "posture": [
        r'"posture"\s*:\s*"([^"]*)"',
        r"'posture'\s*:\s*'([^']*)'",
        r"posture\s*[:：]\s*([A-Za-z_\u4e00-\u9fa5]+)",
        r"姿态类别\s*[:：]\s*([A-Za-z_\u4e00-\u9fa5]+)",
        r"姿态\s*[:：]\s*([A-Za-z_\u4e00-\u9fa5]+)",
        r"类别\s*[:：]\s*([A-Za-z_\u4e00-\u9fa5]+)",
    ],
    "description": [
        r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r"'description'\s*:\s*'((?:[^'\\]|\\.)*)'",
        r"description\s*[:：]\s*([^,}\n]+)",
    ],
    "uncertain": [
        r'"uncertain"\s*:\s*(true|false|True|False)',
        r"uncertain\s*[:：]\s*(true|false|True|False)",
    ],
    "uncertainty_reason": [
        r'"uncertainty_reason"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'"uncertainty_reason"\s*:\s*(null|None)',
    ],
}


@dataclass
class ParseResult:
    """解析结果。"""

    data: Optional[Dict[str, Any]]
    json_valid: bool
    method: str  # direct | fenced | extracted | repaired | regex | failed
    raw_text: str
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.data is not None


def strip_code_fence(text: str) -> Optional[str]:
    """取出第一个 ``` 围栏内的内容。"""
    match = _FENCE_RE.search(text)
    if not match:
        return None
    inner = match.group(1).strip()
    return inner or None


def strip_reasoning(text: str) -> Tuple[str, bool]:
    """剥离 ``<think>…</think>`` 推理过程，返回 (文本, 是否剥离过)。

    部分思考型视觉模型会把推理过程写进 ``<think>`` 块。推理里常出现与结论
    相反或半截的类别词，直接解析会污染结果，因此必须先剥离。
    """
    if not text:
        return text, False
    cleaned, count = _THINK_RE.subn("", text)
    return (cleaned.strip(), count > 0)


def extract_answer(text: str) -> Optional[str]:
    """取出 ``<answer>…</answer>`` 的内容（没有则返回 None）。"""
    match = _ANSWER_RE.search(text or "")
    if not match:
        return None
    inner = match.group(1).strip()
    return inner or None


def extract_freeform(text: str) -> Optional[Dict[str, Any]]:
    """从自然语言回答中抽取姿态类别与描述（A/B 组这类非 JSON 输出）。

    只做"抄写"，不做语义推断：找不到规范类别词就返回 None，绝不猜测。
    """
    if not text:
        return None
    match = _POSTURE_TOKEN_RE.search(text)
    if not match:
        return None
    posture = _canon_posture(match.group(1))
    tail = text[match.end():]
    # 先去掉紧跟类别的"（中文名）"括注，再剥掉分隔符；不吞掉正文
    tail = re.sub(r"^\s*[（(][^）)]{0,12}[）)]", "", tail)
    tail = re.sub(r"^[\s|｜:：，,。.、\-—)）\]】]+", "", tail)
    description = tail.strip()
    if len(description) > 500:
        description = description[:500]
    return {"posture": posture, "description": description}


def find_json_object(text: str) -> Optional[str]:
    """用括号配对扫描出第一个完整的 JSON 对象（能正确处理字符串内的括号）。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        quote: Optional[str] = None
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in ('"', "'"):
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def light_repair(text: str) -> str:
    """对常见的不规范 JSON 做轻量修补。

    只处理不会引入歧义的改法：智能引号、尾随逗号、Python 字面量。
    单引号转双引号有风险（正文里可能含撇号），因此放在最后一档，
    且仅在修补后能成功解析时才会被采用。
    """
    fixed = text
    for src, dst in _SMART_QUOTES.items():
        fixed = fixed.replace(src, dst)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    return fixed


def _try_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _single_quote_to_double(text: str) -> str:
    """把作为键/值定界符的单引号换成双引号。

    仅换 ``'key':`` 与 ``: 'value'`` 两种形态，降低误伤撇号的概率。
    """
    fixed = re.sub(r"'([A-Za-z_][A-Za-z0-9_]*)'\s*:", r'"\1":', text)
    fixed = re.sub(r":\s*'((?:[^'\\]|\\.)*)'", r': "\1"', fixed)
    return fixed


def regex_extract(text: str) -> Dict[str, Any]:
    """从散文中逐字段抽取，作为兜底。

    ``posture`` 只有在捕获值确实是规范类别时才接受，避免把
    "姿态类别：这只猪的四肢是否支撑"这类推理残句当成类别。
    """
    found: Dict[str, Any] = {}
    for name, patterns in _FIELD_PATTERNS.items():
        accepted = False
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                value = match.group(1).strip()
                if name == "posture":
                    canonical = _canon_posture(value)
                    if not re.fullmatch(
                        r"(standing|sitting|unknown|lying_lateral|lying_other)", canonical
                    ):
                        continue  # 不像类别，换下一个匹配/模式
                    found[name] = canonical
                elif name == "uncertain":
                    found[name] = value.lower() == "true"
                elif name == "uncertainty_reason" and value.lower() in ("null", "none"):
                    found[name] = None
                else:
                    found[name] = value
                accepted = True
                break
            if accepted:
                break
    return found


def parse_model_output(text: Optional[str]) -> ParseResult:
    """把模型原始文本解析为字典，并标注解析方式。"""
    if not isinstance(text, str) or not text.strip():
        return ParseResult(
            data=None, json_valid=False, method="failed", raw_text=text or "",
            error="模型返回内容为空",
        )

    raw = text.strip()
    notes: List[str] = []

    # 0) 剥离 <think> 推理过程，优先使用 <answer> 中的结论
    cleaned, had_reasoning = strip_reasoning(raw)
    if had_reasoning:
        notes.append("已剥离 <think> 推理过程")
    answer = extract_answer(cleaned)
    if answer:
        cleaned = answer
        notes.append("已提取 <answer> 内容")

    # 1) 直接解析
    loaded = _try_loads(cleaned)
    if loaded is not None:
        return ParseResult(data=loaded, json_valid=True, method="direct", raw_text=raw, notes=notes)

    # 2) 围栏内解析
    fenced = strip_code_fence(cleaned)
    if fenced:
        loaded = _try_loads(fenced)
        if loaded is not None:
            return ParseResult(data=loaded, json_valid=True, method="fenced", raw_text=raw, notes=notes)

    # 3) 括号配对截取
    snippet = find_json_object(fenced or cleaned)
    if snippet:
        loaded = _try_loads(snippet)
        if loaded is not None:
            notes.append("从夹杂文本中截取 JSON 对象")
            return ParseResult(data=loaded, json_valid=True, method="extracted",
                               raw_text=raw, notes=notes)

        # 4) 轻量修补后再解析
        repaired = light_repair(snippet)
        loaded = _try_loads(repaired)
        if loaded is not None:
            notes.append("修补尾随逗号/引号等后解析成功")
            return ParseResult(data=loaded, json_valid=True, method="repaired",
                               raw_text=raw, notes=notes)

        repaired2 = light_repair(_single_quote_to_double(snippet))
        loaded = _try_loads(repaired2)
        if loaded is not None:
            notes.append("单引号转双引号后解析成功")
            return ParseResult(data=loaded, json_valid=True, method="repaired",
                               raw_text=raw, notes=notes)

    # 5) 正则逐字段抽取（要求 posture 命中规范类别）
    partial = regex_extract(cleaned)
    if partial.get("posture"):
        missing = [k for k in ("description", "uncertain") if k not in partial]
        notes.append("模型未返回合法 JSON，已从文本中抽取部分字段")
        if missing:
            notes.append("缺失字段：{0}".format("、".join(missing)))
        return ParseResult(
            data=partial, json_valid=False, method="regex", raw_text=raw,
            error="模型未返回合法 JSON，已从文本中抽取部分字段",
            notes=notes,
        )

    # 6) 自由文本兜底：只抄写出现的规范类别词，不做语义猜测
    freeform = extract_freeform(cleaned)
    if freeform:
        notes.append("自然语言回答，仅抽取类别与描述")
        return ParseResult(
            data=freeform, json_valid=False, method="freeform", raw_text=raw,
            error="模型以自然语言作答，未返回 JSON",
            notes=notes,
        )

    # 7) 只有零散字段、没有可确认类别
    if partial:
        notes.append("只抽到部分字段，缺少可确认的姿态类别")
        return ParseResult(
            data=partial, json_valid=False, method="regex", raw_text=raw,
            error="未能从回答中确定姿态类别",
            notes=notes,
        )

    return ParseResult(
        data=None, json_valid=False, method="failed", raw_text=raw,
        error="无法从模型返回中解析出任何结构化字段",
        notes=notes,
    )


def describe_parse(result: ParseResult) -> str:
    """给日志用的简短描述。"""
    return "method={0} json_valid={1} error={2}".format(
        result.method, result.json_valid, result.error or "-"
    )
