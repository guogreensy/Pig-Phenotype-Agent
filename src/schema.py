"""模型输出的 Schema 定义与校验。

字段依据需求说明书 5.4 节：

.. code-block:: json

    {"posture": "standing", "description": "...",
     "uncertain": false, "uncertainty_reason": null}

校验原则：
* ``posture`` 必须落在当前类别集合或 ``unknown`` 内，否则整条结果判为非法；
* 类型不符（如 uncertain 是字符串）视为非法，不做"猜测式修复"，只记录原因；
* ``posture == unknown`` 但 ``uncertain == False`` 属于自相矛盾，记为 warning；
* 校验失败**不会**被替换成任何预设答案，调用方必须如实记录失败。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: 规范类别（英文标识）按固定顺序排列，保证混淆矩阵行列稳定。
POSTURES: List[str] = ["standing", "lying_lateral", "lying_other", "sitting"]

#: 模型允许输出的全部取值（含拒答）。
POSTURE_VALUES: List[str] = POSTURES + ["unknown"]

POSTURE_CN: Dict[str, str] = {
    "standing": "站立",
    "lying_lateral": "侧卧",
    "lying_other": "其他卧姿",
    "sitting": "坐姿",
    "unknown": "无法判断",
}

#: 数据集 class_id -> 规范类别（依据 dataset README 的 5 类定义）。
DATASET_CLASS_MAP: Dict[int, str] = {
    0: "lying_lateral",   # Lateral_lying_left
    1: "lying_lateral",   # Lateral_lying_right
    2: "sitting",         # Sitting
    3: "standing",        # Standing
    4: "lying_other",     # Sternal_lying
}

DATASET_CLASS_NAME: Dict[int, str] = {
    0: "Lateral_lying_left",
    1: "Lateral_lying_right",
    2: "Sitting",
    3: "Standing",
    4: "Sternal_lying",
}

#: 常见同义写法 -> 规范类别。模型可能输出中文、缩写或带标点的形式。
_SYNONYMS: Dict[str, str] = {
    # standing
    "standing": "standing", "stand": "standing", "upright": "standing",
    "站立": "standing", "站": "standing", "直立": "standing", "立姿": "standing",
    "站着": "standing", "站立姿态": "standing",
    # lying_lateral
    "lying_lateral": "lying_lateral", "lateral": "lying_lateral",
    "lateral_lying": "lying_lateral", "lateral_lying_left": "lying_lateral",
    "lateral_lying_right": "lying_lateral", "side_lying": "lying_lateral",
    "lying_on_side": "lying_lateral", "side": "lying_lateral",
    "侧卧": "lying_lateral", "侧躺": "lying_lateral", "侧卧姿态": "lying_lateral",
    "侧面躺卧": "lying_lateral", "侧卧在地": "lying_lateral",
    # lying_other
    "lying_other": "lying_other", "sternal": "lying_other",
    "sternal_lying": "lying_other", "chest_down": "lying_other",
    "prone": "lying_other",
    "其他卧姿": "lying_other", "俯卧": "lying_other", "腹卧": "lying_other",
    "胸卧": "lying_other", "趴卧": "lying_other", "趴着": "lying_other",
    # sitting
    "sitting": "sitting", "sit": "sitting", "seated": "sitting",
    "坐姿": "sitting", "坐": "sitting", "半坐": "sitting", "蹲坐": "sitting",
    # unknown
    "unknown": "unknown", "uncertain": "unknown", "undetermined": "unknown",
    "n/a": "unknown", "na": "unknown", "none": "unknown", "null": "unknown",
    "无法判断": "unknown", "无法确定": "unknown", "不确定": "unknown",
    "未知": "unknown", "无法识别": "unknown", "不能判断": "unknown",
}

_STRIP_CHARS = " \t\r\n\"'`。.,，、:：;；!！?？()（）[]【】"


def normalize_posture(value: Any) -> Optional[str]:
    """把模型给出的姿态值归一化为规范类别；无法识别时返回 None。"""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip().strip(_STRIP_CHARS)
    if not text:
        return None
    key = re.sub(r"[\s\-]+", "_", text.lower())
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    # 形如 "standing（站立）"、"侧卧(lying lateral)" 的混合写法：取首个可识别片段
    for token in re.split(r"[\s_,，、/|\\()（）\[\]【】:：]+", key):
        token = token.strip(_STRIP_CHARS)
        if token and token in _SYNONYMS:
            return _SYNONYMS[token]
    return None


def map_dataset_class(class_id: Any) -> Optional[str]:
    """把数据集 class_id 映射为规范类别；越界返回 None。"""
    try:
        return DATASET_CLASS_MAP[int(class_id)]
    except (TypeError, ValueError, KeyError):
        return None


@dataclass
class ValidationResult:
    """一条模型输出的校验结果。"""

    valid: bool
    posture: Optional[str] = None
    description: Optional[str] = None
    uncertain: Optional[bool] = None
    uncertainty_reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        """模型是否拒绝分类（posture 为 unknown）。"""
        return self.posture == "unknown"


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1", "是", "真"):
            return True
        if low in ("false", "no", "0", "否", "假"):
            return False
    return None


def validate_result(data: Any, max_description_chars: int = 500) -> ValidationResult:
    """校验模型返回的字典。

    Args:
        data: 已解析的 JSON 对象。
        max_description_chars: 描述长度上限，超长记为 warning 但不判非法。
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False, errors=["返回值不是 JSON 对象（实际为 {0}）".format(type(data).__name__)]
        )

    posture = normalize_posture(data.get("posture"))
    if posture is None:
        errors.append("posture 缺失或不在合法类别集合内：{0!r}".format(data.get("posture")))

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("description 缺失或不是非空字符串")
        description = None
    elif len(description) > max_description_chars:
        warnings.append(
            "description 长度 {0} 超过上限 {1}，已保留".format(len(description), max_description_chars)
        )

    uncertain = _coerce_bool(data.get("uncertain"))
    if uncertain is None:
        errors.append("uncertain 缺失或不是布尔值：{0!r}".format(data.get("uncertain")))

    reason = data.get("uncertainty_reason")
    if reason is not None and not isinstance(reason, str):
        errors.append("uncertainty_reason 必须是字符串或 null：{0!r}".format(reason))
        reason = None

    if posture == "unknown" and uncertain is False:
        warnings.append("posture=unknown 但 uncertain=false，语义不一致")
    if posture not in (None, "unknown") and uncertain is True:
        warnings.append("给出了具体姿态但 uncertain=true，请结合 uncertainty_reason 解读")

    return ValidationResult(
        valid=not errors,
        posture=posture,
        description=description,
        uncertain=uncertain,
        uncertainty_reason=reason,
        errors=errors,
        warnings=warnings,
    )
