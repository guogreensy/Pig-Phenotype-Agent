"""提示词方案加载与版本管理。

三组提示词必须可追溯（需求 6.2）：内容存为独立文件，运行时计算 SHA256
并写入实验元数据，保证报告中引用的提示词与实际调用完全一致。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from config import PROMPTS_DIR

#: variant key -> (文件名, 中文名称)
VARIANTS: Dict[str, tuple] = {
    "A": ("prompt_a.txt", "A 基础提示词"),
    "B": ("prompt_b.txt", "B 姿态定义提示词"),
    "C": ("prompt_c.txt", "C 定义+结构化输出+不确定性"),
}

VARIANT_ORDER: List[str] = ["A", "B", "C"]


class PromptError(RuntimeError):
    """提示词文件缺失或为空。"""


@dataclass(frozen=True)
class PromptVariant:
    key: str
    title: str
    text: str
    sha256: str
    path: Path

    @property
    def short_sha(self) -> str:
        return self.sha256[:8]


def load_prompt(variant: str, prompts_dir: Path = None) -> PromptVariant:
    """读取指定提示词方案。"""
    key = (variant or "").strip().upper()
    if key not in VARIANTS:
        raise PromptError(
            "未知的提示词方案 {0!r}，可选：{1}".format(variant, "、".join(VARIANT_ORDER))
        )
    filename, title = VARIANTS[key]
    base = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
    path = base / filename
    if not path.is_file():
        raise PromptError("提示词文件不存在：{0}".format(path))
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptError("提示词文件为空：{0}".format(path))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptVariant(key=key, title=title, text=text, sha256=digest, path=path)


def load_all(prompts_dir: Path = None) -> List[PromptVariant]:
    """按 A→B→C 顺序加载全部方案。"""
    return [load_prompt(key, prompts_dir=prompts_dir) for key in VARIANT_ORDER]


def variant_title(key: str) -> str:
    entry = VARIANTS.get((key or "").upper())
    return entry[1] if entry else key
