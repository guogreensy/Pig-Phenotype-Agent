"""默认工具装配（阶段 B 汇总）。

把数据、视觉、知识库三类工具装进一个注册表，并支持按需裁剪——
评测里要对比"只有 RAG"与"RAG + 工具"，就必须能只装配一部分工具。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.tools.data_tools import build_data_tools
from src.tools.rag_tool import build_rag_tools
from src.tools.registry import ToolRegistry
from src.tools.vision_tool import build_vision_tools


def build_default_registry(log_path: Optional[Path] = None,
                           max_calls: int = 0,
                           include_data: bool = True,
                           include_vision: bool = True,
                           include_rag: bool = True) -> ToolRegistry:
    """装配默认工具集。

    Args:
        log_path: 调用日志落盘路径（JSONL），None 时仅保存在内存。
        max_calls: 总调用次数上限，0 表示不限。
        include_*: 是否装配该类工具（评测消融用）。
    """
    registry = ToolRegistry(log_path=log_path, max_calls=max_calls)
    if include_data:
        build_data_tools(registry)
    if include_vision:
        build_vision_tools(registry)
    if include_rag:
        build_rag_tools(registry)
    return registry