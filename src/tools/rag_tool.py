"""知识库检索工具（T10）。

把 :class:`src.rag.retrieve.Retriever` 包成 agent 可调用的工具。两个入口：

* ``search_knowledge`` —— 返回带引用信息的片段（标题、章节、许可、原文偏移）
* ``fetch_passage``    —— 按 chunk_id 取回完整片段，供展开阅读

设计取舍
--------
* **片段只给摘要**：``search_knowledge`` 每条只回 400 字，控制进入提示词的
  token；需要全文时再调 ``fetch_passage``，避免一次塞爆上下文；
* **降级要声明**：向量接口不可用时检索会退回词法，返回值里 ``mode`` /
  ``degraded`` / ``note`` 会如实说明——不能让下游误以为是语义检索的结果；
* **失败不编造**：索引缺失、查询为空都给出明确原因，不返回"看起来像答案"
  的占位内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import load_dotenv
from src.rag.retrieve import RetrievalResult, Retriever
from src.tools.registry import ToolError, ToolRegistry, ToolSpec

#: 片段摘要长度上限（字符）
SNIPPET_CHARS = 400

_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """惰性装配检索器；索引缺失时给出可操作的提示。"""
    global _retriever
    if _retriever is not None:
        return _retriever

    api_key = (load_dotenv().get("ZHIPUAI_API_KEY") or "").strip()
    try:
        _retriever = Retriever(api_key=api_key)
    except FileNotFoundError as exc:
        raise ToolError(
            "知识库索引缺失（{0}）。请先运行 python scripts/build_kb.py 构建。".format(exc))
    except RuntimeError as exc:
        raise ToolError("知识库索引不可用：{0}".format(exc))
    return _retriever


def reset_retriever() -> None:
    """测试用：清空缓存的检索器。"""
    global _retriever
    _retriever = None


def _citation(result: RetrievalResult) -> Dict[str, Any]:
    text = " ".join(result.text.split())
    snippet = text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")
    return {
        "chunk_id": result.chunk_id,
        "title": result.title,
        "section": result.section,
        "url": result.url,
        "license": result.license,
        "score": round(result.score, 4),
        "rank": result.rank,
        "source": result.source,
        "offset": {"start": result.start, "end": result.end},
        "snippet": snippet,
        "chars": len(result.text),
    }


def search_knowledge(query: str, k: int = 5, mode: str = "auto",
                     retriever: Optional[Retriever] = None) -> Dict[str, Any]:
    """检索知识库，返回可引用的片段。"""
    text = (query or "").strip()
    if not text:
        raise ToolError("query 不能为空")

    engine = retriever or get_retriever()
    response = engine.retrieve(text, k=k, mode=mode)

    degraded = response.mode == "dense-fallback-lexical"
    payload: Dict[str, Any] = {
        "query": text,
        "mode": response.mode,
        "degraded": degraded,
        "citations": [_citation(item) for item in response.results],
        "count": len(response.results),
    }
    if degraded:
        payload["note"] = (
            "向量检索不可用（{0}），本次结果来自词法检索，"
            "对中文提问的召回质量明显下降，请谨慎引用。".format(
                (response.fallback_reason or "原因未知")[:120]))
    if not response.results:
        payload["note"] = "知识库中没有找到相关片段，请换一种说法或缩小主题范围。"
    return payload


def fetch_passage(chunk_id: str,
                  retriever: Optional[Retriever] = None) -> Dict[str, Any]:
    """按 chunk_id 取回完整片段与来源信息。"""
    key = (chunk_id or "").strip()
    if not key:
        raise ToolError("chunk_id 不能为空")

    engine = retriever or get_retriever()
    chunk = engine.chunks.get(key)
    if chunk is None:
        raise ToolError("未找到片段 {0}，请先调用 search_knowledge 获取有效 chunk_id".format(key))

    return {
        "chunk_id": key,
        "doc_id": chunk.get("doc_id", ""),
        "title": chunk.get("title", ""),
        "section": chunk.get("section", ""),
        "url": chunk.get("url", ""),
        "license": chunk.get("license", ""),
        "offset": {"start": int(chunk.get("start", 0)),
                   "end": int(chunk.get("end", 0))},
        "text": chunk.get("text", ""),
        "chars": len(chunk.get("text", "")),
    }


def build_rag_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register_all([
        ToolSpec(
            name="search_knowledge",
            description="在项目知识库（论文与项目文档）中检索，返回带出处的片段摘要。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题或关键词"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "mode": {"type": "string", "enum": ["auto", "dense", "lexical", "hybrid"],
                             "default": "auto"},
                },
                "required": ["query"],
            },
            handler=lambda args: search_knowledge(**args),
            returns="citations[]: chunk_id, title, section, url, license, score, snippet",
        ),
        ToolSpec(
            name="fetch_passage",
            description="按 chunk_id 取回知识库片段的完整正文，用于展开引用。",
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string",
                                 "description": "search_knowledge 返回的 chunk_id"},
                },
                "required": ["chunk_id"],
            },
            handler=lambda args: fetch_passage(**args),
            returns="text + doc_id/title/section/url/license/offset",
        ),
    ])
    return registry