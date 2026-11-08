"""知识库检索工具测试（T10）。用假检索器，不加载真实索引、不调用接口。"""

from __future__ import annotations

import unittest

from src.rag.retrieve import RetrievalResponse, RetrievalResult
from src.tools.rag_tool import (
    SNIPPET_CHARS,
    build_rag_tools,
    fetch_passage,
    search_knowledge,
)
from src.tools.registry import ToolError, ToolRegistry


def _result(chunk_id="pmc_1#0001", text=None, score=0.62, rank=1, source="dense"):
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id="pmc_1",
        title="Deep learning for pig posture recognition",
        section="Results",
        url="https://example.org/pmc_1",
        license="CC BY 4.0",
        start=1200,
        end=1900,
        text=text or "本文提出一种基于深度学习的猪只姿态识别方法，准确率达到 0.92。",
        score=score,
        source=source,
        rank=rank,
    )


class FakeRetriever:
    def __init__(self, response, chunks=None):
        self.response = response
        self.calls = []
        self.chunks = chunks or {}

    def retrieve(self, query, k=5, mode="auto"):
        self.calls.append({"query": query, "k": k, "mode": mode})
        return self.response


class TestSearchKnowledge(unittest.TestCase):
    def test_returns_citations(self):
        engine = FakeRetriever(RetrievalResponse(results=[_result()], mode="dense"))
        payload = search_knowledge("猪只姿态识别", retriever=engine)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["mode"], "dense")
        self.assertFalse(payload["degraded"])
        citation = payload["citations"][0]
        self.assertEqual(citation["chunk_id"], "pmc_1#0001")
        self.assertEqual(citation["license"], "CC BY 4.0")
        self.assertEqual(citation["offset"], {"start": 1200, "end": 1900})
        self.assertIn("姿态识别", citation["snippet"])

    def test_passes_k_and_mode(self):
        engine = FakeRetriever(RetrievalResponse(results=[], mode="dense"))
        search_knowledge("q", k=3, mode="hybrid", retriever=engine)
        self.assertEqual(engine.calls[0], {"query": "q", "k": 3, "mode": "hybrid"})

    def test_snippet_truncated(self):
        long_text = "猪" * (SNIPPET_CHARS + 200)
        engine = FakeRetriever(RetrievalResponse(results=[_result(text=long_text)], mode="dense"))
        payload = search_knowledge("q", retriever=engine)
        citation = payload["citations"][0]
        self.assertEqual(len(citation["snippet"]), SNIPPET_CHARS + 1)  # 含省略号
        self.assertTrue(citation["snippet"].endswith("…"))
        self.assertEqual(citation["chars"], len(long_text))

    def test_degraded_mode_disclosed(self):
        engine = FakeRetriever(RetrievalResponse(
            results=[_result(source="lexical")],
            mode="dense-fallback-lexical",
            fallback_reason="ZHIPUAI_API_KEY 未设置，无法使用稠密检索"))
        payload = search_knowledge("q", retriever=engine)
        self.assertTrue(payload["degraded"])
        self.assertIn("词法检索", payload["note"])
        self.assertIn("ZHIPUAI_API_KEY", payload["note"])

    def test_empty_results_has_note(self):
        engine = FakeRetriever(RetrievalResponse(results=[], mode="dense"))
        payload = search_knowledge("q", retriever=engine)
        self.assertEqual(payload["citations"], [])
        self.assertIn("没有找到", payload["note"])

    def test_blank_query_rejected(self):
        with self.assertRaises(ToolError):
            search_knowledge("   ", retriever=FakeRetriever(None))


class TestFetchPassage(unittest.TestCase):
    def setUp(self):
        self.chunks = {
            "pmc_1#0001": {
                "chunk_id": "pmc_1#0001", "doc_id": "pmc_1",
                "title": "T", "section": "Results",
                "url": "https://example.org", "license": "CC BY 4.0",
                "start": 1200, "end": 1900, "text": "完整正文" * 100,
            }
        }
        self.engine = FakeRetriever(None, chunks=self.chunks)

    def test_full_text_returned(self):
        payload = fetch_passage("pmc_1#0001", retriever=self.engine)
        self.assertEqual(payload["chunk_id"], "pmc_1#0001")
        self.assertEqual(payload["chars"], 400)
        self.assertEqual(payload["offset"], {"start": 1200, "end": 1900})

    def test_unknown_chunk_raises(self):
        with self.assertRaises(ToolError) as ctx:
            fetch_passage("nope", retriever=self.engine)
        self.assertIn("未找到片段", str(ctx.exception))

    def test_blank_chunk_id_rejected(self):
        with self.assertRaises(ToolError):
            fetch_passage("", retriever=self.engine)


class TestRegistryIntegration(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        build_rag_tools(self.registry)

    def test_tools_registered(self):
        self.assertEqual(sorted(self.registry.names()),
                         ["fetch_passage", "search_knowledge"])

    def test_defaults_applied(self):
        engine = FakeRetriever(RetrievalResponse(results=[], mode="dense"))
        # 通过 handler 闭包无法注入引擎，这里只验证 schema 默认值
        schema = self.registry.get("search_knowledge").parameters
        self.assertEqual(schema["properties"]["k"]["default"], 5)
        self.assertEqual(schema["properties"]["mode"]["default"], "auto")

    def test_missing_required_reported(self):
        result = self.registry.call("search_knowledge", {})
        self.assertFalse(result.ok)
        self.assertIn("缺少必填参数 query", result.error)

    def test_invalid_mode_reported(self):
        result = self.registry.call("search_knowledge",
                                    {"query": "q", "mode": "fuzzy"})
        self.assertFalse(result.ok)
        self.assertIn("只能是", result.error)


if __name__ == "__main__":
    unittest.main()