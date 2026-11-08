"""统一检索接口测试（T5）。全部离线，注入内存索引，不读写正式索引文件。"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.rag import retrieve
from src.rag.dense import DenseHit, DenseIndex
from src.rag.lexical import LexicalHit, LexicalIndex

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_retrieve"

CHUNKS = [
    {"chunk_id": "docA#0001", "doc_id": "docA", "title": "Pig Posture Paper",
     "section": "Introduction", "url": "https://example.org/a",
     "license": "CC BY", "start": 0, "end": 40,
     "text": "pig posture recognition deep learning computer vision models"},
    {"chunk_id": "docA#0002", "doc_id": "docA", "title": "Pig Posture Paper",
     "section": "Methods", "url": "https://example.org/a",
     "license": "CC BY", "start": 41, "end": 90,
     "text": "sow behaviour tracking farrowing pen activity resting time"},
    {"chunk_id": "docB#0001", "doc_id": "docB", "title": "数据卡",
     "section": "摘要", "url": "https://example.org/b",
     "license": "本项目自有文档", "start": 0, "end": 28,
     "text": "本研究使用公开数据集，共 309 条标注样本，侧卧姿态最多"},
]

#: 正交向量，docA#0001=[1,0,0] 等，便于确定性地控制稠密检索结果
DENSE_VECTORS = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)


class RetrieveFixture(unittest.TestCase):
    def setUp(self):
        self.chunks = {c["chunk_id"]: c for c in CHUNKS}
        self.dense = DenseIndex(DENSE_VECTORS, [c["chunk_id"] for c in CHUNKS])
        self.lexical = LexicalIndex.build(CHUNKS)

    def _retriever(self, api_key="test-key"):
        return retrieve.Retriever(api_key=api_key, dense=self.dense,
                                  lexical=self.lexical, chunks=self.chunks)


class TestDenseRetrieval(RetrieveFixture):
    def test_dense_returns_full_metadata(self):
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            resp = self._retriever().retrieve("posture recognition", k=2, mode="dense")

        self.assertEqual(resp.mode, "dense")
        self.assertIsNone(resp.fallback_reason)
        self.assertEqual(resp.results[0].chunk_id, "docA#0001")
        self.assertEqual(resp.results[0].rank, 1)
        r = resp.results[0]
        self.assertEqual(r.doc_id, "docA")
        self.assertEqual(r.title, "Pig Posture Paper")
        self.assertEqual(r.url, "https://example.org/a")
        self.assertEqual(r.license, "CC BY")
        self.assertEqual(r.start, 0)
        self.assertEqual(r.end, 40)
        self.assertEqual(r.source, "dense")
        self.assertIn("pig posture", r.text)

    def test_dense_respects_k(self):
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            resp = self._retriever().retrieve("posture", k=1, mode="dense")
        self.assertEqual(len(resp), 1)

    def test_ranks_are_sequential(self):
        # 全正分查询向量 → 3 块都返回，rank 应为 1,2,3
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 1.0, 1.0]], dtype=np.float32)):
            resp = self._retriever().retrieve("posture", k=3, mode="dense")
        self.assertEqual([r.rank for r in resp.results], [1, 2, 3])

    def test_missing_api_key_raises_in_dense_mode(self):
        with self.assertRaises(RuntimeError):
            self._retriever(api_key="").retrieve("x", mode="dense")


class TestLexicalAndFallback(RetrieveFixture):
    def test_lexical_mode(self):
        resp = self._retriever().retrieve("侧卧 姿态 数据集", k=3, mode="lexical")
        self.assertEqual(resp.mode, "lexical")
        self.assertTrue(resp.results)
        self.assertEqual(resp.results[0].doc_id, "docB")

    def test_auto_falls_back_on_embedding_error(self):
        def boom(*a, **kw):
            raise RuntimeError("quota exceeded")

        with mock.patch("src.rag.retrieve._call_embedding_api", side_effect=boom):
            resp = self._retriever().retrieve("侧卧 数据集", k=2, mode="auto")
        self.assertEqual(resp.mode, "dense-fallback-lexical")
        self.assertIn("quota exceeded", resp.fallback_reason)
        self.assertTrue(resp.results)
        self.assertEqual(resp.results[0].source, "lexical")

    def test_auto_falls_back_without_api_key(self):
        resp = self._retriever(api_key="").retrieve("pig posture", k=2, mode="auto")
        self.assertEqual(resp.mode, "dense-fallback-lexical")
        self.assertIn("API_KEY", resp.fallback_reason)

    def test_hybrid_flag_equals_hybrid_mode(self):
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            a = self._retriever().retrieve("posture", k=3, hybrid=True)
            b = self._retriever().retrieve("posture", k=3, mode="hybrid")
        self.assertEqual(a.mode, b.mode)
        self.assertEqual([r.chunk_id for r in a], [r.chunk_id for r in b])


class TestHybridFusion(RetrieveFixture):
    def test_hybrid_returns_no_duplicates(self):
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            resp = self._retriever().retrieve("posture pig", k=3, mode="hybrid")
        ids = [r.chunk_id for r in resp.results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_hybrid_marked_for_both_hit(self):
        # 查询向量接近 docA#0001，词法也会命中 docA#0001（"pig posture"）
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            resp = self._retriever().retrieve("pig posture recognition", k=3,
                                              mode="hybrid")
        sources = {r.chunk_id: r.source for r in resp.results}
        self.assertIn("docA#0001", sources)
        # docA#0001 应被稠密且词法同时命中 → hybrid
        self.assertEqual(sources["docA#0001"], "hybrid")

    def test_hybrid_top_is_stable_for_same_query(self):
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[0.0, 1.0, 0.0]], dtype=np.float32)):
            r1 = self._retriever().retrieve("sow behaviour", k=3, mode="hybrid")
            r2 = self._retriever().retrieve("sow behaviour", k=3, mode="hybrid")
        self.assertEqual([x.chunk_id for x in r1], [x.chunk_id for x in r2])


class TestEmptyAndErrors(RetrieveFixture):
    def test_empty_query_returns_none_mode(self):
        resp = self._retriever().retrieve("   ", mode="auto")
        self.assertEqual(resp.mode, "none")
        self.assertEqual(len(resp), 0)

    def test_coverage_mismatch_raises(self):
        ghost_dense = DenseIndex(np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                                 ["ghost#0001"])
        with self.assertRaises(RuntimeError) as ctx:
            retrieve.Retriever(api_key="k", dense=ghost_dense,
                               lexical=self.lexical, chunks=self.chunks)
        self.assertIn("ghost#0001", str(ctx.exception))


class TestGrounding(RetrieveFixture):
    def setUp(self):
        super().setUp()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.raw_dir = TMP_ROOT
        doc_text = "0123456789" * 10   # 100 字符
        (TMP_ROOT / "docA.txt").write_text(doc_text, encoding="utf-8")
        self.doc_text = doc_text

    def tearDown(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_read_document(self):
        r = self._retriever()
        self.assertEqual(r.read_document("docA", raw_dir=self.raw_dir),
                         self.doc_text)

    def test_read_missing_document_returns_none(self):
        self.assertIsNone(self._retriever().read_document("docC",
                                                          raw_dir=self.raw_dir))

    def test_highlight_contains_exact_slice(self):
        r = self._retriever()
        result = retrieve.RetrievalResult(
            chunk_id="docA#0001", doc_id="docA", title="t", section="s",
            url="u", license="l", start=20, end=30, text=self.doc_text[20:30],
            score=0.9, source="dense")
        window = r.highlight(result, raw_dir=self.raw_dir)
        self.assertIsNotNone(window)
        self.assertIn(self.doc_text[20:30], window)


class TestFusionWeights(RetrieveFixture):
    """用桩对象精确控制两路命中，验证加权与噪声下限。"""

    def test_dense_outranks_equally_ranked_lexical(self):
        dense_hits = [DenseHit("docA#0001", 0.90, 0)]
        lex_hits = [LexicalHit("docB#0001", 0.50, 2)]
        retriever = retrieve.Retriever(
            api_key="k",
            dense=_StubDense(dense_hits, ["docA#0001"]),
            lexical=_StubLexical(lex_hits, ["docB#0001"]),
            chunks=self.chunks,
        )
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            resp = retriever.retrieve("query", k=3, mode="hybrid")
        self.assertEqual([r.chunk_id for r in resp.results],
                         ["docA#0001", "docB#0001"])
        self.assertGreater(resp.results[0].score, resp.results[1].score)

    def test_lexical_below_floor_is_ignored(self):
        dense_hits = [DenseHit("docA#0001", 0.90, 0)]
        lex_hits = [LexicalHit("docB#0001", retrieve.LEXICAL_MIN_SCORE / 2, 0)]
        retriever = retrieve.Retriever(
            api_key="k",
            dense=_StubDense(dense_hits, ["docA#0001"]),
            lexical=_StubLexical(lex_hits, ["docB#0001"]),
            chunks=self.chunks,
        )
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            resp = retriever.retrieve("query", k=3, mode="hybrid")
        self.assertEqual([r.chunk_id for r in resp.results], ["docA#0001"])

    def test_component_scores_recorded_for_hybrid_hit(self):
        dense_hits = [DenseHit("docA#0001", 0.88, 0)]
        lex_hits = [LexicalHit("docA#0001", 0.42, 0)]
        retriever = retrieve.Retriever(
            api_key="k",
            dense=_StubDense(dense_hits, ["docA#0001"]),
            lexical=_StubLexical(lex_hits, ["docA#0001"]),
            chunks=self.chunks,
        )
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)):
            resp = retriever.retrieve("query", k=2, mode="hybrid")
        top = resp.results[0]
        self.assertEqual(top.source, "hybrid")
        self.assertAlmostEqual(top.dense_score, 0.88, places=6)
        self.assertAlmostEqual(top.lexical_score, 0.42, places=6)
        payload = top.to_dict()
        self.assertAlmostEqual(payload["dense_score"], 0.88, places=6)
        self.assertAlmostEqual(payload["lexical_score"], 0.42, places=6)


class _StubDense:
    """只实现检索所需接口的稠密索引替身。"""

    def __init__(self, hits, chunk_ids):
        self._hits = list(hits)
        self.chunk_ids = list(chunk_ids)

    def __len__(self):
        return len(self.chunk_ids)

    def search(self, vector, k=5):
        return self._hits[:k]


class _StubLexical:
    """只实现检索所需接口的词法索引替身。"""

    def __init__(self, hits, chunk_ids):
        self._hits = list(hits)
        self.chunk_ids = list(chunk_ids)

    def __len__(self):
        return len(self.chunk_ids)

    def search(self, query, k=5):
        return self._hits[:k]


class TestSerialization(RetrieveFixture):
    def test_result_to_dict(self):
        result = retrieve.RetrievalResult(
            chunk_id="docA#0001", doc_id="docA", title="t", section="s",
            url="u", license="l", start=1, end=2, text="hi", score=0.123456,
            source="dense", rank=7)
        payload = result.to_dict()
        self.assertEqual(payload["chunk_id"], "docA#0001")
        self.assertEqual(payload["score"], 0.123456)
        self.assertEqual(payload["rank"], 7)
        self.assertEqual(payload["source"], "dense")

    def test_response_to_dict(self):
        # [1,1,0] 与 docA 两块的余弦都 >0，与 docB 为 0 → 恰好 2 条
        with mock.patch("src.rag.retrieve._call_embedding_api",
                        return_value=np.array([[1.0, 1.0, 0.0]], dtype=np.float32)):
            resp = self._retriever().retrieve("posture", k=2, mode="dense")
        payload = resp.to_dict()
        self.assertEqual(payload["mode"], "dense")
        self.assertEqual(len(payload["results"]), 2)
        self.assertIn("chunk_id", payload["results"][0])


if __name__ == "__main__":
    unittest.main()