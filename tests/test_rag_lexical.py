"""词法索引测试（T3）。全部离线，不读写正式索引。"""

from __future__ import annotations

import pickle
import shutil
import unittest
from pathlib import Path

from src.rag import lexical

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_lexical"

CHUNKS = [
    {
        "chunk_id": "pmc_a#0001",
        "text": "Pig posture recognition using computer vision and deep learning models. "
                "Standing and lying postures are classified from camera images.",
    },
    {
        "chunk_id": "pmc_b#0001",
        "text": "Automated tracking of sow behaviour changes in farrowing pens, "
                "measuring activity and resting time.",
    },
    {
        "chunk_id": "local_c#0001",
        "text": "本研究使用公开数据集，共 309 条标注样本，其中侧卧姿态数量最多，"
                "坐姿样本最少，站立与侧卧在俯视视角下难以区分。",
    },
    {
        "chunk_id": "local_d#0001",
        "text": "实验结果显示三组提示词的平均准确率均不足四成，模型明显偏向侧卧类别，"
                "坐姿从未被预测出来。",
    },
]


class TestBuildAndSearch(unittest.TestCase):
    def setUp(self):
        self.index = lexical.LexicalIndex.build(CHUNKS)

    def test_index_length(self):
        self.assertEqual(len(self.index), len(CHUNKS))

    def test_english_query_hits_english_chunk(self):
        hits = self.index.search("computer vision posture recognition", k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk_id, "pmc_a#0001")

    def test_chinese_query_hits_chinese_chunk(self):
        hits = self.index.search("侧卧姿态在俯视视角下难以区分", k=2)
        self.assertTrue(hits)
        self.assertIn(hits[0].chunk_id, {"local_c#0001", "local_d#0001"})

    def test_scores_sorted_desc(self):
        hits = self.index.search("posture", k=4)
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.index.search("   ", k=5), [])

    def test_k_larger_than_corpus(self):
        hits = self.index.search("posture behaviour", k=99)
        self.assertLessEqual(len(hits), len(CHUNKS))

    def test_irrelevant_query_returns_nothing(self):
        # 语料里完全没有这些字，TF-IDF 得分为 0，应返回空而不是硬塞结果
        self.assertEqual(self.index.search("zzzz qqqq", k=3), [])

    def test_zero_k(self):
        self.assertEqual(self.index.search("posture", k=0), [])

    def test_build_is_deterministic(self):
        again = lexical.LexicalIndex.build(CHUNKS)
        first = self.index.search("lying posture camera", k=3)
        second = again.search("lying posture camera", k=3)
        self.assertEqual([h.chunk_id for h in first], [h.chunk_id for h in second])
        for a, b in zip(first, second):
            self.assertAlmostEqual(a.score, b.score, places=9)

    def test_empty_corpus_raises(self):
        with self.assertRaises(ValueError):
            lexical.LexicalIndex.build([])


class TestPersistence(unittest.TestCase):
    def setUp(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.path = TMP_ROOT / "lexical.pkl"

    def tearDown(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_roundtrip_preserves_results(self):
        index = lexical.LexicalIndex.build(CHUNKS)
        index.save(self.path)
        self.assertTrue(self.path.is_file())

        restored = lexical.LexicalIndex.load(self.path)
        self.assertEqual(len(restored), len(index))
        query = "sow behaviour activity"
        before = index.search(query, k=3)
        after = restored.search(query, k=3)
        self.assertEqual([h.chunk_id for h in before], [h.chunk_id for h in after])
        for a, b in zip(before, after):
            self.assertAlmostEqual(a.score, b.score, places=6)

    def test_missing_file_raises_with_hint(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            lexical.LexicalIndex.load(TMP_ROOT / "nope.pkl")
        self.assertIn("src.rag.lexical", str(ctx.exception))

    def test_version_mismatch_raises(self):
        index = lexical.LexicalIndex.build(CHUNKS)
        index.save(self.path)
        with self.path.open("rb") as handle:
            payload = pickle.load(handle)
        payload["version"] = 999
        with self.path.open("wb") as handle:
            pickle.dump(payload, handle)
        with self.assertRaises(ValueError) as ctx:
            lexical.LexicalIndex.load(self.path)
        self.assertIn("请重建索引", str(ctx.exception))


class TestHitSerialization(unittest.TestCase):
    def test_hit_to_dict(self):
        index = lexical.LexicalIndex.build(CHUNKS)
        hit = index.search("posture", k=1)[0]
        payload = hit.to_dict()
        self.assertEqual(payload["chunk_id"], hit.chunk_id)
        self.assertIn("score", payload)
        self.assertIsInstance(payload["score"], float)


if __name__ == "__main__":
    unittest.main()
