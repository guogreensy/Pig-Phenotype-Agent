"""稠密索引测试（T4）。全部使用 mock，不调用真实 embedding API。"""

from __future__ import annotations

import hashlib
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.rag import dense

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_dense"

CHUNKS = [
    {"chunk_id": "pmc_a#0001", "text": "Pig posture recognition using computer vision."},
    {"chunk_id": "pmc_b#0001", "text": "Automated tracking of sow behaviour in farrowing pens."},
    {"chunk_id": "local_c#0001", "text": "本研究使用公开数据集，共 309 条标注样本。"},
    {"chunk_id": "local_d#0001", "text": "实验结果显示三组提示词的平均准确率均不足四成。"},
]


def _fake_embeddings(texts, api_key, model=None, batch_size=16,
                     min_interval_s=0, max_retries=1):
    """返回与输入等长的确定性 fake 向量（未归一化，交给 build 归一化）。"""
    rng = np.random.RandomState(42)
    return np.stack([rng.randn(dense._DIM).astype(np.float32) for _ in texts])


class _FakeIndexCase(unittest.TestCase):
    """公共夹具：把 embedding 调用替换成确定性假向量。"""

    def setUp(self):
        self.patcher = mock.patch("src.rag.dense._call_embedding_api",
                                  side_effect=_fake_embeddings)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()


class TestDenseIndex(_FakeIndexCase):
    def test_build_returns_correct_shape(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        self.assertEqual(len(idx), 4)
        self.assertEqual(idx.embeddings.shape, (4, dense._DIM))

    def test_vectors_are_normalized(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        for i in range(len(idx)):
            self.assertAlmostEqual(float(np.linalg.norm(idx.embeddings[i])), 1.0,
                                   places=5)

    def test_search_returns_own_row_as_top(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        hits = idx.search(idx.embeddings[2], k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk_id, "local_c#0001")
        self.assertAlmostEqual(hits[0].score, 1.0, places=4)

    def test_search_normalizes_unnormalized_query(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        unit = idx.search(idx.embeddings[1], k=2)
        scaled = idx.search(idx.embeddings[1] * 37.5, k=2)
        self.assertEqual([h.chunk_id for h in unit], [h.chunk_id for h in scaled])
        for a, b in zip(unit, scaled):
            self.assertAlmostEqual(a.score, b.score, places=5)

    def test_scores_in_valid_range(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        hits = idx.search(idx.embeddings[0] + idx.embeddings[1], k=4)
        for hit in hits:
            self.assertGreaterEqual(hit.score, 0.0)
            self.assertLessEqual(hit.score, 1.0 + 1e-5)

    def test_k_larger_than_corpus(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        self.assertLessEqual(len(idx.search(idx.embeddings[0], k=99)), len(CHUNKS))

    def test_zero_k_returns_empty(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        self.assertEqual(idx.search(idx.embeddings[0], k=0), [])

    def test_zero_query_returns_empty(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        hits = idx.search(np.zeros(dense._DIM, dtype=np.float32), k=3)
        self.assertEqual(hits, [])

    def test_wrong_query_dimension_raises(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        with self.assertRaises(ValueError):
            idx.search(np.zeros(100, dtype=np.float32), k=1)

    def test_empty_corpus_raises(self):
        with self.assertRaises(ValueError):
            dense.DenseIndex.build([], api_key="test-key")

    def test_chunk_ids_follow_input_order(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        self.assertEqual(idx.chunk_ids, [c["chunk_id"] for c in CHUNKS])

    def test_position_matches_chunk_id_order(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        hit = idx.search(idx.embeddings[0], k=1)[0]
        self.assertEqual(hit.position, 0)
        self.assertEqual(idx.chunk_ids[hit.position], hit.chunk_id)

    def test_meta_records_model_and_shape(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        self.assertEqual(idx.meta["model"], "embedding-3")
        self.assertEqual(idx.meta["dim"], dense._DIM)
        self.assertEqual(idx.meta["chunks"], 4)
        self.assertEqual(idx.meta["chunk_ids_sha256"],
                         dense._fingerprint([c["chunk_id"] for c in CHUNKS]))

    def test_rows_and_ids_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dense.DenseIndex(np.zeros((3, dense._DIM), dtype=np.float32), ["a", "b"])


class TestPersistence(_FakeIndexCase):
    def setUp(self):
        super().setUp()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.npz_path = TMP_ROOT / "dense.npz"

    def tearDown(self):
        super().tearDown()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_save_writes_three_files(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        for suffix in (".npz", ".ids.json", ".meta.json"):
            self.assertTrue(dense._sidecar(self.npz_path, suffix).is_file(), suffix)

    def test_meta_content(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        meta = json.loads(dense._sidecar(self.npz_path, ".meta.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(meta["chunks"], 4)
        self.assertEqual(meta["dim"], dense._DIM)
        self.assertEqual(meta["index_version"], dense.INDEX_VERSION)
        self.assertIn("built_at", meta)

    def test_roundtrip_preserves_search_results(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        restored = dense.DenseIndex.load(self.npz_path)

        self.assertEqual(restored.chunk_ids, idx.chunk_ids)
        np.testing.assert_allclose(restored.embeddings, idx.embeddings)

        query = idx.embeddings[2]
        before, after = idx.search(query, k=3), restored.search(query, k=3)
        self.assertEqual([h.chunk_id for h in before], [h.chunk_id for h in after])
        for a, b in zip(before, after):
            self.assertAlmostEqual(a.score, b.score, places=5)

    def test_missing_npz_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            dense.DenseIndex.load(TMP_ROOT / "no_such.npz")
        self.assertIn("稠密索引不存在", str(ctx.exception))

    def test_missing_ids_raises(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        dense._sidecar(self.npz_path, ".ids.json").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            dense.DenseIndex.load(self.npz_path)
        self.assertIn("id 列表不存在", str(ctx.exception))

    def test_custom_path_never_falls_back_to_default_ids(self):
        """回归：自定义路径缺 ids 时，不能静默套用正式索引的 id 列表。"""
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        dense._sidecar(self.npz_path, ".ids.json").unlink()
        with self.assertRaises(FileNotFoundError):
            dense.DenseIndex.load(self.npz_path)

    def test_dim_mismatch_in_meta_raises(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        meta_path = dense._sidecar(self.npz_path, ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["dim"] = 1024
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            dense.DenseIndex.load(self.npz_path)
        self.assertIn("维度与元数据不符", str(ctx.exception))

    def test_count_mismatch_in_meta_raises(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        meta_path = dense._sidecar(self.npz_path, ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["chunks"] = 99
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ValueError):
            dense.DenseIndex.load(self.npz_path)

    def test_fingerprint_mismatch_raises(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        meta_path = dense._sidecar(self.npz_path, ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["chunk_ids_sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            dense.DenseIndex.load(self.npz_path)
        self.assertIn("指纹", str(ctx.exception))

    def test_load_without_meta_still_works(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="test-key")
        idx.save(self.npz_path)
        dense._sidecar(self.npz_path, ".meta.json").unlink()
        restored = dense.DenseIndex.load(self.npz_path)
        self.assertEqual(len(restored), 4)
        self.assertEqual(restored.meta, {})


class TestFreshness(_FakeIndexCase):
    def setUp(self):
        super().setUp()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.chunks_path = TMP_ROOT / "chunks.jsonl"
        self.chunks_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in CHUNKS),
            encoding="utf-8")

    def tearDown(self):
        super().tearDown()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_fresh_index_passes(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="k",
                                     chunks_file=self.chunks_path)
        idx.check_freshness(self.chunks_path)   # 不应抛异常

    def test_stale_index_detected(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="k",
                                     chunks_file=self.chunks_path)
        self.chunks_path.write_text("{}", encoding="utf-8")   # 语料变了
        with self.assertRaises(ValueError) as ctx:
            idx.check_freshness(self.chunks_path)
        self.assertIn("请重建稠密索引", str(ctx.exception))

    def test_no_recorded_digest_is_noop(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="k")
        idx.check_freshness(self.chunks_path)   # 没有记录就不校验

    def test_build_records_chunks_file_digest(self):
        idx = dense.DenseIndex.build(CHUNKS, api_key="k",
                                     chunks_file=self.chunks_path)
        expected = hashlib.sha256(self.chunks_path.read_bytes()).hexdigest()
        self.assertEqual(idx.meta["chunks_file_sha256"], expected)


class TestEmbeddingApiErrors(unittest.TestCase):
    def test_empty_api_key_raises(self):
        with self.assertRaises(RuntimeError):
            dense._call_embedding_api(["text"], api_key="")

    def test_empty_texts_raises(self):
        with self.assertRaises(ValueError):
            dense._call_embedding_api([], api_key="k")

    def test_persistent_500_raises(self):
        with mock.patch("src.rag.dense.requests.Session") as sess_cls:
            resp = mock.MagicMock()
            resp.status_code = 500
            resp.headers = {}
            sess_cls.return_value.post.return_value = resp
            with self.assertRaises(RuntimeError):
                dense._call_embedding_api(["text"], api_key="k", max_retries=2)

    def test_401_raises_without_retry(self):
        with mock.patch("src.rag.dense.requests.Session") as sess_cls:
            resp = mock.MagicMock()
            resp.status_code = 401
            resp.raise_for_status.side_effect = RuntimeError("401")
            sess_cls.return_value.post.return_value = resp
            with self.assertRaises(RuntimeError):
                dense._call_embedding_api(["text"], api_key="bad", max_retries=3)

    def test_count_mismatch_raises(self):
        with mock.patch("src.rag.dense.requests.Session") as sess_cls:
            resp = mock.MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"data": [{"embedding": [0.1] * dense._DIM}]}
            sess_cls.return_value.post.return_value = resp
            with self.assertRaises(RuntimeError):
                dense._call_embedding_api(["a", "b"], api_key="k", max_retries=1)


class TestBatchingAndCli(unittest.TestCase):
    def test_build_and_save_forwards_batch_size(self):
        fake_index = mock.MagicMock()
        with mock.patch("src.rag.dense.load_chunks", return_value=CHUNKS), \
                mock.patch.object(dense.DenseIndex, "build",
                                  return_value=fake_index) as build_mock:
            dense.build_and_save(api_key="k",
                                 chunks_path=TMP_ROOT / "c.jsonl",
                                 out_path=TMP_ROOT / "d.npz",
                                 batch_size=7)
        self.assertEqual(build_mock.call_args.kwargs["batch_size"], 7)
        fake_index.save.assert_called_once()

    def test_all_texts_are_sent_once(self):
        seen = {}

        def spy(texts, api_key, model=None, batch_size=16, min_interval_s=0,
                max_retries=1):
            seen["count"] = len(texts)
            seen["batch_size"] = batch_size
            return _fake_embeddings(texts, api_key)

        many = [{"chunk_id": "c%03d" % i, "text": "text %d" % i} for i in range(30)]
        with mock.patch("src.rag.dense._call_embedding_api", side_effect=spy):
            idx = dense.DenseIndex.build(many, api_key="k", batch_size=16)
        self.assertEqual(len(idx), 30)
        self.assertEqual(seen["count"], 30)
        self.assertEqual(seen["batch_size"], 16)


class TestDenseHitSerialization(unittest.TestCase):
    def test_hit_to_dict(self):
        hit = dense.DenseHit(chunk_id="a#0001", score=0.876543, position=3)
        payload = hit.to_dict()
        self.assertEqual(payload["chunk_id"], "a#0001")
        self.assertEqual(payload["score"], 0.876543)
        self.assertEqual(payload["position"], 3)


if __name__ == "__main__":
    unittest.main()