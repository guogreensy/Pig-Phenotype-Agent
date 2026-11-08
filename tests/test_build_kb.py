"""知识库一键构建脚本测试。全部使用临时目录，不触发下载与接口调用。"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pickle
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_ROOT = Path(__file__).resolve().parent / "_tmp_buildkb"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_kb", PROJECT_ROOT / "scripts" / "build_kb.py")
    module = importlib.util.module_from_spec(spec)
    # dataclass 解析字符串注解时需要能从 sys.modules 找回本模块
    sys.modules["build_kb"] = module
    spec.loader.exec_module(module)
    return module


build_kb = _load_module()


def _touch(path: Path, mtime: float, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (mtime, mtime))


class BuildKbFixture(unittest.TestCase):
    def setUp(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.paths = build_kb.Paths(
            sources=TMP_ROOT / "sources.json",
            chunks=TMP_ROOT / "chunks.jsonl",
            cache_dir=TMP_ROOT / "cache",
            raw_dir=TMP_ROOT / "raw",
            lexical=TMP_ROOT / "index" / "lexical.pkl",
            dense=TMP_ROOT / "index" / "dense.npz",
        )

    def tearDown(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    # ---------------------------------------------------------- 造状态
    def make_all_ready(self) -> None:
        _touch(self.paths.sources, 1000.0, "{}")
        _touch(self.paths.cache_dir / "a.xml", 1000.0, "<x/>")
        _touch(self.paths.chunks, 2000.0, "{\"chunk_id\": \"a#0001\"}\n")
        _touch(self.paths.lexical, 3000.0, "pkl")
        _touch(self.paths.dense, 4000.0, "npz")
        _touch(self.paths.dense.with_suffix(".ids.json"), 4000.0, "[]")
        digest = hashlib.sha256(self.paths.chunks.read_bytes()).hexdigest()
        _touch(self.paths.dense.with_suffix(".meta.json"), 4000.0,
               json.dumps({"chunks_file_sha256": digest}))


class TestStageStatus(BuildKbFixture):
    def test_all_missing(self):
        for name in build_kb.STAGES:
            ready, _ = build_kb.stage_status(name, self.paths)
            self.assertFalse(ready, name)

    def test_fetch_ready_with_manifest_and_cache(self):
        _touch(self.paths.sources, 1000.0, "{}")
        _touch(self.paths.cache_dir / "a.xml", 1000.0)
        ready, detail = build_kb.stage_status("fetch", self.paths)
        self.assertTrue(ready)
        self.assertIn("1 个缓存文件", detail)

    def test_fetch_not_ready_without_cache(self):
        _touch(self.paths.sources, 1000.0, "{}")
        ready, _ = build_kb.stage_status("fetch", self.paths)
        self.assertFalse(ready)

    def test_chunks_stale_when_older_than_sources(self):
        _touch(self.paths.sources, 5000.0, "{}")
        _touch(self.paths.chunks, 1000.0, "{}")
        ready, detail = build_kb.stage_status("chunks", self.paths)
        self.assertFalse(ready)
        self.assertIn("早于 sources", detail)

    def test_chunks_ready_when_newer(self):
        _touch(self.paths.sources, 1000.0, "{}")
        _touch(self.paths.chunks, 2000.0, "{}")
        ready, _ = build_kb.stage_status("chunks", self.paths)
        self.assertTrue(ready)

    def test_lexical_stale_when_older_than_chunks(self):
        _touch(self.paths.chunks, 5000.0, "{}")
        _touch(self.paths.lexical, 1000.0)
        ready, detail = build_kb.stage_status("lexical", self.paths)
        self.assertFalse(ready)
        self.assertIn("早于 chunks", detail)

    def test_lexical_ready(self):
        _touch(self.paths.chunks, 1000.0, "{}")
        _touch(self.paths.lexical, 2000.0)
        self.assertTrue(build_kb.stage_status("lexical", self.paths)[0])

    def test_lexical_fingerprint_beats_mtime(self):
        """仅重写 chunks（内容未变）不应判为过期。"""
        _touch(self.paths.chunks, 5000.0, "{}\n")
        digest = hashlib.sha256(self.paths.chunks.read_bytes()).hexdigest()
        self.paths.lexical.parent.mkdir(parents=True, exist_ok=True)
        self.paths.lexical.write_bytes(pickle.dumps({"chunks_file_sha256": digest}))
        os.utime(self.paths.lexical, (1000.0, 1000.0))   # 时间戳更旧
        ready, detail = build_kb.stage_status("lexical", self.paths)
        self.assertTrue(ready)
        self.assertIn("指纹一致", detail)

    def test_lexical_fingerprint_mismatch(self):
        _touch(self.paths.chunks, 5000.0, "{}\n")
        self.paths.lexical.parent.mkdir(parents=True, exist_ok=True)
        self.paths.lexical.write_bytes(pickle.dumps({"chunks_file_sha256": "0" * 64}))
        ready, detail = build_kb.stage_status("lexical", self.paths)
        self.assertFalse(ready)
        self.assertIn("指纹不一致", detail)

    def test_dense_fingerprint_mismatch(self):
        _touch(self.paths.chunks, 2000.0, "{\"a\": 1}\n")
        _touch(self.paths.dense, 4000.0)
        _touch(self.paths.dense.with_suffix(".ids.json"), 4000.0, "[]")
        _touch(self.paths.dense.with_suffix(".meta.json"), 4000.0,
               json.dumps({"chunks_file_sha256": "0" * 64}))
        ready, detail = build_kb.stage_status("dense", self.paths)
        self.assertFalse(ready)
        self.assertIn("指纹不一致", detail)

    def test_dense_ready_with_matching_fingerprint(self):
        _touch(self.paths.chunks, 2000.0, "{\"a\": 1}\n")
        _touch(self.paths.dense, 4000.0)
        _touch(self.paths.dense.with_suffix(".ids.json"), 4000.0, "[]")
        digest = hashlib.sha256(self.paths.chunks.read_bytes()).hexdigest()
        _touch(self.paths.dense.with_suffix(".meta.json"), 4000.0,
               json.dumps({"chunks_file_sha256": digest}))
        self.assertTrue(build_kb.stage_status("dense", self.paths)[0])

    def test_missing_meta_makes_dense_not_ready(self):
        _touch(self.paths.chunks, 2000.0, "{}")
        _touch(self.paths.dense, 4000.0)
        _touch(self.paths.dense.with_suffix(".ids.json"), 4000.0, "[]")
        self.assertFalse(build_kb.stage_status("dense", self.paths)[0])

    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            build_kb.stage_status("nope", self.paths)


class TestCli(BuildKbFixture):
    def _run(self, argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = build_kb.main(argv, paths=self.paths)
        return code, buffer.getvalue()

    def _patch_runners(self):
        calls = []

        def make(name):
            def runner(paths, force, limit, api_key=None):
                calls.append(name)
            return runner

        patches = {name: make(name) for name in build_kb.STAGES}
        return calls, mock.patch.dict(build_kb._RUNNERS, patches), \
            mock.patch.object(build_kb, "_run_dense", make("dense"))

    def test_status_only_does_not_execute(self):
        calls, runners, dense = self._patch_runners()
        with runners, dense:
            code, output = self._run(["--status"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertIn("待建", output)

    def test_incremental_skips_all_when_ready(self):
        self.make_all_ready()
        calls, runners, dense = self._patch_runners()
        with runners, dense:
            code, output = self._run([])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertIn("均已就绪", output)
        self.assertIn("4 / 4 阶段就绪", output)

    def test_force_executes_every_stage(self):
        self.make_all_ready()
        calls, runners, dense = self._patch_runners()
        with runners, dense:
            code, _ = self._run(["--force"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["fetch", "chunks", "lexical", "dense"])

    def test_only_restricts_stages(self):
        calls, runners, dense = self._patch_runners()
        with runners, dense:
            self._run(["--only", "chunks", "--force"])
        self.assertEqual(calls, ["chunks"])

    def test_skip_excludes_stage(self):
        calls, runners, dense = self._patch_runners()
        with runners, dense:
            self._run(["--force", "--skip", "dense,fetch"])
        self.assertEqual(calls, ["chunks", "lexical"])

    def test_unknown_only_stage_exits(self):
        with self.assertRaises(SystemExit):
            self._run(["--only", "nope"])

    def test_missing_stage_is_executed(self):
        _touch(self.paths.sources, 1000.0, "{}")
        _touch(self.paths.cache_dir / "a.xml", 1000.0)
        calls, runners, dense = self._patch_runners()
        with runners, dense:
            self._run([])
        # fetch 就绪 → 跳过；其余三个阶段都缺产物 → 执行
        self.assertEqual(calls, ["chunks", "lexical", "dense"])

    def test_stage_rebuild_invalidates_downstream_in_same_run(self):
        """chunks 重建后，lexical/dense 必须在同一次运行里跟着重建。"""
        self.make_all_ready()
        calls = []

        def chunks_runner(paths, force, limit, api_key=None):
            calls.append("chunks")
            # 模拟重建：chunks 变新，lexical 因此过期，dense 指纹也不再匹配
            _touch(paths.chunks, 9000.0, '{"chunk_id": "b#0001"}\n')

        patches = {"fetch": lambda *a, **k: calls.append("fetch"),
                   "chunks": chunks_runner,
                   "lexical": lambda *a, **k: calls.append("lexical")}
        with mock.patch.dict(build_kb._RUNNERS, patches), \
                mock.patch.object(build_kb, "_run_dense",
                                  lambda *a, **k: calls.append("dense")):
            code, output = self._run(["--only", "chunks,lexical,dense", "--force"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["chunks", "lexical", "dense"])
        self.assertNotIn("均已就绪", output)


if __name__ == "__main__":
    unittest.main()