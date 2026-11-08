"""``src.experiment_runner`` 的测试（打桩替换 HTTP 客户端，不联网）。

覆盖：记录字段完整性、失败照样落盘、断点续跑、max_calls 成本闸。
"""

from __future__ import annotations

import csv
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import config
from config import Settings
from src import experiment_runner
from src.experiment_runner import ExperimentRunner, Sample, load_samples
from src.vision_client import ModelResponse

TMP = Path(__file__).resolve().parent / "_tmp_runner"


def make_settings(max_calls: int = 100) -> Settings:
    return Settings(
        provider="zhipu", model="glm-4.6v-flash", api_key="sk-test",
        base_url="https://example.invalid/v4/chat/completions",
        temperature=0.0, max_tokens=128, timeout_seconds=5.0, max_retries=1,
        qps=1000.0, concurrency=2, max_calls=max_calls,
        image_max_side=64, jpeg_quality=85, host="127.0.0.1", port=0,
    )


def write_images(count: int) -> list:
    TMP.mkdir(exist_ok=True)
    paths = []
    for index in range(count):
        path = TMP / "img_{0:02d}.jpg".format(index)
        if not path.is_file():
            Image.new("RGB", (80, 80), (index * 10, 60, 90)).save(path, format="JPEG")
        paths.append(path)
    return paths


GOOD_JSON = json.dumps({
    "posture": "standing",
    "description": "猪只四肢支撑站立。",
    "uncertain": False,
    "uncertainty_reason": None,
}, ensure_ascii=False)


class FakeClient:
    """按预设返回固定响应的假客户端。"""

    responses: list = []
    calls: list = []

    def __init__(self, settings, limiter=None, **_kwargs):
        self.settings = settings
        self._index = 0

    def call(self, data_url, prompt, system=None):
        FakeClient.calls.append(prompt[:20])
        index = FakeClient.counter if hasattr(FakeClient, "counter") else 0
        FakeClient.counter = index + 1
        if FakeClient.responses:
            return FakeClient.responses[index % len(FakeClient.responses)]
        return ModelResponse(ok=True, raw_text=GOOD_JSON, http_status=200,
                             latency_seconds=0.5, input_tokens=900, output_tokens=40,
                             model="glm-4.6v-flash")


class RunnerTestBase(unittest.TestCase):
    def setUp(self):
        FakeClient.calls = []
        FakeClient.responses = []
        if hasattr(FakeClient, "counter"):
            del FakeClient.counter
        patcher = mock.patch.object(experiment_runner, "VisionClient", FakeClient)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if TMP.is_dir():
            for path in TMP.glob("*"):
                path.unlink()
            TMP.rmdir()

    def samples(self, count=3):
        paths = write_images(count)
        return [Sample(sample_id="s{0}".format(i), image_path=p, true_label="standing")
                for i, p in enumerate(paths)]

    def runner(self, count=3, variants=("A",), max_calls=100, repeats=1, run_id="t1"):
        output = TMP / "{0}_raw.csv".format(run_id)
        if output.exists():
            output.unlink()
        return ExperimentRunner(
            settings=make_settings(max_calls),
            samples=self.samples(count),
            variants=list(variants),
            run_id=run_id,
            output_path=output,
            concurrency=2,
            max_calls=max_calls,
            repeats=repeats,
        )


class TestRecordContents(RunnerTestBase):
    def test_all_records_written(self):
        runner = self.runner(count=3, variants=("A", "C"))
        summary = runner.run()
        self.assertEqual(summary.n_done, 6)
        self.assertEqual(summary.n_failed, 0)
        with runner.output_path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        required = {"sample_id", "image_path", "true_label", "prompt_variant",
                    "model_name", "predicted_label", "description", "uncertain",
                    "json_valid", "request_success", "latency_seconds",
                    "error_message", "experiment_timestamp"}
        self.assertTrue(required.issubset(set(rows[0].keys())), required - set(rows[0].keys()))
        for row in rows:
            self.assertEqual(row["predicted_label"], "standing")
            self.assertEqual(row["request_success"], "True")
            self.assertEqual(row["schema_valid"], "True")
            self.assertEqual(len(row["prompt_sha256"]), 64)
            self.assertIn(row["prompt_variant"], ("A", "C"))

    def test_tokens_accumulated(self):
        runner = self.runner(count=2, variants=("A",))
        summary = runner.run()
        self.assertEqual(summary.total_tokens, 2 * (900 + 40))


class TestFailureHandling(RunnerTestBase):
    def test_api_failure_recorded_not_dropped(self):
        FakeClient.responses = [
            ModelResponse(ok=False, error="HTTP 401：无效的API密钥", http_status=401,
                          latency_seconds=0.2, model="glm-4.6v-flash", attempt=1)
        ]
        runner = self.runner(count=2, variants=("A",))
        summary = runner.run()
        self.assertEqual(summary.n_failed, 2)
        with runner.output_path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)  # 失败样本同样落盘
        self.assertEqual(rows[0]["request_success"], "False")
        self.assertIn("401", rows[0]["error_message"])
        self.assertEqual(rows[0]["predicted_label"], "")

    def test_unparseable_output_not_fabricated(self):
        FakeClient.responses = [
            ModelResponse(ok=True, raw_text="我看不清这张图。", http_status=200,
                          latency_seconds=0.3, model="glm-4.6v-flash")
        ]
        runner = self.runner(count=1, variants=("A",))
        runner.run()
        with runner.output_path.open(encoding="utf-8-sig") as handle:
            row = list(csv.DictReader(handle))[0]
        self.assertEqual(row["predicted_label"], "")
        self.assertEqual(row["schema_valid"], "False")
        self.assertIn("解析", row["error_message"] + "解析")


class TestResumeAndLimits(RunnerTestBase):
    def test_resume_skips_existing(self):
        runner = self.runner(count=3, variants=("A",))
        first = runner.run()
        self.assertEqual(first.n_done, 3)
        second = runner.run()
        self.assertEqual(second.n_done, 0)
        self.assertEqual(second.n_skipped, 3)
        with runner.output_path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)  # 没有重复写入

    def test_max_calls_guard(self):
        runner = self.runner(count=5, variants=("A",), max_calls=2)
        summary = runner.run()
        self.assertLess(summary.n_done, 5)          # 不可能跑完全部
        self.assertGreater(summary.n_aborted, 0)    # 未执行的条数必须如实报告
        self.assertIn("max_calls", summary.stop_reason)
        with runner.output_path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), summary.n_done)  # 已完成的都有记录

    def test_repeats_duplicate_calls(self):
        runner = self.runner(count=1, variants=("A",), repeats=2)
        summary = runner.run()
        self.assertEqual(summary.n_planned, 2)


class TestLoadSamples(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)

    def tearDown(self):
        for path in TMP.glob("*"):
            path.unlink()
        if TMP.is_dir():
            TMP.rmdir()

    def test_loads_and_filters(self):
        image = write_images(1)[0]
        manifest = TMP / "manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "image_path", "true_label", "split", "filter_reason"])
            writer.writerow(["s1", str(image), "standing", "test", ""])
            writer.writerow(["s2", str(image), "sitting", "dev", ""])
            writer.writerow(["s3", str(image), "sitting", "test", "min_side<64px"])
        test_rows = load_samples(manifest, split="test")
        self.assertEqual([s.sample_id for s in test_rows], ["s1"])
        dev_rows = load_samples(manifest, split="dev")
        self.assertEqual([s.sample_id for s in dev_rows], ["s2"])
        limited = load_samples(manifest, split="", limit=1)
        self.assertEqual(len(limited), 1)


if __name__ == "__main__":
    unittest.main()
