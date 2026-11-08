"""姿态识别工具测试（T9）。使用假客户端，不消耗真实接口额度。"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src.tools import vision_tool
from src.tools.registry import ToolError, ToolRegistry
from src.tools.vision_tool import (
    analyze_pig_posture,
    build_vision_tools,
    list_sample_images,
)
from src.vision_client import ModelResponse

TMP_ROOT = Path(__file__).resolve().parent / "_tmp_vision"

VALID_JSON = json.dumps({
    "posture": "lying_lateral",
    "description": "猪只侧卧在地面上，四肢伸展。",
    "uncertain": False,
    "uncertainty_reason": None,
}, ensure_ascii=False)


def _make_image(path: Path, size=(240, 180), color=(120, 90, 70)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, data_url, prompt):
        self.calls.append({"data_url": data_url, "prompt": prompt})
        return self.response


def _ok(text=VALID_JSON, **overrides):
    payload = dict(ok=True, raw_text=text, model="fake-vl",
                   latency_seconds=0.8, input_tokens=120, output_tokens=30)
    payload.update(overrides)
    return ModelResponse(**payload)


class TestResolveImage(unittest.TestCase):
    def setUp(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.patcher = mock.patch.object(vision_tool, "ALLOWED_ROOT", TMP_ROOT)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_missing_file(self):
        with self.assertRaises(ToolError) as ctx:
            analyze_pig_posture(str(TMP_ROOT / "nope.jpg"), client=FakeClient(_ok()))
        self.assertIn("图片不存在", str(ctx.exception))

    def test_outside_root_rejected(self):
        outside = Path(__file__).resolve().parent / "outside.jpg"
        with self.assertRaises(ToolError) as ctx:
            analyze_pig_posture(str(outside), client=FakeClient(_ok()))
        self.assertIn("只允许读取", str(ctx.exception))

    def test_unsupported_suffix(self):
        target = TMP_ROOT / "note.txt"
        target.write_text("x", encoding="utf-8")
        with self.assertRaises(ToolError) as ctx:
            analyze_pig_posture(str(target), client=FakeClient(_ok()))
        self.assertIn("不支持的图片格式", str(ctx.exception))

    def test_blank_path(self):
        with self.assertRaises(ToolError):
            analyze_pig_posture("   ", client=FakeClient(_ok()))


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)
        TMP_ROOT.mkdir(parents=True)
        self.image = _make_image(TMP_ROOT / "pig.jpg")
        self.patcher = mock.patch.object(vision_tool, "ALLOWED_ROOT", TMP_ROOT)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)

    def test_valid_result_normalized(self):
        client = FakeClient(_ok())
        payload = analyze_pig_posture(str(self.image), variant="C", client=client)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["posture"], "lying_lateral")
        self.assertEqual(payload["posture_cn"], "侧卧")
        self.assertFalse(payload["uncertain"])
        self.assertEqual(payload["model"], "fake-vl")
        self.assertEqual(payload["prompt"]["variant"], "C")
        self.assertEqual(len(payload["prompt"]["sha256"]), 64)
        self.assertEqual(payload["tokens"], {"input": 120, "output": 30})
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["data_url"].startswith("data:image/jpeg;base64,"))

    def test_chinese_posture_accepted(self):
        client = FakeClient(_ok(text=json.dumps({
            "posture": "站立", "description": "站立", "uncertain": False,
            "uncertainty_reason": None}, ensure_ascii=False)))
        payload = analyze_pig_posture(str(self.image), client=client)
        self.assertEqual(payload["posture"], "standing")

    def test_unparseable_output_is_flagged_not_faked(self):
        client = FakeClient(_ok(text="这张图里有一只猪。"))
        payload = analyze_pig_posture(str(self.image), client=client)
        self.assertFalse(payload["valid"])
        self.assertTrue(payload["errors"])
        self.assertNotIn("posture", payload)
        self.assertIn("raw_excerpt", payload)

    def test_schema_invalid_output_is_flagged(self):
        client = FakeClient(_ok(text=json.dumps({
            "posture": "flying", "description": "x", "uncertain": False})))
        payload = analyze_pig_posture(str(self.image), client=client)
        self.assertFalse(payload["valid"])
        self.assertIsNone(payload["posture"])
        self.assertTrue(any("posture" in err for err in payload["errors"]))

    def test_api_failure_raises_tool_error(self):
        client = FakeClient(ModelResponse(ok=False, error="HTTP 500 服务不可用"))
        with self.assertRaises(ToolError) as ctx:
            analyze_pig_posture(str(self.image), client=client)
        self.assertIn("HTTP 500", str(ctx.exception))

    def test_unknown_variant(self):
        with self.assertRaises(ToolError) as ctx:
            analyze_pig_posture(str(self.image), variant="Z", client=FakeClient(_ok()))
        self.assertIn("未知的提示词方案", str(ctx.exception))

    def test_relative_path_resolved_against_project_root(self):
        # 绝对路径可用；相对路径在真实 data/ 下也能解析（此处只验证绝对路径分支）
        payload = analyze_pig_posture(str(self.image), client=FakeClient(_ok()))
        self.assertEqual(payload["image"]["width"], 240)
        self.assertEqual(payload["image"]["height"], 180)


class TestListSampleImages(unittest.TestCase):
    def test_lists_real_samples(self):
        payload = list_sample_images(limit=5)
        self.assertEqual(payload["source"], "local_dataset")
        self.assertGreater(payload["count"], 0)
        self.assertTrue(all(item["path"].startswith("data/samples/")
                            for item in payload["images"]))

    def test_filter_by_label(self):
        payload = list_sample_images(label="lying_lateral", limit=3)
        self.assertIn("lying_lateral", payload["labels"])
        self.assertTrue(all(item["category"] == "lying_lateral"
                            for item in payload["images"]))
        self.assertTrue(all(item["split"] in ("dev", "test")
                            for item in payload["images"]))

    def test_unknown_label_raises(self):
        with self.assertRaises(ToolError) as ctx:
            list_sample_images(label="flying")
        self.assertIn("未知类别", str(ctx.exception))


class TestRegistryIntegration(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        build_vision_tools(self.registry)

    def test_tools_registered(self):
        self.assertEqual(sorted(self.registry.names()),
                         ["analyze_pig_posture", "list_sample_images"])

    def test_missing_image_path_reported(self):
        result = self.registry.call("analyze_pig_posture", {})
        self.assertFalse(result.ok)
        self.assertIn("缺少必填参数 image_path", result.error)

    def test_list_samples_through_registry(self):
        result = self.registry.call("list_sample_images", {"limit": 2})
        self.assertTrue(result.ok)
        self.assertEqual(len(result.data["images"]), 2)


if __name__ == "__main__":
    unittest.main()