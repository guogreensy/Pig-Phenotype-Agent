"""Web 接口测试（tornado 测试客户端）。

除特别说明外，全部为**模拟测试**：不访问真实模型 API。
``test_predict_wiring_with_stub`` 用打桩替换推理函数，只验证 HTTP 层接线。
"""

from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from PIL import Image
from tornado.testing import AsyncHTTPTestCase

from config import Settings
from src import webapp


def make_settings(api_key: str = "") -> Settings:
    return Settings(
        provider="zhipu",
        model="glm-4.6v-flash",
        api_key=api_key,
        base_url="https://example.invalid/v4/chat/completions",
        temperature=0.0,
        max_tokens=256,
        timeout_seconds=10.0,
        max_retries=1,
        qps=10.0,
        concurrency=1,
        max_calls=10,
        image_max_side=256,
        jpeg_quality=85,
        host="127.0.0.1",
        port=0,
    )


def png_bytes(width=64, height=64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 150)).save(buffer, format="PNG")
    return buffer.getvalue()


def multipart(fields, files):
    """构造 multipart/form-data 请求体。"""
    boundary = "----PigTestBoundary"
    parts = []
    for name, value in fields.items():
        parts.append(
            ("--{0}\r\nContent-Disposition: form-data; name=\"{1}\"\r\n\r\n{2}\r\n").format(
                boundary, name, value
            ).encode("utf-8")
        )
    for name, filename, content, content_type in files:
        parts.append(
            ("--{0}\r\nContent-Disposition: form-data; name=\"{1}\"; filename=\"{2}\"\r\n"
             "Content-Type: {3}\r\n\r\n").format(boundary, name, filename, content_type).encode("utf-8")
        )
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(("--{0}--\r\n").format(boundary).encode("utf-8"))
    return b"".join(parts), "multipart/form-data; boundary={0}".format(boundary)


class WebTestCase(AsyncHTTPTestCase):
    api_key = ""

    def get_app(self):
        self.settings = make_settings(self.api_key)
        return webapp.make_app(self.settings)


class TestStaticAndHealth(WebTestCase):
    def test_index_served(self):
        response = self.fetch("/")
        self.assertEqual(response.code, 200)
        body = response.body.decode("utf-8")
        self.assertIn("猪只姿态识别", body)
        self.assertIn("/api/predict", body)
        self.assertNotIn("cdn.", body)  # 不依赖外部 CDN

    def test_health_without_key(self):
        response = self.fetch("/api/health")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertTrue(data["ok"])
        self.assertFalse(data["has_api_key"])
        self.assertEqual(data["model"], "glm-4.6v-flash")
        self.assertIn("密钥", data["hint"])

    def test_prompts_exposed_with_hash(self):
        response = self.fetch("/api/prompts")
        data = json.loads(response.body)
        keys = [item["key"] for item in data["prompts"]]
        self.assertEqual(keys, ["A", "B", "C"])
        for item in data["prompts"]:
            self.assertEqual(len(item["sha256"]), 64)
            self.assertTrue(item["text"].strip())

    def test_unknown_route(self):
        response = self.fetch("/no-such")
        self.assertEqual(response.code, 404)
        data = json.loads(response.body)
        self.assertFalse(data["ok"])


class TestPredictValidation(WebTestCase):
    def test_missing_file(self):
        body, content_type = multipart({"variant": "C"}, [])
        response = self.fetch("/api/predict", method="POST", body=body,
                              headers={"Content-Type": content_type})
        self.assertEqual(response.code, 400)
        data = json.loads(response.body)
        self.assertIn("没有收到图片", data["error_message"])

    def test_no_key_blocks_before_processing(self):
        body, content_type = multipart({}, [("image", "a.png", png_bytes(), "image/png")])
        response = self.fetch("/api/predict", method="POST", body=body,
                              headers={"Content-Type": content_type})
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertFalse(data["ok"])
        self.assertTrue(data.get("need_key"))
        self.assertIn("密钥", data["error_message"])


class TestPredictWithKeyValidation(WebTestCase):
    api_key = "sk-fake-for-validation"

    def test_non_image_rejected_without_calling_api(self):
        body, content_type = multipart({}, [("image", "report.pdf", b"%PDF-1.4", "application/pdf")])
        response = self.fetch("/api/predict", method="POST", body=body,
                              headers={"Content-Type": content_type})
        data = json.loads(response.body)
        self.assertFalse(data["ok"])
        self.assertFalse(data["request_success"])
        self.assertIn("无法识别", data["error_message"])
        # 关键：校验失败时不应产生任何 status（说明未调用 API）
        self.assertNotIn("status", data)

    def test_tiny_image_rejected(self):
        body, content_type = multipart({}, [("image", "tiny.png", png_bytes(5, 5), "image/png")])
        response = self.fetch("/api/predict", method="POST", body=body,
                              headers={"Content-Type": content_type})
        data = json.loads(response.body)
        self.assertFalse(data["ok"])
        self.assertIn("尺寸过小", data["error_message"])


class TestPredictWiringWithStub(WebTestCase):
    """打桩替换推理函数，验证 HTTP 层把结果如实透传。"""

    api_key = "sk-fake"

    def test_result_passthrough(self):
        stub = {
            "ok": True,
            "timestamp": "2026-10-19T12:00:00",
            "variant": {"key": "C", "title": "C 定义+结构化输出+不确定性", "sha256": "a" * 64},
            "request_success": True,
            "error_message": None,
            "posture": "lying_lateral",
            "posture_cn": "侧卧",
            "description": "身体侧向躺卧。",
            "uncertain": False,
            "uncertainty_reason": None,
            "status": {"request_success": True, "http_status": 200, "latency_seconds": 1.23},
            "parse": {"json_valid": True, "method": "direct", "schema_valid": True,
                      "errors": [], "warnings": []},
            "raw_text": '{"posture": "lying_lateral"}',
            "image_meta": {"original": "64×64", "sent": "64×64", "bytes_sent": 500,
                           "estimated_visual_tokens": 6},
        }
        with mock.patch.object(webapp, "run_single", return_value=stub) as patched:
            body, content_type = multipart({"variant": "C"},
                                           [("image", "pig.png", png_bytes(), "image/png")])
            response = self.fetch("/api/predict", method="POST", body=body,
                                  headers={"Content-Type": content_type})
        data = json.loads(response.body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["posture"], "lying_lateral")
        self.assertEqual(data["posture_cn"], "侧卧")
        self.assertEqual(data["variant"]["key"], "C")
        self.assertIn("settings", data)
        self.assertEqual(patched.call_count, 1)
        # 传给推理函数的 variant 应是表单里的值
        self.assertEqual(patched.call_args[0][2], "C")


class TestAgentEndpointValidation(WebTestCase):
    """助手接口的参数与阻塞条件。"""

    def _post(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.fetch("/api/agent", method="POST", body=body,
                          headers={"Content-Type": "application/json"})

    def test_missing_question(self):
        response = self._post({})
        self.assertEqual(response.code, 400)
        self.assertIn("请先输入问题", json.loads(response.body)["error_message"])

    def test_blank_question(self):
        response = self._post({"question": "   "})
        self.assertEqual(response.code, 400)

    def test_too_long_question(self):
        response = self._post({"question": "猪" * 501})
        self.assertEqual(response.code, 400)
        self.assertIn("过长", json.loads(response.body)["error_message"])

    def test_invalid_tools(self):
        response = self._post({"question": "问题", "tools": "bogus"})
        self.assertEqual(response.code, 400)
        self.assertIn("tools", json.loads(response.body)["error_message"])

    def test_bad_json_body(self):
        response = self.fetch("/api/agent", method="POST", body=b"not json",
                              headers={"Content-Type": "application/json"})
        self.assertEqual(response.code, 400)
        self.assertIn("不是合法 JSON", json.loads(response.body)["error_message"])

    def test_no_key_blocks_without_calling_agent(self):
        with mock.patch.object(webapp, "run_agent") as patched:
            response = self._post({"question": "有哪些栏位"})
        data = json.loads(response.body)
        self.assertFalse(data["ok"])
        self.assertTrue(data["need_key"])
        self.assertEqual(patched.call_count, 0)

    def test_busy_lock_does_not_reach_agent(self):
        """未配密钥时先报密钥缺失，不会走到实际执行。"""
        with mock.patch.object(webapp, "run_agent") as patched:
            response = self._post({"question": "有哪些栏位"})
        self.assertEqual(patched.call_count, 0)
        self.assertTrue(json.loads(response.body)["need_key"])

    def test_health_exposes_agent_tools(self):
        response = self.fetch("/api/health")
        data = json.loads(response.body)
        agent = data["agent"]
        self.assertTrue(agent["model"])
        self.assertGreaterEqual(len(agent["tools"]), 8)
        self.assertIn("all", agent["presets"])
        names = {item["name"] for item in agent["tools"]}
        self.assertIn("search_knowledge", names)
        self.assertIn("detect_anomalies", names)


class TestAgentWiringWithStub(WebTestCase):
    """打桩替换 run_agent，验证参数转发与结果透传。"""

    api_key = "sk-fake"

    def _post(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.fetch("/api/agent", method="POST", body=body,
                          headers={"Content-Type": "application/json"})

    def test_result_passthrough(self):
        stub = {
            "ok": True,
            "report": {
                "question": "有哪些栏位",
                "answer": "四个栏位：P1–P4。",
                "ok": True,
                "stop_reason": "answered",
                "stop_reason_cn": "正常作答",
                "citations": [],
                "tool_trace": [{"tool": "list_pens", "args": {}, "ok": True, "ms": 1.2}],
                "flags": {"uses_simulated_data": True},
                "metrics": {"model_calls": 2, "tool_calls": 1},
                "prompt_meta": {"version": "1.0", "sha256": "b" * 64},
            },
            "markdown": "## 回答\n\n四个栏位。",
            "tool_stats": {"calls": 1, "failed": 0},
            "tools": ["list_pens"],
            "model": "glm-4.5-flash",
            "timestamp": "2026-11-07T18:00:00",
            "error_message": None,
        }
        with mock.patch.object(webapp, "run_agent", return_value=stub) as patched:
            response = self._post({"question": "有哪些栏位", "tools": "data", "max_steps": 4})
        data = json.loads(response.body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["report"]["answer"], "四个栏位：P1–P4。")
        self.assertEqual(data["tools"], ["list_pens"])
        self.assertEqual(patched.call_count, 1)
        call = patched.call_args[0]
        self.assertEqual(call[0], "有哪些栏位")
        self.assertEqual(call[1], "data")
        self.assertEqual(call[2], 4)

    def test_max_steps_clamped(self):
        stub = {"ok": True, "report": {"answer": "x"}, "markdown": "", "tool_stats": {},
                "tools": [], "model": "m", "timestamp": "", "error_message": None}
        with mock.patch.object(webapp, "run_agent", return_value=stub) as patched:
            self._post({"question": "问题", "max_steps": 99})
        self.assertEqual(patched.call_args[0][2], webapp.MAX_AGENT_STEPS)

    def test_zero_steps_clamped_to_one(self):
        stub = {"ok": True, "report": {"answer": "x"}, "markdown": "", "tool_stats": {},
                "tools": [], "model": "m", "timestamp": "", "error_message": None}
        with mock.patch.object(webapp, "run_agent", return_value=stub) as patched:
            self._post({"question": "问题", "max_steps": 0})
        self.assertEqual(patched.call_args[0][2], 1)

    def test_unexpected_error_returns_500(self):
        with mock.patch.object(webapp, "run_agent", side_effect=RuntimeError("炸了")):
            response = self._post({"question": "问题"})
        self.assertEqual(response.code, 500)
        self.assertIn("处理失败", json.loads(response.body)["error_message"])

    def test_busy_lock_returns_429(self):
        """占住锁再请求：应直接 429，且不进入实际执行。"""
        acquired = webapp._AGENT_LOCK.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            with mock.patch.object(webapp, "run_agent") as patched:
                response = self._post({"question": "有哪些栏位"})
        finally:
            webapp._AGENT_LOCK.release()
        self.assertEqual(response.code, 429)
        self.assertTrue(json.loads(response.body)["busy"])
        self.assertEqual(patched.call_count, 0)

    def test_lock_released_after_failure(self):
        """异常路径也必须释放锁，否则助手会永久不可用。"""
        with mock.patch.object(webapp, "run_agent", side_effect=RuntimeError("炸了")):
            self._post({"question": "问题"})
        self.assertTrue(webapp._AGENT_LOCK.acquire(blocking=False))
        webapp._AGENT_LOCK.release()


class TestFrontendScriptIntegrity(unittest.TestCase):
    """页面内嵌 JS 的完整性。

    ``INDEX_HTML`` 曾因普通三引号字符串把 JS 里的 ``'\\n'`` 解释成真实换行，
    导致整个 <script> 块语法错误、页面所有交互静默失效。这里把它钉住。
    """

    def test_newline_escape_not_eaten_by_python(self):
        html = webapp.INDEX_HTML
        # 正确形态：JS 源码里是反斜杠 + n（Python 字面量为 '\\n'）
        self.assertIn("split('\\n')", html)
        # 错误形态：被 Python 变成真实换行
        self.assertNotIn("split('\n')", html)

    def test_assistant_panel_present(self):
        html = webapp.INDEX_HTML
        for marker in ("agentQ", "agentRun", "agentResult", "agentTools",
                       "askAgent", "renderAgent", "mdToHtml", "/api/agent"):
            self.assertIn(marker, html, marker)

    def test_script_block_is_balanced(self):
        html = webapp.INDEX_HTML
        self.assertEqual(html.count("<script>"), 1)
        self.assertEqual(html.count("</script>"), 1)


if __name__ == "__main__":
    unittest.main()
