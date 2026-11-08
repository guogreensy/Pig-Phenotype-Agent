"""``src.vision_client`` 的异常处理测试。

**这是模拟测试**：用假的 HTTP 会话替代真实请求，验证错误处理与重试逻辑。
真实 API 集成测试见 ``scripts/verify_api.py``（需要密钥，单独运行）。
"""

from __future__ import annotations

import json
import unittest

import requests

from config import Settings
from src.vision_client import RateLimiter, VisionClient, check_connectivity


def make_settings(**overrides) -> Settings:
    base = dict(
        provider="zhipu",
        model="glm-4.6v-flash",
        api_key="sk-test",
        base_url="https://example.invalid/v4/chat/completions",
        temperature=0.0,
        max_tokens=512,
        timeout_seconds=5.0,
        max_retries=3,
        qps=1000.0,
        concurrency=1,
        max_calls=10,
        image_max_side=512,
        jpeg_quality=90,
        host="127.0.0.1",
        port=7860,
    )
    base.update(overrides)
    return Settings(**base)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self._text = text
        self.headers = headers or {}

    @property
    def text(self):
        if self._text is not None:
            return self._text
        return json.dumps(self._payload) if self._payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def success_payload(text='{"posture": "standing", "description": "站立", "uncertain": false}'):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 30},
    }


def make_client(responses, settings=None):
    settings = settings or make_settings()
    session = FakeSession(responses)
    limiter = RateLimiter(settings.qps, sleep=lambda _s: None)
    client = VisionClient(settings, session=session, limiter=limiter, sleep=lambda _s: None)
    return client, session


DATA_URL = "data:image/jpeg;base64,AAAA"
PROMPT = "请判断姿态"


class TestSuccessPaths(unittest.TestCase):
    def test_normal_success(self):
        client, session = make_client([FakeResponse(200, success_payload())])
        response = client.call(DATA_URL, PROMPT)
        self.assertTrue(response.ok)
        self.assertEqual(response.http_status, 200)
        self.assertIn("posture", response.raw_text)
        self.assertEqual(response.input_tokens, 900)
        self.assertGreaterEqual(response.latency_seconds, 0.0)
        self.assertEqual(len(session.calls), 1)

    def test_payload_shape(self):
        client, session = make_client([FakeResponse(200, success_payload())])
        client.call(DATA_URL, PROMPT)
        payload = session.calls[0]["json"]
        self.assertEqual(payload["model"], "glm-4.6v-flash")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[0]["image_url"]["url"], DATA_URL)
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertIn("Authorization", session.calls[0]["headers"])

    def test_content_as_list(self):
        payload = {
            "choices": [{"message": {"content": [{"type": "text", "text": '{"posture":"sitting"}'}]}}],
            "usage": {},
        }
        client, _ = make_client([FakeResponse(200, payload)])
        response = client.call(DATA_URL, PROMPT)
        self.assertTrue(response.ok)
        self.assertIn("sitting", response.raw_text)


class TestAuthErrors(unittest.TestCase):
    def test_401_not_retried(self):
        client, session = make_client([FakeResponse(401, {"error": {"code": "invalid_api_key",
                                                                  "message": "无效的API密钥"}})])
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertEqual(response.http_status, 401)
        self.assertEqual(len(session.calls), 1)  # 不重试
        self.assertIn("密钥", response.error)

    def test_403_not_retried(self):
        client, session = make_client([FakeResponse(403, {"error": "forbidden"})])
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertEqual(len(session.calls), 1)


class TestRetryBehaviour(unittest.TestCase):
    def test_429_then_success(self):
        client, session = make_client([
            FakeResponse(429, {"error": "rate limit"}, headers={"Retry-After": "0"}),
            FakeResponse(200, success_payload()),
        ])
        response = client.call(DATA_URL, PROMPT)
        self.assertTrue(response.ok)
        self.assertEqual(response.attempt, 2)
        self.assertEqual(len(session.calls), 2)

    def test_500_persists_then_fail(self):
        settings = make_settings(max_retries=3)
        client, session = make_client(
            [FakeResponse(500, {"error": "boom"}) for _ in range(3)], settings=settings
        )
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(response.http_status, 500)

    def test_timeout_then_fail(self):
        settings = make_settings(max_retries=2)
        client, _ = make_client([requests.Timeout("slow"), requests.Timeout("slow")], settings=settings)
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertIn("超时", response.error)

    def test_connection_error(self):
        settings = make_settings(max_retries=2)
        client, _ = make_client(
            [requests.ConnectionError("down"), requests.ConnectionError("down")], settings=settings
        )
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertIn("网络请求失败", response.error)


class TestBadResponses(unittest.TestCase):
    def test_empty_content(self):
        payload = {"choices": [{"message": {"content": "   "}}], "usage": {}}
        client, _ = make_client([FakeResponse(200, payload)])
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertIn("为空", response.error)

    def test_non_json_body(self):
        client, _ = make_client([FakeResponse(200, None, text="<html>oops</html>")])
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertIn("JSON", response.error)

    def test_no_choices(self):
        client, _ = make_client([FakeResponse(200, {"choices": [], "usage": {}})])
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)


class TestThinkingFallback(unittest.TestCase):
    def test_400_on_thinking_drops_param(self):
        reject = FakeResponse(
            400, {"error": {"code": "invalid_parameter",
                            "message": "unknown field thinking"}}
        )
        client, session = make_client([reject, FakeResponse(200, success_payload())])
        response = client.call(DATA_URL, PROMPT)
        self.assertTrue(response.ok)
        self.assertNotIn("thinking", session.calls[1]["json"])
        self.assertTrue(any("thinking" in note for note in response.notes))


class TestMissingKey(unittest.TestCase):
    def test_no_key_returns_error_without_request(self):
        client, session = make_client([], settings=make_settings(api_key=""))
        response = client.call(DATA_URL, PROMPT)
        self.assertFalse(response.ok)
        self.assertIn("密钥", response.error)
        self.assertEqual(len(session.calls), 0)


class TestThinkingParameter(unittest.TestCase):
    def test_hybrid_model_sends_disabled_thinking(self):
        client, session = make_client([FakeResponse(200, success_payload())])
        client.call(DATA_URL, PROMPT)
        self.assertEqual(session.calls[0]["json"]["thinking"], {"type": "disabled"})

    def test_forced_thinking_model_omits_parameter(self):
        settings = make_settings(model="glm-4.1v-thinking-flash")
        client, session = make_client([FakeResponse(200, success_payload())], settings=settings)
        client.call(DATA_URL, PROMPT)
        self.assertNotIn("thinking", session.calls[0]["json"])


class TestRateLimiter(unittest.TestCase):
    def test_interval_respected(self):
        sleeps = []
        limiter = RateLimiter(2.0, clock=lambda: 0.0, sleep=sleeps.append)
        limiter.acquire()
        limiter.acquire()
        self.assertGreaterEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[-1], 0.5, places=3)


class TestConnectivityProbe(unittest.TestCase):
    def test_invalid_host(self):
        ok, message = check_connectivity(make_settings(base_url="https://127.0.0.1:1/none"))
        self.assertFalse(ok)
        self.assertIn("无法连接", message)


if __name__ == "__main__":
    unittest.main()
