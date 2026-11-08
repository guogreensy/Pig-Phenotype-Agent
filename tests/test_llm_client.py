"""对话客户端测试（T11）。全部使用假 session，不发起真实请求。"""

from __future__ import annotations

import json
import unittest

from src.agent.llm_client import (
    ChatClient,
    parse_tool_calls,
    tool_model_name,
)
from src.vision_client import RateLimiter


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """按顺序返回预设响应，并记录每次请求体。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.payloads.append({"url": url, "headers": headers, "json": json})
        if not self.responses:
            raise AssertionError("没有更多预设响应")
        return self.responses.pop(0)


def _client(responses, model="glm-4.5-flash"):
    return ChatClient(
        api_key="test-key",
        base_url="https://example.invalid/chat/completions",
        model=model,
        session=FakeSession(responses),
        limiter=RateLimiter(1000.0),
        sleep=lambda _s: None,
        max_retries=2,
    )


def _text_body(text, prompt_tokens=10, completion_tokens=5):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _tool_body(name, arguments, call_id="call_1"):
    return {
        "choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": arguments}}],
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }


class TestParseToolCalls(unittest.TestCase):
    def test_valid_json(self):
        message = {"tool_calls": [{"id": "c1", "function": {
            "name": "list_pens", "arguments": '{"limit": 3}'}}]}
        calls = parse_tool_calls(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "list_pens")
        self.assertEqual(calls[0].arguments, {"limit": 3})
        self.assertIsNone(calls[0].parse_error)

    def test_invalid_json_records_error(self):
        message = {"tool_calls": [{"id": "c1", "function": {
            "name": "list_pens", "arguments": "{not json"}}]}
        calls = parse_tool_calls(message)
        self.assertEqual(calls[0].arguments, {})
        self.assertIn("不是合法 JSON", calls[0].parse_error)

    def test_non_object_json_rejected(self):
        message = {"tool_calls": [{"id": "c1", "function": {
            "name": "x", "arguments": "[1, 2]"}}]}
        calls = parse_tool_calls(message)
        self.assertIn("应为 JSON 对象", calls[0].parse_error)

    def test_dict_arguments_accepted(self):
        message = {"tool_calls": [{"id": "c1", "function": {
            "name": "x", "arguments": {"a": 1}}}]}
        calls = parse_tool_calls(message)
        self.assertEqual(calls[0].arguments, {"a": 1})

    def test_missing_id_generated(self):
        message = {"tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]}
        self.assertEqual(parse_tool_calls(message)[0].id, "call_0")

    def test_no_tool_calls(self):
        self.assertEqual(parse_tool_calls({"content": "hi"}), [])
        self.assertEqual(parse_tool_calls(None), [])
        self.assertEqual(parse_tool_calls({"tool_calls": "oops"}), [])


class TestPayload(unittest.TestCase):
    def test_tools_included(self):
        client = _client([])
        payload = client.build_payload([{"role": "user", "content": "hi"}],
                                       tools=[{"type": "function", "function": {"name": "f"}}],
                                       tool_choice="auto")
        self.assertIn("tools", payload)
        self.assertEqual(payload["tool_choice"], "auto")

    def test_thinking_disabled_for_normal_model(self):
        client = _client([])
        payload = client.build_payload([{"role": "user", "content": "hi"}])
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_thinking_not_sent_to_thinking_model(self):
        client = _client([], model="glm-4.5-thinking")
        payload = client.build_payload([{"role": "user", "content": "hi"}])
        self.assertNotIn("thinking", payload)

    def test_temperature_override(self):
        client = _client([])
        payload = client.build_payload([{"role": "user", "content": "hi"}],
                                       temperature=0.7, max_tokens=64)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 64)


class TestChat(unittest.TestCase):
    def test_plain_answer(self):
        client = _client([FakeResponse(200, _text_body("你好"))])
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "你好")
        self.assertFalse(response.wants_tools)
        self.assertEqual(response.input_tokens, 10)

    def test_tool_calls_parsed(self):
        client = _client([FakeResponse(200, _tool_body("list_pens", "{}"))])
        response = client.chat([{"role": "user", "content": "hi"}],
                               tools=[{"type": "function", "function": {"name": "list_pens"}}])
        self.assertTrue(response.ok)
        self.assertTrue(response.wants_tools)
        self.assertEqual(response.tool_calls[0].name, "list_pens")

    def test_empty_message_fails(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "  "}}]}
        client = _client([FakeResponse(200, body)])
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertFalse(response.ok)
        self.assertIn("既没有返回文本也没有请求工具", response.error)

    def test_missing_api_key(self):
        client = ChatClient(api_key="", base_url="https://x", model="m",
                            limiter=RateLimiter(1000.0), sleep=lambda _s: None)
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertFalse(response.ok)
        self.assertIn("未配置 API 密钥", response.error)

    def test_auth_error_gives_hint(self):
        client = _client([FakeResponse(401, {"error": {"code": "invalid_api_key"}})])
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertFalse(response.ok)
        self.assertIn("密钥无效", response.error)
        self.assertEqual(response.attempts, 1)

    def test_thinking_rejected_then_retried(self):
        client = _client([
            FakeResponse(400, {"error": {"message": "unknown parameter: thinking"}}),
            FakeResponse(200, _text_body("ok")),
        ])
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "ok")
        self.assertTrue(any("thinking" in note for note in response.notes))
        # 第二次请求不应该再带 thinking
        self.assertNotIn("thinking", client._session.payloads[1]["json"])

    def test_retryable_error_retried(self):
        client = _client([
            FakeResponse(503, {"error": {"message": "busy"}}),
            FakeResponse(200, _text_body("ok")),
        ])
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertTrue(response.ok)
        self.assertEqual(response.attempts, 2)

    def test_non_json_body(self):
        client = _client([FakeResponse(200, None)])
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertFalse(response.ok)
        self.assertIn("不是合法 JSON", response.error)

    def test_tool_model_name_default(self):
        self.assertTrue(tool_model_name())
        self.assertIn("glm", tool_model_name().lower())


if __name__ == "__main__":
    unittest.main()