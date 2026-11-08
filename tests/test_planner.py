"""规划器测试（T12）。"""

from __future__ import annotations

import unittest
from datetime import date

from src.agent.planner import (
    PLANNER_VERSION,
    build_messages,
    build_system_prompt,
    describe_prompt,
    prompt_fingerprint,
)
from src.tools.registry import ToolRegistry, ToolSpec


def _registry(with_tools: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    if with_tools:
        registry.register(ToolSpec(
            name="list_pens",
            description="列出栏位",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: {"pens": []},
        ))
        registry.register(ToolSpec(
            name="search_knowledge",
            description="检索知识库",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]},
            handler=lambda args: {"citations": []},
        ))
    return registry


class TestSystemPrompt(unittest.TestCase):
    def test_contains_tool_list(self):
        prompt = build_system_prompt(_registry())
        self.assertIn("list_pens", prompt)
        self.assertIn("search_knowledge", prompt)
        self.assertIn("列出栏位", prompt)

    def test_contains_hard_rules(self):
        prompt = build_system_prompt(_registry())
        self.assertIn("先调用工具", prompt)
        self.assertIn("模拟", prompt)
        self.assertIn("不要虚构文献", prompt)
        self.assertIn("不要用推测替代失败的结果", prompt)

    def test_no_tool_variant(self):
        prompt = build_system_prompt(_registry(with_tools=False))
        self.assertIn("（无）", prompt)
        self.assertIn("没有可调用的工具", prompt)
        self.assertNotIn("先调用工具", prompt)

    def test_max_steps_mentioned(self):
        prompt = build_system_prompt(_registry(), max_steps=4)
        self.assertIn("最多可进行 4 步", prompt)

    def test_data_range_included(self):
        prompt = build_system_prompt(_registry(), today=date(2026, 11, 7))
        self.assertIn("2026-10-08", prompt)
        self.assertIn("2026-11-06", prompt)
        self.assertIn("2026-11-07", prompt)

    def test_stable_across_calls(self):
        first = build_system_prompt(_registry(), today=date(2026, 11, 7))
        second = build_system_prompt(_registry(), today=date(2026, 11, 7))
        self.assertEqual(first, second)


class TestMessages(unittest.TestCase):
    def test_system_first_question_last(self):
        messages = build_messages("猪只均重多少", _registry())
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "猪只均重多少"})

    def test_history_inserted(self):
        history = [{"role": "user", "content": "上一问"},
                   {"role": "assistant", "content": "上一答"},
                   {"role": "tool", "content": "应被忽略"}]
        messages = build_messages("新问题", _registry(), history=history)
        roles = [item["role"] for item in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("   ", _registry())

    def test_question_stripped(self):
        messages = build_messages("  有空格  ", _registry())
        self.assertEqual(messages[-1]["content"], "有空格")


class TestFingerprint(unittest.TestCase):
    def test_sha_shape(self):
        digest = prompt_fingerprint("abc")
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, prompt_fingerprint("abc"))
        self.assertNotEqual(digest, prompt_fingerprint("abd"))

    def test_describe_prompt(self):
        meta = describe_prompt(_registry(), max_steps=5)
        self.assertEqual(meta["version"], PLANNER_VERSION)
        self.assertEqual(len(meta["sha256"]), 64)
        self.assertEqual(meta["tools"], ["list_pens", "search_knowledge"])
        self.assertEqual(meta["max_steps"], 5)
        self.assertGreater(meta["chars"], 200)


if __name__ == "__main__":
    unittest.main()