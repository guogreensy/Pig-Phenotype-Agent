"""执行器测试（T13）。用脚本化假客户端驱动完整循环，不调用真实模型。"""

from __future__ import annotations

import json
import unittest

from src.agent.executor import AgentExecutor
from src.agent.llm_client import ChatResponse, ToolCall
from src.tools.registry import ToolRegistry, ToolSpec


class ScriptedClient:
    """按脚本依次返回响应，并记录每次收到的消息与工具。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice=None, **kwargs):
        self.calls.append({
            "messages": [dict(item) for item in messages],
            "tools": tools,
        })
        if not self.responses:
            raise AssertionError("脚本已用尽")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _answer(text="结论：一切正常。"):
    return ChatResponse(ok=True, text=text, input_tokens=10, output_tokens=5,
                        latency_seconds=0.5, model="fake")


def _tool_request(name="list_pens", arguments="{}", call_id="call_1"):
    try:
        parsed = json.loads(arguments)
    except ValueError as exc:
        call = ToolCall(id=call_id, name=name, arguments_raw=arguments, arguments={},
                        parse_error="参数不是合法 JSON：{0}".format(exc))
    else:
        call = ToolCall(id=call_id, name=name, arguments_raw=arguments,
                        arguments=parsed if isinstance(parsed, dict) else {})
    return ChatResponse(
        ok=True, text="", input_tokens=12, output_tokens=6, latency_seconds=0.4,
        model="fake", tool_calls=[call],
    )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    def pens(args):
        return {"pens": [{"pen_id": "P1", "pigs": 12}], "source": "simulated"}

    def knowledge(args):
        return {"citations": [
            {"chunk_id": "pmc_1#0001", "title": "论文甲", "section": "Results",
             "url": "https://example.org/1", "license": "CC BY", "score": 0.7},
        ], "count": 1}

    def broken(args):
        raise RuntimeError("数据库连接失败")

    registry.register(ToolSpec(name="list_pens", description="栏位",
                               parameters={"type": "object", "properties": {}},
                               handler=pens))
    registry.register(ToolSpec(name="search_knowledge", description="检索",
                               parameters={"type": "object", "properties": {
                                   "query": {"type": "string", "default": "q"}},
                                   "required": []},
                               handler=knowledge))
    registry.register(ToolSpec(name="broken", description="会失败",
                               parameters={"type": "object", "properties": {}},
                               handler=broken))
    return registry


class TestLoop(unittest.TestCase):
    def test_direct_answer(self):
        client = ScriptedClient([_answer()])
        result = AgentExecutor(_registry(), client).run("猪只均重多少")
        self.assertTrue(result.ok)
        self.assertEqual(result.stop_reason, "answered")
        self.assertEqual(result.answer, "结论：一切正常。")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.tool_calls, 0)

    def test_tool_then_answer(self):
        client = ScriptedClient([_tool_request("list_pens"), _answer("四个栏位")])
        result = AgentExecutor(_registry(), client).run("有哪些栏位")
        self.assertTrue(result.ok)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.used_tools, ["list_pens"])
        self.assertEqual(result.answer, "四个栏位")
        # 第二轮消息里必须带上 tool 角色结果
        second_messages = client.calls[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_1")
        self.assertIn("simulated", second_messages[-1]["content"])

    def test_citations_collected_and_deduped(self):
        arguments = '{"query": "猪只姿态"}'
        client = ScriptedClient([
            _tool_request("search_knowledge", arguments, "call_a"),
            _tool_request("search_knowledge", arguments, "call_b"),
            _answer("见文献"),
        ])
        result = AgentExecutor(_registry(), client).run("猪只姿态识别有哪些方法")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0]["chunk_id"], "pmc_1#0001")

    def test_tool_failure_fed_back(self):
        client = ScriptedClient([_tool_request("broken"), _answer("工具坏了，无法查询")])
        result = AgentExecutor(_registry(), client).run("查一下")
        self.assertTrue(result.ok)
        tool_step = [step for step in result.steps if step.kind == "tool"][0]
        self.assertFalse(tool_step.ok)
        self.assertIn("RuntimeError", tool_step.error)
        fed_back = client.calls[1]["messages"][-1]["content"]
        self.assertIn("数据库连接失败", fed_back)

    def test_invalid_json_arguments_not_executed(self):
        client = ScriptedClient([_tool_request("list_pens", "{bad json"),
                                 _answer("参数有误，改为直接回答")])
        result = AgentExecutor(_registry(), client).run("有哪些栏位")
        tool_step = [step for step in result.steps if step.kind == "tool"][0]
        self.assertFalse(tool_step.ok)
        self.assertIn("参数无法解析", tool_step.error)
        self.assertEqual(result.tool_calls, 1)   # 计一次，但没有真正执行工具

    def test_model_error_stops(self):
        client = ScriptedClient([ChatResponse(ok=False, error="HTTP 500 服务异常")])
        result = AgentExecutor(_registry(), client).run("问题")
        self.assertFalse(result.ok)
        self.assertEqual(result.stop_reason, "model_error")
        self.assertIn("HTTP 500", result.error)
        self.assertEqual(result.answer, "")

    def test_max_steps_forces_answer(self):
        # 两步循环各要一次工具，第三次调用是"不带工具"的收尾
        responses = [_tool_request("list_pens", call_id="c1"),
                     _tool_request("list_pens", call_id="c2"),
                     _answer("根据已有结果：四个栏位。")]
        client = ScriptedClient(responses)
        result = AgentExecutor(_registry(), client, max_steps=2).run("有哪些栏位")
        self.assertTrue(result.ok)
        self.assertEqual(result.stop_reason, "forced_answer")
        self.assertIn("四个栏位", result.answer)
        # 收尾调用必须不带工具
        self.assertIsNone(client.calls[-1]["tools"])
        self.assertTrue(any("步数达到上限" in note for note in result.degraded))

    def test_forced_call_returning_tool_calls_is_failure(self):
        """收尾调用若仍请求工具，不能把空回答当成功。"""
        responses = [_tool_request("list_pens", call_id="c1"),
                     _tool_request("list_pens", call_id="c2"),
                     _tool_request("list_pens", call_id="c3")]
        client = ScriptedClient(responses)
        result = AgentExecutor(_registry(), client, max_steps=2).run("有哪些栏位")
        self.assertFalse(result.ok)
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.answer, "")

    def test_forced_answer_failure(self):
        responses = [_tool_request("list_pens", call_id="c1"),
                     ChatResponse(ok=False, error="收尾也失败")]
        client = ScriptedClient(responses)
        result = AgentExecutor(_registry(), client, max_steps=1).run("有哪些栏位")
        self.assertFalse(result.ok)
        self.assertEqual(result.stop_reason, "max_steps")

    def test_repeat_guard_blocks_third_identical_call(self):
        responses = [_tool_request("list_pens", call_id="c{0}".format(i)) for i in range(3)]
        responses.append(_answer("好的"))
        client = ScriptedClient(responses)
        result = AgentExecutor(_registry(), client, max_steps=4).run("有哪些栏位")
        tool_steps = [step for step in result.steps if step.kind == "tool"]
        self.assertEqual(len(tool_steps), 3)
        self.assertTrue(tool_steps[0].ok)
        self.assertTrue(tool_steps[1].ok)
        self.assertFalse(tool_steps[2].ok)
        self.assertIn("不再重复执行", tool_steps[2].error)

    def test_empty_question(self):
        client = ScriptedClient([])
        result = AgentExecutor(_registry(), client).run("   ")
        self.assertFalse(result.ok)
        self.assertEqual(result.stop_reason, "invalid_question")
        self.assertEqual(client.calls, [])

    def test_token_accounting(self):
        client = ScriptedClient([_tool_request("list_pens"), _answer()])
        result = AgentExecutor(_registry(), client).run("问题")
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.input_tokens, 22)
        self.assertEqual(result.output_tokens, 11)

    def test_prompt_meta_recorded(self):
        client = ScriptedClient([_answer()])
        result = AgentExecutor(_registry(), client).run("问题")
        self.assertEqual(len(result.prompt_meta["sha256"]), 64)
        self.assertIn("list_pens", result.prompt_meta["tools"])

    def test_tools_passed_on_every_call(self):
        client = ScriptedClient([_tool_request("list_pens"), _answer()])
        AgentExecutor(_registry(), client).run("问题")
        self.assertEqual(len(client.calls[0]["tools"]), 3)
        self.assertEqual(len(client.calls[1]["tools"]), 3)


if __name__ == "__main__":
    unittest.main()