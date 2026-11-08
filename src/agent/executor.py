"""工具调用循环（T13）。

一轮问答的控制流
----------------
1. 用 planner 装配消息（system + 问题）；
2. 把工具 schema 一起发给模型；
3. 模型要么给最终回答，要么请求调用工具：
   - 请求工具 → 逐个执行 → 把结果按 ``tool_call_id`` 回填 → 回到第 2 步；
   - 给回答   → 结束，记为 ``answered``；
4. 步数用尽仍没结论 → 再做**一次不带工具**的收尾调用，要求"基于已有结果作答"，
   记为 ``forced_answer``。宁可明确说"信息不足"，也不让循环空转或凭空编造。

两个防御性设计
--------------
* **重复调用拦截**：同样的工具 + 同样的参数在短窗口内重复出现时不再执行，
  而是回传"已调用过，请换方式"，避免模型陷入死循环烧掉额度；
* **参数解析失败不回传垃圾**：模型给出非法 JSON 时，把解析错误告诉它去修正，
  而不是拿空参去执行一个可能有害的调用。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from src.agent.llm_client import ChatClient, ChatResponse, ToolCall
from src.agent.planner import build_messages, describe_prompt
from src.tools.registry import ToolRegistry, ToolResult

#: 默认最大模型步数
DEFAULT_MAX_STEPS = 6

#: 默认单轮工具调用上限
DEFAULT_MAX_TOOL_CALLS = 8

#: 同一 (工具, 参数) 允许出现的次数，超过即拦截
DEFAULT_REPEAT_GUARD = 2


@dataclass
class StepRecord:
    """一步调用记录（模型一步或工具一步）。"""

    index: int
    kind: str                      # model | tool
    text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: Optional[str] = None
    preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "text": self.text,
            "tool_calls": self.tool_calls,
            "latency_seconds": round(self.latency_seconds, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "error": self.error,
            "preview": self.preview,
        }


@dataclass
class AgentResult:
    """一次完整问答的结果。"""

    question: str
    answer: str = ""
    ok: bool = True
    error: Optional[str] = None
    stop_reason: str = "answered"
    steps: List[StepRecord] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0.0
    used_tools: List[str] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    prompt_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "ok": self.ok,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "tool_calls": self.tool_calls,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "used_tools": self.used_tools,
            "degraded": self.degraded,
            "citations": self.citations,
            "steps": [step.to_dict() for step in self.steps],
            "prompt_meta": self.prompt_meta,
        }


class AgentExecutor:
    """按"模型 → 工具 → 模型"循环推进，直到得到回答或触达上限。"""

    def __init__(self, registry: ToolRegistry, client: ChatClient,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
                 repeat_guard: int = DEFAULT_REPEAT_GUARD):
        self.registry = registry
        self.client = client
        self.max_steps = max(1, max_steps)
        self.max_tool_calls = max(1, max_tool_calls)
        self.repeat_guard = max(1, repeat_guard)

    # ------------------------------------------------------------------ 主流程
    def run(self, question: str, history: Optional[List[Dict[str, Any]]] = None,
            today: Optional[date] = None) -> AgentResult:
        started = time.monotonic()
        result = AgentResult(question=(question or "").strip())
        result.prompt_meta = describe_prompt(self.registry, self.max_steps)

        try:
            messages = build_messages(question, self.registry, self.max_steps,
                                      history=history, today=today)
        except ValueError as exc:
            result.ok = False
            result.error = str(exc)
            result.stop_reason = "invalid_question"
            result.elapsed_seconds = time.monotonic() - started
            return result

        tools = self.registry.schemas()
        seen: Dict[str, int] = {}
        step_index = 0

        for _ in range(self.max_steps):
            step_index += 1
            response = self.client.chat(messages, tools=tools)
            self._record_model(result, step_index, response)
            if not response.ok:
                result.ok = False
                result.error = response.error
                result.stop_reason = "model_error"
                result.messages = messages
                result.elapsed_seconds = time.monotonic() - started
                return result

            messages.append(self._assistant_message(response))

            if not response.wants_tools:
                result.answer = response.text.strip()
                result.stop_reason = "answered"
                break

            if result.tool_calls + len(response.tool_calls) > self.max_tool_calls:
                result.degraded.append(
                    "工具调用次数将超过上限 {0}，已停止继续调用".format(self.max_tool_calls))
                break

            for call in response.tool_calls:
                step_index += 1
                self._execute_call(result, step_index, call, messages, seen)

            if result.tool_calls > self.max_tool_calls:
                break
        else:
            # for-else：循环跑满 max_steps 仍未拿到回答
            result.degraded.append("模型步数达到上限 {0}".format(self.max_steps))

        if not result.answer:
            self._force_answer(result, messages, step_index)

        result.messages = messages
        result.elapsed_seconds = time.monotonic() - started
        return result

    # ------------------------------------------------------------------ 单步
    def _record_model(self, result: AgentResult, index: int,
                      response: ChatResponse) -> None:
        result.model_calls += 1
        result.input_tokens += response.input_tokens or 0
        result.output_tokens += response.output_tokens or 0
        result.steps.append(StepRecord(
            index=index, kind="model", text=response.text[:400],
            tool_calls=[call.to_dict() for call in response.tool_calls],
            latency_seconds=response.latency_seconds,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            ok=response.ok, error=response.error,
        ))

    @staticmethod
    def _assistant_message(response: ChatResponse) -> Dict[str, Any]:
        """按接口要求重建 assistant 消息（不要直接回传服务端的原样结构）。"""
        message: Dict[str, Any] = {"role": "assistant", "content": response.text or ""}
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name,
                                 "arguments": call.arguments_raw or "{}"},
                }
                for call in response.tool_calls
            ]
        return message

    def _execute_call(self, result: AgentResult, index: int, call: ToolCall,
                      messages: List[Dict[str, Any]],
                      seen: Dict[str, int]) -> None:
        if not call.name:
            self._append_tool_error(result, index, call,
                                    "工具名为空，无法执行", messages)
            return

        if call.parse_error:
            self._append_tool_error(
                result, index, call,
                "参数无法解析，请修正后重试：{0}".format(call.parse_error), messages)
            return

        key = "{0}|{1}".format(call.name, json.dumps(call.arguments,
                                                     sort_keys=True, ensure_ascii=False))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > self.repeat_guard:
            self._append_tool_error(
                result, index, call,
                "相同工具与参数已调用 {0} 次，不再重复执行；请换一种查询方式或直接作答".format(
                    self.repeat_guard), messages, count_call=False)
            return

        tool_result: ToolResult = self.registry.call(call.name, call.arguments)
        result.tool_calls += 1
        if call.name not in result.used_tools:
            result.used_tools.append(call.name)

        preview = tool_result.to_message(max_chars=600)
        result.steps.append(StepRecord(
            index=index, kind="tool", tool=call.name, args=call.arguments,
            ok=tool_result.ok, error=tool_result.error,
            latency_seconds=tool_result.duration_ms / 1000.0, preview=preview,
        ))
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": tool_result.to_message(),
        })

        self._collect_evidence(result, call.name, tool_result)

    def _append_tool_error(self, result: AgentResult, index: int, call: ToolCall,
                           message: str, messages: List[Dict[str, Any]],
                           count_call: bool = True) -> None:
        """把工具层的失败如实回传给模型，让它有机会自我修正。"""
        if count_call:
            result.tool_calls += 1
        result.steps.append(StepRecord(
            index=index, kind="tool", tool=call.name or "(unknown)",
            args=call.arguments, ok=False, error=message, preview=message,
        ))
        messages.append({
            "role": "tool",
            "tool_call_id": call.id or "call_unknown",
            "content": json.dumps({"ok": False, "tool": call.name,
                                   "error": message}, ensure_ascii=False),
        })

    @staticmethod
    def _collect_evidence(result: AgentResult, name: str, tool_result: ToolResult) -> None:
        """从工具结果里收集可引用的证据与降级说明。"""
        if not tool_result.ok or not isinstance(tool_result.data, dict):
            return
        if name == "search_knowledge":
            for item in tool_result.data.get("citations") or []:
                if isinstance(item, dict) and item.get("chunk_id"):
                    if not any(c.get("chunk_id") == item["chunk_id"]
                               for c in result.citations):
                        result.citations.append(item)
            note = tool_result.data.get("note")
            if tool_result.data.get("degraded") and note:
                result.degraded.append(str(note))

    def _force_answer(self, result: AgentResult,
                      messages: List[Dict[str, Any]], index: int) -> None:
        """步数用尽后的收尾：不带工具再问一次，要求基于已有结果作答。"""
        if result.answer:
            return
        messages.append({
            "role": "user",
            "content": ("工具调用到此为止。请只依据上面的工具结果作答："
                        "如果信息足够，给出结论与依据；如果不足，直接说明缺少什么，"
                        "不要编造，也不要再请求调用工具。"),
        })
        response = self.client.chat(messages, tools=None)
        self._record_model(result, index + 1, response)
        if response.ok and response.text.strip():
            result.answer = response.text.strip()
            result.stop_reason = "forced_answer"
            messages.append({"role": "assistant", "content": result.answer})
        else:
            result.ok = False
            result.stop_reason = "max_steps"
            result.error = response.error or "步数用尽且未能生成回答"
            if not result.degraded:
                result.degraded.append("模型步数达到上限，且收尾调用失败")