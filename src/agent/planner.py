"""系统提示词与消息装配（T12）。

提示词的作用不是"让模型更聪明"，而是把**本项目必须遵守的约束**写死：
不编造、标明模拟数据、引用要给出处、工具失败要如实说。这些约束和评测口径
直接相关，因此必须以常量形式留存并计算指纹，报告里引用的就是实际发送的文本。
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, Dict, List, Optional

from src.tools.registry import ToolRegistry

#: 提示词版本，改动必须同步更新（与报告中的指纹一起）
PLANNER_VERSION = "1.0"

#: 骨架。``{tools}`` 与 ``{rules_extra}`` 由 :func:`build_system_prompt` 填充。
SYSTEM_TEMPLATE = """你是"猪只表型智能分析助手"，服务对象是养殖场的技术人员。

## 你可以做的事
1. 查询养殖数据（栏位、猪只、日增重、采食、活动量等），并做聚合与异常筛查；
2. 对给定的猪只图片做姿态识别；
3. 检索项目知识库（论文与项目文档），并给出出处。

## 可用工具
{tools}

## 必须遵守的规则
- {evidence_rule}
- 养殖数据来自**模拟数据集**，回答中必须明确说明"数据为模拟"，不得表述为真实场户数据。
- 引用知识库内容时，要写出处（标题或章节）；没有检索到就说没检索到，不要虚构文献。
- 提到具体方法、模型名或数字时，只能使用**同一篇文献片段里明确出现**的内容；
  不要把不同论文的方法或结论合并到一条引用下，也不要把某篇的做法说成另一篇的结论。
- 工具返回失败时，如实说明失败原因，不要用推测替代失败的结果。
- 需要均值、合计、极值等统计量时，**优先调用 aggregate_metric 之类的工具算出结果**，
  不要自己从明细数据里心算；直接引用工具返回的数字，并说明统计口径（时间范围、分组）。
- 同一个工具不要用完全相同的参数反复调用。
- 用中文回答，先给结论，再给依据；数字要带单位和时间范围。
- 回答里不要出现工具调用的 JSON、函数名或内部字段名。
{rules_extra}"""

#: 有工具时对"证据优先"的要求
EVIDENCE_RULE_WITH_TOOLS = (
    "涉及具体数据、事实、文献结论时，**先调用工具**再回答；不要凭印象编造数字。")

#: 无工具时不能要求"先调用工具"，改为要求明确说不确定
EVIDENCE_RULE_NO_TOOLS = (
    "涉及具体数据或文献结论时，如果无法确认，请明确说明无法核实；不要凭印象编造数字。")

#: 无工具时追加的说明（对应"直接回答"消融条件）
NO_TOOL_RULE = "- 本次没有可调用的工具。请基于你的知识回答，并明确标注哪些内容无法核实。"

#: 有工具时的补充说明
TOOL_RULE = "- 最多可进行 {max_steps} 步工具调用，够用即止。"


def build_system_prompt(registry: ToolRegistry, max_steps: int = 6,
                        today: Optional[date] = None) -> str:
    """生成系统提示词。

    Args:
        registry: 工具注册表；为空时提示词会说明"无工具可用"。
        max_steps: 最大工具调用步数，写进提示词帮助模型控制节奏。
        today: 当天日期，用于说明数据时间范围（测试时可固定）。
    """
    names = registry.names()
    if names:
        tools = registry.describe()
        rules_extra = TOOL_RULE.format(max_steps=max_steps)
        evidence_rule = EVIDENCE_RULE_WITH_TOOLS
    else:
        tools = "（无）"
        rules_extra = NO_TOOL_RULE
        evidence_rule = EVIDENCE_RULE_NO_TOOLS

    prompt = SYSTEM_TEMPLATE.format(tools=tools, rules_extra=rules_extra,
                                    evidence_rule=evidence_rule)

    # 数据时间范围是模型理解"最近一周"这类说法的基础；无法得知时明确说不知道
    try:
        from src.tools.mock_data import get_farm

        farm = get_farm()
        prompt += ("\n\n## 数据范围\n模拟数据覆盖 {0} 至 {1}，"
                   "共 {2} 个栏位、{3} 头猪。今天日期：{4}。").format(
            farm.start_date.isoformat(), farm.end_date.isoformat(),
            len(farm.pens), len(farm.pigs),
            (today or date.today()).isoformat())
    except Exception:  # noqa: BLE001 —— 拿不到数据范围不影响主流程
        pass

    return prompt


def build_messages(question: str, registry: ToolRegistry,
                   max_steps: int = 6,
                   history: Optional[List[Dict[str, Any]]] = None,
                   today: Optional[date] = None) -> List[Dict[str, Any]]:
    """装配首轮消息：system + 可选历史 + 当前问题。"""
    text = (question or "").strip()
    if not text:
        raise ValueError("问题不能为空")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(registry, max_steps, today)}
    ]
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": str(item.get("content", ""))})
    messages.append({"role": "user", "content": text})
    return messages


def prompt_fingerprint(text: str) -> str:
    """提示词内容指纹，写入运行记录以便追溯。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe_prompt(registry: ToolRegistry, max_steps: int = 6) -> Dict[str, Any]:
    """提示词元信息（不含正文），用于运行记录与报告。"""
    text = build_system_prompt(registry, max_steps)
    return {
        "version": PLANNER_VERSION,
        "sha256": prompt_fingerprint(text),
        "chars": len(text),
        "tools": registry.names(),
        "max_steps": max_steps,
    }