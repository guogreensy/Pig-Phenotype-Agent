"""结果组装（T14）。

把执行器的原始结果整理成"可以直接给人看、也可以直接进实验记录"的形态：
回答、编号引用、工具轨迹、来源标注与运行指标。

为什么单独一层
--------------
* 前端、命令行、实验统计吃的是同一份结构，避免三处各拼一套；
* **来源标注是硬要求**：用了模拟数据就必须出现"模拟"字样，检索降级就必须
  说明降级——这些判断放在这里集中做，比散落在界面代码里可靠；
* 引用统一编号，方便人工核对回答里的每个论断能不能对应到出处。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.executor import AgentResult

#: 视为"使用了养殖数据"的工具
DATA_TOOLS = {"list_pens", "query_pig_records", "aggregate_metric", "detect_anomalies"}

#: 结束原因的中文说明
STOP_REASON_CN = {
    "answered": "正常作答",
    "forced_answer": "步数用尽后收尾作答",
    "max_steps": "步数用尽且未能作答",
    "model_error": "模型调用失败",
    "invalid_question": "问题为空",
}


@dataclass
class AnswerReport:
    """一次问答的交付形态。"""

    question: str
    answer: str
    ok: bool = True
    error: Optional[str] = None
    stop_reason: str = "answered"
    citations: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    prompt_meta: Dict[str, Any] = field(default_factory=dict)
    degraded: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "ok": self.ok,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "stop_reason_cn": STOP_REASON_CN.get(self.stop_reason, self.stop_reason),
            "citations": self.citations,
            "tool_trace": self.tool_trace,
            "flags": self.flags,
            "metrics": self.metrics,
            "degraded": self.degraded,
            "prompt_meta": self.prompt_meta,
        }


def build_report(result: AgentResult) -> AnswerReport:
    """由执行结果组装报告。"""
    used = set(result.used_tools)
    flags = {
        "uses_simulated_data": bool(used & DATA_TOOLS),
        "uses_knowledge": "search_knowledge" in used,
        "uses_vision": "analyze_pig_posture" in used,
        "has_citations": bool(result.citations),
        "degraded": bool(result.degraded),
    }

    citations = []
    for order, item in enumerate(result.citations, start=1):
        citations.append({
            "index": order,
            "chunk_id": item.get("chunk_id", ""),
            "title": item.get("title", ""),
            "section": item.get("section", ""),
            "url": item.get("url", ""),
            "license": item.get("license", ""),
            "score": item.get("score"),
        })

    tool_trace = []
    for step in result.steps:
        if step.kind != "tool":
            continue
        tool_trace.append({
            "tool": step.tool,
            "args": step.args,
            "ok": step.ok,
            "error": step.error,
            "ms": round(step.latency_seconds * 1000, 1),
        })

    return AnswerReport(
        question=result.question,
        answer=result.answer,
        ok=result.ok,
        error=result.error,
        stop_reason=result.stop_reason,
        citations=citations,
        tool_trace=tool_trace,
        flags=flags,
        degraded=list(result.degraded),
        prompt_meta=dict(result.prompt_meta),
        metrics={
            "model_calls": result.model_calls,
            "tool_calls": result.tool_calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "used_tools": list(result.used_tools),
        },
    )


def render_markdown(report: AnswerReport) -> str:
    """渲染为便于阅读与留档的 Markdown。"""
    lines: List[str] = ["## 问题", "", report.question, "", "## 回答", ""]

    if report.answer:
        lines.append(report.answer)
    elif not report.ok:
        lines.append("（未能生成回答：{0}）".format(report.error or "原因未知"))
    else:
        lines.append("（模型未给出回答）")

    if report.citations:
        lines.extend(["", "## 引用", ""])
        for item in report.citations:
            source = item.get("title") or item.get("chunk_id")
            section = " · {0}".format(item["section"]) if item.get("section") else ""
            license_text = "（{0}）".format(item["license"]) if item.get("license") else ""
            lines.append("[{0}] {1}{2}{3}".format(item["index"], source, section, license_text))
            if item.get("url"):
                lines.append("    <{0}>".format(item["url"]))

    notes: List[str] = []
    if report.flags.get("uses_simulated_data"):
        count = sum(1 for step in report.tool_trace if step["tool"] in DATA_TOOLS)
        notes.append("- 本回答使用了**模拟养殖数据**（{0} 次数据工具调用），"
                     "不代表任何真实养殖场，不可用于生产决策。".format(count))
    if report.flags.get("uses_vision"):
        notes.append("- 姿态识别由多模态视觉模型给出，结果为模型判断，非人工标注。")
    if report.flags.get("uses_knowledge") and not report.citations:
        notes.append("- 检索了知识库，但未找到可引用片段。")
    for item in report.degraded:
        notes.append("- 注意：{0}".format(item))
    if not report.ok:
        notes.append("- 本次未能正常完成：{0}".format(report.error or "原因未知"))
    if notes:
        lines.extend(["", "## 说明", ""] + notes)

    metrics = report.metrics
    lines.extend([
        "", "## 运行信息", "",
        "- 结束原因：{0}".format(STOP_REASON_CN.get(report.stop_reason, report.stop_reason)),
        "- 模型调用 {0} 次，工具调用 {1} 次".format(
            metrics.get("model_calls", 0), metrics.get("tool_calls", 0)),
        "- tokens：输入 {0} / 输出 {1}".format(
            metrics.get("input_tokens", 0), metrics.get("output_tokens", 0)),
        "- 耗时 {0} 秒".format(metrics.get("elapsed_seconds", 0)),
    ])
    return "\n".join(lines)