"""结果组装测试（T14）。"""

from __future__ import annotations

import unittest

from src.agent.executor import AgentResult, StepRecord
from src.agent.report import build_report, render_markdown


def _result(**overrides) -> AgentResult:
    payload = dict(
        question="猪只均重多少",
        answer="P1 栏均重 97.3 kg。",
        ok=True,
        stop_reason="answered",
        tool_calls=2,
        model_calls=2,
        input_tokens=100,
        output_tokens=40,
        elapsed_seconds=3.2,
        used_tools=["list_pens", "search_knowledge"],
        citations=[{"chunk_id": "pmc_1#0001", "title": "论文甲", "section": "Results",
                    "url": "https://example.org/1", "license": "CC BY", "score": 0.71}],
        steps=[
            StepRecord(index=1, kind="model", text="", tool_calls=[{"name": "list_pens"}]),
            StepRecord(index=2, kind="tool", tool="list_pens", args={}, ok=True,
                       latency_seconds=0.05, preview="{}"),
            StepRecord(index=3, kind="tool", tool="search_knowledge",
                       args={"query": "均重"}, ok=True, latency_seconds=0.4, preview="{}"),
        ],
        prompt_meta={"version": "1.0", "sha256": "a" * 64, "tools": ["list_pens"]},
    )
    payload.update(overrides)
    return AgentResult(**payload)


class TestBuildReport(unittest.TestCase):
    def test_flags_detected(self):
        report = build_report(_result())
        self.assertTrue(report.flags["uses_simulated_data"])
        self.assertTrue(report.flags["uses_knowledge"])
        self.assertFalse(report.flags["uses_vision"])
        self.assertTrue(report.flags["has_citations"])

    def test_vision_flag(self):
        report = build_report(_result(used_tools=["analyze_pig_posture"]))
        self.assertTrue(report.flags["uses_vision"])
        self.assertFalse(report.flags["uses_simulated_data"])

    def test_citations_numbered(self):
        report = build_report(_result())
        self.assertEqual(len(report.citations), 1)
        self.assertEqual(report.citations[0]["index"], 1)

    def test_tool_trace_excludes_model_steps(self):
        report = build_report(_result())
        self.assertEqual([item["tool"] for item in report.tool_trace],
                         ["list_pens", "search_knowledge"])
        self.assertEqual(report.tool_trace[1]["args"], {"query": "均重"})

    def test_metrics_copied(self):
        report = build_report(_result())
        self.assertEqual(report.metrics["model_calls"], 2)
        self.assertEqual(report.metrics["tool_calls"], 2)
        self.assertEqual(report.metrics["elapsed_seconds"], 3.2)

    def test_to_dict_has_chinese_stop_reason(self):
        payload = build_report(_result()).to_dict()
        self.assertEqual(payload["stop_reason_cn"], "正常作答")


class TestRender(unittest.TestCase):
    def test_markdown_sections(self):
        text = render_markdown(build_report(_result()))
        self.assertIn("## 问题", text)
        self.assertIn("## 回答", text)
        self.assertIn("P1 栏均重 97.3 kg。", text)
        self.assertIn("## 引用", text)
        self.assertIn("[1] 论文甲", text)
        self.assertIn("Results", text)
        self.assertIn("CC BY", text)

    def test_simulated_warning_present(self):
        text = render_markdown(build_report(_result()))
        self.assertIn("模拟养殖数据", text)
        self.assertIn("不可用于生产决策", text)

    def test_degraded_notes_rendered(self):
        text = render_markdown(build_report(_result(
            degraded=["向量检索不可用，已退回词法检索"])))
        self.assertIn("向量检索不可用", text)

    def test_failure_message(self):
        report = build_report(_result(answer="", ok=False, error="HTTP 500", 
                                      stop_reason="model_error"))
        text = render_markdown(report)
        self.assertIn("未能生成回答：HTTP 500", text)
        self.assertIn("模型调用失败", text)
        self.assertIn("本次未能正常完成", text)

    def test_no_citation_note_when_searched(self):
        report = build_report(_result(citations=[]))
        text = render_markdown(report)
        self.assertIn("未找到可引用片段", text)
        self.assertNotIn("## 引用", text)

    def test_metrics_rendered(self):
        text = render_markdown(build_report(_result()))
        self.assertIn("模型调用 2 次，工具调用 2 次", text)
        self.assertIn("输入 100 / 输出 40", text)
        self.assertIn("耗时 3.2 秒", text)


if __name__ == "__main__":
    unittest.main()