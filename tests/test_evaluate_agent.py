"""阶段 E 指标计算的单元测试。

只测纯函数（数字抽取、容差匹配、单次判定、汇总口径），不读真实实验记录。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_agent", PROJECT_ROOT / "scripts" / "evaluate_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_agent"] = module
    spec.loader.exec_module(module)
    return module


evaluate_agent = _load_module()


class TestExtractNumbers(unittest.TestCase):
    def test_plain_and_decimal(self):
        values = evaluate_agent.extract_numbers("平均采食量 1.63 kg，体重 97.78 公斤")
        self.assertIn(1.63, values)
        self.assertIn(97.78, values)

    def test_dates_are_ignored(self):
        """日期里的 2026/10/31 不能当成数值。"""
        values = evaluate_agent.extract_numbers("2026-10-31 至 2026-11-06 共 7 天")
        self.assertNotIn(2026.0, values)
        self.assertNotIn(31.0, values)
        self.assertIn(7.0, values)

    def test_pig_ids_are_ignored(self):
        values = evaluate_agent.extract_numbers("P2-007 这头猪的采食量为 1.5 kg")
        self.assertNotIn(2.0, values)
        self.assertNotIn(7.0, values)
        self.assertIn(1.5, values)

    def test_percent_gives_both_forms(self):
        values = evaluate_agent.extract_numbers("躺卧比例 58.5%")
        self.assertIn(58.5, values)
        self.assertIn(0.585, values)

    def test_empty_text(self):
        self.assertEqual(evaluate_agent.extract_numbers(""), [])


class TestMatchNumbers(unittest.TestCase):
    def test_within_tolerance(self):
        expected = [{"value": 1.6257, "tol": 0.024}]
        hit, flags = evaluate_agent.match_numbers(expected, [1.63])
        self.assertEqual(hit, 1)
        self.assertEqual(flags, [True])

    def test_outside_tolerance(self):
        expected = [{"value": 1.6257, "tol": 0.024}]
        hit, _ = evaluate_agent.match_numbers(expected, [1.69])
        self.assertEqual(hit, 0)

    def test_candidates_cannot_be_reused(self):
        """两个期望值不能由同一个数字满足。"""
        expected = [{"value": 4.0, "tol": 0.5}, {"value": 4.2, "tol": 0.5}]
        hit, flags = evaluate_agent.match_numbers(expected, [4.1])
        self.assertEqual(hit, 1)
        self.assertEqual(flags, [True, False])

    def test_distinct_candidates_match_both(self):
        expected = [{"value": 4.0, "tol": 0.5}, {"value": 48.0, "tol": 0.5}]
        hit, _ = evaluate_agent.match_numbers(expected, [4.0, 48.0])
        self.assertEqual(hit, 2)

    def test_picks_closest_candidate(self):
        expected = [{"value": 2.4018, "tol": 0.036}]
        hit, _ = evaluate_agent.match_numbers(expected, [2.44, 2.40])
        self.assertEqual(hit, 1)


class TestCitationDocIds(unittest.TestCase):
    def test_doc_id_derived_from_chunk_id(self):
        record = {"citations": [
            {"chunk_id": "pmc_8147168#0051"},
            {"chunk_id": "local_experiment_report#0003"},
        ]}
        self.assertEqual(evaluate_agent.citation_doc_ids(record),
                         ["pmc_8147168", "local_experiment_report"])

    def test_empty(self):
        self.assertEqual(evaluate_agent.citation_doc_ids({}), [])


class TestEvaluateRun(unittest.TestCase):
    def _record(self, **overrides):
        record = {
            "condition": "full", "question_id": "D1", "category": "data",
            "ok": True, "stop_reason": "answered", "answer": "",
            "expected": {"numbers": [{"value": 1.6257, "tol": 0.024,
                                      "label": "均值"}], "tokens": ["P2-007"]},
            "citations": [], "tool_trace": [], "metrics": {}, "flags": {},
            "wall_seconds": 1.0,
        }
        record.update(overrides)
        return record

    def test_data_hit(self):
        row = evaluate_agent.evaluate_run(self._record(answer="平均为 1.63 kg"))
        self.assertEqual(row["numbers_hit"], 1)
        self.assertTrue(row["numbers_ok"])

    def test_data_miss(self):
        row = evaluate_agent.evaluate_run(self._record(answer="平均为 1.90 kg"))
        self.assertEqual(row["numbers_hit"], 0)
        self.assertFalse(row["numbers_ok"])

    def test_knowledge_retrieval_hit(self):
        record = self._record(
            question_id="K1", category="knowledge",
            expected={"doc_ids": ["pmc_8147168"]},
            citations=[{"chunk_id": "pmc_8147168#0002"}])
        row = evaluate_agent.evaluate_run(record)
        self.assertTrue(row["retrieval_hit"])

    def test_knowledge_retrieval_miss(self):
        record = self._record(
            question_id="K1", category="knowledge",
            expected={"doc_ids": ["pmc_8147168"]},
            citations=[{"chunk_id": "pmc_9999999#0001"}])
        row = evaluate_agent.evaluate_run(record)
        self.assertFalse(row["retrieval_hit"])

    def test_vision_label_hit_by_chinese(self):
        record = self._record(
            question_id="V1", category="vision",
            expected={"label": "lying_lateral", "label_cn": "侧卧"},
            tool_trace=[{"tool": "analyze_pig_posture"}],
            answer="这头猪是侧卧姿态。")
        self.assertTrue(evaluate_agent.evaluate_run(record)["label_hit"])

    def test_vision_label_hit_by_english(self):
        record = self._record(
            question_id="V1", category="vision",
            expected={"label": "standing", "label_cn": "站立"},
            tool_trace=[{"tool": "analyze_pig_posture"}],
            answer="姿态为 standing。")
        self.assertTrue(evaluate_agent.evaluate_run(record)["label_hit"])

    def test_vision_without_tool_is_not_a_hit(self):
        """没有视觉工具时，回答里罗列候选姿态不能算命中。"""
        record = self._record(
            question_id="V1", category="vision",
            expected={"label": "lying_lateral", "label_cn": "侧卧"},
            answer="我无法查看图片。可能的姿态有：站立、侧卧、坐姿。")
        row = evaluate_agent.evaluate_run(record)
        self.assertIsNone(row["label_hit"])
        self.assertFalse(row["vision_tool_used"])
        self.assertTrue(row["vision_refused"])

    def test_unanswerable_admit(self):
        record = self._record(
            question_id="U1", category="unanswerable",
            expected={"must_admit": True},
            answer="模拟数据中没有品种字段，无法回答。")
        self.assertTrue(evaluate_agent.evaluate_run(record)["admit"])

    def test_unanswerable_fabrication(self):
        record = self._record(
            question_id="U1", category="unanswerable",
            expected={"must_admit": True},
            answer="这批猪是大白猪。")
        self.assertFalse(evaluate_agent.evaluate_run(record)["admit"])

    def test_entity_token_in_question_is_not_counted(self):
        """题面里已经出现猪号时，回答照抄不算命中。"""
        record = self._record(question="P2-007 最近 7 天的平均采食量是多少？",
                              answer="P2-007 的平均采食量为 1.63 kg")
        self.assertIsNone(evaluate_agent.evaluate_run(record)["entity_hit"])

    def test_entity_identification_question(self):
        """指认题（题面不含猪号）答出正确猪号才算命中。"""
        record = self._record(
            question_id="D3", question="最近 7 天有没有采食量异常的猪？是哪一头？",
            expected={"tokens": ["P2-007"]},
            answer="有一头猪采食量异常，是 P2-007。")
        self.assertTrue(evaluate_agent.evaluate_run(record)["entity_hit"])

    def test_entity_identification_wrong_pig(self):
        record = self._record(
            question_id="D3", question="最近 7 天有没有采食量异常的猪？是哪一头？",
            expected={"tokens": ["P2-007"]},
            answer="P1-002 采食量异常。")
        self.assertFalse(evaluate_agent.evaluate_run(record)["entity_hit"])

    def test_exclusion_flag(self):
        row = evaluate_agent.evaluate_run(self._record(question_id="D6"),
                                          {"D6": "提示词内信息"})
        self.assertTrue(row["excluded"])


class TestSummarizeCondition(unittest.TestCase):
    def _row(self, **overrides):
        row = {
            "condition": "full", "question_id": "D1", "category": "data",
            "excluded": False, "ok": True, "answer_chars": 100,
            "tool_calls": 1, "model_calls": 2, "wall_seconds": 10.0,
            "citations": 0, "degraded": False,
            "numbers_hit": 1, "numbers_total": 1, "numbers_ok": True,
            "entity_hit": True, "retrieval_hit": None, "label_hit": None,
            "admit": None,
        }
        row.update(overrides)
        return row

    def test_prompt_answerable_question_is_separated(self):
        rows = [
            self._row(question_id="D1", numbers_ok=True),
            self._row(question_id="D6", numbers_ok=True, excluded=True),
        ]
        summary = evaluate_agent.summarize_condition(rows, "full")
        self.assertEqual(summary["data_runs"], 1)          # D6 不计入
        self.assertEqual(summary["data_question_accuracy"], 1.0)
        self.assertEqual(summary["data_prompt_answerable_accuracy"], 1.0)

    def test_rates_and_latency(self):
        rows = [
            self._row(wall_seconds=10.0, tool_calls=1),
            self._row(question_id="D2", wall_seconds=30.0, tool_calls=3, ok=False),
        ]
        summary = evaluate_agent.summarize_condition(rows, "full")
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["ok_rate"], 0.5)
        self.assertEqual(summary["tool_calls_mean"], 2.0)
        self.assertEqual(summary["latency_mean"], 20.0)

    def test_empty_condition_returns_empty(self):
        self.assertEqual(evaluate_agent.summarize_condition([], "rag"), {})


if __name__ == "__main__":
    unittest.main()