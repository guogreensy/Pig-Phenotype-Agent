"""``src.evaluation`` 的单元测试：手算 + 与 scikit-learn 交叉核对。"""

from __future__ import annotations

import csv
import unittest

from src.evaluation import (
    evaluate,
    evaluate_all,
    percentile,
    write_per_class_csv,
    write_summary_csv,
)
from src.schema import POSTURES


def rec(sample_id, truth, pred, success=True, json_valid=True, schema_valid=True,
        latency="1.0", error=""):
    return {
        "sample_id": sample_id,
        "image_path": "img/{0}.jpg".format(sample_id),
        "true_label": truth,
        "prompt_variant": "A",
        "model_name": "glm-4.6v-flash",
        "predicted_label": pred if success else "",
        "description": "描述",
        "uncertain": "True" if pred == "unknown" else "False",
        "json_valid": str(json_valid),
        "schema_valid": str(schema_valid),
        "request_success": str(success),
        "latency_seconds": latency,
        "error_message": error,
        "experiment_timestamp": "2026-10-19T12:00:00",
    }


def standard_set():
    return [
        rec("s1", "standing", "standing", latency="1.0"),
        rec("s2", "standing", "lying_lateral", latency="3.0"),
        rec("s3", "sitting", "sitting", latency="2.0"),
        rec("s4", "sitting", "unknown", latency="1.5"),
        rec("s5", "lying_other", "", success=False, schema_valid=False,
            json_valid=False, latency="5.0", error="HTTP 500：服务端错误"),
        rec("s6", "lying_lateral", "lying_lateral", latency="4.0"),
    ]


class TestHandComputed(unittest.TestCase):
    def setUp(self):
        self.result = evaluate(standard_set(), variant="A", model="glm-4.6v-flash")

    def test_counts(self):
        r = self.result
        self.assertEqual(r.n_total, 6)
        self.assertEqual(r.n_success, 5)
        self.assertEqual(r.n_failed, 1)
        self.assertAlmostEqual(r.failure_rate, 1 / 6)
        self.assertEqual(r.n_schema_valid, 5)
        self.assertAlmostEqual(r.json_valid_rate, 5 / 6)
        self.assertEqual(r.n_unknown, 1)
        self.assertAlmostEqual(r.unknown_rate, 1 / 5)

    def test_accuracy(self):
        self.assertEqual(self.result.n_classified, 4)
        self.assertEqual(self.result.n_correct, 3)
        self.assertAlmostEqual(self.result.accuracy_valid, 0.75)
        self.assertAlmostEqual(self.result.accuracy_strict, 0.5)

    def test_per_class(self):
        metrics = {m.label: m for m in self.result.per_class}
        # standing：2 个真实样本都进入分类子集，其中 1 个被判成侧卧
        self.assertAlmostEqual(metrics["standing"].precision, 1.0)
        self.assertAlmostEqual(metrics["standing"].recall, 0.5)
        self.assertEqual(metrics["standing"].support_classified, 2)
        self.assertEqual(metrics["standing"].support_total, 2)
        self.assertAlmostEqual(metrics["standing"].f1, 2 * 1.0 * 0.5 / 1.5)

        # lying_lateral：1 个真实样本，被 2 次预测（其中 1 次是误判）
        self.assertAlmostEqual(metrics["lying_lateral"].precision, 0.5)
        self.assertAlmostEqual(metrics["lying_lateral"].recall, 1.0)
        self.assertAlmostEqual(metrics["lying_lateral"].f1, 2 * 0.5 * 1.0 / 1.5)

        # lying_other：唯一真实样本调用失败，不在分类子集内
        self.assertEqual(metrics["lying_other"].support_classified, 0)
        self.assertEqual(metrics["lying_other"].support_total, 1)
        self.assertAlmostEqual(metrics["lying_other"].precision, 0.0)
        self.assertAlmostEqual(metrics["lying_other"].recall, 0.0)
        self.assertAlmostEqual(metrics["lying_other"].f1, 0.0)

        # sitting：2 个真实样本，1 个被拒答（unknown）
        self.assertAlmostEqual(metrics["sitting"].precision, 1.0)
        self.assertAlmostEqual(metrics["sitting"].recall, 1.0)
        self.assertAlmostEqual(metrics["sitting"].f1, 1.0)
        self.assertEqual(metrics["sitting"].support_classified, 1)
        self.assertEqual(metrics["sitting"].support_total, 2)
        self.assertAlmostEqual(metrics["sitting"].recall_strict, 0.5)

    def test_macro_and_weighted_f1(self):
        expected_macro = (2 / 3 + 2 / 3 + 0.0 + 1.0) / 4
        self.assertAlmostEqual(self.result.macro_f1, expected_macro, places=6)
        # 权重为分类子集内的各类样本数：2, 1, 0, 1
        expected_weighted = ((2 / 3) * 2 + (2 / 3) * 1 + 0.0 * 0 + 1.0 * 1) / 4
        self.assertAlmostEqual(self.result.weighted_f1, expected_weighted, places=6)

    def test_confusion_matrix(self):
        self.assertEqual(self.result.confusion_true_labels, POSTURES)
        self.assertEqual(self.result.confusion_pred_labels, POSTURES + ["unknown"])
        expected = [
            [1, 1, 0, 0, 0],   # standing
            [0, 1, 0, 0, 0],   # lying_lateral
            [0, 0, 0, 0, 0],   # lying_other（唯一样本调用失败，不计入）
            [0, 0, 0, 1, 1],   # sitting
        ]
        self.assertEqual(self.result.confusion, expected)

    def test_latency(self):
        self.assertAlmostEqual(self.result.latency_mean, 11.5 / 5)
        self.assertAlmostEqual(self.result.latency_median, 2.0)
        self.assertAlmostEqual(self.result.latency_p95, 4.0)

    def test_failure_reason_recorded(self):
        self.assertTrue(any("HTTP 500" in key for key in self.result.failure_reasons))


class TestEdgeCases(unittest.TestCase):
    def test_empty(self):
        result = evaluate([])
        self.assertEqual(result.n_total, 0)
        self.assertIsNone(result.accuracy_valid)
        self.assertIsNone(result.macro_f1)
        self.assertTrue(result.notes)

    def test_all_refusals(self):
        records = [rec("s1", "standing", "unknown"), rec("s2", "sitting", "unknown")]
        result = evaluate(records)
        self.assertEqual(result.n_classified, 0)
        self.assertEqual(result.unknown_rate, 1.0)
        self.assertIsNone(result.accuracy_valid)
        self.assertAlmostEqual(result.accuracy_strict, 0.0)
        self.assertIsNone(result.macro_f1)

    def test_all_failures(self):
        records = [rec("s1", "standing", "", success=False, schema_valid=False,
                       json_valid=False, error="请求超时")]
        result = evaluate(records)
        self.assertEqual(result.failure_rate, 1.0)
        self.assertAlmostEqual(result.accuracy_strict, 0.0)
        self.assertIsNone(result.latency_mean)

    def test_schema_invalid_still_scored_by_extracted_label(self):
        """格式不合规但类别可抽取时，仍应参与分类评分（格式由 JSON 有效率单独衡量）。"""
        records = [rec("s1", "standing", "standing", schema_valid=False)]
        result = evaluate(records)
        self.assertEqual(result.n_classified, 1)
        self.assertAlmostEqual(result.accuracy_valid, 1.0)
        self.assertEqual(result.schema_valid_rate, 0.0)

    def test_ignores_unknown_truth_label(self):
        records = [rec("s1", "", "standing")]
        result = evaluate(records)
        self.assertEqual(result.n_classified, 0)

    def test_percentile_recent_rank(self):
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([5.0], 95), 5.0)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 95), 5)


class TestAgainstSklearn(unittest.TestCase):
    """用 scikit-learn 交叉验证 Macro-F1 与各类指标。"""

    def test_matches_sklearn(self):
        from sklearn.metrics import precision_recall_fscore_support

        records = standard_set()
        result = evaluate(records)

        pairs = [
            (row["predicted_label"], row["true_label"])
            for row in records
            if row["request_success"] == "True"
            and row["schema_valid"] == "True"
            and row["predicted_label"] in POSTURES
        ]
        y_pred = [p for p, _ in pairs]
        y_true = [t for _, t in pairs]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=POSTURES, zero_division=0
        )

        for index, label in enumerate(POSTURES):
            metric = next(m for m in result.per_class if m.label == label)
            self.assertAlmostEqual(metric.precision, precision[index], places=6)
            self.assertAlmostEqual(metric.recall, recall[index], places=6)
            self.assertAlmostEqual(metric.f1, f1[index], places=6)

        self.assertAlmostEqual(result.macro_f1, float(sum(f1) / len(f1)), places=6)


class TestGroupingAndExport(unittest.TestCase):
    def test_evaluate_all_order(self):
        rows_a = standard_set()
        rows_b = [dict(row, prompt_variant="B") for row in standard_set()]
        results = evaluate_all(rows_a + rows_b, variants=["A", "B"])
        self.assertEqual([r.variant for r in results], ["A", "B"])
        self.assertEqual([r.n_total for r in results], [6, 6])

    def test_evaluate_all_missing_variant(self):
        results = evaluate_all(standard_set(), variants=["A", "B", "C"])
        missing = [r for r in results if r.variant in ("B", "C")]
        self.assertTrue(all(r.n_total == 0 for r in missing))

    def test_write_csvs(self):
        import tempfile
        from pathlib import Path

        base = Path(__file__).resolve().parent / "_tmp_out"
        base.mkdir(exist_ok=True)
        try:
            results = evaluate_all(standard_set(), variants=["A"])
            summary = write_summary_csv(results, base / "summary.csv")
            per_class = write_per_class_csv(results, base / "per_class.csv")
            with summary.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_variant"], "A")
            self.assertAlmostEqual(float(rows[0]["accuracy_valid"]), 0.75, places=4)
            with per_class.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(POSTURES))
        finally:
            for name in ("summary.csv", "per_class.csv"):
                target = base / name
                if target.exists():
                    target.unlink()
            if base.exists():
                base.rmdir()


if __name__ == "__main__":
    unittest.main()
