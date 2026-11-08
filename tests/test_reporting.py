"""``src.reporting`` 的测试：图表与报告草稿能从真实计算结果生成。

注意：本测试使用**构造的夹具记录**验证绘图/渲染代码，产物写到
``tests/_tmp_figures/`` 并在结束时删除，**不会**污染 ``results/``，
也不会被当作实验结果。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.evaluation import evaluate_all
from src.reporting import generate_all, render_markdown, setup_font

TMP = Path(__file__).resolve().parent / "_tmp_figures"


def fixture_records():
    """构造三组表现不同的夹具记录（仅用于验证绘图与渲染代码）。"""
    rows = []
    labels = ["standing", "lying_lateral", "lying_other", "sitting"]
    offsets = {"A": 0, "B": 1, "C": 2}
    for variant in ("A", "B", "C"):
        offset = offsets[variant]
        for index in range(24):
            truth = labels[index % 4]
            correct = (index + offset) % 3 != 0
            if variant == "C" and index % 7 == 0:
                pred, json_valid, schema_valid = "unknown", "True", "True"
            elif variant == "A" and index % 11 == 0:
                pred, json_valid, schema_valid = "", "False", "False"
            else:
                pred = truth if correct else labels[(index + 1) % 4]
                json_valid = schema_valid = "True"
            rows.append({
                "sample_id": "s{0}".format(index),
                "image_path": "data/samples/test/x.jpg",
                "true_label": truth,
                "prompt_variant": variant,
                "prompt_sha256": "a" * 64,
                "model_name": "glm-4.6v-flash",
                "model_params": "temperature=0",
                "predicted_label": pred,
                "description": "描述",
                "uncertain": "True" if pred == "unknown" else "False",
                "uncertainty_reason": "",
                "json_valid": json_valid,
                "schema_valid": schema_valid,
                "request_success": "True",
                "http_status": "200",
                "latency_seconds": str(0.8 + (index % 5) * 0.3),
                "input_tokens": "900",
                "output_tokens": "60",
                "error_message": "",
                "attempt": "1",
                "parse_method": "direct",
                "experiment_timestamp": "2026-10-19T12:00:00",
                "run_id": "fixture",
                "raw_text": "{}",
            })
    return rows


class TestReporting(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)

    def tearDown(self):
        if TMP.is_dir():
            for path in TMP.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(TMP.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            TMP.rmdir()

    def test_generate_all_creates_figures(self):
        setup_font()
        results = evaluate_all(fixture_records(), variants=["A", "B", "C"])
        figures = generate_all(results, TMP)
        expected = {"accuracy", "macro_f1", "json_rate", "unknown_rate", "latency",
                    "per_class_f1", "confusion_C"}
        self.assertTrue(expected.issubset(set(figures)), expected - set(figures))
        for name, path in figures.items():
            self.assertTrue(Path(path).is_file(), name)
            self.assertGreater(Path(path).stat().st_size, 2000, name)

    def test_markdown_contains_real_numbers(self):
        results = evaluate_all(fixture_records(), variants=["A", "B", "C"])
        figures = generate_all(results, TMP)
        meta = {"run_id": "fixture", "模型": "glm-4.6v-flash"}
        path = render_markdown(results, meta, TMP / "report.md", figures)
        text = path.read_text(encoding="utf-8")
        self.assertIn("指标汇总", text)
        self.assertIn("glm-4.6v-flash", text)
        expected = "{0:.4f}".format(results[-1].accuracy_valid)
        self.assertIn(expected, text)
        self.assertIn("混淆矩阵", text)

    def test_no_data_produces_no_figures(self):
        results = evaluate_all([], variants=["A"])
        figures = generate_all(results, TMP)
        self.assertEqual(figures, {})


if __name__ == "__main__":
    unittest.main()
