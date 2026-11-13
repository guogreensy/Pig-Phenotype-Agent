"""阶段 E 图表：把指标画成报告里能直接用的图。

依赖 ``results/metrics/`` 下的产物（先跑 scripts/evaluate_agent.py）。

    python scripts/plot_agent.py

产出（results/figures/）：
    agent_task_metrics.png   三类题目的完成质量对比
    agent_cost.png           调用次数与耗时对比
    agent_data_heatmap.png   10 道数据题在三个条件下的数值命中情况
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.reporting import save, setup_font  # noqa: E402

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

CONDITIONS = ["direct", "rag", "full"]
CONDITION_CN = {"direct": "直接回答", "rag": "检索增强", "full": "检索+工具"}
COLORS = {"direct": "#94a3b8", "rag": "#60a5fa", "full": "#2563eb"}


def load_summary(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["condition"]: row for row in payload.get("conditions", [])}


def plot_task_metrics(index: Dict[str, Dict[str, Any]], path: Path) -> Optional[Path]:
    metrics = [
        ("数据题\n数值命中率", "data_number_hit_rate"),
        ("数据题\n整题正确率", "data_question_accuracy"),
        ("知识题\n检索命中率", "knowledge_retrieval_hit_rate"),
        ("视觉题\n姿态命中率", "vision_label_accuracy"),
        ("拒答题\n诚实拒答率", "unanswerable_admit_rate"),
    ]
    available = [c for c in CONDITIONS if c in index]
    if not available:
        return None

    # 画布按印刷宽度（13.6 cm）出图，图内字号即纸面字号，避免整幅缩放后文字过小
    fig, ax = plt.subplots(figsize=(13.6 / 2.54, 2.75))
    width = 0.8 / len(available)
    positions = list(range(len(metrics)))
    for offset, condition in enumerate(available):
        values = [index[condition].get(key) or 0.0 for _, key in metrics]
        xs = [p + offset * width - (len(available) - 1) * width / 2 for p in positions]
        bars = ax.bar(xs, values, width * 0.92, label=CONDITION_CN[condition],
                      color=COLORS[condition])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                    "{0:.0%}".format(value), ha="center", va="bottom", fontsize=6.4)

    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in metrics], fontsize=7.0)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("比例", fontsize=7.2)
    ax.set_title("三个条件下的任务完成质量（越高越好）", fontsize=7.8)
    ax.legend(fontsize=6.6, ncol=len(available), loc="upper center")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, path)


def plot_cost(index: Dict[str, Dict[str, Any]], path: Path) -> Optional[Path]:
    available = [c for c in CONDITIONS if c in index]
    if not available:
        return None
    labels = [CONDITION_CN[c] for c in available]
    model_calls = [index[c].get("model_calls_mean") or 0 for c in available]
    tool_calls = [index[c].get("tool_calls_mean") or 0 for c in available]
    latency = [index[c].get("latency_mean") or 0 for c in available]
    p95 = [index[c].get("latency_p95") or 0 for c in available]

    fig, axes = plt.subplots(1, 2, figsize=(14.2 / 2.54, 2.35))
    xs = range(len(available))
    width = 0.36

    ax = axes[0]
    bars1 = ax.bar([x - width / 2 for x in xs], model_calls, width, label="模型调用",
                   color="#60a5fa")
    bars2 = ax.bar([x + width / 2 for x in xs], tool_calls, width, label="工具调用",
                   color="#2563eb")
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                    "{0:.1f}".format(height), ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("每次对话平均次数", fontsize=7.0)
    ax.set_title("调用开销", fontsize=7.4)
    ax.legend(fontsize=6.2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar(list(xs), latency, 0.55, color=[COLORS[c] for c in available],
                  label="平均耗时")
    ax.plot(list(xs), p95, "o--", color="#dc2626", markersize=3, linewidth=1,
            label="P95 耗时")
    for x, value in zip(xs, latency):
        ax.text(x, value + 1.5, "{0:.0f}s".format(value), ha="center",
                va="bottom", fontsize=6.0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("秒", fontsize=7.0)
    ax.set_title("单次对话耗时", fontsize=7.4)
    ax.legend(fontsize=6.2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return save(fig, path)


def plot_data_heatmap(path: Path, out: Path) -> Optional[Path]:
    """10 道数据题 × 3 条件的数值命中率热力图。"""
    if not path.exists():
        return None
    rows: List[Dict[str, str]] = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    data_rows = [r for r in rows if r.get("category") == "data"]
    if not data_rows:
        return None

    questions = sorted({r["question_id"] for r in data_rows})
    matrix: List[List[float]] = []
    for condition in CONDITIONS:
        line = []
        for question_id in questions:
            match = [r for r in data_rows if r["question_id"] == question_id
                     and r["condition"] == condition]
            if not match or not match[0].get("numbers_total"):
                line.append(float("nan"))
            else:
                hit = float(match[0]["numbers_hit"])
                total = float(match[0]["numbers_total"])
                line.append(hit / total)
        matrix.append(line)

    fig, ax = plt.subplots(figsize=(14.2 / 2.54, 1.85))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(questions)))
    ax.set_xticklabels(questions, fontsize=6.6)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([CONDITION_CN[c] for c in CONDITIONS], fontsize=6.8)
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            if value != value:  # NaN
                continue
            ax.text(j, i, "{0:.0f}%".format(value * 100), ha="center", va="center",
                    fontsize=6.0, color="white" if value > 0.6 else "#1e293b")
    ax.set_title("数据题：期望数值被回答命中的比例", fontsize=7.4)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return save(fig, out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 E 图表")
    parser.add_argument("--metrics", default=str(METRICS_DIR))
    parser.add_argument("--figures", default=str(FIGURES_DIR))
    args = parser.parse_args(argv)

    metrics_dir = Path(args.metrics)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_font()

    summary_path = metrics_dir / "agent_summary.json"
    if not summary_path.exists():
        raise SystemExit("缺少 {0}，请先运行 scripts/evaluate_agent.py".format(
            summary_path.relative_to(PROJECT_ROOT)))
    index = load_summary(summary_path)

    produced = []
    for name, func in (
        ("agent_task_metrics.png", lambda p: plot_task_metrics(index, p)),
        ("agent_cost.png", lambda p: plot_cost(index, p)),
        ("agent_data_heatmap.png",
         lambda p: plot_data_heatmap(metrics_dir / "agent_per_question.csv", p)),
    ):
        result = func(figures_dir / name)
        if result is not None:
            produced.append(result)
            print("已生成 {0}".format(result.relative_to(PROJECT_ROOT)))
        else:
            print("跳过 {0}（数据不足）".format(name))

    if not produced:
        raise SystemExit("没有生成任何图，请检查指标文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())