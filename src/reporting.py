"""实验图表与报告草稿生成。

所有图表**只读取真实计算结果**：没有数据就报错跳过，绝不生成占位图，
也不填任何虚构数值（需求 6.6）。
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 无界面环境
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation import EvaluationResult  # noqa: E402
from src.prompts import variant_title  # noqa: E402
from src.schema import POSTURES, POSTURE_CN  # noqa: E402

PALETTE = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7"]


def setup_font() -> None:
    """启用系统中文字体（Windows 自带微软雅黑/黑体）。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def _labels(results: Sequence[EvaluationResult]) -> List[str]:
    return ["{0}\n{1}".format(r.variant, variant_title(r.variant).split(" ", 1)[-1]) for r in results]


def _require(results: Sequence[EvaluationResult]) -> bool:
    return bool(results) and any(r.n_total for r in results)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metric_bars(results: Sequence[EvaluationResult], path: Path,
                     title: str, ylabel: str,
                     value_getter, value_getter2=None, label2: str = "") -> Optional[Path]:
    """通用柱状对比图（可叠加两组指标）。"""
    if not _require(results):
        return None
    labels = _labels(results)
    values = [value_getter(r) for r in results]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    positions = range(len(results))
    width = 0.38 if value_getter2 else 0.6
    if value_getter2:
        values2 = [value_getter2(r) for r in results]
        bars1 = ax.bar([p - width / 2 for p in positions],
                       [v if v is not None else 0 for v in values], width,
                       label="有效分类子集", color=PALETTE[0])
        bars2 = ax.bar([p + width / 2 for p in positions],
                       [v if v is not None else 0 for v in values2], width,
                       label=label2 or "严格口径", color=PALETTE[1])
        for bars, vals in ((bars1, values), (bars2, values2)):
            _annotate(ax, bars, vals)
        ax.legend(frameon=False)
    else:
        bars = ax.bar(positions, [v if v is not None else 0 for v in values],
                      width, color=[PALETTE[i % len(PALETTE)] for i in positions])
        _annotate(ax, bars, values)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return save(fig, path)


def _annotate(ax, bars, values) -> None:
    for bar, value in zip(bars, values):
        text = "N/A" if value is None else "{0:.3f}".format(value)
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)


def plot_accuracy(results: Sequence[EvaluationResult], path: Path) -> Optional[Path]:
    return plot_metric_bars(
        results, path,
        title="不同提示词方案的姿态识别准确率",
        ylabel="准确率",
        value_getter=lambda r: r.accuracy_valid,
        value_getter2=lambda r: r.accuracy_strict,
        label2="严格口径（拒答/失败算错）",
    )


def plot_macro_f1(results: Sequence[EvaluationResult], path: Path) -> Optional[Path]:
    return plot_metric_bars(
        results, path,
        title="不同提示词方案的 Macro-F1",
        ylabel="Macro-F1",
        value_getter=lambda r: r.macro_f1,
    )


def plot_json_rate(results: Sequence[EvaluationResult], path: Path) -> Optional[Path]:
    return plot_metric_bars(
        results, path,
        title="不同提示词方案的 JSON 有效率",
        ylabel="有效率",
        value_getter=lambda r: r.json_valid_rate,
    )


def plot_unknown_rate(results: Sequence[EvaluationResult], path: Path) -> Optional[Path]:
    return plot_metric_bars(
        results, path,
        title="不同提示词方案的“无法判断”比例",
        ylabel="比例",
        value_getter=lambda r: r.unknown_rate,
    )


def plot_latency(results: Sequence[EvaluationResult], path: Path) -> Optional[Path]:
    if not _require(results):
        return None
    labels = _labels(results)
    means = [r.latency_mean if r.latency_mean is not None else 0 for r in results]
    medians = [r.latency_median if r.latency_median is not None else 0 for r in results]
    p95 = [r.latency_p95 if r.latency_p95 is not None else 0 for r in results]
    import numpy as np

    x = np.arange(len(results))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width, means, width, label="均值", color=PALETTE[0])
    ax.bar(x, medians, width, label="中位数", color=PALETTE[1])
    ax.bar(x + width, p95, width, label="p95", color=PALETTE[2])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("响应时间（秒）")
    ax.set_title("不同提示词方案的模型响应时间")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return save(fig, path)


def plot_confusion(result: EvaluationResult, path: Path) -> Optional[Path]:
    """混淆矩阵热图：行=真实类别，列=预测类别（含 unknown）。"""
    if not result or not result.confusion or not any(any(row) for row in result.confusion):
        return None
    import numpy as np

    matrix = np.array(result.confusion)
    true_labels = ["{0}\n{1}".format(l, POSTURE_CN.get(l, l)) for l in result.confusion_true_labels]
    pred_labels = ["{0}\n{1}".format(l, POSTURE_CN.get(l, l)) for l in result.confusion_pred_labels]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(pred_labels)))
    ax.set_xticklabels(pred_labels, fontsize=8)
    ax.set_yticks(range(len(true_labels)))
    ax.set_yticklabels(true_labels, fontsize=8)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵（方案 {0}）".format(result.variant))
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                ax.text(j, i, value, ha="center", va="center", fontsize=9,
                        color="white" if value > threshold else "#12304a")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="样本数")
    return save(fig, path)


def plot_per_class_f1(results: Sequence[EvaluationResult], path: Path) -> Optional[Path]:
    """各类别 F1 的分组柱状图。"""
    if not _require(results):
        return None
    import numpy as np

    x = np.arange(len(POSTURES))
    count = len(results)
    width = min(0.8 / max(count, 1), 0.26)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for index, result in enumerate(results):
        values = [next((m.f1 for m in result.per_class if m.label == label), 0.0) for label in POSTURES]
        offset = (index - (count - 1) / 2) * width
        ax.bar(x + offset, values, width, label="方案 {0}".format(result.variant),
               color=PALETTE[index % len(PALETTE)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["{0}\n{1}".format(l, POSTURE_CN.get(l, l)) for l in POSTURES], fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("各类别 F1 对比（有效分类子集）")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return save(fig, path)


def generate_all(results: Sequence[EvaluationResult], figures_dir: Path,
                 confusion_variant: Optional[str] = None) -> Dict[str, Path]:
    """生成全套图表，返回 {名称: 路径}。无数据的图会跳过。"""
    setup_font()
    figures_dir = Path(figures_dir)
    outputs: Dict[str, Path] = {}
    planners = [
        ("accuracy", lambda p: plot_accuracy(results, p)),
        ("macro_f1", lambda p: plot_macro_f1(results, p)),
        ("json_rate", lambda p: plot_json_rate(results, p)),
        ("unknown_rate", lambda p: plot_unknown_rate(results, p)),
        ("latency", lambda p: plot_latency(results, p)),
        ("per_class_f1", lambda p: plot_per_class_f1(results, p)),
    ]
    for name, planner in planners:
        produced = planner(figures_dir / "{0}.png".format(name))
        if produced:
            outputs[name] = produced

    target_variant = confusion_variant or (results[-1].variant if results else None)
    for result in results:
        if result.variant == target_variant:
            produced = plot_confusion(result, figures_dir / "confusion_{0}.png".format(result.variant))
            if produced:
                outputs["confusion_{0}".format(result.variant)] = produced
    return outputs


def _pct(value: Optional[float]) -> str:
    return "待实验" if value is None else "{0:.2%}".format(value)


def _num(value: Optional[float], digits: int = 4) -> str:
    return "待实验" if value is None else ("{0:.{1}f}".format(value, digits))


def render_markdown(results: Sequence[EvaluationResult], meta: Dict[str, object],
                    path: Path, figures: Optional[Dict[str, Path]] = None) -> Path:
    """生成实验结果 Markdown（数字全部来自真实计算）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = figures or {}
    lines: List[str] = []
    lines.append("# 实验结果（自动生成）\n")
    lines.append("> 本文件由 `scripts/run_experiment.py` 依据原始 CSV 自动生成，")
    lines.append("> 其中的每个数字都可以用 `scripts/plot_results.py` 从 `results/raw/` 重新计算得到。\n")
    lines.append("## 运行元数据\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for key, value in meta.items():
        lines.append("| {0} | {1} |".format(key, value))
    lines.append("| 生成时间 | {0} |".format(_dt.datetime.now().isoformat(timespec="seconds")))
    lines.append("")

    lines.append("## 指标汇总\n")
    lines.append("| 方案 | 调用数 | 成功 | 失败率 | JSON 有效率 | 无法判断比例 | 有效分类数 "
                 "| Accuracy(有效) | Accuracy(严格) | Macro-F1 | 平均响应(s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} |".format(
            r.variant, r.n_total, r.n_success, _pct(r.failure_rate), _pct(r.json_valid_rate),
            _pct(r.unknown_rate), r.n_classified, _num(r.accuracy_valid),
            _num(r.accuracy_strict), _num(r.macro_f1),
            "待实验" if r.latency_mean is None else "{0:.2f}".format(r.latency_mean),
        ))
    lines.append("")

    lines.append("## 各类别指标（有效分类子集）\n")
    lines.append("| 方案 | 类别 | Precision | Recall | F1 | 分类子集样本 | 全部样本 | Recall(严格) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        for metric in r.per_class:
            lines.append("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |".format(
                r.variant, "{0}({1})".format(metric.label, POSTURE_CN.get(metric.label, "")),
                _num(metric.precision), _num(metric.recall), _num(metric.f1),
                metric.support_classified, metric.support_total, _num(metric.recall_strict),
            ))
    lines.append("")

    if figures:
        lines.append("## 图表\n")
        for name, path_obj in figures.items():
            rel = Path(path_obj)
            lines.append("### {0}\n".format(_figure_title(name)))
            lines.append("![{0}]({1})\n".format(name, rel.as_posix()))

    failures = [(r.variant, r.failure_reasons) for r in results if r.failure_reasons]
    if failures:
        lines.append("## 调用失败原因分布\n")
        for variant, reasons in failures:
            lines.append("- 方案 {0}：{1}".format(variant, reasons))
        lines.append("")

    notes = [(r.variant, r.notes) for r in results if r.notes]
    if notes:
        lines.append("## 计算备注\n")
        for variant, items in notes:
            for item in items:
                lines.append("- 方案 {0}：{1}".format(variant, item))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_FIGURE_TITLES = {
    "accuracy": "Accuracy 对比",
    "macro_f1": "Macro-F1 对比",
    "json_rate": "JSON 有效率对比",
    "unknown_rate": "无法判断比例对比",
    "latency": "响应时间对比",
    "per_class_f1": "各类别 F1 对比",
}


def _figure_title(name: str) -> str:
    """图表文件名 -> 中文标题（混淆矩阵带方案后缀）。"""
    if name.startswith("confusion_"):
        return "混淆矩阵（方案 {0}）".format(name.split("_", 1)[1])
    return _FIGURE_TITLES.get(name, name)
