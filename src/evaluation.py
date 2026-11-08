"""实验指标计算。

口径约定（会原样写进实验报告，务必与实现保持一致）：

* **总调用数** ``n_total``：写入原始记录的样本数（每样本每提示词方案一条）。
* **调用成功率**：``request_success=True`` 的比例。
* **JSON 有效率**：``json_valid`` **且** ``schema_valid`` 的比例，分母为 ``n_total``。
  它衡量"结构化输出质量"，与分类对错无关。
* **分类得分的前提**：请求成功 **且** 从回答中抽出了合法类别（四类之一或 ``unknown``）。
  **不要求** Schema 完全合规——把"输出格式"和"分类正确性"混在一起会把
  格式劣势误记为识别能力劣势，二者必须分开统计（A/B 组为自然语言输出，
  只有 C 组才是严格 JSON；若要求 Schema 合规才能得分，A/B 的分类能力就无从比较）。
* **无法判断比例**：模型输出 ``unknown`` 的比例，分母为**成功调用数**。
  拒答**绝不算作正确**，单独报告。
* **Accuracy（有效分类子集）**：分母为"成功且给出具体四类之一"的样本数。
* **Accuracy（严格）**：分母为 ``n_total``，拒答、非法输出、调用失败一律记为错误。
* **Macro-F1**：四类各自 F1 的等权平均，只在有效分类子集上计算；
  为使口径可核查，同时给出 ``recall_strict``（分母含该类全部样本，拒答算漏检）。
* **响应时间**：仅统计成功调用，单位秒；p95 采用最近秩法。

所有指标都由原始 CSV 重新计算，保证"结果可复现"（需求 6.4 / 阶段四验收）。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.schema import POSTURES

UNKNOWN = "unknown"
LABELS_ALL: List[str] = POSTURES + [UNKNOWN]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "是")
    return bool(value)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """最近秩法分位数（不插值），便于人工核对。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(math.ceil(pct / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass
class ClassMetric:
    label: str
    precision: float
    recall: float              # 分母 = 该类在有效分类子集中的样本数
    f1: float
    support_classified: int    # 该类在有效分类子集中的样本数
    support_total: int         # 该类全部样本数（含拒答/失败）
    predicted: int             # 被预测成该类别的次数
    recall_strict: float       # 分母 = support_total，拒答算漏检


@dataclass
class EvaluationResult:
    variant: str = ""
    model: str = ""
    n_total: int = 0
    n_success: int = 0
    n_failed: int = 0
    failure_rate: float = 0.0
    n_json_valid: int = 0
    json_valid_rate: float = 0.0
    n_schema_valid: int = 0
    schema_valid_rate: float = 0.0
    n_unknown: int = 0
    unknown_rate: float = 0.0
    n_classified: int = 0
    n_correct: int = 0
    accuracy_valid: Optional[float] = None
    accuracy_strict: float = 0.0
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    per_class: List[ClassMetric] = field(default_factory=list)
    confusion: List[List[int]] = field(default_factory=list)
    confusion_pred_labels: List[str] = field(default_factory=list)
    confusion_true_labels: List[str] = field(default_factory=list)
    latency_mean: Optional[float] = None
    latency_median: Optional[float] = None
    latency_p95: Optional[float] = None
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_row(self) -> Dict[str, Any]:
        """展平成一行，便于写入指标汇总 CSV。"""
        return {
            "prompt_variant": self.variant,
            "model_name": self.model,
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "failure_rate": _fmt(self.failure_rate),
            "n_json_valid": self.n_json_valid,
            "json_valid_rate": _fmt(self.json_valid_rate),
            "n_schema_valid": self.n_schema_valid,
            "schema_valid_rate": _fmt(self.schema_valid_rate),
            "n_unknown": self.n_unknown,
            "unknown_rate": _fmt(self.unknown_rate),
            "n_classified": self.n_classified,
            "n_correct": self.n_correct,
            "accuracy_valid": _fmt(self.accuracy_valid),
            "accuracy_strict": _fmt(self.accuracy_strict),
            "macro_f1": _fmt(self.macro_f1),
            "weighted_f1": _fmt(self.weighted_f1),
            "latency_mean_s": _fmt(self.latency_mean),
            "latency_median_s": _fmt(self.latency_median),
            "latency_p95_s": _fmt(self.latency_p95),
        }


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return "{0:.4f}".format(value)


def _record_prediction(record: Dict[str, Any]) -> Optional[str]:
    """取出可用于计分的预测类别；不可用时返回 None。

    只要求：请求成功 + 抽出了合法类别。Schema 是否合规由 JSON 有效率单独统计。
    """
    if not _truthy(record.get("request_success")):
        return None
    label = (record.get("predicted_label") or "").strip()
    if label in POSTURES:
        return label
    if label == UNKNOWN:
        return UNKNOWN
    return None


def evaluate(records: Iterable[Dict[str, Any]], variant: str = "", model: str = "") -> EvaluationResult:
    """计算一组实验记录的全部指标。"""
    rows = list(records)
    result = EvaluationResult(variant=variant, model=model, n_total=len(rows))
    if not rows:
        result.notes.append("没有可统计的记录")
        result.confusion_pred_labels = LABELS_ALL[:]
        result.confusion_true_labels = POSTURES[:]
        result.confusion = [[0] * len(LABELS_ALL) for _ in POSTURES]
        return result

    result.model = model or str(rows[0].get("model_name") or "")

    latencies: List[float] = []
    predictions: List[Optional[str]] = []
    truths: List[str] = []
    failure_reasons: Dict[str, int] = {}

    for row in rows:
        success = _truthy(row.get("request_success"))
        if success:
            result.n_success += 1
            latency = _to_float(row.get("latency_seconds"))
            if latency is not None:
                latencies.append(latency)
        else:
            result.n_failed += 1
            reason = (row.get("error_message") or "未记录原因").strip()
            key = reason.split("：")[0][:60]
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

        if _truthy(row.get("json_valid")) and _truthy(row.get("schema_valid")):
            result.n_json_valid += 1
        if _truthy(row.get("schema_valid")):
            result.n_schema_valid += 1

        pred = _record_prediction(row)
        truth = (row.get("true_label") or "").strip()
        predictions.append(pred)
        truths.append(truth)
        if pred == UNKNOWN:
            result.n_unknown += 1

    result.failure_rate = result.n_failed / result.n_total
    result.json_valid_rate = result.n_json_valid / result.n_total
    result.schema_valid_rate = result.n_schema_valid / result.n_total
    result.unknown_rate = result.n_unknown / result.n_success if result.n_success else 0.0
    result.failure_reasons = failure_reasons

    # ---------------- 混淆矩阵：行=真实四类，列=预测四类+unknown
    pred_labels = LABELS_ALL[:]
    index_pred = {label: i for i, label in enumerate(pred_labels)}
    index_true = {label: i for i, label in enumerate(POSTURES)}
    confusion = [[0] * len(pred_labels) for _ in POSTURES]
    for pred, truth in zip(predictions, truths):
        if truth not in index_true or pred is None:
            continue
        confusion[index_true[truth]][index_pred[pred]] += 1
    result.confusion = confusion
    result.confusion_pred_labels = pred_labels
    result.confusion_true_labels = POSTURES[:]

    # ---------------- 有效分类子集
    classified = [
        (pred, truth)
        for pred, truth in zip(predictions, truths)
        if pred in POSTURES and truth in POSTURES
    ]
    result.n_classified = len(classified)
    result.n_correct = sum(1 for pred, truth in classified if pred == truth)
    if result.n_classified:
        result.accuracy_valid = result.n_correct / result.n_classified
    result.accuracy_strict = result.n_correct / result.n_total

    # ---------------- 每类指标
    per_class: List[ClassMetric] = []
    support_total = {label: sum(1 for truth in truths if truth == label) for label in POSTURES}
    support_classified = {
        label: sum(1 for _, truth in classified if truth == label) for label in POSTURES
    }
    f1_weighted_numerator = 0.0
    f1_sum = 0.0
    for label in POSTURES:
        tp = sum(1 for pred, truth in classified if pred == label and truth == label)
        predicted = sum(1 for pred, _ in classified if pred == label)
        support = support_classified[label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        recall_strict = tp / support_total[label] if support_total[label] else 0.0
        per_class.append(
            ClassMetric(
                label=label,
                precision=precision,
                recall=recall,
                f1=f1,
                support_classified=support,
                support_total=support_total[label],
                predicted=predicted,
                recall_strict=recall_strict,
            )
        )
        f1_sum += f1
        f1_weighted_numerator += f1 * support
    result.per_class = per_class

    if POSTURES:
        result.macro_f1 = f1_sum / len(POSTURES) if result.n_classified else None
    classified_support = sum(support_classified.values())
    if classified_support and result.n_classified:
        result.weighted_f1 = f1_weighted_numerator / classified_support

    # ---------------- 延迟
    if latencies:
        result.latency_mean = sum(latencies) / len(latencies)
        result.latency_median = percentile(latencies, 50)
        result.latency_p95 = percentile(latencies, 95)

    if result.n_classified == 0:
        result.notes.append("没有任何样本被有效分类，Accuracy/Macro-F1 不可计算")
    if result.n_unknown:
        result.notes.append(
            "有 {0} 条拒答（unknown），未计入 Accuracy（有效分类子集）".format(result.n_unknown)
        )
    return result


REQUIRED_FIELDS = [
    "sample_id",
    "image_path",
    "true_label",
    "prompt_variant",
    "model_name",
    "predicted_label",
    "description",
    "uncertain",
    "json_valid",
    "request_success",
    "latency_seconds",
    "error_message",
    "experiment_timestamp",
]


def load_records(csv_path) -> List[Dict[str, Any]]:
    """读取原始结果 CSV。"""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_variant(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按提示词方案分组。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        key = (row.get("prompt_variant") or "").strip() or "?"
        grouped.setdefault(key, []).append(row)
    return grouped


def evaluate_all(records: Iterable[Dict[str, Any]],
                 variants: Optional[Sequence[str]] = None) -> List[EvaluationResult]:
    """对每个提示词方案分别计算指标。"""
    rows = list(records)
    grouped = group_by_variant(rows)
    order = list(variants) if variants else sorted(grouped)
    results: List[EvaluationResult] = []
    for key in order:
        subset = grouped.get(key, [])
        model = subset[0].get("model_name", "") if subset else ""
        results.append(evaluate(subset, variant=key, model=str(model)))
    return results


def write_summary_csv(results: Sequence[EvaluationResult], path) -> Path:
    """把指标写成汇总 CSV（utf-8-sig，Excel 可直接打开）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.as_row() for result in results]
    fieldnames = list(rows[0].keys()) if rows else ["prompt_variant"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_per_class_csv(results: Sequence[EvaluationResult], path) -> Path:
    """各类别 Precision/Recall/F1 明细。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["prompt_variant", "label", "precision", "recall", "f1",
              "support_classified", "support_total", "predicted", "recall_strict"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for metric in result.per_class:
                writer.writerow({
                    "prompt_variant": result.variant,
                    "label": metric.label,
                    "precision": _fmt(metric.precision),
                    "recall": _fmt(metric.recall),
                    "f1": _fmt(metric.f1),
                    "support_classified": metric.support_classified,
                    "support_total": metric.support_total,
                    "predicted": metric.predicted,
                    "recall_strict": _fmt(metric.recall_strict),
                })
    return target


def format_table(results: Sequence[EvaluationResult]) -> str:
    """终端用的纯文本对比表。"""
    header = ("方案", "n", "成功", "失败率", "JSON率", "无法判断率", "有效n",
              "Acc(有效)", "Acc(严格)", "Macro-F1", "延迟均值")
    lines = [" | ".join(header), "-" * 96]
    for r in results:
        lines.append(" | ".join([
            r.variant or "-",
            str(r.n_total),
            str(r.n_success),
            "{0:.1%}".format(r.failure_rate),
            "{0:.1%}".format(r.json_valid_rate),
            "{0:.1%}".format(r.unknown_rate),
            str(r.n_classified),
            "-" if r.accuracy_valid is None else "{0:.3f}".format(r.accuracy_valid),
            "{0:.3f}".format(r.accuracy_strict),
            "-" if r.macro_f1 is None else "{0:.3f}".format(r.macro_f1),
            "-" if r.latency_mean is None else "{0:.2f}s".format(r.latency_mean),
        ]))
    return "\n".join(lines)
