"""阶段 E 指标计算：把 90 次对话记录变成可引用、可复核的指标。

指标全部是**可从记录自动算出**的客观量，不做"回答质量"的主观打分：

数据题
  - 数值命中率：期望值（从原始记录独立算出）是否出现在回答中，容差按题记录
  - 题目正确率：一道题的所有期望值都被命中
  - 实体命中率：该题的猪号/栏位号是否出现

知识题
  - 检索命中率：引用里是否出现期望文献（doc_id 由 chunk_id 推导）
  - 引用覆盖率：回答是否带引用

视觉题
  - 姿态命中率：回答是否给出期望类别（中英文任一）

不可作答题
  - 拒答率：回答是否明确表示"没有这个信息"，而不是编造

产物：
  results/metrics/agent_metrics_by_condition.csv
  results/metrics/agent_per_question.csv
  results/metrics/agent_summary.json
  results/metrics/agent_manual_review.md（供人工抽查）

    python scripts/evaluate_agent.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_FILE = PROJECT_ROOT / "data" / "experiment" / "agent_questions.jsonl"


def load_exclusions() -> Dict[str, str]:
    """读出被排除在汇总之外的数据题（题号 -> 原因）。"""
    if not QUESTIONS_FILE.exists():
        return {}
    out: Dict[str, str] = {}
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("exclude_from_aggregate"):
            out[item["id"]] = item.get("exclude_reason", "")
    return out


RUNS_DIR = PROJECT_ROOT / "results" / "agent" / "runs"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

CONDITIONS = ["direct", "rag", "full"]
CATEGORIES = ["data", "knowledge", "vision", "unanswerable"]

#: 表示"确实没有这个信息"的措辞；命中任一即视为诚实拒答
ADMIT_MARKERS = [
    "无法", "没有", "不包含", "未提供", "不存在", "不能", "不清楚", "未记录",
    "查不到", "尚无", "不涉及", "未包含", "缺乏", "不具备", "未采集", "无相关",
]

#: 抽取数字前先剔除日期与编号，避免 2026、10、31、007 之类误当数值
DATE_RE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?")
ID_RE = re.compile(r"\b[A-Za-z]{1,3}\d+\s*-\s*\d+\b")
NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%)?")


def extract_numbers(text: str) -> List[float]:
    """从回答中抽取候选数值；百分比同时给出原值与小数两种形态。"""
    cleaned = DATE_RE.sub(" ", text or "")
    cleaned = ID_RE.sub(" ", cleaned)
    values: List[float] = []
    for match in NUM_RE.finditer(cleaned):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        values.append(value)
        if match.group(2):
            values.append(value / 100.0)
    return values


def match_numbers(expected: List[Dict[str, Any]],
                  candidates: List[float]) -> Tuple[int, List[bool]]:
    """逐个期望值找最近的未占用候选值；距差在容差内算命中。"""
    used = set()
    hits: List[bool] = []
    for item in expected:
        target = float(item["value"])
        tol = float(item.get("tol") or 0.02)
        best_index, best_diff = None, None
        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            diff = abs(candidate - target)
            if best_diff is None or diff < best_diff:
                best_index, best_diff = index, diff
        if best_index is not None and best_diff is not None and best_diff <= tol:
            used.add(best_index)
            hits.append(True)
        else:
            hits.append(False)
    return sum(hits), hits


def citation_doc_ids(record: Dict[str, Any]) -> List[str]:
    ids = []
    for citation in record.get("citations") or []:
        chunk_id = str(citation.get("chunk_id") or "")
        doc_id = chunk_id.split("#")[0] if "#" in chunk_id else chunk_id
        if doc_id:
            ids.append(doc_id)
    return ids


def evaluate_run(record: Dict[str, Any],
                 exclusions: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """单次对话的判定结果。"""
    answer = record.get("answer") or ""
    expected = record.get("expected") or {}
    category = record.get("category")
    docs = citation_doc_ids(record)
    excluded = bool(exclusions and record.get("question_id") in exclusions)
    row: Dict[str, Any] = {
        "condition": record.get("condition"),
        "question_id": record.get("question_id"),
        "category": category,
        "excluded": excluded,
        "ok": bool(record.get("ok")),
        "stop_reason": record.get("stop_reason"),
        "answer_chars": len(answer),
        "tool_calls": (record.get("metrics") or {}).get("tool_calls", 0),
        "model_calls": (record.get("metrics") or {}).get("model_calls", 0),
        "wall_seconds": record.get("wall_seconds") or 0,
        "citations": len(record.get("citations") or []),
        "degraded": bool((record.get("flags") or {}).get("degraded")),
        "numbers_hit": None, "numbers_total": None, "numbers_ok": None,
        "entity_hit": None, "retrieval_hit": None, "label_hit": None,
        "vision_tool_used": None, "vision_refused": None, "admit": None,
    }

    if category == "data":
        expected_numbers = expected.get("numbers") or []
        if expected_numbers:
            hit, _ = match_numbers(expected_numbers, extract_numbers(answer))
            row["numbers_hit"] = hit
            row["numbers_total"] = len(expected_numbers)
            row["numbers_ok"] = hit == len(expected_numbers)
        # 只统计"题面里没有出现"的实体词，否则模型照抄问题就算命中，指标失去意义
        question_text = record.get("question") or ""
        tokens = [t for t in (expected.get("tokens") or [])
                  if t and t not in question_text]
        if tokens:
            row["entity_hit"] = all(t in answer for t in tokens)

    elif category == "knowledge":
        expected_docs = set(expected.get("doc_ids") or [])
        row["retrieval_hit"] = bool(expected_docs & set(docs)) if expected_docs else None
        row["citation_present"] = bool(docs)

    elif category == "vision":
        label = expected.get("label")
        label_cn = expected.get("label_cn")
        used_tool = any((t.get("tool") == "analyze_pig_posture")
                        for t in (record.get("tool_trace") or []))
        row["vision_tool_used"] = used_tool
        # 没有视觉工具时，模型连图片都看不到：回答里罗列"站立/侧卧/坐姿"
        # 只是举例，不能算命中，这里记为不适用。
        if used_tool:
            row["label_hit"] = bool(
                (label and label in answer) or (label_cn and label_cn in answer))
        else:
            row["label_hit"] = None
            row["vision_refused"] = any(m in answer for m in ADMIT_MARKERS)

    elif category == "unanswerable":
        row["admit"] = any(marker in answer for marker in ADMIT_MARKERS)

    return row


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.mean(values), 4) if values else None


def _rate(flags: List[Optional[bool]]) -> Optional[float]:
    clean = [f for f in flags if f is not None]
    return round(sum(1 for f in clean if f) / len(clean), 4) if clean else None


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return round(ordered[index], 3)


def summarize_condition(rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    subset = [r for r in rows if r["condition"] == condition]
    if not subset:
        return {}
    by_cat = {cat: [r for r in subset if r["category"] == cat] for cat in CATEGORIES}
    data_rows = [r for r in by_cat["data"] if not r.get("excluded")]
    data_excluded = [r for r in by_cat["data"] if r.get("excluded")]
    know_rows = by_cat["knowledge"]
    vision_rows = by_cat["vision"]
    unans_rows = by_cat["unanswerable"]

    number_hits = sum(r["numbers_hit"] for r in data_rows if r["numbers_hit"] is not None)
    number_total = sum(r["numbers_total"] for r in data_rows if r["numbers_total"])
    return {
        "condition": condition,
        "runs": len(subset),
        "ok_rate": _rate([r["ok"] for r in subset]),
        "answer_chars_mean": _mean([r["answer_chars"] for r in subset]),
        "model_calls_mean": _mean([r["model_calls"] for r in subset]),
        "tool_calls_mean": _mean([r["tool_calls"] for r in subset]),
        "latency_mean": _mean([r["wall_seconds"] for r in subset]),
        "latency_p95": _p95([r["wall_seconds"] for r in subset]),
        "citations_mean": _mean([r["citations"] for r in subset]),
        "degraded_rate": _rate([r["degraded"] for r in subset]),
        # 数据题
        "data_runs": len(data_rows),
        "data_number_hit_rate": (round(number_hits / number_total, 4)
                                 if number_total else None),
        "data_question_accuracy": _rate([r["numbers_ok"] for r in data_rows]),
        # 异常指认题（如 D3）：题面不含实体名，答案是"哪头猪"，检不出数值
        "data_entity_accuracy": _rate([r["entity_hit"] for r in data_rows]),
        "data_numbers_total": number_total,
        # 提示词里已给出答案的题（如 D6），三个条件都能答，单独列出
        "data_prompt_answerable_accuracy": _rate(
            [r["numbers_ok"] for r in data_excluded]),
        # 知识题
        "knowledge_runs": len(know_rows),
        "knowledge_retrieval_hit_rate": _rate([r["retrieval_hit"] for r in know_rows]),
        "knowledge_citation_rate": _rate([r.get("citation_present") for r in know_rows]),
        # 视觉题
        "vision_runs": len(vision_rows),
        "vision_tool_used_rate": _rate([r["vision_tool_used"] for r in vision_rows]),
        "vision_label_accuracy": _rate([r["label_hit"] for r in vision_rows]),
        "vision_refusal_rate": _rate([r.get("vision_refused") for r in vision_rows]),
        # 不可作答题
        "unanswerable_runs": len(unans_rows),
        "unanswerable_admit_rate": _rate([r["admit"] for r in unans_rows]),
    }


def load_runs() -> List[Dict[str, Any]]:
    records = []
    for condition in CONDITIONS:
        folder = RUNS_DIR / condition
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def write_csv(rows: List[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_manual_review(records: List[Dict[str, Any]], path: Path) -> Path:
    """把需要人工判断的回答整理成一份可复核的清单。"""
    lines = ["# 阶段 E 人工抽查清单", "",
             "自动指标无法判断的部分（引用归因是否准确、拒答是否合理、",
             "姿态描述是否与图片一致）在这里逐条列出，供人工核对。", ""]

    lines.append("## 一、不可作答题（检查是否诚实说明「没有这个信息」）")
    lines.append("")
    for condition in CONDITIONS:
        lines.append("### 条件 {0}".format(condition))
        lines.append("")
        for record in records:
            if record["condition"] != condition or record["category"] != "unanswerable":
                continue
            lines.append("- **{0}** {1}".format(record["question_id"], record["question"]))
            answer = (record.get("answer") or "").strip().replace("\n", " ")
            lines.append("  > {0}".format(answer[:300] or "（空回答）"))
        lines.append("")

    lines.append("## 二、视觉题（检查姿态类别与图片是否一致）")
    lines.append("")
    for condition in CONDITIONS:
        lines.append("### 条件 {0}".format(condition))
        lines.append("")
        for record in records:
            if record["condition"] != condition or record["category"] != "vision":
                continue
            expected = record.get("expected") or {}
            answer = (record.get("answer") or "").strip().replace("\n", " ")
            lines.append("- **{0}** 期望 `{1}`（{2}）".format(
                record["question_id"], expected.get("label"), expected.get("label_cn")))
            lines.append("  > {0}".format(answer[:220] or "（空回答）"))
        lines.append("")

    lines.append("## 三、知识题引用归因（检查引用是否支撑结论）")
    lines.append("")
    for condition in ("rag", "full"):
        lines.append("### 条件 {0}".format(condition))
        lines.append("")
        for record in records:
            if record["condition"] != condition or record["category"] != "knowledge":
                continue
            expected_docs = (record.get("expected") or {}).get("doc_ids") or []
            got = citation_doc_ids(record)
            lines.append("- **{0}** 期望来源 {1}；实际引用 {2}".format(
                record["question_id"], "、".join(expected_docs) or "—",
                "、".join(got) or "—"))
            answer = (record.get("answer") or "").strip().replace("\n", " ")
            lines.append("  > {0}".format(answer[:260] or "（空回答）"))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_markdown_table(per_condition: List[Dict[str, Any]],
                         exclusions: Dict[str, str], path: Path) -> Path:
    """把主指标表写成 Markdown，便于直接引用到报告里。"""
    index = {row["condition"]: row for row in per_condition}
    conditions = [c for c in CONDITIONS if c in index]
    lines = ["# 三条件对照指标", ""]
    lines.append("| 指标 | " + " | ".join(conditions) + " |")
    lines.append("|---|" + "---|" * len(conditions))
    rows = [
        ("对话次数", "runs", "{0}"),
        ("完成率", "ok_rate", "{0:.1%}"),
        ("平均模型调用", "model_calls_mean", "{0:.2f}"),
        ("平均工具调用", "tool_calls_mean", "{0:.2f}"),
        ("平均耗时（秒）", "latency_mean", "{0:.1f}"),
        ("P95 耗时（秒）", "latency_p95", "{0:.1f}"),
        ("平均引用数", "citations_mean", "{0:.2f}"),
        ("数据题·数值命中率", "data_number_hit_rate", "{0:.1%}"),
        ("数据题·整题正确率", "data_question_accuracy", "{0:.1%}"),
        ("数据题·异常指认", "data_entity_accuracy", "{0:.1%}"),
        ("数据题·提示词内题（D6）", "data_prompt_answerable_accuracy", "{0:.1%}"),
        ("数据题·数值项合计", "data_numbers_total", "{0:.0f}"),
        ("知识题·检索命中率", "knowledge_retrieval_hit_rate", "{0:.1%}"),
        ("知识题·引用覆盖率", "knowledge_citation_rate", "{0:.1%}"),
        ("视觉题·姿态命中率", "vision_label_accuracy", "{0:.1%}"),
        ("视觉题·调用视觉工具", "vision_tool_used_rate", "{0:.1%}"),
        ("视觉题·如实说看不到", "vision_refusal_rate", "{0:.1%}"),
        ("拒答题·诚实拒答率", "unanswerable_admit_rate", "{0:.1%}"),
        ("检索降级率", "degraded_rate", "{0:.1%}"),
    ]
    for label, key, fmt in rows:
        cells = []
        for condition in conditions:
            value = index[condition].get(key)
            cells.append("—" if value is None else fmt.format(value))
        lines.append("| {0} | {1} |".format(label, " | ".join(cells)))

    if exclusions:
        lines += ["", "## 未计入汇总的题目", ""]
        for question_id, reason in exclusions.items():
            lines.append("- **{0}**：{1}".format(question_id, reason))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 E 指标计算")
    parser.add_argument("--out", default=str(METRICS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out)

    records = load_runs()
    if not records:
        raise SystemExit("没有找到任何实验记录，请先运行 scripts/run_agent_experiment.py")

    exclusions = load_exclusions()

    per_run = [evaluate_run(record, exclusions) for record in records]
    per_condition = [summarize_condition(per_run, c) for c in CONDITIONS]
    per_condition = [row for row in per_condition if row]

    run_csv = write_csv(per_run, out_dir / "agent_per_question.csv")
    cond_csv = write_csv(per_condition, out_dir / "agent_metrics_by_condition.csv")
    review_md = write_manual_review(records, out_dir / "agent_manual_review.md")
    table_md = write_markdown_table(per_condition, exclusions,
                                    out_dir / "agent_metrics_table.md")

    summary = {
        "total_runs": len(records),
        "conditions": per_condition,
        "counts": {c: sum(1 for r in records if r["condition"] == c) for c in CONDITIONS},
        "questions": len({r["question_id"] for r in records}),
    }
    summary_path = out_dir / "agent_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print("记录 {0} 次（{1} 题）".format(len(records), summary["questions"]))
    print("=" * 96)
    header = "指标".ljust(30) + "".join(c.rjust(14) for c in CONDITIONS)
    print(header)
    rows = [
        ("完成率", "ok_rate"),
        ("平均模型调用", "model_calls_mean"),
        ("平均工具调用", "tool_calls_mean"),
        ("平均耗时（秒）", "latency_mean"),
        ("P95 耗时（秒）", "latency_p95"),
        ("数据题·数值命中率", "data_number_hit_rate"),
        ("数据题·整题正确率", "data_question_accuracy"),
        ("数据题·异常指认", "data_entity_accuracy"),
        ("数据题·提示词内题", "data_prompt_answerable_accuracy"),
        ("知识题·检索命中率", "knowledge_retrieval_hit_rate"),
        ("知识题·引用覆盖率", "knowledge_citation_rate"),
        ("视觉题·姿态命中率", "vision_label_accuracy"),
        ("视觉题·调用视觉工具", "vision_tool_used_rate"),
        ("视觉题·如实说看不到", "vision_refusal_rate"),
        ("拒答题·诚实拒答率", "unanswerable_admit_rate"),
        ("检索降级率", "degraded_rate"),
    ]
    index = {row["condition"]: row for row in per_condition}
    for label, key in rows:
        cells = []
        for condition in CONDITIONS:
            value = index.get(condition, {}).get(key)
            cells.append(("—" if value is None else "{0:.3f}".format(value)).rjust(14))
        print(label.ljust(30) + "".join(cells))

    print("=" * 96)
    print("按条件表：{0}".format(cond_csv.relative_to(PROJECT_ROOT)))
    print("逐题明细：{0}".format(run_csv.relative_to(PROJECT_ROOT)))
    print("汇总 JSON：{0}".format(summary_path.relative_to(PROJECT_ROOT)))
    print("人工抽查：{0}".format(review_md.relative_to(PROJECT_ROOT)))
    print("指标表：{0}".format(table_md.relative_to(PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())