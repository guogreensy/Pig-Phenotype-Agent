"""阶段 E 批量实验：30 题 × 3 个条件，共 90 次真实对话。

三个条件对应消融设计：

- ``direct``：不给任何工具，只靠模型自身知识作答（基线）
- ``rag``   ：只给知识库检索（search_knowledge / fetch_passage）
- ``full``  ：给全部 8 个工具（数据 + 视觉 + 检索）

每次对话完成后**立即落盘**到 ``results/agent/runs/<条件>/<题号>.json``，
因此可以随时中断、重跑时自动跳过已完成的题（``--force`` 覆盖）。

    python scripts/run_agent_experiment.py --dry-run
    python scripts/run_agent_experiment.py
    python scripts/run_agent_experiment.py --conditions full --ids D1,K1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.executor import AgentExecutor  # noqa: E402
from src.agent.llm_client import build_chat_client  # noqa: E402
from src.agent.report import build_report  # noqa: E402
from src.tools.builder import build_default_registry  # noqa: E402

QUESTIONS_FILE = PROJECT_ROOT / "data" / "experiment" / "agent_questions.jsonl"
RUNS_DIR = PROJECT_ROOT / "results" / "agent" / "runs"
INDEX_FILE = RUNS_DIR / "index.json"

#: 条件 -> 装配哪些工具
CONDITIONS: Dict[str, Dict[str, bool]] = {
    "direct": {"include_data": False, "include_vision": False, "include_rag": False},
    "rag": {"include_data": False, "include_vision": False, "include_rag": True},
    "full": {"include_data": True, "include_vision": True, "include_rag": True},
}

CONDITION_CN = {"direct": "直接回答（无工具）", "rag": "检索增强（仅知识库）",
                "full": "检索增强 + 工具调用（全部工具）"}

#: 实验统一参数（三条件必须一致）
MAX_STEPS = 5
MAX_TOOL_CALLS = 8


def load_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_FILE.exists():
        raise SystemExit("缺少 {0}，请先运行 scripts/build_questions.py".format(
            QUESTIONS_FILE.relative_to(PROJECT_ROOT)))
    return [json.loads(line) for line in
            QUESTIONS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_path(condition: str, question_id: str) -> Path:
    return RUNS_DIR / condition / "{0}.json".format(question_id)


def run_one(question: Dict[str, Any], condition: str, client) -> Dict[str, Any]:
    """跑一次对话并返回完整记录。"""
    registry = build_default_registry(
        log_path=RUNS_DIR / "tool_calls_{0}.jsonl".format(condition),
        max_calls=MAX_TOOL_CALLS,
        **CONDITIONS[condition]
    )
    executor = AgentExecutor(registry, client, max_steps=MAX_STEPS)
    started = time.perf_counter()
    result = executor.run(question["question"])
    elapsed = time.perf_counter() - started
    report = build_report(result)

    return {
        "question_id": question["id"],
        "category": question["category"],
        "condition": condition,
        "question": question["question"],
        "answerable": question.get("answerable", True),
        "expected": question.get("expected", {}),
        "evidence_hint": question.get("evidence_hint", ""),
        "image_path": question.get("image_path"),
        "ok": report.ok,
        "answer": report.answer,
        "stop_reason": report.stop_reason,
        "citations": report.citations,
        "tool_trace": report.tool_trace,
        "flags": report.flags,
        "metrics": report.metrics,
        "prompt_meta": report.prompt_meta,
        "error": report.error,
        "wall_seconds": round(elapsed, 3),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _save(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def collect_from_disk() -> List[Dict[str, Any]]:
    """把磁盘上所有结果重新读一遍，保证断点续跑后统计仍准确。"""
    records: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for question in load_questions():
            path = run_path(condition, question["id"])
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = [r for r in records if r["condition"] == condition]
        if not subset:
            continue
        summary[condition] = {
            "runs": len(subset),
            "ok": sum(1 for r in subset if r.get("ok")),
            "tool_calls": sum((r.get("metrics") or {}).get("tool_calls", 0) for r in subset),
            "model_calls": sum((r.get("metrics") or {}).get("model_calls", 0) for r in subset),
            "total_seconds": round(sum(r.get("wall_seconds") or 0 for r in subset), 1),
        }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 E 批量实验")
    parser.add_argument("--conditions", default="direct,rag,full",
                        help="逗号分隔：direct,rag,full")
    parser.add_argument("--ids", default="", help="只跑指定题号，逗号分隔，如 D1,K3")
    parser.add_argument("--force", action="store_true", help="覆盖已完成的结果")
    parser.add_argument("--dry-run", action="store_true", help="只列出计划，不调用接口")
    parser.add_argument("--sleep", type=float, default=1.0, help="每次对话之间停顿秒数")
    parser.add_argument("--max-runs", type=int, default=0, help="最多跑几次（0 不限）")
    args = parser.parse_args(argv)

    questions = load_questions()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit("未知条件：{0}".format("、".join(unknown)))
    if args.ids:
        wanted = {i.strip().upper() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q["id"].upper() in wanted]
        if not questions:
            raise SystemExit("没有匹配的题号")

    tasks = [(q, c) for c in conditions for q in questions]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("计划执行 {0} 次对话（{1} 题 × {2} 条件）".format(
            len(tasks), len(questions), len(conditions)))
        for condition in conditions:
            print("  {0:<7} {1}".format(condition, CONDITION_CN[condition]))
        print("已有结果：")
        for condition in conditions:
            done = sum(1 for q in questions if run_path(condition, q["id"]).exists())
            print("  {0:<7} {1}/{2}".format(condition, done, len(questions)))
        return 0

    client = build_chat_client()
    print("模型：{0}　最多 {1} 步 / {2} 次工具调用".format(
        client.model, MAX_STEPS, MAX_TOOL_CALLS))
    print("计划 {0} 次对话，结果写入 {1}".format(
        len(tasks), RUNS_DIR.relative_to(PROJECT_ROOT)))
    print("=" * 78)

    ran = 0
    failures = 0
    started_all = time.perf_counter()

    for question, condition in tasks:
        path = run_path(condition, question["id"])
        if path.exists() and not args.force:
            print("[跳过] {0}/{1} 已完成".format(condition, question["id"]))
            continue
        if args.max_runs and ran >= args.max_runs:
            print("达到 --max-runs 上限，停止")
            break

        label = "{0}/{1}".format(condition, question["id"])
        print("[{0}/{1}] {2}  {3}".format(ran + 1, len(tasks), label,
                                          question["question"][:38]))
        try:
            record = run_one(question, condition, client)
        except Exception as exc:  # 单次失败不应中断整批实验
            record = {
                "question_id": question["id"], "category": question["category"],
                "condition": condition, "question": question["question"],
                "answerable": question.get("answerable", True),
                "expected": question.get("expected", {}),
                "image_path": question.get("image_path"),
                "ok": False, "answer": "", "stop_reason": "exception",
                "error": "{0}: {1}".format(type(exc).__name__, exc),
                "citations": [], "tool_trace": [], "flags": {}, "metrics": {},
                "wall_seconds": 0.0,
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            print("    异常：{0}".format(record["error"]))

        _save(path, record)
        ran += 1
        if not record.get("ok"):
            failures += 1
        metrics = record.get("metrics") or {}
        print("    完成={0} 模型{1}次 工具{2}次 {3}秒 引用{4}条".format(
            record.get("ok"), metrics.get("model_calls", 0),
            metrics.get("tool_calls", 0), record.get("wall_seconds"),
            len(record.get("citations") or [])))
        if args.sleep:
            time.sleep(args.sleep)

    elapsed = time.perf_counter() - started_all
    print("=" * 78)
    print("本次实际运行 {0} 次，失败 {1} 次，耗时 {2:.1f} 分钟".format(
        ran, failures, elapsed / 60))

    all_records = collect_from_disk()
    summary = summarize(all_records)
    INDEX_FILE.write_text(json.dumps({
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": client.model,
        "max_steps": MAX_STEPS,
        "max_tool_calls": MAX_TOOL_CALLS,
        "total_runs": len(all_records),
        "conditions": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    total_expected = len(CONDITIONS) * len(load_questions())
    print("累计完成 {0}/{1} 次".format(len(all_records), total_expected))
    for condition, item in summary.items():
        print("  {0:<7} {1:>2} 次　成功 {2:>2}　模型 {3:>3} 次　工具 {4:>3} 次　{5} 秒".format(
            condition, item["runs"], item["ok"], item["model_calls"],
            item["tool_calls"], item["total_seconds"]))
    print("索引：{0}".format(INDEX_FILE.relative_to(PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())