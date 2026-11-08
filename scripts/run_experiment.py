"""正式实验：三组提示词 × 测试集 → 原始记录 → 指标 → 图表 → 报告草稿。

用法::

    python scripts/run_experiment.py --split test --variants A,B,C
    python scripts/run_experiment.py --split dev --limit 12   # 先小规模验证流程

只有配置了 API 密钥才会真正调用模型；没有密钥会立即退出并说明原因，
**不会**生成任何假数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from config import MissingAPIKeyError, get_settings  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_all,
    format_table,
    load_records,
    write_per_class_csv,
    write_summary_csv,
)
from src.experiment_runner import ExperimentRunner, load_samples  # noqa: E402
from src.prompts import VARIANT_ORDER, load_all  # noqa: E402
from src.reporting import generate_all, render_markdown  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行三组提示词对照实验")
    parser.add_argument("--split", default="test", choices=["test", "dev"],
                        help="使用哪个划分（默认 test）")
    parser.add_argument("--variants", default=",".join(VARIANT_ORDER),
                        help="逗号分隔的提示词方案，默认 A,B,C")
    parser.add_argument("--limit", type=int, default=0, help="限制样本数（0=不限）")
    parser.add_argument("--run-id", default="", help="运行标识，默认按时间生成")
    parser.add_argument("--concurrency", type=int, default=0, help="并发线程数")
    parser.add_argument("--qps", type=float, default=0.0, help="每秒请求数上限")
    parser.add_argument("--max-calls", type=int, default=0, help="调用次数硬上限")
    parser.add_argument("--repeats", type=int, default=1, help="每个样本重复次数")
    parser.add_argument("--no-report", action="store_true", help="只跑实验，不生成图表")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        settings = get_settings(require_key=True)
    except MissingAPIKeyError as exc:
        print("[阻塞] {0}".format(exc), file=sys.stderr)
        print("       提示：本项目不会用模拟或预制结果替代真实调用。", file=sys.stderr)
        return 2

    config.ensure_dirs()
    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    try:
        prompts = {v.key: v for v in load_all()}
    except Exception as exc:
        print("[错误] 提示词加载失败：{0}".format(exc), file=sys.stderr)
        return 2
    for key in variants:
        if key not in prompts:
            print("[错误] 未知提示词方案 {0}".format(key), file=sys.stderr)
            return 2

    if not config.MANIFEST_PATH.is_file():
        print("[错误] 未找到 {0}，请先运行 python scripts/build_dataset.py".format(
            config.MANIFEST_PATH.relative_to(ROOT)), file=sys.stderr)
        return 2

    samples = load_samples(config.MANIFEST_PATH, split=args.split,
                           limit=args.limit or None)
    if not samples:
        print("[错误] {0} 划分下没有可用样本。".format(args.split), file=sys.stderr)
        return 2

    if args.qps:
        # 允许命令行覆盖 QPS
        object.__setattr__(settings, "qps", args.qps)

    run_id = args.run_id or "{0}_{1}".format(
        args.split, __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"))
    planned = len(samples) * len(variants) * max(1, args.repeats)

    print("=" * 66)
    print("实验配置")
    print("-" * 66)
    print("  run_id   : {0}".format(run_id))
    print("  样本数   : {0}（{1} 划分）".format(len(samples), args.split))
    print("  方案     : {0}".format("、".join(
        "{0}({1})".format(k, prompts[k].short_sha) for k in variants)))
    print("  模型     : {0} / {1}".format(settings.provider, settings.model))
    print("  计划调用 : {0} 次（含重复 {1} 次）".format(planned, args.repeats))
    print("  并发/QPS : {0} / {1}".format(args.concurrency or settings.concurrency, settings.qps))
    print("  成本闸   : {0} 次".format(args.max_calls or settings.max_calls))
    print("=" * 66)

    runner = ExperimentRunner(
        settings=settings,
        samples=samples,
        variants=variants,
        run_id=run_id,
        concurrency=args.concurrency or None,
        max_calls=args.max_calls or None,
        repeats=args.repeats,
        progress=_progress_printer(planned),
    )
    summary = runner.run()

    print("\n" + "-" * 66)
    print("运行结束：完成 {0} 条，跳过（已存在）{1} 条，失败 {2} 条，用时 {3:.1f}s".format(
        summary.n_done, summary.n_skipped, summary.n_failed, summary.elapsed_seconds))
    if summary.n_aborted:
        print("提前停止：{0} 条未执行（{1}）".format(summary.n_aborted, summary.stop_reason))
    print("原始记录：{0}".format(Path(summary.raw_path).relative_to(ROOT)))

    if args.no_report:
        return 0

    records = load_records(summary.raw_path)
    results = evaluate_all(records, variants=variants)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = write_summary_csv(results, config.METRICS_DIR / "{0}_metrics.csv".format(run_id))
    per_class_csv = write_per_class_csv(results, config.METRICS_DIR / "{0}_per_class.csv".format(run_id))

    print("\n" + format_table(results))

    figures = {}
    try:
        figures = generate_all(results, config.FIGURES_DIR, confusion_variant=variants[-1])
    except Exception as exc:
        print("[警告] 图表生成失败：{0}".format(exc), file=sys.stderr)

    meta = {
        "run_id": run_id,
        "数据划分": "{0}（样本 {1} 条）".format(args.split, len(samples)),
        "模型": "{0} / {1}".format(settings.provider, settings.model),
        "调用参数": "temperature={0}, max_tokens={1}, image_max_side={2}".format(
            settings.temperature, settings.max_tokens, settings.image_max_side),
        "提示词 SHA256": "；".join("{0}={1}".format(k, prompts[k].sha256) for k in variants),
        "重复次数": args.repeats,
        "计划调用": planned,
        "实际完成": summary.n_done,
        "调用失败": summary.n_failed,
        "总 Token": summary.total_tokens,
        "原始记录": str(Path(summary.raw_path).relative_to(ROOT)),
        "指标汇总": str(summary_csv.relative_to(ROOT)),
        "分类明细": str(per_class_csv.relative_to(ROOT)),
    }
    report_path = render_markdown(
        results, meta, config.RESULTS_DIR / "{0}_report.md".format(run_id), figures
    )

    print("\n产出文件：")
    print("  指标汇总: {0}".format(summary_csv.relative_to(ROOT)))
    print("  分类明细: {0}".format(per_class_csv.relative_to(ROOT)))
    for name, path in figures.items():
        print("  图表    : {0}".format(Path(path).relative_to(ROOT)))
    print("  报告草稿: {0}".format(report_path.relative_to(ROOT)))

    import csv as _csv
    with (config.RESULTS_DIR / "{0}_run_meta.json".format(run_id)).open("w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "summary": summary.__dict__}, handle, ensure_ascii=False,
                  indent=2, default=str)
    return 0


def _progress_printer(total: int):
    state = {"last": 0}

    def printer(info):
        done = info["done"]
        if done - state["last"] >= 10 or done == info["total"]:
            state["last"] = done
            print("  进度 {0}/{1}（失败 {2}）".format(done, info["total"], info["failed"]))

    return printer


if __name__ == "__main__":
    raise SystemExit(main())
