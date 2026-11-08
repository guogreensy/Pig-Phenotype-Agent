"""从已有的原始结果 CSV 重新计算指标与图表（不调用任何 API）。

用途：验证"结果可复现"——把 ``results/raw/*.csv`` 重新算一遍，
指标与图表应与正式实验完全一致（需求 阶段四验收）。

用法::

    python scripts/plot_results.py --raw results/raw/test_20261019_120000_raw.csv
    python scripts/plot_results.py            # 自动选最新的原始记录
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_all,
    format_table,
    load_records,
    write_per_class_csv,
    write_summary_csv,
)
from src.reporting import generate_all, render_markdown  # noqa: E402


def latest_raw() -> Optional[Path]:
    if not config.RAW_DIR.is_dir():
        return None
    candidates = sorted(config.RAW_DIR.glob("*_raw.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="重算指标与图表")
    parser.add_argument("--raw", default="", help="原始 CSV 路径；缺省取最新")
    parser.add_argument("--run-id", default="", help="输出文件名前缀；缺省从 CSV 名推断")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw).resolve() if args.raw else latest_raw()
    if raw_path is None or not raw_path.is_file():
        print("[错误] 找不到原始结果 CSV，请先运行 scripts/run_experiment.py", file=sys.stderr)
        return 2

    run_id = args.run_id or raw_path.stem.replace("_raw", "")
    records = load_records(raw_path)
    if not records:
        print("[错误] {0} 中没有数据行".format(raw_path), file=sys.stderr)
        return 2

    variants = sorted({(row.get("prompt_variant") or "").upper() for row in records if row.get("prompt_variant")})
    results = evaluate_all(records, variants=variants)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = write_summary_csv(results, config.METRICS_DIR / "{0}_metrics.csv".format(run_id))
    per_class_csv = write_per_class_csv(results, config.METRICS_DIR / "{0}_per_class.csv".format(run_id))

    print(format_table(results))

    figures = generate_all(results, config.FIGURES_DIR, confusion_variant=variants[-1] if variants else None)

    model = records[0].get("model_name", "")
    timestamps = sorted({row.get("experiment_timestamp", "") for row in records if row.get("experiment_timestamp")})
    meta = {
        "run_id": run_id,
        "来源文件": str(raw_path.relative_to(ROOT)),
        "模型": model,
        "记录数": len(records),
        "实验时间范围": "{0} ~ {1}".format(timestamps[0], timestamps[-1]) if timestamps else "未记录",
        "提示词 SHA256": "；".join(
            "{0}={1}".format(row.get("prompt_variant"), row.get("prompt_sha256"))
            for row in records[:len(variants)]
        ),
        "指标汇总": str(summary_csv.relative_to(ROOT)),
        "分类明细": str(per_class_csv.relative_to(ROOT)),
    }
    report = render_markdown(results, meta, config.RESULTS_DIR / "{0}_report.md".format(run_id), figures)

    print("\n重算完成：")
    print("  指标: {0}".format(summary_csv.relative_to(ROOT)))
    print("  明细: {0}".format(per_class_csv.relative_to(ROOT)))
    for name, path in figures.items():
        print("  图  : {0}".format(Path(path).relative_to(ROOT)))
    print("  报告: {0}".format(report.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
