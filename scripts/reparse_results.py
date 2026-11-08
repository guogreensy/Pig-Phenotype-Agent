"""用当前解析器重新解析已有原始记录（不重新调用 API）。

用途：解析策略改进后（例如模型把 ``lying_lateral`` 写成 ``lying lateral``、
或推理过程混入答案），无需重新付费调用，即可在**同一批原始返回**上
重新抽取字段并重算指标。

约束与诚信说明
--------------
* **不修改** ``raw_text``、``request_success``、``http_status``、``latency_seconds``
  等调用类字段，只重算由文本解析得到的字段；
* 输入输出分开存放（``--out``），原始文件保留以便审计；
* 脚本会把重解析前后的"可计分样本数"打印出来，便于核对改进幅度。

用法::

    python scripts/reparse_results.py --raw results/raw/test_main_raw.csv \
        --out results/raw/test_main_v2_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.experiment_runner import RAW_FIELDS  # noqa: E402
from src.output_parser import parse_model_output  # noqa: E402
from src.schema import validate_result  # noqa: E402

#: 由文本决定的字段，重解析时会被覆盖
DERIVED_FIELDS = [
    "predicted_label", "description", "uncertain", "uncertainty_reason",
    "json_valid", "schema_valid", "parse_method", "error_message",
]


def reparse_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """就地重算派生字段，返回新行列表。"""
    output: List[Dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        success = str(row.get("request_success", "")).strip().lower() == "true"
        raw_text = row.get("raw_text") or ""
        if not success or not raw_text.strip():
            updated.update({k: "" for k in DERIVED_FIELDS})
            updated["json_valid"] = "False"
            updated["schema_valid"] = "False"
            if not updated.get("error_message"):
                updated["error_message"] = "调用失败或返回为空"
            output.append(updated)
            continue

        parsed = parse_model_output(raw_text)
        updated["json_valid"] = str(parsed.json_valid)
        updated["parse_method"] = parsed.method
        updated["predicted_label"] = ""
        updated["description"] = ""
        updated["uncertain"] = ""
        updated["uncertainty_reason"] = ""
        updated["schema_valid"] = "False"
        updated["error_message"] = ""

        if parsed.data is None:
            updated["error_message"] = parsed.error or "无法解析模型返回"
            output.append(updated)
            continue

        validation = validate_result(parsed.data)
        updated["schema_valid"] = str(validation.valid)
        updated["predicted_label"] = validation.posture or ""
        updated["description"] = (validation.description or "").replace("\n", " ")
        updated["uncertain"] = "" if validation.uncertain is None else str(validation.uncertain)
        updated["uncertainty_reason"] = validation.uncertainty_reason or ""
        if not validation.valid:
            updated["error_message"] = "Schema 校验失败：" + "；".join(validation.errors)
        elif parsed.error:
            updated["error_message"] = parsed.error
        output.append(updated)
    return output


def scoreable(rows: Sequence[Dict[str, str]]) -> int:
    return sum(
        1 for r in rows
        if r.get("predicted_label") in ("standing", "lying_lateral", "lying_other", "sitting", "unknown")
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="重新解析已有原始记录")
    parser.add_argument("--raw", required=True, help="输入原始 CSV")
    parser.add_argument("--out", required=True, help="输出 CSV（不要覆盖原文件）")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw).resolve()
    out_path = Path(args.out).resolve()
    if raw_path == out_path:
        print("[错误] --out 不能与 --raw 相同，请保留原始文件用于审计。", file=sys.stderr)
        return 2
    if not raw_path.is_file():
        print("[错误] 找不到 {0}".format(raw_path), file=sys.stderr)
        return 2

    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("[错误] 输入文件没有数据行", file=sys.stderr)
        return 2

    before_scoreable = scoreable(rows)
    before_methods = {}
    for row in rows:
        key = row.get("parse_method") or "(空)"
        before_methods[key] = before_methods.get(key, 0) + 1

    updated_rows = reparse_rows(rows)
    after_scoreable = scoreable(updated_rows)
    after_methods = {}
    for row in updated_rows:
        key = row.get("parse_method") or "(空)"
        after_methods[key] = after_methods.get(key, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updated_rows)

    print("=" * 62)
    print("重解析完成（未调用任何 API）")
    print("-" * 62)
    print("  记录数        : {0}".format(len(rows)))
    print("  可计分样本    : {0} -> {1}（+{2}）".format(
        before_scoreable, after_scoreable, after_scoreable - before_scoreable))
    print("  解析方式(前)  : {0}".format(before_methods))
    print("  解析方式(后)  : {0}".format(after_methods))
    print("  原始文件      : {0}".format(raw_path))
    print("  重解析文件    : {0}".format(out_path))
    print("=" * 62)
    print("下一步：python scripts/plot_results.py --raw {0} --run-id <新 run_id>".format(
        out_path.relative_to(ROOT) if str(out_path).startswith(str(ROOT)) else out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
