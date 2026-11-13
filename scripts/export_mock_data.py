# -*- coding: utf-8 -*-
"""把模拟养殖数据导出为 CSV/JSON，便于审阅与外部复核。

数据由 ``src/tools/mock_data.py`` 以固定种子（20261107）确定性生成：
4 个栏位 × 12 头猪 × 30 天 = 1440 条日记录。生成代码是唯一权威来源，
本脚本只做导出，不引入新的随机性；重复运行得到的文件逐字节一致。

用法::

    python scripts\\export_mock_data.py
    python scripts\\export_mock_data.py --out data/records --seed 20261107
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.mock_data import SEED, build_farm, summary  # noqa: E402

RECORD_FIELDS = ["date", "pig_id", "pen_id", "weight_kg", "feed_kg", "water_l",
                 "activity_index", "lying_ratio", "temp_c", "humidity_pct"]


def _write_csv(path: Path, fields, rows) -> int:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        n = 0
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
            n += 1
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="导出模拟养殖数据（CSV/JSON）")
    parser.add_argument("--out", default=str(ROOT / "data" / "records"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    farm = build_farm(seed=args.seed)
    pens = [pen.__dict__ for pen in farm.pens]
    pigs = [pig.__dict__ for pig in farm.pigs]
    records = [record.__dict__ for record in farm.records]

    n_pens = _write_csv(out / "pens.csv",
                        ["pen_id", "zone", "capacity", "camera_id"], pens)
    n_pigs = _write_csv(out / "pigs.csv",
                        ["pig_id", "pen_id", "sex", "entry_weight_kg"], pigs)
    n_records = _write_csv(out / "daily_records.csv", RECORD_FIELDS, records)

    meta = summary(farm)
    meta["files"] = {
        "pens.csv": n_pens,
        "pigs.csv": n_pigs,
        "daily_records.csv": n_records,
    }
    meta["note"] = ("模拟数据，全部由 src/tools/mock_data.py 以固定种子生成，"
                    "不代表任何真实养殖场；日记录中的重量、采食量与活动指数"
                    "均为按预设轨迹合成的数值。")
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("已导出到 %s" % out)
    print("  栏位 %d 个、猪 %d 头、日记录 %d 条（%s ~ %s）"
          % (n_pens, n_pigs, n_records, meta["start_date"], meta["end_date"]))
    for name in ("pens.csv", "pigs.csv", "daily_records.csv", "meta.json"):
        size = (out / name).stat().st_size
        print("  %-18s %6.1f KB" % (name, size / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())