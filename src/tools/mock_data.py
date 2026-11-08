"""模拟养殖数据（T7）。

用途
----
工具层需要一个"可以真查"的数据源，才能验证 ``query_pig_records`` /
``aggregate_metric`` / ``detect_anomalies`` 这条链路。真实养殖数据涉及隐私与
授权，本项目**不**使用任何真实场户数据，因此这里构造一份**确定性模拟数据**。

三条硬约束
----------
1. **确定性**：固定随机种子，任何机器任何时间生成的结果完全一致，
   报告里的数字因此可复现；
2. **如实标注**：所有返回值都带 ``source="simulated"`` 与免责说明，
   不允许被误当成真实数据引用；
3. **植入已知异常**：模拟数据里预先放了三处异常，用来检验异常检测工具
   是否真能发现它们——而不是让工具"碰巧有输出"。

数据规模
--------
4 个栏位 × 12 头猪 × 30 天 = 1440 条日记录，日期区间 2026-10-08 ~ 2026-11-06。

预设场景
--------
* **P3 热应激**（第 18–24 天）：温度 +4.5 ℃、饮水量 +25 %、采食 −12 %、
  躺卧比例上升、活动量下降；
* **P2-007 采食下降**（第 20 天起）：采食 −35 %，增重明显放缓；
* **P4-003 活动量下降**（第 12 天起）：活动指数 −0.22，用于模拟跛行。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: 固定种子：改动它等于换一份数据，报告中的数字必须同步更新
SEED = 20261107
DAYS = 30
START_DATE = date(2026, 10, 8)
PEN_COUNT = 4
PIGS_PER_PEN = 12

PEN_IDS = ["P1", "P2", "P3", "P4"]
ZONES = ["东区", "东区", "西区", "西区"]

#: 预设异常（供文档与自检引用）
SCENARIOS: List[Dict[str, Any]] = [
    {"id": "heat_stress_P3", "target": "P3", "kind": "environment",
     "days": [18, 24], "desc": "P3 栏第 18–24 天热应激：温度 +4.5℃、饮水 +25%、采食 −12%"},
    {"id": "low_feed_P2-007", "target": "P2-007", "kind": "individual",
     "days": [20, 30], "desc": "P2-007 自第 20 天起采食 −35%、增重放缓"},
    {"id": "low_activity_P4-003", "target": "P4-003", "kind": "individual",
     "days": [12, 30], "desc": "P4-003 自第 12 天起活动指数 −0.22（疑似跛行）"},
]

DISCLAIMER = ("模拟数据，仅用于验证工具调用链路与报表口径，"
              "不代表任何真实养殖场，不可用于生产决策。")


@dataclass(frozen=True)
class Pen:
    pen_id: str
    zone: str
    capacity: int
    camera_id: str
    area_m2: float


@dataclass(frozen=True)
class Pig:
    pig_id: str
    pen_id: str
    sex: str
    entry_weight_kg: float


@dataclass(frozen=True)
class DailyRecord:
    date: str            # ISO 日期，便于直接 JSON 化
    pig_id: str
    pen_id: str
    weight_kg: float
    feed_kg: float
    water_l: float
    activity_index: float
    lying_ratio: float
    temp_c: float
    humidity_pct: float


@dataclass
class FarmData:
    pens: List[Pen]
    pigs: List[Pig]
    records: List[DailyRecord]
    seed: int = SEED
    start_date: date = START_DATE
    days: int = DAYS
    scenarios: List[Dict[str, Any]] = field(default_factory=lambda: list(SCENARIOS))

    # ---------------------------------------------------------------- 元信息
    @property
    def end_date(self) -> date:
        return self.start_date + timedelta(days=self.days - 1)

    def meta(self) -> Dict[str, Any]:
        return {
            "source": "simulated",
            "seed": self.seed,
            "pens": len(self.pens),
            "pigs": len(self.pigs),
            "records": len(self.records),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "days": self.days,
            "scenarios": [dict(item) for item in self.scenarios],
            "disclaimer": DISCLAIMER,
        }


# ------------------------------------------------------------------ 生成
def _in_window(day_index: int, window: Tuple[int, int]) -> bool:
    """day_index 从 1 开始计数。"""
    return window[0] <= day_index <= window[1]


def build_farm(seed: int = SEED, days: int = DAYS,
               start: date = START_DATE) -> FarmData:
    """确定性生成整个养殖场数据。"""
    rng = random.Random(seed)

    pens = [
        Pen(pen_id=pen_id, zone=ZONES[index], capacity=PIGS_PER_PEN,
            camera_id="cam{0}".format(index + 1),
            area_m2=round(28.0 + rng.uniform(-2.0, 2.0), 1))
        for index, pen_id in enumerate(PEN_IDS)
    ]

    pigs: List[Pig] = []
    for pen in pens:
        for index in range(1, PIGS_PER_PEN + 1):
            pigs.append(Pig(
                pig_id="{0}-{1:03d}".format(pen.pen_id, index),
                pen_id=pen.pen_id,
                sex=rng.choice(["M", "F"]),
                entry_weight_kg=round(rng.uniform(78.0, 96.0), 1),
            ))

    # 栏位级环境（温度/湿度按天变化，P3 有热应激窗口）
    pen_env: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for day_index in range(1, days + 1):
        for pen in pens:
            temp = 22.0 + rng.uniform(-0.8, 0.8)
            humidity = 62.0 + rng.uniform(-4.0, 4.0)
            if pen.pen_id == "P3" and _in_window(day_index, (18, 24)):
                temp += 4.5
                humidity -= 3.0
            pen_env[(pen.pen_id, day_index)] = (round(temp, 1), round(humidity, 1))

    records: List[DailyRecord] = []
    for pig in pigs:
        weight = pig.entry_weight_kg
        for day_index in range(1, days + 1):
            current_date = start + timedelta(days=day_index - 1)

            growth = max(-0.2, rng.gauss(0.72, 0.12))
            feed = max(0.8, rng.gauss(2.45, 0.18))
            water = feed * rng.uniform(2.3, 2.9)
            activity = min(1.0, max(0.0, rng.gauss(0.46, 0.07)))
            lying = min(0.95, max(0.05, rng.gauss(0.58, 0.06)))

            # ---------------- 预设场景：P2-007 采食下降 ----------------
            if pig.pig_id == "P2-007" and day_index >= 20:
                feed *= 0.65
                water *= 0.72
                growth *= 0.35
                activity *= 0.9

            # ---------------- 预设场景：P4-003 活动量下降 ----------------
            if pig.pig_id == "P4-003" and day_index >= 12:
                activity = max(0.0, activity - 0.22)
                lying = min(0.95, lying + 0.10)

            # ---------------- 预设场景：P3 热应激 ----------------
            if pig.pen_id == "P3" and _in_window(day_index, (18, 24)):
                feed *= 0.88
                water *= 1.25
                activity = min(1.0, max(0.0, activity - 0.15))
                lying = min(0.95, lying + 0.12)
                growth *= 0.55

            weight = round(max(40.0, weight + growth), 2)
            temp_c, humidity = pen_env[(pig.pen_id, day_index)]

            records.append(DailyRecord(
                date=current_date.isoformat(),
                pig_id=pig.pig_id,
                pen_id=pig.pen_id,
                weight_kg=weight,
                feed_kg=round(feed, 2),
                water_l=round(water, 2),
                activity_index=round(activity, 3),
                lying_ratio=round(lying, 3),
                temp_c=temp_c,
                humidity_pct=humidity,
            ))

    return FarmData(pens=pens, pigs=pigs, records=records, seed=seed,
                    start_date=start, days=days)


@lru_cache(maxsize=1)
def get_farm() -> FarmData:
    """进程内缓存的默认数据集。"""
    return build_farm()


# ------------------------------------------------------------------ 自检/CLI
def summary(farm: Optional[FarmData] = None) -> Dict[str, Any]:
    """汇总统计，同时用于冒烟检查。"""
    data = farm or get_farm()
    latest_date = data.end_date.isoformat()
    latest = [r for r in data.records if r.date == latest_date]
    per_pen: Dict[str, Dict[str, Any]] = {}
    for pen in data.pens:
        rows = [r for r in latest if r.pen_id == pen.pen_id]
        per_pen[pen.pen_id] = {
            "pigs": len(rows),
            "avg_weight_kg": round(sum(r.weight_kg for r in rows) / len(rows), 2) if rows else None,
            "avg_feed_kg": round(sum(r.feed_kg for r in rows) / len(rows), 2) if rows else None,
            "camera": pen.camera_id,
        }
    meta = data.meta()
    meta["per_pen_latest"] = per_pen
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="生成/检查模拟养殖数据")
    parser.add_argument("--dump", default="", help="把完整数据写成 JSON 文件")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    farm = build_farm(seed=args.seed)
    payload = summary(farm)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("-" * 60)
    print("记录数：{0}（{1} ~ {2}）".format(
        payload["records"], payload["start_date"], payload["end_date"]))
    print("预设场景：")
    for item in payload["scenarios"]:
        print("  - [{0}] {1}".format(item["id"], item["desc"]))

    if args.dump:
        target = Path(args.dump)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "meta": payload,
            "pens": [pen.__dict__ for pen in farm.pens],
            "pigs": [pig.__dict__ for pig in farm.pigs],
            "records": [record.__dict__ for record in farm.records],
        }
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print("已写出：{0}（{1:.1f} KB）".format(
            target, target.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())