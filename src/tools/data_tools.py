"""数据查询工具（T8）。

四个工具覆盖"看全貌 → 查明细 → 做聚合 → 找异常"的典型问法：

* ``list_pens``          —— 有哪些栏位、各多少头猪、最近均重
* ``query_pig_records``  —— 按栏位/猪只/日期/指标查日记录
* ``aggregate_metric``   —— 按栏位/日期/猪只分组求均值等统计量
* ``detect_anomalies``   —— 用 z 分数找离群个体或栏位

所有返回值都带 ``source="simulated"`` 与免责说明。参数错误一律抛
:class:`~src.tools.registry.ToolError`，由注册表转成 ``ok=False`` 的结果，
模型因此能读到"错在哪、可选值是什么"，而不是收到一句笼统的失败。
"""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from src.tools.mock_data import DISCLAIMER, FarmData, get_farm
from src.tools.registry import ToolRegistry, ToolError, ToolSpec

#: 可查询的数值指标
METRICS: List[str] = [
    "weight_kg", "feed_kg", "water_l",
    "activity_index", "lying_ratio", "temp_c", "humidity_pct",
]

#: 聚合统计量
STATS = ["mean", "min", "max", "sum", "count"]

_ISO_HINT = "日期格式应为 YYYY-MM-DD"


def _parse_date(value: Optional[str], field_name: str) -> Optional[date]:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ToolError("参数 {0} 不合法：{1!r}（{2}）".format(
            field_name, value, _ISO_HINT)) from exc


def _check_pen(farm: FarmData, pen_id: Optional[str]) -> Optional[str]:
    if pen_id in (None, ""):
        return None
    valid = [pen.pen_id for pen in farm.pens]
    if pen_id not in valid:
        raise ToolError("未知栏位 {0!r}，可选：{1}".format(pen_id, "、".join(valid)))
    return pen_id


def _check_pig(farm: FarmData, pig_id: Optional[str]) -> Optional[str]:
    if pig_id in (None, ""):
        return None
    valid = {pig.pig_id for pig in farm.pigs}
    if pig_id not in valid:
        sample = "、".join(sorted(valid)[:5])
        raise ToolError("未知猪只 {0!r}，例如：{1}".format(pig_id, sample))
    return pig_id


def _filter_records(farm: FarmData, pen_id: Optional[str] = None,
                    pig_id: Optional[str] = None,
                    start: Optional[date] = None,
                    end: Optional[date] = None) -> List[Any]:
    rows = farm.records
    if pen_id:
        rows = [r for r in rows if r.pen_id == pen_id]
    if pig_id:
        rows = [r for r in rows if r.pig_id == pig_id]
    if start:
        rows = [r for r in rows if date.fromisoformat(r.date) >= start]
    if end:
        rows = [r for r in rows if date.fromisoformat(r.date) <= end]
    return rows


def _envelope(payload: Dict[str, Any]) -> Dict[str, Any]:
    """统一加上来源标注，避免模拟数据被误当真实数据。"""
    payload.setdefault("source", "simulated")
    payload.setdefault("disclaimer", DISCLAIMER)
    return payload


# ------------------------------------------------------------------ 工具实现
def list_pens(farm: Optional[FarmData] = None) -> Dict[str, Any]:
    data = farm or get_farm()
    latest = data.end_date.isoformat()
    rows = [r for r in data.records if r.date == latest]
    pens = []
    for pen in data.pens:
        own = [r for r in rows if r.pen_id == pen.pen_id]
        pigs = [p for p in data.pigs if p.pen_id == pen.pen_id]
        pens.append({
            "pen_id": pen.pen_id,
            "zone": pen.zone,
            "capacity": pen.capacity,
            "pigs": len(pigs),
            "camera_id": pen.camera_id,
            "area_m2": pen.area_m2,
            "avg_weight_kg": round(mean(r.weight_kg for r in own), 2) if own else None,
            "latest_date": latest,
        })
    return _envelope({"pens": pens, "date": latest})


def query_pig_records(pen_id: Optional[str] = None, pig_id: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      metrics: Optional[List[str]] = None,
                      limit: int = 50,
                      farm: Optional[FarmData] = None) -> Dict[str, Any]:
    data = farm or get_farm()
    pen_id = _check_pen(data, pen_id)
    pig_id = _check_pig(data, pig_id)
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start and end and start > end:
        raise ToolError("start_date 不能晚于 end_date")

    selected = [m for m in (metrics or ["weight_kg", "feed_kg", "water_l",
                                       "activity_index", "lying_ratio"]) if m in METRICS]
    if metrics:
        unknown = [m for m in metrics if m not in METRICS]
        if unknown:
            raise ToolError("未知指标 {0}，可选：{1}".format(
                "、".join(unknown), "、".join(METRICS)))

    rows = _filter_records(data, pen_id, pig_id, start, end)
    rows = sorted(rows, key=lambda r: (r.date, r.pig_id))
    truncated = len(rows) > limit
    rows = rows[:limit]

    records = []
    for row in rows:
        item = {"date": row.date, "pig_id": row.pig_id, "pen_id": row.pen_id}
        for metric in selected:
            item[metric] = getattr(row, metric)
        records.append(item)

    return _envelope({
        "records": records,
        "returned": len(records),
        "truncated": truncated,
        "filters": {
            "pen_id": pen_id, "pig_id": pig_id,
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
        },
        "metrics": selected,
    })


def aggregate_metric(metric: str, group_by: str = "pen",
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     stat: str = "mean",
                     farm: Optional[FarmData] = None) -> Dict[str, Any]:
    data = farm or get_farm()
    if metric not in METRICS:
        raise ToolError("未知指标 {0!r}，可选：{1}".format(metric, "、".join(METRICS)))
    if group_by not in ("pen", "day", "pig"):
        raise ToolError("group_by 只能是 pen / day / pig，实际 {0!r}".format(group_by))
    if stat not in STATS:
        raise ToolError("stat 只能是 {0}，实际 {1!r}".format("、".join(STATS), stat))

    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start and end and start > end:
        raise ToolError("start_date 不能晚于 end_date")

    rows = _filter_records(data, start=start, end=end)
    if not rows:
        return _envelope({"metric": metric, "group_by": group_by,
                          "stat": stat, "groups": [],
                          "note": "所选条件下没有记录"})

    buckets: Dict[str, List[float]] = {}
    for row in rows:
        key = {"pen": row.pen_id, "day": row.date, "pig": row.pig_id}[group_by]
        buckets.setdefault(key, []).append(float(getattr(row, metric)))

    def _apply(values: List[float]) -> float:
        if stat == "mean":
            return round(mean(values), 3)
        if stat == "min":
            return round(min(values), 3)
        if stat == "max":
            return round(max(values), 3)
        if stat == "sum":
            return round(sum(values), 3)
        return float(len(values))

    groups = [{"group": key, "value": _apply(values), "n": len(values)}
              for key, values in buckets.items()]
    groups.sort(key=lambda item: item["group"])

    return _envelope({
        "metric": metric, "stat": stat, "group_by": group_by,
        "range": {"start_date": start.isoformat() if start else data.start_date.isoformat(),
                  "end_date": end.isoformat() if end else data.end_date.isoformat()},
        "groups": groups,
    })


def detect_anomalies(metric: str = "feed_kg", z_threshold: float = 2.0,
                     days: int = 7, scope: str = "pig",
                     farm: Optional[FarmData] = None) -> Dict[str, Any]:
    """用 z 分数找离群对象。

    ``scope="pig"``：比较每头猪在窗口内的均值与全场均值；
    ``scope="pen"``：先按栏位聚合再比较。
    """
    data = farm or get_farm()
    if metric not in METRICS:
        raise ToolError("未知指标 {0!r}，可选：{1}".format(metric, "、".join(METRICS)))
    if scope not in ("pig", "pen"):
        raise ToolError("scope 只能是 pig / pen，实际 {0!r}".format(scope))
    if days < 1:
        raise ToolError("days 必须 >= 1")

    window_start = data.end_date - timedelta(days=days - 1)
    rows = [r for r in data.records if date.fromisoformat(r.date) >= window_start]
    if not rows:
        return _envelope({"metric": metric, "flags": [],
                          "note": "窗口内没有记录"})

    buckets: Dict[str, List[float]] = {}
    for row in rows:
        key = row.pig_id if scope == "pig" else row.pen_id
        buckets.setdefault(key, []).append(float(getattr(row, metric)))

    per_unit = {key: mean(values) for key, values in buckets.items()}
    values = list(per_unit.values())
    herd_mean = mean(values)
    herd_std = pstdev(values) if len(values) > 1 else 0.0

    flags: List[Dict[str, Any]] = []
    if herd_std > 0:
        for key, value in sorted(per_unit.items()):
            z = (value - herd_mean) / herd_std
            if abs(z) >= z_threshold:
                flags.append({
                    "target": key,
                    "mean": round(value, 3),
                    "z_score": round(z, 3),
                    "direction": "偏低" if z < 0 else "偏高",
                })
    flags.sort(key=lambda item: -abs(item["z_score"]))

    return _envelope({
        "metric": metric,
        "scope": scope,
        "days": days,
        "z_threshold": z_threshold,
        "window": {"start_date": window_start.isoformat(),
                   "end_date": data.end_date.isoformat()},
        "herd_mean": round(herd_mean, 3),
        "herd_std": round(herd_std, 3),
        "units_compared": len(per_unit),
        "flags": flags,
        "note": ("未发现超过阈值的离群对象" if not flags
                 else "共 {0} 个对象超过 |z| >= {1}".format(len(flags), z_threshold)),
    })


# ------------------------------------------------------------------ 注册
def build_data_tools(registry: ToolRegistry) -> ToolRegistry:
    """把四个数据工具注册进给定注册表。"""
    registry.register_all([
        ToolSpec(
            name="list_pens",
            description="列出所有栏位及其猪只数量、最近均重、摄像头编号。",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda args: list_pens(),
            returns="pens[]: pen_id, zone, pigs, avg_weight_kg, camera_id",
        ),
        ToolSpec(
            name="query_pig_records",
            description="按栏位、猪只、日期范围查询日记录明细（可指定指标）。",
            parameters={
                "type": "object",
                "properties": {
                    "pen_id": {"type": "string", "description": "栏位编号，如 P1"},
                    "pig_id": {"type": "string", "description": "猪只编号，如 P2-007"},
                    "start_date": {"type": "string", "description": "起始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    "metrics": {"type": "array", "items": {"type": "string", "enum": METRICS},
                                "description": "要返回的指标，缺省返回常用五项"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500,
                              "default": 50, "description": "最多返回多少条"},
                },
                "required": [],
            },
            handler=lambda args: query_pig_records(**args),
            returns="records[] + returned/truncated/filters",
        ),
        ToolSpec(
            name="aggregate_metric",
            description="按栏位/日期/猪只分组，对某个指标求均值、极值或合计。",
            parameters={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": METRICS},
                    "group_by": {"type": "string", "enum": ["pen", "day", "pig"],
                                 "default": "pen"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "stat": {"type": "string", "enum": STATS, "default": "mean"},
                },
                "required": ["metric"],
            },
            handler=lambda args: aggregate_metric(**args),
            returns="groups[]: group, value, n",
        ),
        ToolSpec(
            name="detect_anomalies",
            description="用 z 分数在最近若干天内找出指标异常的猪只或栏位。",
            parameters={
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": METRICS, "default": "feed_kg"},
                    "z_threshold": {"type": "number", "minimum": 0.5, "maximum": 5.0,
                                    "default": 2.0},
                    "days": {"type": "integer", "minimum": 1, "maximum": 30, "default": 7},
                    "scope": {"type": "string", "enum": ["pig", "pen"], "default": "pig"},
                },
                "required": [],
            },
            handler=lambda args: detect_anomalies(**args),
            returns="flags[]: target, mean, z_score, direction",
        ),
    ])
    return registry