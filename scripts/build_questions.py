"""构建阶段 E 的评测题集（30 题）。

设计原则：**每道题的标准答案都可独立验证**，不靠主观判断。

- 数据题：答案直接从模拟数据集的原始记录算出（按日期、猪号、栏位过滤后统计），
  与工具实现无关——工具若算错，指标会暴露出来。
- 知识题：期望文献按关键词在正文片段中逐篇核实；关键词对不上就构建失败。
- 视觉题：期望类别来自图片所在的目录名（人工标注的类别）。
- 不可作答题：构建时反向验证"确实没有这个信息"（字段不存在、日期越界、
  栏位不存在、关键词在知识库中零命中），避免把"其实能答"的题当成拒答题。

产物：``data/experiment/agent_questions.jsonl`` + 一份统计摘要。

    python scripts/build_questions.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.mock_data import DISCLAIMER, DAYS, SEED, START_DATE, get_farm  # noqa: E402

QUESTION_DIR = PROJECT_ROOT / "data" / "experiment"
QUESTION_FILE = QUESTION_DIR / "agent_questions.jsonl"
CHUNKS_FILE = PROJECT_ROOT / "data" / "kb" / "chunks.jsonl"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples" / "dev"

#: 数值容差：相对 1.5%，绝对不小于 0.02（容忍模型合理的四舍五入）
def _tolerance(value: float) -> float:
    return round(max(0.02, abs(value) * 0.015), 4)


# --------------------------------------------------------------- 数据题定义
# 每条：(id, 问题, 过滤条件, 要报告的数值, 必须出现的实体词)
DATA_SPECS: List[Dict[str, Any]] = [
    {
        "id": "D1",
        "question": "P2-007 这头猪最近 7 天的平均采食量是多少？",
        "select": {"pig_id": "P2-007", "start": "2026-10-31", "end": "2026-11-06"},
        "numbers": [("feed_kg", "mean", "平均采食量")],
        "tokens": ["P2-007"],
        "note": "query_pig_records 或 detect_anomalies 的均值",
    },
    {
        "id": "D2",
        "question": "最近 7 天各栏位的平均采食量分别是多少？",
        "select": {"start": "2026-10-31", "end": "2026-11-06"},
        "group_by": "pen",
        "numbers": [("feed_kg", "mean", "P{0} 平均采食量")],
        "tokens": ["P1", "P2", "P3", "P4"],
        "note": "aggregate_metric(group_by=pen) 的四个分组均值",
    },
    {
        "id": "D3",
        "question": "最近 7 天有没有采食量异常的猪？如果有，是哪一头？",
        "select": {"start": "2026-10-31", "end": "2026-11-06"},
        "anomaly": ("feed_kg", "pig", "P2-007"),
        "tokens": ["P2-007"],
        "note": "注入场景 low_feed_P2-007；构建时验证其 z 分数确实低于 -3",
    },
    {
        "id": "D4",
        "question": "P3 栏在 2026-10-25 到 2026-10-31 期间的平均温度是多少？",
        "select": {"pen_id": "P3", "start": "2026-10-25", "end": "2026-10-31"},
        "numbers": [("temp_c", "mean", "P3 平均温度")],
        "tokens": ["P3"],
        "note": "注入场景 heat_stress_P3 的时间窗",
    },
    {
        "id": "D5",
        "question": "P2-007 在这 30 天里体重从多少长到了多少？",
        "select": {"pig_id": "P2-007", "start": "2026-10-08", "end": "2026-11-06"},
        "numbers": [("weight_kg", "first", "起始体重"), ("weight_kg", "last", "结束体重")],
        "tokens": ["P2-007"],
        "note": "query_pig_records 返回的首末体重",
    },
    {
        "id": "D6",
        "question": "这个猪场一共有几个栏位、多少头猪？",
        "counts": True,
        "numbers": [("pen", "count", "栏位数"), ("pig", "count", "猪只数")],
        "tokens": [],
        "note": "list_pens 返回的栏位与猪只数量",
        # 系统提示词的"数据范围"一节已写明栏位数与猪只数，
        # 因此三个条件都能直接答出，测不到工具使用——单独报告，不计入汇总。
        "exclude_from_aggregate": True,
        "exclude_reason": "答案在系统提示词的数据范围说明中，三个条件均可直接回答",
    },
    {
        "id": "D7",
        "question": "最近 7 天各栏位的平均体重是多少？",
        "select": {"start": "2026-10-31", "end": "2026-11-06"},
        "group_by": "pen",
        "numbers": [("weight_kg", "mean", "P{0} 平均体重")],
        "tokens": ["P1", "P2", "P3", "P4"],
        "note": "aggregate_metric(group_by=pen, metric=weight_kg)",
    },
    {
        "id": "D8",
        "question": "P4-003 这头猪最近 7 天的活动指数是否偏低？",
        "select": {"pig_id": "P4-003", "start": "2026-10-31", "end": "2026-11-06"},
        "anomaly": ("activity_index", "pig", "P4-003"),
        "tokens": ["P4-003"],
        "note": "注入场景 low_activity_P4-003；构建时验证 z < -3",
    },
    {
        "id": "D9",
        "question": "2026-11-06 这一天各栏位的平均饮水量是多少？",
        "select": {"start": "2026-11-06", "end": "2026-11-06"},
        "group_by": "pen",
        "numbers": [("water_l", "mean", "P{0} 平均饮水量")],
        "tokens": ["P1", "P2", "P3", "P4"],
        "note": "单日聚合，检验模型是否守住时间范围",
    },
    {
        "id": "D10",
        "question": "P1 栏最近 7 天的平均躺卧比例是多少？",
        "select": {"pen_id": "P1", "start": "2026-10-31", "end": "2026-11-06"},
        "numbers": [("lying_ratio", "mean", "P1 平均躺卧比例")],
        "tokens": ["P1"],
        "note": "比例型指标，检验小数精度",
    },
]


# --------------------------------------------------------------- 知识题定义
# keywords 用于构建时核实"这篇文献确实讲这个"；doc_ids 是可接受的正确来源
KNOWLEDGE_SPECS: List[Dict[str, Any]] = [
    {
        "id": "K1",
        "question": "猪只姿态识别常用的深度学习方法有哪些？请给出出处。",
        "doc_ids": ["pmc_8147168", "pmc_7070994", "pmc_6749226"],
        "keywords": ["YOLOv5", "two-stream", "Faster R-CNN"],
    },
    {
        "id": "K2",
        "question": "有哪些公开可用的猪只姿态识别数据集？",
        "doc_ids": ["hf_dataset_readme", "pmc_8147168"],
        "keywords": ["dataset", "Dataset"],
    },
    {
        "id": "K3",
        "question": "YOLOv5 是怎么用于单头猪的姿态检测的？",
        "doc_ids": ["pmc_8147168"],
        "keywords": ["YOLOv5"],
    },
    {
        "id": "K4",
        "question": "如何用目标检测模型确定猪只的姿态和位置？",
        "doc_ids": ["pmc_11635830"],
        "keywords": ["object detection", "posture"],
    },
    {
        "id": "K5",
        "question": "母猪行走姿态（locomotion）的深度学习姿态检测模型是怎么做的？",
        "doc_ids": ["pmc_11252330"],
        "keywords": ["locomotion", "pose"],
    },
    {
        "id": "K6",
        "question": "双流卷积网络如何识别猪只的视频行为？",
        "doc_ids": ["pmc_7070994"],
        "keywords": ["two-stream", "Two-Stream"],
    },
    {
        "id": "K7",
        "question": "基于 RGB 图像的机器视觉如何用于猪只疾病症状监测？",
        "doc_ids": ["pmc_11833201"],
        "keywords": ["RGB"],
    },
    {
        "id": "K8",
        "question": "如何通过行为变化早期发现猪只的健康与福利问题？",
        "doc_ids": ["pmc_5110645", "pmc_5730557"],
        "keywords": ["health", "welfare"],
    },
    {
        "id": "K9",
        "question": "妊娠期限位栏母猪的行为识别用了什么计算机视觉方法？",
        "doc_ids": ["pmc_8071751"],
        "keywords": ["gestating sows", "sows"],
    },
    {
        "id": "K10",
        "question": "本项目的姿态识别实验用了哪几组提示词、多少张样本？",
        "doc_ids": ["local_experiment_report"],
        "keywords": ["提示词", "样本"],
    },
]


# --------------------------------------------------------------- 不可作答题
# verify 字段说明"为什么答不出来"，构建时逐条验证
UNANSWERABLE_SPECS: List[Dict[str, Any]] = [
    {
        "id": "U1",
        "question": "这批猪是什么品种？是大白、长白还是杜洛克？",
        "verify": "no_field:breed",
        "note": "模拟数据只有性别和体重，没有品种字段",
    },
    {
        "id": "U2",
        "question": "2025 年 3 月这批猪的平均采食量是多少？",
        "verify": "out_of_range",
        "note": "数据从 2026-10-08 开始，2025 年 3 月没有记录",
    },
    {
        "id": "U3",
        "question": "这个猪场的疫苗免疫程序是怎样的？",
        "verify": "absent_keywords:免疫程序,vaccination program",
        "note": "知识库只泛泛提到疫苗研究，没有任何免疫程序内容",
    },
    {
        "id": "U4",
        "question": "P5 栏最近一周的采食量怎么样？",
        "verify": "missing_pen:P5",
        "note": "只有 P1–P4 四个栏位，P5 不存在",
    },
    {
        "id": "U5",
        "question": "这批猪的屠宰胴体品质如何？",
        "verify": "absent_keywords:胴体,carcass",
        "note": "只有在栏体重，没有屠宰或胴体测定数据",
    },
]


# --------------------------------------------------------------- 视觉题定义
# 原图按类别存放在 data/samples/dev/<类别>/ 下；直接引用会把类别写进题目文本，
# 模型能从路径读出答案。因此复制到 data/experiment/images/ 下的中性文件名，
# 题目里只出现"图片 1/2/3"这样的编号。
VISION_SPECS: List[Dict[str, Any]] = [
    {"id": "V1", "label": "lying_lateral", "index": 0},
    {"id": "V2", "label": "standing", "index": 0},
    {"id": "V3", "label": "sitting", "index": 0},
    {"id": "V4", "label": "lying_other", "index": 0},
    {"id": "V5", "label": "standing", "index": 10},
]

IMAGES_DIR = PROJECT_ROOT / "data" / "experiment" / "images"

LABEL_CN = {
    "lying_lateral": "侧卧",
    "lying_other": "其他卧姿",
    "sitting": "坐姿",
    "standing": "站立",
}


# --------------------------------------------------------------- 计算工具
def _filter(records, select: Dict[str, str]):
    rows = records
    if "pig_id" in select:
        rows = [r for r in rows if r.pig_id == select["pig_id"]]
    if "pen_id" in select:
        rows = [r for r in rows if r.pen_id == select["pen_id"]]
    if "start" in select:
        rows = [r for r in rows if r.date >= select["start"]]
    if "end" in select:
        rows = [r for r in rows if r.date <= select["end"]]
    return rows


def _stat(values: List[float], how: str) -> float:
    if how == "mean":
        return statistics.mean(values)
    if how == "min":
        return min(values)
    if how == "max":
        return max(values)
    if how == "sum":
        return sum(values)
    if how == "count":
        return float(len(values))
    if how == "first":
        return values[0]
    if how == "last":
        return values[-1]
    raise ValueError("未知统计量：{0}".format(how))


def _z_score(value: float, population: List[float]) -> float:
    mu = statistics.mean(population)
    sd = statistics.pstdev(population)
    return 0.0 if sd == 0 else (value - mu) / sd


def build_data_questions(farm) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for spec in DATA_SPECS:
        item: Dict[str, Any] = {
            "id": spec["id"],
            "category": "data",
            "question": spec["question"],
            "answerable": True,
            "evidence_hint": spec["note"],
            "expected": {"numbers": [], "tokens": spec.get("tokens", [])},
        }
        if spec.get("exclude_from_aggregate"):
            item["exclude_from_aggregate"] = True
            item["exclude_reason"] = spec.get("exclude_reason", "")
        numbers: List[Dict[str, Any]] = item["expected"]["numbers"]

        if spec.get("counts"):
            numbers.append({"label": "栏位数", "value": float(len(farm.pens)),
                            "tol": 0.5, "unit": "个"})
            numbers.append({"label": "猪只数", "value": float(len(farm.pigs)),
                            "tol": 0.5, "unit": "头"})
            item["evidence_hint"] += "（{0} 个栏位，共 {1} 头猪）".format(
                len(farm.pens), len(farm.pigs))

        if spec.get("group_by") == "pen":
            rows = _filter(farm.records, spec["select"])
            for pen in sorted({r.pen_id for r in rows}):
                subset = [r for r in rows if r.pen_id == pen]
                for metric, how, label in spec["numbers"]:
                    value = _stat([getattr(r, metric) for r in subset], how)
                    numbers.append({
                        "label": label.format(pen.replace("P", "")),
                        "value": round(value, 4),
                        "tol": _tolerance(value),
                        "unit": "kg" if metric.endswith("_kg") else (
                            "L" if metric == "water_l" else ""),
                    })

        elif spec.get("anomaly"):
            metric, scope, entity = spec["anomaly"]
            # 总体必须是同期全体个体，不能沿用 select 里的个体过滤，
            # 否则总体只有一行、标准差为 0，z 恒等于 0。
            window = {k: v for k, v in spec["select"].items() if k in ("start", "end")}
            rows = _filter(farm.records, window)
            if scope == "pig":
                grouped = {}
                for row in rows:
                    grouped.setdefault(row.pig_id, []).append(getattr(row, metric))
                means = {k: statistics.mean(v) for k, v in grouped.items()}
                if len(means) < 2:
                    raise SystemExit("{0}: 总体只有 {1} 个个体".format(spec["id"], len(means)))
                z = _z_score(means[entity], list(means.values()))
                if z > -3:  # 期望是显著偏低；不成立就说明题集与数据不一致
                    raise SystemExit(
                        "{0}: 期望 {1} 的 z 分数低于 -3，实际 {2:.2f}".format(
                            spec["id"], entity, z))
                item["expected"]["z_score"] = round(z, 3)
                item["evidence_hint"] += "（实测 z={0:.2f}）".format(z)

        elif spec.get("numbers") and "select" in spec:
            rows = _filter(farm.records, spec["select"])
            if not rows:
                raise SystemExit("{0}: 过滤后没有记录".format(spec["id"]))
            for metric, how, label in spec["numbers"]:
                if how in ("first", "last"):
                    ordered = sorted(rows, key=lambda r: r.date)
                    value = getattr(ordered[0 if how == "first" else -1], metric)
                else:
                    value = _stat([getattr(r, metric) for r in rows], how)
                numbers.append({
                    "label": label,
                    "value": round(value, 4),
                    "tol": _tolerance(value),
                    "unit": "kg" if metric.endswith("_kg") else (
                        "L" if metric == "water_l" else (
                            "℃" if metric == "temp_c" else "")),
                })
        out.append(item)
    return out


def build_knowledge_questions(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_doc: Dict[str, str] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], "")
    texts = {doc_id: [] for doc_id in by_doc}
    for chunk in chunks:
        texts[chunk["doc_id"]].append(chunk["text"])

    out: List[Dict[str, Any]] = []
    for spec in KNOWLEDGE_SPECS:
        verified = {}
        for doc_id in spec["doc_ids"]:
            if doc_id not in texts:
                raise SystemExit("{0}: 文献 {1} 不在知识库中".format(spec["id"], doc_id))
            body = "\n".join(texts[doc_id])
            hits = {kw: body.count(kw) for kw in spec["keywords"]}
            if not any(count > 0 for count in hits.values()):
                raise SystemExit(
                    "{0}: 文献 {1} 未命中任何关键词 {2}".format(
                        spec["id"], doc_id, spec["keywords"]))
            verified[doc_id] = {k: v for k, v in hits.items() if v}
        out.append({
            "id": spec["id"],
            "category": "knowledge",
            "question": spec["question"],
            "answerable": True,
            "expected": {"doc_ids": spec["doc_ids"], "keyword_hits": verified},
            "evidence_hint": "期望来源需命中关键词：" + "、".join(spec["keywords"]),
        })
    return out


def build_vision_questions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for spec in VISION_SPECS:
        label_dir = SAMPLES_DIR / spec["label"]
        if not label_dir.is_dir():
            raise SystemExit("{0}: 类别目录不存在 {1}".format(spec["id"], label_dir))
        images = sorted(label_dir.glob("*.jpg"))
        if len(images) <= spec["index"]:
            raise SystemExit("{0}: 类别 {1} 只有 {2} 张图".format(
                spec["id"], spec["label"], len(images)))
        source = images[spec["index"]]

        # 复制到中性文件名：题目里不能出现类别名，否则模型读路径就能答对
        target = IMAGES_DIR / "{0}.jpg".format(spec["id"].lower())
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, target)

        rel = target.relative_to(PROJECT_ROOT).as_posix()
        out.append({
            "id": spec["id"],
            "category": "vision",
            "question": "这张图片里的猪是什么姿态？{0}".format(rel),
            "answerable": True,
            "image_path": rel,
            "source_image": source.relative_to(PROJECT_ROOT).as_posix(),
            "expected": {"label": spec["label"], "label_cn": LABEL_CN[spec["label"]],
                         "doc_ids": []},
            "evidence_hint": "类别来自原图所在目录名（人工标注）；题目使用中性文件名避免泄漏",
        })
    return out


def build_unanswerable_questions(farm, chunks) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    corpus = "\n".join(c["text"] for c in chunks)
    for spec in UNANSWERABLE_SPECS:
        verify = spec["verify"]
        if verify.startswith("no_field:"):
            field_name = verify.split(":", 1)[1]
            if hasattr(farm.records[0], field_name):
                raise SystemExit("{0}: 模拟数据其实有字段 {1}".format(spec["id"], field_name))
        elif verify == "out_of_range":
            if any(r.date < START_DATE.isoformat() for r in farm.records):
                raise SystemExit("{0}: 数据早于起始日期，题不成立".format(spec["id"]))
        elif verify.startswith("absent_keywords:"):
            for keyword in verify.split(":", 1)[1].split(","):
                if keyword and keyword in corpus:
                    raise SystemExit(
                        "{0}: 知识库中出现关键词「{1}」，题不成立".format(spec["id"], keyword))
        elif verify.startswith("missing_pen:"):
            pen_id = verify.split(":", 1)[1]
            if any(p.pen_id == pen_id for p in farm.pens):
                raise SystemExit("{0}: 栏位 {1} 其实存在".format(spec["id"], pen_id))
        out.append({
            "id": spec["id"],
            "category": "unanswerable",
            "question": spec["question"],
            "answerable": False,
            "expected": {"doc_ids": [], "must_admit": True},
            "evidence_hint": spec["note"],
        })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建阶段 E 题集")
    parser.add_argument("--out", default=str(QUESTION_FILE))
    args = parser.parse_args(argv)

    if not CHUNKS_FILE.exists():
        raise SystemExit("缺少 {0}，请先构建知识库".format(CHUNKS_FILE))
    chunks = [json.loads(line) for line in
              CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    farm = get_farm()

    questions: List[Dict[str, Any]] = []
    questions += build_data_questions(farm)
    questions += build_knowledge_questions(chunks)
    questions += build_vision_questions()
    questions += build_unanswerable_questions(farm, chunks)

    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise SystemExit("题目 ID 重复")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "built_at": date.today().isoformat(),
        "seed": SEED,
        "days": DAYS,
        "start_date": START_DATE.isoformat(),
        "total": len(questions),
        "disclaimer": DISCLAIMER,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            question["meta"] = header
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")

    counts: Dict[str, int] = {}
    for question in questions:
        counts[question["category"]] = counts.get(question["category"], 0) + 1

    print("题集：{0}".format(out_path.relative_to(PROJECT_ROOT)))
    print("总数 {0}：{1}".format(len(questions), "、".join(
        "{0} {1} 题".format(k, v) for k, v in counts.items())))
    print("=" * 78)
    for question in questions:
        expected = question["expected"]
        if question["category"] == "data":
            detail = "；".join("{0}={1}{2}".format(
                n["label"], n["value"], n["unit"] and " " + n["unit"])
                for n in expected["numbers"][:4]) or "、".join(expected["tokens"])
            if len(expected["numbers"]) > 4:
                detail += " 等 {0} 项".format(len(expected["numbers"]))
        elif question["category"] == "knowledge":
            detail = "来源 " + "、".join(expected["doc_ids"])
        elif question["category"] == "vision":
            detail = "期望 " + expected["label_cn"]
        else:
            detail = "应明确说不知道"
        print("{0:<4} {1:<12} {2}".format(question["id"], question["category"], detail))
        print("     {0}".format(question["question"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())