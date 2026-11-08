"""阶段 B 冒烟检查：把工具层真跑一遍。

默认**离线**：只调数据工具、示例图片列举与词法检索，不消耗任何接口额度。
加 ``--live`` 才会真实调用向量检索与视觉模型，各一次。

    python scripts/smoke_tools.py
    python scripts/smoke_tools.py --live

任何一步失败都会打印真实错误并以非零码退出——冒烟脚本不掩盖失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.builder import build_default_registry  # noqa: E402
from src.tools.registry import ToolResult  # noqa: E402

LOG_PATH = PROJECT_ROOT / "logs" / "tool_smoke_calls.jsonl"


def _show(title: str, result: ToolResult, brief: bool = True) -> Dict[str, Any]:
    mark = "OK  " if result.ok else "FAIL"
    print("[{0}] {1:<20} {2:>7.0f} ms".format(mark, result.tool, result.duration_ms))
    if not result.ok:
        print("       错误：{0}".format(result.error))
        return {}
    data = result.data
    if brief:
        print("       {0}".format(_summarize(data)))
    return data


def _summarize(data: Any) -> str:
    if not isinstance(data, dict):
        return repr(data)[:160]
    keys = [k for k in ("count", "returned", "pens", "records", "flags", "groups",
                        "posture", "posture_cn", "mode", "degraded", "note")
            if k in data]
    bits = []
    for key in keys:
        value = data[key]
        if isinstance(value, list):
            bits.append("{0}={1}项".format(key, len(value)))
        else:
            text = str(value)
            bits.append("{0}={1}".format(key, text[:60]))
    return "，".join(bits) or json.dumps(data, ensure_ascii=False)[:160]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 B 工具层冒烟")
    parser.add_argument("--live", action="store_true", help="真实调用向量检索与视觉模型")
    args = parser.parse_args(argv)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    registry = build_default_registry(log_path=LOG_PATH, max_calls=12)
    print("已注册 {0} 个工具：{1}".format(len(registry), "、".join(registry.names())))
    print("=" * 68)

    failures = 0

    # ---------------- 数据工具（离线）
    pens = _show("栏位总览", registry.call("list_pens", {}))
    failures += 0 if pens else 1

    anomalies = _show("采食异常检测", registry.call(
        "detect_anomalies", {"metric": "feed_kg", "days": 7, "z_threshold": 2.0}))
    failures += 0 if anomalies else 1
    if anomalies:
        for flag in anomalies.get("flags", [])[:3]:
            print("       · {0} 均值 {1} z={2}（{3}）".format(
                flag["target"], flag["mean"], flag["z_score"], flag["direction"]))

    records = _show("单只明细", registry.call("query_pig_records", {
        "pig_id": "P2-007", "start_date": "2026-11-01", "end_date": "2026-11-03"}))
    failures += 0 if records else 1

    agg = _show("栏位均重", registry.call(
        "aggregate_metric", {"metric": "weight_kg", "group_by": "pen"}))
    failures += 0 if agg else 1
    if agg:
        for group in agg.get("groups", []):
            print("       · {0}: {1} kg（n={2}）".format(
                group["group"], group["value"], group["n"]))

    # ---------------- 参数校验（失败路径必须如实报错）
    bad = registry.call("aggregate_metric", {"metric": "weight_kg", "group_by": "barn"})
    print("[{0}] {1:<20} 预期失败：{2}".format(
        "OK  " if not bad.ok else "FAIL", "参数校验", (bad.error or "")[:60]))
    failures += 0 if not bad.ok else 1

    # ---------------- 视觉工具（列举，不调模型）
    samples = _show("示例图片", registry.call("list_sample_images", {"limit": 3}))
    failures += 0 if samples else 1
    if samples:
        for item in samples["images"]:
            print("       · {0}（{1}/{2}）".format(
                item["path"], item["split"], item["category"]))

    # ---------------- 知识库检索
    if args.live:
        rag = _show("知识库检索(稠密)", registry.call(
            "search_knowledge", {"query": "如何用深度学习识别猪只姿态", "k": 3}))
        failures += 0 if rag else 1
        if rag:
            for cite in rag.get("citations", []):
                print("       · [{0}] {1}（{2}）".format(
                    cite["chunk_id"], cite["title"][:48], cite["license"]))
        mode = "dense"
    else:
        rag = _show("知识库检索(词法)", registry.call(
            "search_knowledge", {"query": "deep learning pig posture", "k": 3,
                                 "mode": "lexical"}))
        failures += 0 if rag else 1
        mode = "lexical"
    print("       检索模式：{0}，降级：{1}".format(mode, rag.get("degraded")))

    # ---------------- 视觉模型（一次真实调用）
    if args.live and samples and samples.get("images"):
        target = samples["images"][0]["path"]
        vision = _show("姿态识别(真实调用)", registry.call(
            "analyze_pig_posture", {"image_path": target, "variant": "C"}))
        failures += 0 if vision else 1
        if vision:
            print("       · 姿态 {0}（{1}），valid={2}，uncertain={3}".format(
                vision.get("posture"), vision.get("posture_cn"),
                vision.get("valid"), vision.get("uncertain")))
            print("       · 解析方式 {0}，提示词 {1}#{2}".format(
                vision.get("parse", {}).get("method"),
                vision.get("prompt", {}).get("variant"),
                str(vision.get("prompt", {}).get("sha256"))[:8]))
    else:
        print("[SKIP] analyze_pig_posture  未加 --live，跳过真实模型调用")

    print("=" * 68)
    stats = registry.stats()
    print("调用统计：共 {0} 次，失败 {1} 次，成功率 {2}".format(
        stats["calls"], stats["failed"], stats["success_rate"]))
    print("按工具：{0}".format(json.dumps(stats["by_tool"], ensure_ascii=False)))
    print("调用日志：{0}".format(LOG_PATH.relative_to(PROJECT_ROOT)))
    print("=" * 68)
    if failures:
        print("冒烟未通过：{0} 处异常".format(failures))
        return 1
    print("冒烟通过" + ("（含真实接口调用）" if args.live else "（离线）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())