"""阶段 C 端到端冒烟：问题 → 规划 → 工具 → 回答 → 报告。

默认**真实调用**，三个问题分别覆盖数据查询、知识检索、姿态识别。
``--dry-run`` 用脚本化假客户端跑通全链路，不消耗接口额度，用于验证装配。

    python scripts/smoke_agent.py --dry-run
    python scripts/smoke_agent.py
    python scripts/smoke_agent.py --question "P2-007 最近采食怎么样？"

每次运行都会把完整记录（回答、引用、工具轨迹、指标、提示词指纹）写入
``results/agent/smoke_<时间戳>.json``，作为报告里的过程证据。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.executor import AgentExecutor  # noqa: E402
from src.agent.llm_client import ChatResponse, ToolCall, build_chat_client  # noqa: E402
from src.agent.report import build_report, render_markdown  # noqa: E402
from src.tools.builder import build_default_registry  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "agent"

#: 三个代表性问题：数据 / 知识 / 视觉
QUESTIONS: List[Dict[str, str]] = [
    {"kind": "data",
     "text": "最近一周哪个栏位的猪采食量偏低？P2-007 这头猪的情况要不要重点关注？"},
    {"kind": "knowledge",
     "text": "猪只姿态识别常用的深度学习方法有哪些？请给出出处。"},
    {"kind": "vision",
     "text": ("这张图片里的猪是什么姿态？"
              "data/samples/dev/lying_lateral/test_pen1_tur_cam2_20250921_051726_0005.jpg")},
]


class DryRunClient:
    """脚本化客户端：先请求 list_pens，再给出固定回答。"""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice=None, **kwargs):
        self.calls += 1
        if tools and self.calls % 2 == 1:
            return ChatResponse(
                ok=True, text="", model="dry-run", latency_seconds=0.01,
                input_tokens=100, output_tokens=20,
                tool_calls=[ToolCall(id="dry_{0}".format(self.calls),
                                     name="list_pens", arguments_raw="{}", arguments={})],
            )
        return ChatResponse(
            ok=True, text="（试运行）已依次完成规划、工具调用与回答，链路正常。",
            model="dry-run", latency_seconds=0.01, input_tokens=100, output_tokens=20,
        )


def _select_registry(tools: str):
    """按 --tools 装配（逗号分隔：data,vision,rag；all 表示全部）。"""
    if tools.strip() == "all":
        return build_default_registry(log_path=OUT_DIR / "tool_calls.jsonl", max_calls=40)
    names = {item.strip() for item in tools.split(",") if item.strip()}
    unknown = names - {"data", "vision", "rag"}
    if unknown:
        raise SystemExit("未知工具类别：{0}".format("、".join(sorted(unknown))))
    return build_default_registry(
        log_path=OUT_DIR / "tool_calls.jsonl", max_calls=40,
        include_data="data" in names,
        include_vision="vision" in names,
        include_rag="rag" in names,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="阶段 C 端到端冒烟")
    parser.add_argument("--dry-run", action="store_true", help="用假客户端跑通链路，不调接口")
    parser.add_argument("--question", default="", help="只跑单个问题")
    parser.add_argument("--tools", default="all", help="工具类别：all 或 data,vision,rag")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = _select_registry(args.tools)
    client = DryRunClient() if args.dry_run else build_chat_client()
    executor = AgentExecutor(registry, client, max_steps=args.max_steps)

    questions = ([{"kind": "custom", "text": args.question}] if args.question
                 else QUESTIONS)

    print("工具：{0}".format("、".join(registry.names()) or "（无）"))
    print("客户端：{0}".format("dry-run" if args.dry_run else client.model))
    print("=" * 74)

    records: List[Dict[str, Any]] = []
    failures = 0
    for index, item in enumerate(questions, start=1):
        print("\n[{0}/{1}]（{2}）{3}".format(index, len(questions), item["kind"],
                                            item["text"][:60]))
        result = executor.run(item["text"])
        report = build_report(result)
        records.append({"kind": item["kind"], "report": report.to_dict()})

        print("-" * 74)
        print(render_markdown(report))
        if not report.ok:
            failures += 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = OUT_DIR / "smoke_{0}{1}.json".format(stamp, "_dryrun" if args.dry_run else "")
    target.write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "dry-run" if args.dry_run else "live",
        "tools": registry.names(),
        "tool_stats": registry.stats(),
        "prompts": records and records[0]["report"]["prompt_meta"],
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("记录已写入：{0}".format(target.relative_to(PROJECT_ROOT)))
    print("工具调用统计：{0}".format(json.dumps(registry.stats(), ensure_ascii=False)))
    if failures:
        print("冒烟未通过：{0} 个问题未正常完成".format(failures))
        return 1
    print("冒烟通过" + ("（dry-run）" if args.dry_run else "（真实调用）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())