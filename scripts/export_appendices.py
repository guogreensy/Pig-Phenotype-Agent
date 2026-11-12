# -*- coding: utf-8 -*-
"""从数据文件导出台账，生成结课报告附录中被折叠的两份完整清单。

结课报告只保留「按类别的汇总表」，逐篇语料与逐题题目放在仓库里，
由本脚本从 data/kb/sources.json、data/kb/chunks.jsonl 与
data/experiment/agent_questions.jsonl 生成，保证与实验所用数据一致：

    python scripts/export_appendices.py

输出：
    docs/appendix_corpus.md     附录 A  检索语料清单（20 篇）
    docs/appendix_questions.md  附录 B  测试题清单（30 题）

不引入任何第三方依赖，仅用标准库。
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data" / "kb"
EXP = ROOT / "data" / "experiment"
DOCS = ROOT / "docs"

KIND_CN = OrderedDict([
    ("pmc_xml", "期刊论文（PMC 开放获取）"),
    ("local_md", "项目自有文档"),
    ("hf_md", "数据集卡片"),
])

CAT_CN = OrderedDict([
    ("data", "数据"),
    ("knowledge", "知识"),
    ("vision", "视觉"),
    ("refuse", "拒答"),
    ("unanswerable", "拒答"),
])

# 题集里拒答题用 refuse 或 unanswerable 两种标签，统计时合并
CAT_ALIAS = {"unanswerable": "refuse"}


def cat_key(category):
    return CAT_ALIAS.get(category, category)


def fmt_num(value):
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def md_escape(text):
    return text.replace("|", "\\|").replace("\n", " ")


def export_corpus():
    sources = json.loads((KB / "sources.json").read_text(encoding="utf-8"))
    docs = sources["documents"]
    chunks = read_jsonl(KB / "chunks.jsonl")

    per_doc = OrderedDict()
    for ch in chunks:
        rec = per_doc.setdefault(ch["doc_id"], {"chunks": 0, "chars": 0})
        rec["chunks"] += 1
        rec["chars"] += int(ch.get("chars") or 0)

    total_chunks = sum(v["chunks"] for v in per_doc.values())
    total_chars = sum(v["chars"] for v in per_doc.values())

    kind_order = {k: i for i, k in enumerate(KIND_CN)}
    docs = sorted(docs, key=lambda d: (kind_order.get(d["kind"], 99), d["doc_id"]))

    lines = []
    lines.append("# 附录 A  检索语料清单")
    lines.append("")
    lines.append("本文件是结课报告附录 A 的完整版：报告正文只保留按类别的汇总表，"
                 "逐篇明细（标题、许可、片段数、链接）列于此，便于核对报告中每一条引用的出处。")
    lines.append("")
    lines.append("由 `scripts/export_appendices.py` 从 `data/kb/sources.json` 与 "
                 "`data/kb/chunks.jsonl` 生成，与实验所用索引一致。")
    lines.append("")
    lines.append("- 语料总数：%d 篇" % len(docs))
    lines.append("- 片段总数：%d 个（平均 %.0f 字符）" % (
        total_chunks, total_chars / max(1, total_chunks)))
    lines.append("- 片段字符合计：%d（相邻片段重叠部分已去除）" % total_chars)
    lines.append("- 片段编号形如 `pmc_7070994#0053`，由 `doc_id` 与段内序号构成，可回溯至下文对应行")
    lines.append("")
    lines.append("| 序号 | 编号 | 类别 | 标题 | 许可 | 片段数 | 字符数 | 链接 |")
    lines.append("|---:|---|---|---|---|---:|---:|---|")
    for i, d in enumerate(docs, 1):
        rec = per_doc.get(d["doc_id"], {"chunks": 0, "chars": 0})
        url = d.get("url") or ""
        link = "[链接](%s)" % url if url.startswith("http") else "`%s`" % url
        lines.append("| %d | `%s` | %s | %s | %s | %d | %d | %s |" % (
            i, d["doc_id"], KIND_CN.get(d["kind"], d["kind"]),
            md_escape(d["title"]), md_escape(d["license"].replace("本项目自有文档", "项目自有")),
            rec["chunks"], rec["chars"], link))
    lines.append("")
    lines.append("## 按类别汇总")
    lines.append("")
    lines.append("| 类别 | 篇数 | 片段数 | 片段字符数 |")
    lines.append("|---|---:|---:|---:|")
    for kind, name in KIND_CN.items():
        subset = [d for d in docs if d["kind"] == kind]
        if not subset:
            continue
        c = sum(per_doc.get(d["doc_id"], {}).get("chunks", 0) for d in subset)
        n = sum(per_doc.get(d["doc_id"], {}).get("chars", 0) for d in subset)
        lines.append("| %s | %d | %d | %d |" % (name, len(subset), c, n))
    lines.append("| 合计 | %d | %d | %d |" % (len(docs), total_chunks, total_chars))
    lines.append("")
    lines.append("> 本项目不分发任何语料原文，仅在本地建立索引；CC BY-NC 与 CC BY-NC-ND 语料"
                 "仅用于非商业研究。")
    lines.append("")

    out = DOCS / "appendix_corpus.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("已生成：%s（%d 篇 / %d 片段）" % (out, len(docs), total_chunks))


def export_questions():
    questions = read_jsonl(EXP / "agent_questions.jsonl")

    def expected_text(q):
        exp = q.get("expected") or {}
        parts = []
        for item in exp.get("numbers") or []:
            tol = item.get("tol")
            unit = item.get("unit") or ""
            if tol is None:
                parts.append("%s = %s%s" % (item.get("label", ""), fmt_num(item.get("value")), unit))
            else:
                parts.append("%s = %s ± %s%s" % (item.get("label", ""), fmt_num(item.get("value")),
                                                 fmt_num(tol), unit))
        tokens = exp.get("tokens") or []
        if tokens:
            parts.append("实体：" + "、".join(tokens))
        return "；".join(parts) if parts else "—"

    counts = OrderedDict((k, 0) for k in ("data", "knowledge", "vision", "refuse"))
    for q in questions:
        k = cat_key(q["category"])
        counts[k] = counts.get(k, 0) + 1

    lines = []
    lines.append("# 附录 B  测试题清单")
    lines.append("")
    lines.append("本文件是结课报告附录 B 的完整版：报告正文只保留按类别的构成表，"
                 "30 道题的题面、期望值与判定要点列于此。")
    lines.append("")
    lines.append("由 `scripts/export_appendices.py` 从 `data/experiment/agent_questions.jsonl` "
                 "生成；期望值由独立脚本从原始记录计算，不取自工具输出。")
    lines.append("")
    lines.append("- 题目总数：%d" % len(questions))
    for cat, name in (("data", "数据"), ("knowledge", "知识"), ("vision", "视觉"), ("refuse", "拒答")):
        if counts.get(cat):
            ids = [q["id"] for q in questions if cat_key(q["category"]) == cat]
            lines.append("- %s类 %d 题：%s" % (name, counts[cat], "、".join(ids)))
    lines.append("")
    lines.append("| 编号 | 类别 | 题目 | 期望值 / 判定要点 | 证据提示 |")
    lines.append("|---|---|---|---|---|")
    for q in questions:
        lines.append("| %s | %s | %s | %s | %s |" % (
            q["id"], CAT_CN.get(q["category"], q["category"]),
            md_escape(q["question"]), md_escape(expected_text(q)),
            md_escape(q.get("evidence_hint") or "—")))
    lines.append("")
    lines.append("## 判定口径")
    lines.append("")
    lines.append("| 类别 | 题数 | 判定方式 | 报告中的分母 |")
    lines.append("|---|---:|---|---|")
    lines.append("| 数据 | %d | 期望数值按 1.5%% 容差匹配（计数类 ±0.5）；"
                 "无期望数值者按实体是否指认判定 | 17 个期望数值项 / 9 道整题 / 4 个实体 |"
                 % counts.get("data", 0))
    lines.append("| 知识 | %d | 引用集合与期望文档集合是否有交集 | 10 道题 |" % counts.get("knowledge", 0))
    lines.append("| 视觉 | %d | 姿态类别标签完全一致 | 5 道题 |" % counts.get("vision", 0))
    lines.append("| 拒答 | %d | 是否声明信息缺失而非编造 | 5 道题 |" % counts.get("refuse", 0))
    lines.append("")
    lines.append("> D6 题面自带两个数字，回答仅需复述，不能反映计算能力，"
                 "其 2 个数值项不计入数值命中率的汇总分母。")
    lines.append("")

    out = DOCS / "appendix_questions.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("已生成：%s（%d 题）" % (out, len(questions)))


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    export_corpus()
    export_questions()
    return 0


if __name__ == "__main__":
    sys.exit(main())