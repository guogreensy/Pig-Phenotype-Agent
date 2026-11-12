"""系统设计示意图：总体架构、检索融合流程、智能体执行循环。

    python scripts/plot_architecture.py

产出（results/figures/）：

    architecture.png     三层体系结构与调用关系
    retrieval_flow.png   双通道检索、融合排序与降级路径
    agent_loop.png       多步循环、重复调用拦截与强制收尾

由于示意图无法靠单元测试验证，本脚本在绘图时登记每个方框的矩形，
绘制结束后程序化检查「是否越界」与「是否相互重叠」，把版式问题
变成可复现的断言，而不是靠肉眼。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.reporting import save, setup_font  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

INK = "#1f2a37"
MUTED = "#6b7684"
LINE = "#cbd5e1"
BLUE = "#2563eb"
BLUE_SOFT = "#dbeafe"
BLUE_MID = "#60a5fa"
GREY_SOFT = "#f1f5f9"
GREEN_SOFT = "#dcfce7"
GREEN_EDGE = "#86efac"
GREEN = "#15803d"
AMBER_SOFT = "#fef3c7"
AMBER_EDGE = "#fcd34d"
AMBER = "#a16207"
RED_SOFT = "#fee2e2"
RED_EDGE = "#fca5a5"
RED = "#b91c1c"

_BOXES: list[tuple[str, float, float, float, float]] = []
_TEXTS: list = []
_CANVAS = [10.0, 10.0]


def _register(name, x, y, w, h):
    _BOXES.append((name, x, y, w, h))


def _box(ax, name, x, y, w, h, text, *, face=GREY_SOFT, edge=LINE,
         weight="normal", size=10, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.01,rounding_size=0.06",
                                linewidth=1.0, facecolor=face, edgecolor=edge,
                                zorder=2))
    artist = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                     fontsize=size, color=color, weight=weight, zorder=3,
                     linespacing=1.7)
    _register(name, x, y, w, h)
    _TEXTS.append((name, artist, ax, (x, y, w, h)))


def _arrow(ax, start, end, *, color=BLUE, lw=1.3, ls="-", rad=0.0,
           label=None, label_offset=(0, 0), size=9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle="arc3,rad=%s" % rad,
                                 shrinkA=3, shrinkB=3, zorder=1))
    if label:
        ax.text((start[0] + end[0]) / 2 + label_offset[0],
                (start[1] + end[1]) / 2 + label_offset[1], label,
                ha="center", va="center", fontsize=size, color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none"))


def _canvas(width, height):
    _BOXES.clear()
    _TEXTS.clear()
    top = 10.0 * height / width
    _CANVAS[0], _CANVAS[1] = 10.0, top
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.axis("off")
    return fig, ax, top


def _verify(title):
    """越界与重叠检查：矩形相交面积超过阈值即视为版式错误。"""
    problems = []
    xmax, ymax = _CANVAS
    for name, x, y, w, h in _BOXES:
        if x < 0 or y < 0 or x + w > xmax + 1e-9 or y + h > ymax + 1e-9:
            problems.append("越界：%s (%.2f,%.2f,%.2f,%.2f)" % (name, x, y, w, h))
    for i in range(len(_BOXES)):
        for j in range(i + 1, len(_BOXES)):
            n1, x1, y1, w1, h1 = _BOXES[i]
            n2, x2, y2, w2, h2 = _BOXES[j]
            ox = min(x1 + w1, x2 + w2) - max(x1, x2)
            oy = min(y1 + h1, y2 + h2) - max(y1, y2)
            if ox > 0.01 and oy > 0.01:
                problems.append("重叠：%s ↔ %s  面积 %.3f" % (n1, n2, ox * oy))
    if _TEXTS:
        from matplotlib.transforms import Bbox
        fig = _TEXTS[0][2].figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for name, artist, ax, rect in _TEXTS:
            tb = artist.get_window_extent(renderer)
            x, y, w, h = rect
            (dx0, dy0) = ax.transData.transform((x, y))
            (dx1, dy1) = ax.transData.transform((x + w, y + h))
            box = Bbox.from_extents(dx0, dy0, dx1, dy1)
            over_w = (tb.width - box.width) / box.width
            over_h = (tb.height - box.height) / box.height
            if over_w > 0.02 or over_h > 0.02:
                problems.append("文字溢出：%s 宽超 %.0f%% 高超 %.0f%%"
                                % (name, max(over_w, 0) * 100, max(over_h, 0) * 100))
    if problems:
        raise SystemExit("%s 版式检查未通过：\n  %s" % (title, "\n  ".join(problems)))
    print("  %s 版式检查通过（%d 个方框，无越界、无重叠）" % (title, len(_BOXES)))


def draw_architecture() -> Path:
    fig, ax, top = _canvas(11.6, 5.6)

    for label, y in (("交互层", 4.22), ("智能体层", 2.62), ("工具与数据层", 0.95)):
        ax.text(0.14, y, label, ha="center", va="center", fontsize=10,
                color=MUTED, weight="bold", rotation=90)

    _box(ax, "交互层", 1.00, 3.75, 8.55, 0.95,
         "Web 单页应用　·　图像识别　|　智能问答　|　提示词查看\n"
         "四个接口：首页　/　健康自检　/　图像识别　/　智能问答",
         face=BLUE_SOFT, edge=BLUE_MID, size=10)

    _box(ax, "提示词构造", 1.00, 1.95, 2.55, 1.35,
         "提示词构造\n工具说明书\n+ 七条规则", face="white", edge=BLUE_MID, size=9.6)
    _box(ax, "多步循环", 3.72, 1.95, 2.55, 1.35,
         "多步循环\n≤ 6 步\n同参数重复 ≤ 2 次", face="white", edge=BLUE_MID, size=9.6)
    _box(ax, "结果组装", 6.44, 1.95, 3.11, 1.35,
         "结果组装\n引用按出现顺序编号\n工具轨迹随回答返回",
         face="white", edge=BLUE_MID, size=9.6)

    _box(ax, "结构化记录", 1.00, 0.30, 2.55, 1.30,
         "结构化记录\n4 栏位 / 48 头\n30 天日记录",
         face=GREEN_SOFT, edge=GREEN_EDGE, size=9.5)
    _box(ax, "文献知识库", 3.72, 0.30, 2.55, 1.30,
         "文献知识库\n20 篇 / 943 片段\n双通道索引",
         face=GREEN_SOFT, edge=GREEN_EDGE, size=9.5)
    _box(ax, "工具与视觉", 6.44, 0.30, 3.11, 1.30,
         "8 个只读工具 + 视觉模型\nJSON Schema 约束 · 结构化错误回填",
         face=GREEN_SOFT, edge=GREEN_EDGE, size=9.2)

    _arrow(ax, (5.30, 3.75), (5.30, 3.30), label="提问", label_offset=(0.45, 0))
    _arrow(ax, (5.30, 3.30), (5.30, 3.75), color=BLUE_MID, ls="--",
           label="引用 + 轨迹", label_offset=(-0.85, 0))
    _arrow(ax, (4.95, 1.95), (4.95, 1.60), color=GREEN, label="调用",
           label_offset=(0.42, 0))
    _arrow(ax, (5.65, 1.60), (5.65, 1.95), color=GREEN, ls="--",
           label="观察结果", label_offset=(0.58, 0))

    fig.tight_layout()
    _verify("architecture")
    return save(fig, FIGURES_DIR / "architecture.png")


def draw_retrieval_flow() -> Path:
    fig, ax, top = _canvas(11.6, 4.8)

    _box(ax, "问题", 0.20, 1.62, 1.55, 1.05, "问题 q\n（含工具上下文）",
         face=BLUE_SOFT, edge=BLUE_MID, size=9.2)
    _box(ax, "词法通道", 2.30, 2.62, 2.65, 1.10,
         "词法通道　char_wb 2–3 gram\nTF-IDF 28 932 维\n索引 4.2 MB",
         face="white", edge=BLUE_MID, size=9.2)
    _box(ax, "稠密通道", 2.30, 1.15, 2.65, 1.10,
         "稠密通道　2048 维嵌入\n943 × 2048 矩阵\n余弦相似度",
         face="white", edge=BLUE_MID, size=9.2)
    _box(ax, "降级", 2.30, 0.25, 2.65, 0.62,
         "稠密通道异常 → 仅词法（带降级标志）", face=RED_SOFT, edge=RED_EDGE,
         size=8.6, color=RED)
    _box(ax, "融合", 5.70, 1.55, 1.95, 1.15,
         "加权 RRF\nk = 60\n权重 1.0 / 0.5", face=AMBER_SOFT, edge=AMBER_EDGE,
         size=9.2, color=AMBER)
    _box(ax, "证据", 8.05, 1.55, 1.75, 1.15, "Top-k 片段\n→ 证据与引用",
         face=GREEN_SOFT, edge=GREEN_EDGE, size=9.2)

    _arrow(ax, (1.75, 2.30), (2.30, 3.05), label="召回 20", label_offset=(0.02, 0.28))
    _arrow(ax, (1.75, 2.00), (2.30, 1.75), label="召回 20", label_offset=(0.02, -0.28))
    _arrow(ax, (4.95, 3.17), (5.70, 2.42))
    _arrow(ax, (4.95, 1.70), (5.70, 1.95))
    _arrow(ax, (7.65, 2.12), (8.05, 2.12))
    _arrow(ax, (3.62, 1.15), (3.62, 0.87), color=RED, ls="--", lw=1.0)
    _arrow(ax, (4.95, 0.56), (6.10, 1.55), color=RED, ls="--", lw=1.0, rad=-0.12,
           label="降级时跳过稠密", label_offset=(0.30, -0.28), size=8.4)

    fig.tight_layout()
    _verify("retrieval_flow")
    return save(fig, FIGURES_DIR / "retrieval_flow.png")


def draw_agent_loop() -> Path:
    fig, ax, top = _canvas(10.6, 5.4)

    _box(ax, "题面", 0.25, 2.30, 1.75, 1.00, "题面 + 工具说明书",
         face=BLUE_SOFT, edge=BLUE_MID, size=9.2)
    _box(ax, "模型决策", 2.45, 2.30, 1.85, 1.00, "模型决策\n（本轮输出）",
         face="white", edge=BLUE_MID, size=9.2)
    _box(ax, "执行工具", 4.75, 3.75, 2.05, 0.95, "执行工具\n参数校验 · 只读",
         face=GREEN_SOFT, edge=GREEN_EDGE, size=9.2)
    _box(ax, "观察结果", 4.75, 0.95, 2.05, 0.95, "观察结果回填\n（或结构化错误）",
         face=GREEN_SOFT, edge=GREEN_EDGE, size=9.2)
    _box(ax, "最终回答", 7.55, 2.30, 1.95, 1.00, "最终回答\n引用 + 轨迹",
         face=BLUE_SOFT, edge=BLUE_MID, size=9.2)
    _box(ax, "约束", 2.45, 0.20, 7.05, 0.55,
         "同一「工具 + 参数」≤ 2 次　·　总步数 ≤ 6　·　步数耗尽：去除工具定义，强制给出阶段性结论",
         face=AMBER_SOFT, edge=AMBER_EDGE, size=8.8, color=AMBER)

    _arrow(ax, (2.00, 2.80), (2.45, 2.80))
    _arrow(ax, (4.30, 3.05), (4.75, 4.05), label="要工具", label_offset=(-0.05, 0.28), size=8.8)
    _arrow(ax, (4.30, 2.55), (7.55, 2.80), label="可直接作答", label_offset=(0.05, 0.28), size=8.8)
    _arrow(ax, (5.78, 3.75), (5.78, 1.90), color=GREEN)
    _arrow(ax, (4.75, 1.42), (4.30, 2.42), color=GREEN, rad=0.18)
    _arrow(ax, (5.78, 0.95), (5.78, 0.75), color=AMBER, ls="--", lw=1.0)

    fig.tight_layout()
    _verify("agent_loop")
    return save(fig, FIGURES_DIR / "agent_loop.png")


def main() -> int:
    setup_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (draw_architecture, draw_retrieval_flow, draw_agent_loop):
        print("绘制 %s ……" % fn.__name__)
        path = fn()
        print("  已生成：%s（%.1f KB）" % (path, path.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())