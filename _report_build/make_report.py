"""生成《大语言模型技术与应用》结课报告（论文格式 .docx）。

封面沿用作者已填写的模板：本脚本从 ``cover_values.json`` 读取学号、姓名等
字段回填，因此不会覆盖作者的个人信息。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).parent))

from docx_helpers import (  # noqa: E402
    HEI, SONG, TIMES,
    add_page_number_footer, body, caption, clear_footer, figure, formula,
    heading, para, set_run, three_line_table,
)

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
OUT = ROOT / "大语言模型技术与应用-结课报告.docx"
TITLE = "基于检索增强生成与工具调用的大模型猪只表型分析助手设计与评估"
FIG = ROOT / "results" / "figures"


def fill_cover(doc):
    """按作者已填内容回填封面，避免覆盖个人信息。

    ``cover_values.json`` 不入库（含学号、姓名），缺失时保留空封面，
    使脚本在他人机器上仍可运行。
    """
    path = BUILD / "cover_values.json"
    if not path.exists():
        print("未找到 cover_values.json：保留空白封面")
        return
    values = json.loads(path.read_text(encoding="utf-8"))
    order = ["学号", "专业", "姓名", "题目", "指导教师", "所属学院"]
    table = doc.tables[0]
    for row_index, key in enumerate(order):
        text = values.get(key, "").strip()
        if not text:
            continue
        row = table.rows[row_index]
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if key == "题目":
            row.cells[0].vertical_alignment = WD_ALIGN_VERTICAL.TOP
        cell = row.cells[1]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run(p.add_run(text), ea=SONG, size=12)


def front_matter(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    set_run(p.add_run(TITLE), ea=HEI, size=16, bold=True)

    para(doc, "（桂林电子科技大学  广西 桂林  541004）", size=10.5, indent_chars=0,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10, first_line=False)

    abstract = (
        "广西地方品种猪的人工智能育种需要同时处理三类异质信息：自动采集设备产生的表型记录、"
        "学术文献中的方法与结论，以及巡检设备采集的图像。数据量的增长并未降低使用门槛——"
        "查询、汇总与文献佐证仍依赖人工。针对这一问题，提出并实现一个以检索增强生成与工具调用"
        "为核心的表型分析助手，作为育种数据融合与综合管控平台的智能分析入口。系统将数值计算、"
        "记录查询、文献检索与图像判读外化为 8 个只读工具，由 JSON Schema 约束参数并以结构化结果回填；"
        "构建含 20 篇文献、943 个片段的双通道索引，采用字符 n-gram 词法检索与 2048 维稠密检索并行召回，"
        "经加权倒数排名融合后排序，稠密通道异常时降级为词法检索；以“规划—调用—观察”循环驱动"
        "多步推理，对同一工具与参数设置重复调用上限，步数耗尽时去除工具定义以强制收敛。"
        "在 30 题测试集上以纯对话、仅检索、全工具三种条件各执行 30 次，90 次运行全部完成。"
        "结果表明：数据题数值命中率由 0%（95% 置信区间 0—11.6%）提升至 100%（88.4%—100%），"
        "整题正确率与异常指认率同步达到 100%；知识题检索命中率由 0% 提升至 100%；"
        "视觉题工具调用率 100%，姿态标签命中率 60%（14.7%—94.7%）；三种条件的诚实拒答率分别为"
        "80%（28.4%—99.5%）、100% 与 100%。代价为平均时延由 15.6 s 增至 49.0 s，95 分位达 168.3 s，"
        "单次模型调用的平均输入 token 由 460 增至 3 040。实验进一步定位两类失效模式："
        "检索证据覆盖题面所述事实，以及检索命中与论断归因不一致。结果说明，将计算与检索外化"
        "可显著提升表型数据的可利用性与答案的可验证性，但证据的权威排序与归因核验仍需外部机制保障。"
    )
    p = para(doc, "", indent_chars=2, line=1.3, space_after=4)
    set_run(p.add_run("摘要："), ea=HEI, size=10.5, bold=True)
    set_run(p.add_run(abstract), ea=SONG, size=10.5)

    p = para(doc, "", indent_chars=0, space_after=2, first_line=False)
    set_run(p.add_run("关键词："), ea=HEI, size=10.5, bold=True)
    set_run(p.add_run("检索增强生成；工具调用；大语言模型；猪只表型；智能育种；消融实验"),
            ea=SONG, size=10.5)

    p = para(doc, "", indent_chars=0, space_after=10, first_line=False)
    set_run(p.add_run("中图分类号：TP391.4        文献标识码：A"), ea=SONG, size=10.5)


def english_abstract(doc):
    p = para(doc, "", indent_chars=0, space_before=4, space_after=4, line=1.3,
             first_line=False)
    set_run(p.add_run("Abstract: "), ascii_font=TIMES, ea=TIMES, size=10.5, bold=True)
    text = (
        "AI-driven breeding of local pig breeds in Guangxi must handle three heterogeneous "
        "sources: phenotype records produced by automated acquisition devices, methods and "
        "conclusions scattered in the literature, and images captured by inspection equipment. "
        "Growing data volume has not lowered the barrier to use—querying, aggregating and "
        "literature verification remain manual. This paper presents an LLM assistant built on "
        "retrieval-augmented generation (RAG) and tool calling, serving as the intelligent "
        "analysis entry of a phenotype data-fusion and management platform. Eight read-only tools "
        "externalize computation, record querying, literature search and image interpretation; "
        "their parameters are constrained by JSON Schema and their results are fed back in "
        "structured form. A dual-channel index over 20 documents and 943 chunks combines character "
        "n-gram lexical retrieval with 2048-dimensional dense retrieval, fused by weighted "
        "reciprocal rank fusion, with automatic lexical fallback when the dense channel fails. A "
        "plan-act-observe loop drives multi-step reasoning, with a repeat-call guard and a "
        "tool-free forced-finish stage. On a 30-question set evaluated under three conditions "
        "(dialogue-only, retrieval-only, full), 90 runs completed without failure. The numeric hit "
        "rate of data questions rose from 0% (95% CI 0—11.6%) to 100% (88.4%—100%); the retrieval "
        "hit rate on knowledge questions rose from 0% to 100%; posture-label accuracy was 60% "
        "(14.7%—94.7%) with a 100% tool-use rate; and the honest-refusal rate was 80% "
        "(28.4%—99.5%), 100% and 100%. Mean latency increased from 15.6 s to 49.0 s (P95 168.3 s), "
        "and mean input tokens per model call from 460 to 3 040. Two failure modes are identified: "
        "retrieved evidence overriding facts stated in the question, and retrieval hits that do "
        "not imply correct attribution. We conclude that externalizing computation and retrieval "
        "substantially improves both data usability and answer verifiability, while authority "
        "ranking of evidence and attribution checking remain unsolved without external mechanisms."
    )
    set_run(p.add_run(text), ascii_font=TIMES, ea=TIMES, size=10.5)

    p = para(doc, "", indent_chars=0, space_after=10, first_line=False)
    set_run(p.add_run("Keywords: "), ascii_font=TIMES, ea=TIMES, size=10.5, bold=True)
    set_run(p.add_run("retrieval-augmented generation; tool calling; large language model; "
                      "pig phenotyping; intelligent breeding; ablation study"),
            ascii_font=TIMES, ea=TIMES, size=10.5)


def section_intro(doc):
    heading(doc, "1  引言", 1)

    heading(doc, "1.1  课题来源与研究背景", 2)
    body(doc, "本研究的工程需求来自课题组承担的广西基础研究计划（“夯基”行动计划）项目"
              "《广西地方品种猪人工智能育种技术研究》。该项目以可见光、热红外、深度等多模态数据"
              "融合人工智能技术，研究猪肌内脂肪含量、背膘厚等表型信息的智能感知算法，研制猪舍"
              "手持式与自走式移动巡检智能机器人；结合身份识别、传感器与物联网技术，研发猪只个体"
              "性状信息的自动采集技术，构建不同生长阶段的动态预测模型；并在此基础上整合基因组、"
              "转录组与蛋白组等多组学数据，挖掘肉质性状的关键功能基因与分子标记，研发基因组遗传"
              "评估新技术与肉质性状基因组筛选新方法。项目设三项研究内容：非侵入式智能测定技术研发"
              "与应用、功能基因与分子标记挖掘鉴定、遗传评估新方法构建与应用，并要求形成面向"
              "数据融合与综合管控的平台能力。")
    body(doc, "表型测定是遗传评估的前置环节，也是整条技术链条中人力投入最密集的一环。传统测定"
              "依赖人工逐头记录与事后汇总，通量低、周期长，且不同测定人员对同一性状的判读存在差异。"
              "项目引入自动采集设备与巡检机器人后，数据规模将显著增长，随之产生新的矛盾："
              "数据变多，并不等于数据变得可用。研究者若要回答“某栏位近一周采食量是否偏低”，"
              "仍需自行定位记录、确定统计口径并完成聚合；若要确认某项方法在文献中的既有结论，"
              "仍需逐篇检索并核对出处。数据与知识同时增长，而获取它们的方式仍停留在人工层面。")
    body(doc, "本课题的前期工作从反面印证了这一判断。以多模态大模型直接识别生猪姿态，"
              "在 309 张样本、三组提示词、600 次调用的条件下，最优配置的准确率为 0.377，"
              "宏平均 F1 为 0.232；其中侧卧类召回率 0.836 而精确率 0.321，坐姿类 F1 为 0，"
              "其他卧姿类 F1 为 0.077。通用视觉模型在细粒度姿态判别上尚未达到可直接部署的水平，"
              "且三组提示词的宏平均 F1 极差不足 0.07，说明在给定模型能力下提示词优化的边际收益"
              "已接近饱和。这一结果决定了本研究的取向：不追求单一模型能力的提升，而是把模型能力"
              "嵌入一条可验证的证据链。")

    heading(doc, "1.2  现实意义与工程定位", 2)
    body(doc, "本研究的意义体现在三个层面。对育种决策而言，可信度优先于流畅度。育种场依据表型"
              "数据作出留种与淘汰判断，一旦数值来源不明，结论再合理也无法使用。系统因此要求每个"
              "数值都能由工具调用复算，每条论断都能指向具体片段，使结论具备被复核的可能。")
    body(doc, "对平台建设而言，本项目所要求的“数据融合与综合管控平台”在完成数据汇聚之后，"
              "普遍面临同一个问题：数据已入库，但业务人员仍不会用。本研究提供的问答式分析入口，"
              "把“定位记录—确定口径—完成计算—核对文献”这一串操作压缩为一次提问，其价值不在于"
              "替代分析人员，而在于把重复性查询的成本降到可以忽略。")
    body(doc, "对资源受限场景而言，项目经费与算力条件决定了方案必须轻量。本研究坚持零新增依赖、"
              "以标准库实现检索与评测、在外部服务异常时自动降级，正是为了让系统能够在普通工作站上"
              "长期运行，而不依赖额外的基础设施。")
    body(doc, "需要明确本文的边界：本文工作定位于表型数据的查询、汇总、图像判读与文献佐证，"
              "即任务书中“数据融合与综合管控平台”的智能分析组件，不涉及功能基因挖掘与基因组遗传"
              "评估本身。表 1 给出本文工作与项目任务书内容的对应关系。")
    three_line_table(
        doc,
        ["任务书内容", "本文的支撑点", "实现位置", "验证方式"],
        [["研究内容（1）：非侵入式智能测定与数据融合平台",
          "为结构化表型记录提供问答式查询、聚合与异常识别",
          "3.3、3.5 节", "数据题 10 题，数值命中 100%"],
         ["研究内容（1）：动态预测与优质猪肉关联模型",
          "以工具完成可复算的均值、极值与分组统计，避免人工誊抄",
          "3.5 节", "期望值由独立脚本复算"],
         ["考核指标：表型多模态智能感知算法模型 1 套",
          "将姿态识别封装为工具，并如实标定其精度边界",
          "3.5 节", "视觉题 5 题，逐图判定"],
         ["平台需支撑文献与方法查阅",
          "20 篇文献双通道检索，引用可溯源至具体片段",
          "3.3、3.4 节", "知识题 10 题，检索命中 100%"],
         ["经费与算力受限（项目经费 30 万元，模型走免费额度）",
          "零新增依赖、词法降级、串行限流，保证受限条件下可用",
          "3.4、3.7 节", "21 次降级均未中断会话"]],
        widths_cm=[3.6, 5.2, 2.2, 3.4],
        caption_text="表 1  本文工作与项目任务书的对应关系",
        align_center=False,
    )

    heading(doc, "1.3  待解决的三个缺陷", 2)
    body(doc, "上述需求可分解为三个可观测的缺陷，且均在本项目的实测中出现。")
    body(doc, "（1）数值计算不可靠。要求模型直接阅读七日记录并给出均值，其输出为 1.69，"
              "而真实值为 1.6257，相对偏差 3.9%。由于该数值将作为异常判定的基准，"
              "误差会在后续推理中被放大。")
    body(doc, "（2）知识边界不可判。模型在前期实验中给出过三个无法检索到的文献标题，"
              "其表述方式与真实引用并无区别。该缺陷的实质并非记忆错误，而是模型缺乏对自身"
              "知识状态的判别能力。")
    body(doc, "（3）结论来源不可溯。回答未指明依据来自哪一条记录或哪一篇文献的哪一段，"
              "使用者只能整体接受或整体拒绝，无法定点复核。")

    heading(doc, "1.4  本文工作", 2)
    body(doc, "本文的设计思路是：把“计算”与“检索”从模型内部迁移到模型外部，并以工程手段保证"
              "每次输出都能回溯到具体证据。主要工作包括以下四项。")
    body(doc, "（1）形式化分析任务的可验证性要求，将可复算性与可归因性定义为两个可检验的性质，"
              "并据此确定系统边界（第 3.1 节）。")
    body(doc, "（2）实现 8 个只读工具与双通道检索知识库。工具参数由 JSON Schema 约束，执行失败时"
              "返回结构化错误而非抛出异常；检索采用字符 n-gram 与稠密向量并行召回，经加权倒数排名"
              "融合排序，并在稠密通道异常时降级（第 3.3—3.5 节）。")
    body(doc, "（3）设计“规划—调用—观察”循环与证据链机制：限制步数、拦截重复调用、步数耗尽时"
              "去除工具定义强制收敛，引用编号由程序按证据首次出现顺序生成（第 3.6 节）。")
    body(doc, "（4）在 30 题测试集上以三种条件开展 90 次受控实验，报告点估计与 95% 置信区间，"
              "并给出两类失效模式的具体案例（第 4—5 节）。")

    heading(doc, "1.5  组织结构", 2)
    body(doc, "第 2 节综述相关工作；第 3 节阐述系统设计；第 4 节说明实验设计；第 5 节报告结果；"
              "第 6 节讨论证据权威性、归因边界与效度威胁；第 7 节总结。")


def section_related(doc):
    heading(doc, "2  相关工作", 1)

    heading(doc, "2.1  视觉表型识别", 2)
    body(doc, "基于视觉的动物表型研究大致沿两条路线展开。第一条是监督式检测与分割，以 YOLO[1] 等"
              "单阶段检测器为代表，通过大规模边界框标注训练专用模型，在特定场景下精度较高，"
              "但标注成本与场景迁移成本同样显著。第二条是视觉—语言预训练，以 CLIP[2] 为代表的"
              "对比学习模型借助图文对获得开放词汇的零样本判别能力，使自然语言描述可直接用于目标检索。"
              "多模态大模型将这一能力扩展至通用问答，但其对细粒度、专业化的类别边界仍不稳定，"
              "这一现象在本课题的前期实验中已得到量化印证。本文据此将视觉能力定位为可被调用的"
              "外部工具，而非系统的决策核心。")

    heading(doc, "2.2  检索增强生成", 2)
    body(doc, "检索增强生成（RAG）由 Lewis 等[3]提出，其基本思想是在生成阶段引入非参数化外部记忆，"
              "使模型输出受检索证据约束。该框架的两个关键组件是检索器与融合策略。稠密检索以双塔"
              "编码器将问句与段落映射至同一向量空间，DPR[4] 是其代表性实现；稀疏检索以 BM25[5] 为"
              "代表，依赖词项匹配与文档长度归一化，在缺乏训练数据的场景中通常更为稳健。"
              "当多路检索并行使用时，如何合并排序成为独立问题：Cormack 等[6]提出的倒数排名融合"
              "仅利用排名而非分数，避免不同检索器分数量纲不一致带来的可比性问题，因而被广泛采用。")
    body(doc, "中文场景对该框架提出了额外要求。中文词边界模糊，同一术语常有多种切分与书写形式，"
              "基于词的检索容易因分词歧义导致漏召回。本文因此在词法通道中采用字符级 n-gram 特征，"
              "以牺牲部分语义粒度换取召回的稳定性。")

    heading(doc, "2.3  工具调用与智能体", 2)
    body(doc, "工具调用使模型能够将符号运算、数据访问与外部服务委托给确定性程序执行。ReAct[7] 将"
              "推理与行动交替组织为“思考—行动—观察”循环，使模型可在获得中间结果后调整策略，"
              "构成当前多数工具调用智能体的基本形态。")
    body(doc, "既有研究多关注工具选择的准确性与调用格式的合法性，而对长循环下的资源约束关注较少。"
              "工程实践中的典型困难包括：工具选择错误、参数类型不符、同一调用被反复重试导致额度耗尽、"
              "以及步数耗尽而未能收敛。本文的设计重点正落在这些约束性问题上，并在第 5 节给出量化结果。")

    heading(doc, "2.4  研究定位", 2)
    body(doc, "与面向开放域问答的通用 RAG 系统相比，本文的约束更为严格：回答中的每个数值须可复算，"
              "每个论断须可归因。为此系统引入结构化工具承担计算，并由程序而非模型生成引用编号。"
              "此外，本文在规模有限的私有语料上验证降级策略，这一取舍更贴近资源受限的实际部署条件。")


def section_design(doc):
    heading(doc, "3  系统设计", 1)

    heading(doc, "3.1  问题形式化", 2)
    body(doc, "设问题为 q，可用工具集合为 T={t₁,…,t₈}，文献知识库为 K，结构化记录为 D。系统输出"
              "二元组 (a, C)，其中 a 为自然语言回答，C 为按出现顺序编号的证据序列。"
              "在此基础上定义两个性质。")
    body(doc, "定义 1（可复算性）  若 a 中出现的任一数值 v 均存在工具调用序列 τ 属于本次运行的调用轨迹，"
              "使得 v 可由 τ 的返回值经确定性运算复现，则称 a 关于数值是可复算的。")
    body(doc, "定义 2（可归因性）  若 a 中每一实质性论断均可映射至 C 中某一片段的陈述，则称 a 是可归因的。")
    body(doc, "本文的设计目标即在保证回答完整性的前提下，尽可能提升上述两个性质。二者共同决定了"
              "系统的边界：模型负责语义理解、证据选择与表述组织，凡涉及数值与外部事实的部分均外化为"
              "工具调用，凡涉及引用的部分均由程序生成。")

    heading(doc, "3.2  体系结构", 2)
    body(doc, "系统分为三层，如图 1 所示。交互层为单页 Web 应用，提供图像识别、智能问答与"
              "提示词查看三个功能区块；智能体层负责提示词构造、多步循环与结果组装；"
              "工具与数据层包含结构化记录、文献知识库、只读工具与视觉模型四类资源。"
              "层间以函数调用与结构化返回值耦合，任一工具可独立替换而不影响其余部分。")
    figure(doc, FIG / "architecture.png", 13.6,
           "图 1  系统三层体系结构与调用关系")
    body(doc, "一次提问的处理路径为：前端提交问题与工具组合；服务端校验参数并拼装系统提示词"
              "（含工具说明书）；模型判断是否需要调用工具；工具执行后以观察结果回填消息序列；"
              "循环直至模型给出最终回答或步数耗尽；程序抽取引用并统一编号；返回回答、引用、"
              "工具轨迹与运行指标。工具轨迹随回答一并返回，为人工复核提供依据。")

    heading(doc, "3.3  知识库构建", 2)
    body(doc, "语料由三部分构成：开放获取期刊论文 13 篇、项目自有文档 6 篇、数据集卡片 1 篇，共 20 篇。"
              "清洗阶段去除导航栏、版权声明与参考文献列表，保留正文及图表标题，正文合计 569 057 个字符。"
              "切分按段落与标题层级进行，目标长度 300—800 字符，相邻片段重叠 80 字符，最终得到 943 个片段，"
              "平均长度 603 字符。")
    body(doc, "为支持后续复用，入库时对每个片段标注许可类型，按片段数统计如表 2 所示。"
              "其中 CC BY-NC 与 CC BY-NC-ND 合计 197 个片段，仅用于非商业研究，不随系统分发原文。")
    three_line_table(
        doc,
        ["许可类型", "片段数", "占比/%"],
        [["CC BY", "697", "73.9"],
         ["CC BY-NC", "131", "13.9"],
         ["CC BY-NC-ND", "66", "7.0"],
         ["项目自有", "39", "4.1"],
         ["CC BY 4.0（数据集卡片）", "10", "1.1"],
         ["合计", "943", "100.0"]],
        widths_cm=[5.6, 2.9, 2.9],
        caption_text="表 2  语料片段构成（按许可类型）",
    )
    body(doc, "向量化采用 2048 维文本嵌入接口，一次性消耗 144 651 个 token。索引以矩阵形式持久化，"
              "未引入向量数据库：在 943 个片段的规模下，一次精确余弦检索的耗时不足 10 ms，"
              "显著低于单次模型调用的数百毫秒，近似最近邻在此规模下不产生收益，反而增加部署负担。")

    heading(doc, "3.4  双通道检索与融合", 2)
    body(doc, "检索流程如图 2 所示。词法通道采用字符级 n-gram 的 TF-IDF 表示：以 char_wb 方式在"
              "词边界内取 2—3 元组，共 28 932 个特征，启用次线性词频抑制以降低高频字串的支配作用，"
              "索引体积 4.2 MB。稠密通道将 943 个片段编码为 943×2048 的矩阵，检索时对归一化问句"
              "向量取点积，等价于余弦相似度。两通道各返回前 20 个候选，按下式融合排序：")
    formula(doc, [
        ("s(r) = Σ", "n"), ("c", "sub"), (" w", "n"), ("c", "sub"),
        (" / (k + rank", "n"), ("c", "sub"), ("(r))", "n"),
    ], 1)
    body(doc, "其中 s(r) 为片段 r 的融合分数，求和遍历稠密与词法两个通道 c，rank 表示该片段在相应"
              "通道中的排名，k 取 60 以抑制头部排名的过度影响，两通道权重分别取 1.0 与 0.5，"
              "词法原始分数低于 0.01 的候选在融合前被剔除。该形式的优点在于仅依赖排名，"
              "无需归一化不同通道的分数。")
    figure(doc, FIG / "retrieval_flow.png", 13.6,
           "图 2  双通道召回、加权融合与降级路径")
    body(doc, "降级策略是本层的工程重点。当稠密通道出现网络超时、配额限制或维度不匹配等异常时，"
              "系统不中断会话，而是退化为纯词法检索，并在返回结果中标注降级标志与原因。"
              "实验期内共发生 21 次降级，无一次导致对话失败，但相关性明显下降，其影响在第 5.6 节量化。")

    heading(doc, "3.5  工具层设计", 2)
    body(doc, "工具在注册表中以四元组描述：名称、自然语言说明、JSON Schema 参数定义与执行函数。"
              "模型所见的正是这份说明书，它决定了工具选择与参数填充的正确性。8 个工具及其在"
              "实验期的调用次数如表 3 所示。")
    three_line_table(
        doc,
        ["序号", "工具名", "功能", "调用/次"],
        [["1", "list_pens", "列出栏位、存栏数、最近均重与摄像头编号", "5"],
         ["2", "query_pig_records", "按栏位、猪只、日期范围查询日记录明细", "6"],
         ["3", "aggregate_metric", "按分组求均值、极值或合计", "12"],
         ["4", "detect_anomalies", "以 z 分数识别异常猪只或栏位", "1"],
         ["5", "list_sample_images", "枚举可用于分析的示例图片", "1"],
         ["6", "analyze_pig_posture", "调用视觉模型识别姿态并返回不确定性", "5"],
         ["7", "search_knowledge", "知识库检索，返回带出处的片段摘要", "107"],
         ["8", "fetch_passage", "按片段编号取回全文以展开引用", "58"],
         ["", "合计", "", "195"]],
        widths_cm=[1.5, 4.0, 7.1, 1.8],
        caption_text="表 3  工具清单与实验期调用分布",
    )
    body(doc, "工具层遵循四条约束。其一，参数校验置于工具侧而非提示词侧，对类型、取值范围与日期格式"
              "逐一检查，非法参数返回失败标志与具体原因，使模型获得修正机会。其二，全部工具只读，"
              "系统不提供写操作，从机制上排除模型构造虚假写入的可能。其三，返回值附带单位与来源，"
              "数据类工具显式标注“模拟数据”，避免结论被误认为生产事实。其四，未知工具名返回明确错误："
              "实验中模型曾调用不存在的 analyze_image，注册表拒绝执行且不产生任何副作用。")
    body(doc, "结构化数据层以固定随机种子生成 4 个栏位、48 头猪、30 天的日记录，并植入三个可检验事件："
              "P3 栏的热应激、P2-007 的采食量下降与 P4-003 的活动量下降。如此设计使异常检测类问题"
              "具有唯一标准答案，也使失败案例可被精确定位。采用模拟数据的原因是真实养殖数据不可获取，"
              "而评测必须依赖已知答案；其表结构与真实系统一致，替换数据源不影响上层逻辑。")

    heading(doc, "3.6  智能体循环与证据链", 2)
    body(doc, "系统提示词包含七条规则，其中五条直接对应已观测到的失败：仅可使用工具返回的事实；"
              "数值必须来自工具而非自行推算；信息不足时应说明缺项；引用由系统统一编号；"
              "不得将不同文献的结论合并为单一论断。每条规则均可追溯至一次具体失败，而非一般性建议。")
    body(doc, "循环结构如图 3 所示。循环默认最多执行 6 步（接口允许的上限为 8 步，"
              "实验统一取 6），每步一次模型调用。模型返回工具调用时执行并"
              "回填观察结果，返回纯文本时循环终止。为避免在同一调用上反复试探，系统记录"
              "“工具名+参数”的指纹，同一指纹执行超过 2 次即拒绝执行，并回传提示要求更换方式。"
              "若 6 步用尽仍未收敛，系统移除工具定义后再行询问，强制其给出阶段性结论并声明局限，"
              "该情形在指标中记为强制收尾。")
    figure(doc, FIG / "agent_loop.png", 12.4,
           "图 3  多步执行循环与两条资源约束")
    body(doc, "证据链由程序维护。回答中允许出现编号占位，最终编号按证据在工具轨迹中首次出现的顺序"
              "重排，且仅接受工具实际返回过的片段。由此，每个编号均可回溯至原文，模型自行生成的"
              "文献条目无法进入引用列表。")
    body(doc, "另需说明一处工程妥协：视觉模型与工具调度模型相分离。实测的免费视觉模型不支持原生"
              "工具调用协议，若令其同时承担调度职责，函数调用将被降级为纯文本而解析失败。"
              "因此视觉模型仅作为被调用方存在，对话与调度由另一模型负责。")

    heading(doc, "3.7  服务集成与工程约束", 2)
    body(doc, "服务端基于 Tornado 实现，暴露首页、健康自检、图像识别与智能问答四个接口。"
              "问答请求在线程池中执行，并以串行锁保证同一时刻仅有一个会话运行，避免配额被并发耗尽。"
              "端口占用时于 7860—7869 范围内顺延，并将实际端口输出至控制台。未配置密钥时，"
              "接口返回明确的阻塞原因，界面不产生任何伪造结果。")
    body(doc, "工程上遵循零新增依赖原则：未引入向量数据库、中文分词库与测试框架，检索、索引与评测"
              "均以标准库及既有科学计算库实现。代码规模为业务代码 27 个文件 6 746 行、脚本 17 个文件"
              "4 299 行、测试 28 个文件 5 348 行，测试用例 493 项全部通过。90 次实验的完整记录"
              "（含工具轨迹、token 用量与耗时）以 JSON 格式随项目留存，指标可由脚本从记录重新计算，"
              "避免手工誊抄引入误差。")


def section_experiment(doc):
    heading(doc, "4  实验设计", 1)

    heading(doc, "4.1  研究问题", 2)
    body(doc, "实验围绕五个问题展开。RQ1：工具调用能否消除数值类回答的错误；RQ2：检索能否使知识类回答"
              "具备可核查的依据；RQ3：视觉能力作为工具被调用时的表现如何；RQ4：对无法回答的问题，"
              "系统是否如实声明；RQ5：上述收益对应的成本为何。")

    heading(doc, "4.2  测试集构造", 2)
    body(doc, "测试集共 30 题，分四类，构成见表 4。数据类 10 题覆盖单猪均值、栏位聚合、极值统计与"
              "异常指认；知识类 10 题覆盖文献中的方法、指标与结论；视觉类 5 题要求判别指定图片的姿态；"
              "拒答类 5 题询问知识库与数据均不涉及的事实。")
    three_line_table(
        doc,
        ["类别", "题号", "题量", "考查点", "判定依据"],
        [["数据查询", "D1—D10", "10", "数值计算、异常指认、事实边界", "数值容差匹配与结论方向"],
         ["知识问答", "K1—K10", "10", "文献检索与归纳", "引用片段是否属于期望文档集合"],
         ["视觉分析", "V1—V5", "5", "姿态识别工具的调用与判读", "标签完全一致"],
         ["无法回答", "U1—U5", "5", "诚实拒答", "是否声明信息缺失且未编造"],
         ["合计", "—", "30", "—", "—"]],
        widths_cm=[1.9, 2.3, 1.2, 4.2, 4.8],
        caption_text="表 4  测试集构成",
    )
    body(doc, "题目构造遵循三条原则。其一，答案唯一：标准答案均可由数据或文献确定，不依赖主观判断。"
              "其二，防泄漏：视觉题图片在题面中以中性文件名 v1—v5 出现，文件名与目录均不含类别信息，"
              "以避免模型由路径推断答案——该缺陷在预实验中确实发生，导致 15 次视觉运行作废并重跑。"
              "其三，可复核：数据题期望值由独立脚本从原始记录计算，而非取自工具输出，否则构成自证。")
    body(doc, "数值类判定共涉及 19 个期望数值，其中 D6 的 2 项不计入汇总：该题题面已给出两个数字，"
              "回答仅需复述，不能反映计算能力。因此参与汇总的数值项为 17 个。")

    heading(doc, "4.3  实验条件", 2)
    body(doc, "三种条件仅改变可用工具集合，其余设置完全一致，见表 5。direct 为基线，模型仅依据题面"
              "与参数化记忆作答；rag 仅开放检索类工具，用于隔离检索贡献；full 开放全部 8 个工具，"
              "即完整系统。")
    three_line_table(
        doc,
        ["条件", "可用工具", "设计意图"],
        [["direct", "无", "基线：考察模型独立作答能力"],
         ["rag", "search_knowledge、fetch_passage", "隔离检索贡献，排除计算工具"],
         ["full", "全部 8 个工具", "完整系统"]],
        widths_cm=[2.2, 6.2, 5.9],
        caption_text="表 5  三种实验条件",
    )
    body(doc, "每个条件执行 30 题，合计 90 次会话。模型温度设为 0，最大输出 2048 token，单次超时 120 s，"
              "失败重试上限 8 次，请求速率限制为 0.5 次每秒。全部运行于 2026 年 11 月完成，累计 52.1 分钟，"
              "无一次失败。")

    heading(doc, "4.4  指标定义与统计方法", 2)
    body(doc, "记某数据题的期望数值集合为 V，回答中出现的数值集合为 V(a)，容差函数 τ(v) 取期望值的 1.5%，"
              "小数值另设下限，计数类取 ±0.5。单个数值的命中指示函数为：")
    formula(doc, "h(v) = 1[∃ v̂ ∈ V(a) : |v̂ − v| ≤ τ(v)]", 2)
    body(doc, "数值命中率定义为 V 上指示函数的均值，匹配采用“最近未用”配对，以避免同一数值被重复计入：")
    formula(doc, [
        ("Acc", "n"), ("num", "sub"), (" = (1/|V|)·Σ", "n"), ("v∈V", "sub"), (" h(v)", "n"),
    ], 3)
    body(doc, "整题正确率要求该题全部期望数值命中且结论方向正确。记知识题集合为 Q，题意对应的期望文档"
              "集合为 D(q)，回答引用集合为 C(q)，命中题数 m 为满足 C(q) 与 D(q) 交集非空的题目个数，"
              "则检索命中率为：")
    formula(doc, "RH = m / |Q|", 4)
    body(doc, "诚实拒答率要求回答明确声明信息缺失，且未给出具体数值或结论。降级率为运行期间发生"
              "词法兜底的比例。成本以模型调用次数、工具调用次数、输入输出 token 与墙钟耗时计量。")
    body(doc, "考虑到每个条件仅执行一次、样本量为 30，本文对所有比率同时报告 Clopper-Pearson 精确"
              "二项置信区间（95%），以明确点估计的不确定程度，而不将“100%”理解为确定性结论。")


def section_results(doc):
    heading(doc, "5  结果与分析", 1)

    heading(doc, "5.1  总体结果", 2)
    body(doc, "三种条件的核心指标见表 6，任务质量对比见图 4。")
    three_line_table(
        doc,
        ["指标", "direct", "rag", "full"],
        [["完成率/%", "100", "100", "100"],
         ["平均模型调用/次", "1.00", "4.77", "3.73"],
         ["平均工具调用/次", "0.00", "3.77", "2.73"],
         ["平均耗时/s", "15.6", "39.7", "49.0"],
         ["95 分位耗时/s", "40.1", "110.4", "168.3"],
         ["平均引用数/条", "0.00", "8.77", "3.10"],
         ["数据题·数值命中率/%", "0", "0", "100"],
         ["数据题·整题正确率/%", "0", "0", "100"],
         ["数据题·异常指认率/%", "0", "0", "100"],
         ["知识题·检索命中率/%", "0", "100", "100"],
         ["视觉题·工具调用率/%", "0", "0", "100"],
         ["视觉题·姿态命中率/%", "—", "—", "60"],
         ["拒答题·诚实拒答率/%", "80", "100", "100"],
         ["检索降级率/%", "0", "53.3", "16.7"]],
        widths_cm=[5.9, 2.8, 2.8, 2.8],
        caption_text="表 6  三种条件下的主要指标",
    )
    figure(doc, FIG / "agent_task_metrics.png", 13.6,
           "图 4  三个条件下的任务完成质量对比（越高越好）")

    heading(doc, "5.2  RQ1：数值计算的可靠性", 2)
    body(doc, "纯对话条件下，17 个参与汇总的期望数值无一命中，整题正确率为 0（95% CI：0—11.6%）。"
              "该结果与预期一致：模型无法访问原始记录，只能依据题面出现的少量数字推测趋势。"
              "典型错误为将七条记录的均值表述为 1.69，而真值为 1.6257。")
    body(doc, "接入工具后，数值命中率、整题正确率与异常指认率同时达到 100%（88.4%—100%）。"
              "逐题结果如图 5 所示：full 条件下数据题全部命中，rag 与 direct 则整体落空。"
              "值得注意的是，后两者的失败呈系统性而非随机分布——凡要求对多行记录做聚合的题目"
              "均无法通过。这说明模型的局限并非运算能力不足，而是缺乏调用确定性计算的机制；"
              "当工具可用时，模型倾向于将算术外化。")
    figure(doc, FIG / "agent_data_heatmap.png", 14.2,
           "图 5  数据题逐题数值命中情况（行：条件；列：题目）")
    body(doc, "一处反例需单独说明。D6 题面自带两个数字，direct 条件因复述而命中率达 100%，"
              "rag 条件反而降为 0。原因是检索将前期文档中“2 个猪栏、8 台相机”的描述引入上下文，"
              "模型据此覆盖了题面事实。这一现象既表明检索并非越多越好，也说明本题不应计入数值能力汇总，"
              "故其 2 个数值项被排除（详见第 6.1 节）。")

    heading(doc, "5.3  RQ2：知识检索的有效性", 2)
    body(doc, "知识题检索命中率由 0 升至 100%，平均引用数由 0 增至 8.77（rag）与 3.10（full）。"
              "full 的引用数低于 rag，原因有二：其一，full 具备任务规划能力，平均步数与工具调用更少；"
              "其二，部分知识题可由数据工具间接回答，检索需求相应下降。引用数并非越多越好，"
              "关键在于每条引用是否被实际使用。")
    body(doc, "指标的高值同时掩盖了部分问题。以 K9 为例，期望文献确实出现在引用集合中，"
              "但回答的核心论断由相邻文献支撑。按第 4.4 节的定义该题记为命中，但从归因角度看并不严格。"
              "这说明检索命中只是可归因性的必要条件，自动指标无法完全替代人工核查，相关讨论见第 6.2 节。")

    heading(doc, "5.4  RQ3：视觉工具的调用与判读", 2)
    body(doc, "视觉题的工具调用率为 100%：full 条件下模型均调用了姿态识别工具，未出现仅凭文字推测的情形。"
              "姿态标签命中率为 60%（14.7%—94.7%），逐图结果见表 7。")
    three_line_table(
        doc,
        ["编号", "期望姿态", "系统输出", "判定"],
        [["V1", "侧卧（lying_lateral）", "侧卧", "一致"],
         ["V2", "站立（standing）", "站立", "一致"],
         ["V3", "坐（sitting）", "侧卧", "不一致"],
         ["V4", "其他卧姿（lying_other）", "侧卧", "不一致"],
         ["V5", "站立（standing）", "站立", "一致"]],
        widths_cm=[1.8, 5.8, 4.0, 2.7],
        caption_text="表 7  视觉题逐图判定结果（full 条件）",
    )
    body(doc, "两处错误指向同一倾向：模型将“坐”与“其他卧姿”均归为侧卧，与前期实验的混淆方向一致——"
              "侧卧类召回高而精确率低，边界类别被大量吸收。更值得注意的是稳定性：同一样本在两次独立运行中"
              "给出不同标签（V1 曾在另一次运行中被判为“其他卧姿”），而温度已设为 0。"
              "这表明该视觉模型的输出并非确定，视觉结论只能作为参考证据。需要指出，"
              "60% 的置信区间宽度达 0.80，样本量为 5 时该点估计的推断价值有限。")

    heading(doc, "5.5  RQ4：拒答行为的诚实性", 2)
    body(doc, "三种条件的拒答率分别为 80%（28.4%—99.5%）、100% 与 100%。direct 条件下的失败案例为 U4："
              "模型为不存在的 P5 栏位编造了完整报表，包含 2.35 kg/头/天的日均采食、164.5 kg 的总采食量，"
              "甚至给出“11 月 2 日最高 2.52”等细节。full 条件面对同一问题时先调用栏位枚举工具，"
              "得到 P1—P4 的列表，遂声明“系统中没有 P5 栏位”，并给出可用栏位的均值作为替代信息。")
    body(doc, "该对比说明工具的作用并非提升模型的诚实程度，而是降低诚实行为的成本："
              "查询一次即可确证不存在，编造反而需要更多步骤。")

    heading(doc, "5.6  RQ5：成本与时延", 2)
    body(doc, "收益伴随明确代价，见表 8 与图 6。平均时延由 15.6 s 增至 49.0 s，95 分位达 168.3 s，"
              "单次最长 316 s（full 条件 K4 题，因检索多次重试）。输入 token 总量由 13 809 增至 340 530，"
              "而单次模型调用的平均输入 token 由 460 增至 3 040，即上下文随工具结果累积而膨胀。")
    three_line_table(
        doc,
        ["条件", "模型调用", "工具调用", "耗时/min", "输入 token", "输出 token", "降级次数"],
        [["direct", "30", "0", "7.8", "13 809", "4 824", "0"],
         ["rag", "143", "113", "19.8", "468 254", "10 668", "16"],
         ["full", "112", "82", "24.5", "340 530", "10 200", "5"],
         ["合计", "285", "195", "52.1", "822 593", "25 692", "21"]],
        widths_cm=[1.6, 2.0, 2.0, 2.0, 2.4, 2.3, 2.0],
        caption_text="表 8  三种条件的成本构成",
    )
    figure(doc, FIG / "agent_cost.png", 14.2, "图 6  调用开销与单次对话耗时")
    body(doc, "降级率呈现与直觉相反的结果：rag 为 53.3%（16 次），full 仅 16.7%（5 次）。"
              "原因不在于 full 更稳定，而在于其检索调用更少——工具越多，模型越少依赖文献检索。"
              "此外降级事件集中于同一时段，与接口限流的分布吻合，说明降级多由外部服务波动引起。")

    heading(doc, "5.7  典型案例", 2)
    body(doc, "从 90 次运行中选取五个具有代表性的片段，见表 9。案例既包含成功路径，也包含失败路径，"
              "其价值在于揭示指标背后的具体机制。")
    three_line_table(
        doc,
        ["编号", "条件", "观测到的行为", "结果"],
        [["C1", "direct", "为不存在的 P5 栏给出 2.35 kg/头/天等 5 项细节", "未拒答"],
         ["C2", "full", "栏位枚举返回 P1—P4，声明无 P5 并给出可用范围均值", "拒答并给替代"],
         ["C3", "full", "首次调用携带非法参数被拒，据错误信息修正后重试成功", "26.59 ℃"],
         ["C4", "rag", "检索证据覆盖题面给定事实", "答“2 个栏位”"],
         ["C5", "full", "将“坐”判为“侧卧”，且表述确定", "标签错误"]],
        widths_cm=[1.5, 1.8, 8.3, 2.8],
        caption_text="表 9  典型运行片段",
    )
    body(doc, "C3 与其余案例性质不同，它证明的是错误回填机制的有效性：工具返回“不支持的参数 pen_id”后，"
              "模型在下一步丢弃该字段并补充正确参数，最终得到 26.59 ℃ 的正确结果。"
              "若工具以异常形式中断会话，这一自我修正便无从发生。C4 与 C5 则分别对应第 6.1 与 6.2 节"
              "讨论的两类失效。")


def section_discussion(doc):
    heading(doc, "6  讨论", 1)

    heading(doc, "6.1  证据权威性：题面与检索的优先级", 2)
    body(doc, "D6 的结果揭示了一个在 RAG 系统设计中常被忽略的问题：当检索证据与题面陈述冲突时，"
              "应以何者为准。rag 条件从前期文档中检索到“2 个猪栏、8 台相机”，与题面给定的两个栏位、"
              "八台相机恰好同义，但系统实际有四个栏位，模型据此给出了错误来源的正确答案。"
              "full 条件因先行调用栏位枚举工具而未受影响。")
    body(doc, "本文的处理是将该题从数值能力汇总中剔除，并在提示词中要求不得以检索内容覆盖题面事实。"
              "更彻底的方案是为证据引入权威等级（题面 > 工具实时返回 > 历史文档），由程序而非模型"
              "决定取舍，这属于后续工作。")

    heading(doc, "6.2  归因准确性：指标的可操作化边界", 2)
    body(doc, "K9 暴露的是指标定义层面的局限。以“引用是否落在期望文档集合”判定命中，"
              "无法区分“引用了”与“依据了”。更严格的做法是逐句核对论断与引用片段的蕴含关系，"
              "但该判定需要人工标注或更强的判别模型，成本显著。本文选择如实标注这一局限，"
              "而非放宽指标以提升表观数值。")

    heading(doc, "6.3  工具选择偏差与循环收敛", 2)
    body(doc, "工具调用分布显示，检索类工具占 165 次，计算类仅 19 次，说明模型的默认倾向是“查”而非“算”。"
              "在纯数据题上该倾向被系统提示词纠正，但在混合型问题上仍出现先检索后计算的顺序倒置，"
              "平均多消耗一至两步。这构成了将单智能体拆分为“规划者—执行者”的动机。")
    body(doc, "循环收敛性同样值得关注。full 条件的强制收尾为 5 次，rag 为 16 次。rag 反复检索却"
              "难以获得可用信息，最终触及步数上限，说明在步数受限的场景下，工具集设计应优先保证"
              "“能够结束”，而非“尽可能查全”。")

    heading(doc, "6.4  效度威胁", 2)
    body(doc, "内部效度方面，主要风险来自评测口径。本文识别的三处风险均已修正：视觉题标签泄漏通过"
              "统一使用中性文件名并重跑 15 次视觉会话消除；判定口径通过剔除题面已出现的数值修正；"
              "数值容差经三档试验后定为 1.5%，使“计算错误”与“合理舍入”可被区分。")
    body(doc, "外部效度方面，结论受限于单一模型族、模拟数据与 20 篇语料。真实养殖场景中的缺失值、"
              "传感器漂移与多源冲突未被覆盖，扩大语料后检索结论的普适性亦有待验证。")
    body(doc, "构念效度方面，“正确”被操作化为“数值在容差内命中且结论方向一致”，该定义忽略了解释质量、"
              "论证完整性与可读性；因而本文报告的提升应理解为可验证性的提升，而非回答质量的全面提升。")
    body(doc, "统计结论效度方面，每一条件仅执行一次，样本量为 30，置信区间较宽。以视觉题为例，"
              "多命中一题即可使命中率由 60% 升至 80%，故该类小样本结论只能作为方向性证据。")

    heading(doc, "6.5  研究局限", 2)
    body(doc, "除上述效度问题外，本研究尚存四点局限。其一，系统不支持多轮对话，每题独立成会话，"
              "追问与上下文消解场景未被评估。其二，视觉模型与调度模型分离，视觉能力的上限即系统上限。"
              "其三，检索未引入重排模型，融合后仍依赖排名启发式。其四，异常检测采用固定 z 分数阈值，"
              "对季节性波动不具适应性。")


def section_conclusion(doc):
    heading(doc, "7  结论", 1)
    body(doc, "本文面向广西地方品种猪人工智能育种中的表型数据利用问题，设计并实现了一个以检索增强"
              "生成与工具调用为核心的分析助手，并通过 30 题、三种条件、90 次运行的受控实验检验其"
              "有效性。结论可归纳为三点。")
    body(doc, "第一，将计算外化为工具是消除数值错误的决定性手段。数值命中率由 0% 提升至 100%，"
              "且该提升不依赖更强的模型，仅依赖“不让模型自行计算”这一设计选择。")
    body(doc, "第二，检索使知识类回答具备可核查依据，但检索质量与归因准确性是两个独立问题。"
              "系统可将检索命中率提升至 100%，却无法保证被引用的片段正是论断所依据的片段。")
    body(doc, "第三，可验证性的工程投入产出比高于单纯的模型能力提升。使每个数值来自工具、"
              "每个论断指向片段、每次调用留有轨迹，其成本远低于模型微调，而使用者由此获得复核的可能，"
              "这正是育种决策场景的基本要求。")
    body(doc, "后续工作将沿四个方向推进：引入多轮对话与追问；以真实养殖数据替换模拟数据并评估分布漂移；"
              "在扩大语料的条件下比较不同融合与重排策略；以及引入人工评审，建立更贴近实际使用需求的"
              "评价体系。")


REFS = [
    "REDMON J, DIVVALA S, GIRSHICK R, et al. You only look once: unified, real-time object "
    "detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. "
    "2016: 779-788.",
    "RADFORD A, KIM J W, HALLACY C, et al. Learning transferable visual models from natural "
    "language supervision[C]//International Conference on Machine Learning. 2021: 8748-8763.",
    "LEWIS P, PEREZ E, PIKTUS A, et al. Retrieval-augmented generation for knowledge-intensive NLP "
    "tasks[C]//Advances in Neural Information Processing Systems. 2020: 9459-9474.",
    "KARPUKHIN V, OGUZ B, MIN S, et al. Dense passage retrieval for open-domain question "
    "answering[C]//Proceedings of the 2020 Conference on Empirical Methods in Natural Language "
    "Processing. 2020: 6769-6781.",
    "ROBERTSON S, ZARAGOZA H. The probabilistic relevance framework: BM25 and beyond[J]. "
    "Foundations and Trends in Information Retrieval, 2009, 3(4): 333-389.",
    "CORMACK G V, CLARKE C L A, BUETTCHER S. Reciprocal rank fusion outperforms Condorcet and "
    "individual rank learning methods[C]//Proceedings of the 32nd International ACM SIGIR "
    "Conference. 2009: 758-759.",
    "YAO S, ZHAO J, YU D, et al. ReAct: synergizing reasoning and acting in language models[C]//"
    "International Conference on Learning Representations. 2023.",
]


def references(doc):
    heading(doc, "参考文献", 1)
    for index, text in enumerate(REFS, start=1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.2
        pf.left_indent = Pt(21)
        pf.first_line_indent = Pt(-21)
        set_run(p.add_run("[%d]  " % index), ea=SONG, size=10.5)
        set_run(p.add_run(text), ea=SONG, size=10.5)


def main():
    doc = docx.Document(str(BUILD / "cover.docx"))
    fill_cover(doc)

    section = doc.add_section(WD_SECTION.NEW_PAGE)
    clear_footer(doc.sections[0])
    add_page_number_footer(section, start_at=1)
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)

    front_matter(doc)
    english_abstract(doc)
    section_intro(doc)
    section_related(doc)
    section_design(doc)
    section_experiment(doc)
    section_results(doc)
    section_discussion(doc)
    section_conclusion(doc)
    references(doc)

    doc.save(str(OUT))
    print("已生成：%s" % OUT)
    print("大小：%.1f KB" % (OUT.stat().st_size / 1024))


if __name__ == "__main__":
    main()