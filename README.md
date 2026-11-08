# 猪只表型智能分析助手

**基于检索增强生成（RAG）与工具调用的猪只表型问答系统**，构建在一个已完成的视觉姿态识别系统之上。

上传一张图 → 识别姿态；提一个问题 → 查数据、查文献、看图，并给出带来源的回答。

> 本项目**不训练任何模型**，全部推理通过云端 API 完成。
> 养殖数据为**模拟数据**，视觉识别结论**不构成**疾病诊断或体尺、背膘、肌内脂肪的估计。

---

## 0. 为什么需要一层"助手"

第一阶段用 600 次真实调用验证了视觉大模型能否识别猪只姿态，结论是：**最好的提示词方案准确率 37.7%**，仅略高于四分类随机水平 25%（详见[实验报告](docs/experiment_report.md)）。

也就是说，单靠视觉模型回答不了养殖场的实际问题：

| 用户会问 | 视觉模型能答吗 | 助手怎么答 |
|---|---|---|
| "P2-007 最近一周采食量怎么样？" | ❌ 参数里没有这头猪 | 调聚合工具，给出 1.63 kg/天 |
| "姿态识别常用哪些深度学习方法？" | ❌ 会编文献标题 | 检索本地 943 个片段，附 13 条真实引用 |
| "这张图是什么姿态？" | ⚠️ 能看，但只有 37.7% 准确率 | 调视觉工具，转述判断并标注"模型判断" |
| "P5 栏呢？"（栏位不存在） | ❌ 会编出一份报表 | "系统中没有 P5 栏位，可用的是 P1–P4" |

所以本项目把视觉识别降级为 8 个工具中的一个，在其上加检索与工具调用。**关键是让模型"知道自己不知道"**——而这个"知道"，来自工具能给出可验证的否定答案，不来自提示词祈祷。

---

## 1. 能力速览

四个场景，都是实测通过的：

**查数**（`full` 条件，数据题数值命中率 100%）

> 问：P2-007 最近 7 天的平均采食量是多少？
> 答：1.63 kg/头/天（期望 1.6257，容差内命中）。回答附 7 天明细与"数据为模拟"标注。

**查文献**（检索命中率 100%）

> 问：猪只姿态识别常用的深度学习方法有哪些？请给出出处。
> 答：按方法分类陈述，每条附文献标题、章节、许可与链接，编号 [1]…[13] 可对回原文片段。

**看图**（调用率 100%）

> 问：这张图片里的猪是什么姿态？`data/experiment/images/v1.jpg`
> 答：侧卧（lying_lateral），附描述与"模型判断，非人工标注"。

**该拒答就拒答**（`rag`/`full` 拒答率 100%）

> 问：P5 栏最近一周的采食量怎么样？
> 答：抱歉，系统中没有 P5 栏位。目前可用的栏位有：P1、P2、P3、P4。

同一个问题，**不给工具**时模型的回答是（真实记录）：

> 根据模拟数据集，P5 栏最近一周的采食量情况如下：平均采食量 2.35 kg/头/天，总采食量 164.5 kg，最高的一天 2026-11-02 为 2.52 kg/头/天……

一个不存在的栏位，被编出了带日期的完整报表。这个对比就是整个项目的动机。

---

## 2. 实验结果速览

**三条件消融**：90 次对话、285 次模型调用、195 次工具调用、82 万输入 token、52 分钟、**0 失败**。

| 指标 | 无工具 | 仅检索 | 全部工具 |
|---|---:|---:|---:|
| 数据题·数值命中率 | 0.0% | 0.0% | **100%** |
| 知识题·检索命中率 | 0.0% | **100%** | **100%** |
| 视觉题·姿态命中率 | — | — | 60%（3/5） |
| 拒答题·诚实拒答率 | 80% | **100%** | **100%** |
| 平均模型调用 / 耗时 | 1.00 / 15.6 s | 4.77 / 39.7 s | 3.73 / 49.0 s |

**一句话**：工具让数据题从 0% 到 100%，检索让知识题从 0% 到 100%，代价是 P95 延迟从 40 秒升到 168 秒。

详细分析（含 8 个失败案例、3 处指标修正、有效性威胁）见[实验报告](docs/experiment_report_agent.md)。

---

## 3. 快速开始

### 3.1 环境

本项目在当前机器上**零安装**运行。使用已存在的 conda 环境：

```bat
C:\Users\Programs\Miniconda3\envs\data_science\python.exe
```

Python 3.9，依赖（requests、tornado、numpy、scikit-learn、Pillow、matplotlib）均已预装。

> 项目期间唯一新增安装的包是 `pymupdf==1.24.14`，但最终语料全部是 HTML 与 Markdown，**它从未被调用**。项目收尾时已卸载，当前环境零新增依赖。

### 3.2 配置密钥

```bat
copy .env.example .env
```

```ini
# 智谱开放平台（对话、视觉、embedding 共用一个密钥）
ZHIPUAI_API_KEY=你的密钥
PIG_PROVIDER=zhipu
PIG_MODEL=glm-4.1v-thinking-flash     # 视觉：实测比 glm-4.6v-flash 稳定
PIG_TOOL_MODEL=glm-4.5-flash          # 工具调用
```

> 视觉模型实测提示：`glm-4.6v-flash` 在当前离线时段限流严重，`glm-4.1v-thinking-flash` 稳定得多。
> 工具调用模型需要支持原生 `tool_calls`——`glm-4.6v-flash` 不支持，故单独配置。

密钥缺失时，界面与接口会**明确说明阻塞原因**，不会显示任何假的识别或回答结果。

### 3.3 启动

```bat
python app.py
```

浏览器打开 <http://127.0.0.1:7860>，四个区块：

1. **服务自检** —— 解释器、服务商、模型、密钥状态、提示词指纹
2. **图像识别** —— 上传图片 → 选提示词方案 → 看结构化结果
3. **智能助手** —— 提问，可选工具组合（全部 / 仅数据 / 仅知识库 / 不用工具）
4. **提示词查看** —— 三组提示词全文与哈希

助手区"工具组合"下拉与实验的三个条件一一对应，可现场演示消融。

---

## 4. 命令行使用

```bat
REM ① 构建知识库（抓取 → 分块 → 词法索引 → 向量索引，带状态检查）
python scripts\build_kb.py

REM 　 只看状态，不重建
python scripts\build_kb.py --status

REM ② 构建实验题集（含自校验断言，答不出标准答案就失败）
python scripts\build_questions.py

REM ③ 分层冒烟（离线工具 → 真实 Agent → HTTP 端到端）
python scripts\smoke_tools.py
python scripts\smoke_agent.py --dry-run
python scripts\smoke_web.py

REM ④ 跑对照实验（90 次对话，逐题落盘，可断点续跑）
python scripts\run_agent_experiment.py --sleep 0.5
python scripts\run_agent_experiment.py --conditions full --ids D1,V1 --force

REM ⑤ 指标与图表
python scripts\evaluate_agent.py
python scripts\plot_agent.py
```

产物：

| 路径 | 内容 |
|---|---|
| `data/kb/sources.json` | 20 篇文献的来源、许可、链接 |
| `data/kb/chunks.jsonl` | 943 个片段（含章节、字符区间） |
| `data/kb/dense_index.npz` | 943 × 2048 向量矩阵 |
| `data/experiment/agent_questions.jsonl` | 30 题与标准答案 |
| `results/agent/runs/<cond>/<qid>.json` | 单次对话完整记录（回答、工具轨迹、引用、token、提示词指纹） |
| `results/metrics/` | 指标表、逐题明细、人工抽查清单 |
| `results/figures/agent_*.png` | 三张图 |

---

## 5. 项目结构

```
├─ app.py                       Web 入口
├─ config.py                    配置与 .env 解析
├─ PROJECT_REQUIREMENTS.md      需求说明书
├─ src/
│  ├─ vision_client.py          视觉客户端（阶段一，复用）
│  ├─ output_parser.py          JSON 抽取（围栏/修补/正则降级）
│  ├─ schema.py  evaluation.py  reporting.py   阶段一的 Schema/指标/图表
│  ├─ webapp.py                 tornado 单页：上传识别 + 助手对话
│  ├─ rag/                      corpus 语料 · dense 向量 · lexical 词法 · retrieve 融合
│  ├─ tools/                    registry 注册表 · mock_data 模拟数据
│  │                            data_tools 4 个 · vision_tool 2 个 · rag_tool 2 个
│  └─ agent/                    llm_client 客户端 · planner 提示词
│                               executor 循环 · report 组装
├─ scripts/                     构建、实验、评测、绘图、冒烟
├─ tests/                       478 项单元测试
├─ data/                        samples 样本 · kb 知识库 · experiment 题集
├─ results/                     agent 运行记录 · metrics 指标 · figures 图表
└─ docs/                        technical_report_agent · experiment_report_agent
                                （另有阶段一的可行性、数据卡、技术/实验报告、局限）
```

8 个工具：`list_pens`、`query_pig_records`、`aggregate_metric`、`detect_anomalies`、`analyze_pig_posture`、`list_sample_images`、`search_knowledge`、`fetch_passage`。

---

## 6. 常见问题

**Q：为什么不用 Gradio？**
`gradio==4.44.1` 要求 `pillow<11`，会把现有 pillow 降级，影响同一环境下的其它项目。零安装约束下改用 tornado（随 Jupyter 已装）+ 原生 JS 单页。

**Q：为什么不用 Chroma 做向量库？**
943 个片段用 numpy 全量点积检索约 1 ms，精确且无需近似索引，也省去一个重依赖。

**Q：为什么用 unittest 而不是 pytest？**
pytest 未安装。`unittest` 是标准库，覆盖等价：
`python -m unittest discover -s tests -t .`

**Q：为什么工具调用和视觉用两个模型？**
视觉模型 `glm-4.1v-thinking-flash` 不支持原生 `tool_calls`；`glm-4.5-flash` 支持且稳定。两者分工明确，互不干扰。

**Q：为什么不做多智能体？**
任务边界清晰（查数/查文献/看图），单循环已覆盖；多智能体在免费模型上延迟翻倍，且让轨迹难以复核。

**Q：界面提示未配置密钥怎么办？**
把密钥写入项目根目录 `.env`（参考 `.env.example`），重启 `python app.py`。

**Q：图片会被发到哪里？**
发送至 `PIG_PROVIDER` 指定的服务商（默认智谱开放平台）。请仅使用许可允许的公开图片。

**Q：一次提问最长要等多久？**
实测 P95 为 110 秒（仅检索）/ 168 秒（全部工具），最长一次观测到 **316 秒**（知识题，多轮检索）。界面已提示"最长可能要几分钟"。

---

## 7. 已知限制

| 限制 | 依据 | 影响 |
|---|---|---|
| 视觉识别准确率低 | 阶段一 200 样本 37.7%，坐姿 F1 = 0 | 视觉工具只作参考 |
| 同一图两次结论不同 | 温度 0 仍出现分歧 | 单次结果不可复现 |
| 检索语料仅 20 篇 | 抓取阶段只有 PMC 稳定可获取 | 超范围只能拒答 |
| 词法兜底相关性差 | 21 次降级，中文查询常召回无关文档 | 降级只保证不崩 |
| 检索命中≠归因准确 | K9 期望文献被引用，内容却来自相邻文献 | 自动指标不能替代人工核查 |
| 缺能力时会死循环 | `rag` 16/30 次撞步数上限 | 延迟与 token 浪费 |
| 单进程并发 | 助手同时只处理一个提问 | 多人使用需队列 |
| 无多轮对话 | 每题独立会话 | 无法评估追问场景 |

完整的 15 条限制与对策见 [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md) 第十五章。

---

## 8. 文档

| 文档 | 内容 |
|---|---|
| [PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md) | 需求说明书：架构、模块、实验设计、验收、限制 |
| [docs/technical_report_agent.md](docs/technical_report_agent.md) | 技术报告：知识库构建、工具层、Agent、工程决策、缺陷清单 |
| [docs/experiment_report_agent.md](docs/experiment_report_agent.md) | 实验报告：90 次对照实验、指标、8 个案例、有效性威胁 |
| [docs/experiment_report.md](docs/experiment_report.md) | 阶段一：提示词对姿态识别的影响（600 次调用） |
| [docs/technical_report.md](docs/technical_report.md) | 阶段一：技术报告 |
| [docs/feasibility_report.md](docs/feasibility_report.md) | 阶段一：可行性与 API 核查 |
| [docs/data_card.md](docs/data_card.md) | 数据集来源、许可、标签映射 |
| [docs/limitations.md](docs/limitations.md) | 阶段一的局限与适用边界 |