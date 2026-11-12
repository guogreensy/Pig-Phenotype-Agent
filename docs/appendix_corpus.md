# 附录 A  检索语料清单

本文件是结课报告附录 A 的完整版：报告正文只保留按类别的汇总表，逐篇明细（标题、许可、片段数、链接）列于此，便于核对报告中每一条引用的出处。

由 `scripts/export_appendices.py` 从 `data/kb/sources.json` 与 `data/kb/chunks.jsonl` 生成，与实验所用索引一致。

- 语料总数：20 篇
- 片段总数：943 个（平均 603 字符）
- 片段字符合计：568555（相邻片段重叠部分已去除）
- 片段编号形如 `pmc_7070994#0053`，由 `doc_id` 与段内序号构成，可回溯至下文对应行

| 序号 | 编号 | 类别 | 标题 | 许可 | 片段数 | 字符数 | 链接 |
|---:|---|---|---|---|---:|---:|---|
| 1 | `pmc_11013991` | 期刊论文（PMC 开放获取） | Pattern Mining-Based Pig Behavior Analysis for Health and Welfare Monitoring | CC BY | 110 | 65795 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC11013991/) |
| 2 | `pmc_11252330` | 期刊论文（PMC 开放获取） | Deep learning pose detection model for sow locomotion | CC BY | 76 | 43328 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC11252330/) |
| 3 | `pmc_11635830` | 期刊论文（PMC 开放获取） | Determining the posture and location of pigs using an object detection model under different lighting conditions | CC BY-NC-ND | 66 | 36726 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC11635830/) |
| 4 | `pmc_11833201` | 期刊论文（PMC 开放获取） | RGB-based machine vision for enhanced pig disease symptoms monitoring and health management: a review | CC BY-NC | 89 | 55447 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC11833201/) |
| 5 | `pmc_5110645` | 期刊论文（PMC 开放获取） | Early detection of health and welfare compromises through automated detection of behavioural changes in pigs | CC BY | 57 | 34907 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC5110645/) |
| 6 | `pmc_5730557` | 期刊论文（PMC 开放获取） | Automated tracking to measure behavioural changes in pigs for health and welfare monitoring | CC BY | 75 | 43587 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC5730557/) |
| 7 | `pmc_6739619` | 期刊论文（PMC 开放获取） | Design and evaluation of an imager for assessing wound inflammatory responses and bioburden in a pig model | CC BY | 56 | 32932 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC6739619/) |
| 8 | `pmc_6749226` | 期刊论文（PMC 开放获取） | Deep Learning and Machine Vision Approaches for Posture Detection of Individual Pigs | CC BY | 45 | 27235 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC6749226/) |
| 9 | `pmc_7070994` | 期刊论文（PMC 开放获取） | Automated Video Behavior Recognition of Pigs Using Two-Stream Convolutional Networks | CC BY | 58 | 34593 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC7070994/) |
| 10 | `pmc_7423952` | 期刊论文（PMC 开放获取） | Automated recognition of postures and drinking behaviour for the detection of compromised health in pigs | CC BY | 93 | 58573 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC7423952/) |
| 11 | `pmc_8071751` | 期刊论文（PMC 开放获取） | A computer vision-based approach for behavior recognition of gestating sows fed different fiber levels during high ambient temperature | CC BY-NC | 42 | 25775 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC8071751/) |
| 12 | `pmc_8147168` | 期刊论文（PMC 开放获取） | Pig-Posture Recognition Based on Computer Vision: Dataset and Exploration | CC BY | 52 | 28865 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC8147168/) |
| 13 | `pmc_8614322` | 期刊论文（PMC 开放获取） | Deep-Learning-Based Automatic Monitoring of Pigs’ Physico-Temporal Activities at Different Greenhouse Gas Concentrations | CC BY | 75 | 46052 | [链接](https://pmc.ncbi.nlm.nih.gov/articles/PMC8614322/) |
| 14 | `local_README` | 项目自有文档 | README.md | 项目自有 | 3 | 2156 | `data/README.md` |
| 15 | `local_data_card` | 项目自有文档 | data_card.md | 项目自有 | 3 | 1953 | `docs/data_card.md` |
| 16 | `local_experiment_report` | 项目自有文档 | experiment_report.md | 项目自有 | 12 | 8359 | `docs/experiment_report.md` |
| 17 | `local_feasibility_report` | 项目自有文档 | feasibility_report.md | 项目自有 | 6 | 4269 | `docs/feasibility_report.md` |
| 18 | `local_limitations` | 项目自有文档 | limitations.md | 项目自有 | 3 | 2311 | `docs/limitations.md` |
| 19 | `local_technical_report` | 项目自有文档 | technical_report.md | 项目自有 | 12 | 8684 | `docs/technical_report.md` |
| 20 | `hf_dataset_readme` | 数据集卡片 | Viewpoint-Aware Pig Posture Recognition（数据集说明） | CC BY 4.0（数据集卡片声明） | 10 | 7008 | [链接](https://hf-mirror.com/datasets/anilbhujel/viewpoint-aware-pig-posture-recognition/raw/main/README.md) |

## 按类别汇总

| 类别 | 篇数 | 片段数 | 片段字符数 |
|---|---:|---:|---:|
| 期刊论文（PMC 开放获取） | 13 | 894 | 533815 |
| 项目自有文档 | 6 | 39 | 27732 |
| 数据集卡片 | 1 | 10 | 7008 |
| 合计 | 20 | 943 | 568555 |

> 本项目不分发任何语料原文，仅在本地建立索引；CC BY-NC 与 CC BY-NC-ND 语料仅用于非商业研究。
