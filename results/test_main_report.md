# 实验结果（自动生成）

> 本文件由 `scripts/run_experiment.py` 依据原始 CSV 自动生成，
> 其中的每个数字都可以用 `scripts/plot_results.py` 从 `results/raw/` 重新计算得到。

## 运行元数据

| 项 | 值 |
|---|---|
| run_id | test_main |
| 数据划分 | test（样本 200 条） |
| 模型 | zhipu / glm-4.1v-thinking-flash |
| 调用参数 | temperature=0.0, max_tokens=2048, image_max_side=512 |
| 提示词 SHA256 | A=098f7653cb3ec2854bea8e91e27e1703cfdb68f0756460426c03dc6dfbfb58fe；B=914878c32ff2c99d9384861cec5ae33e7a0edfc38ee28bd78880397d7b2eb998；C=e5c513d404ddacb484c5743abcf5c2f34d32bd457ead29bea2dd40f95db89b2f |
| 重复次数 | 1 |
| 计划调用 | 600 |
| 实际完成 | 600 |
| 调用失败 | 0 |
| 总 Token | 388407 |
| 原始记录 | results\raw\test_main_raw.csv |
| 指标汇总 | results\metrics\test_main_metrics.csv |
| 分类明细 | results\metrics\test_main_per_class.csv |
| 生成时间 | 2026-10-19T13:38:52 |

## 指标汇总

| 方案 | 调用数 | 成功 | 失败率 | JSON 有效率 | 无法判断比例 | 有效分类数 | Accuracy(有效) | Accuracy(严格) | Macro-F1 | 平均响应(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 200 | 200 | 0.00% | 0.00% | 0.00% | 192 | 0.3906 | 0.3750 | 0.2401 | 4.21 |
| B | 200 | 200 | 0.00% | 0.00% | 0.00% | 198 | 0.3889 | 0.3850 | 0.2333 | 3.66 |
| C | 200 | 200 | 0.00% | 99.50% | 0.00% | 200 | 0.3300 | 0.3300 | 0.1672 | 3.59 |

## 各类别指标（有效分类子集）

| 方案 | 类别 | Precision | Recall | F1 | 分类子集样本 | 全部样本 | Recall(严格) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | standing(站立) | 0.6286 | 0.2933 | 0.4000 | 75 | 78 | 0.2821 |
| A | lying_lateral(侧卧) | 0.3355 | 0.8361 | 0.4789 | 61 | 62 | 0.8226 |
| A | lying_other(其他卧姿) | 0.4000 | 0.0455 | 0.0816 | 44 | 47 | 0.0426 |
| A | sitting(坐姿) | 0.0000 | 0.0000 | 0.0000 | 12 | 13 | 0.0000 |
| B | standing(站立) | 0.5854 | 0.3158 | 0.4103 | 76 | 78 | 0.3077 |
| B | lying_lateral(侧卧) | 0.3399 | 0.8387 | 0.4837 | 62 | 62 | 0.8387 |
| B | lying_other(其他卧姿) | 0.2500 | 0.0213 | 0.0392 | 47 | 47 | 0.0213 |
| B | sitting(坐姿) | 0.0000 | 0.0000 | 0.0000 | 13 | 13 | 0.0000 |
| C | standing(站立) | 0.5000 | 0.1282 | 0.2041 | 78 | 78 | 0.1282 |
| C | lying_lateral(侧卧) | 0.3128 | 0.9032 | 0.4647 | 62 | 62 | 0.9032 |
| C | lying_other(其他卧姿) | 0.0000 | 0.0000 | 0.0000 | 47 | 47 | 0.0000 |
| C | sitting(坐姿) | 0.0000 | 0.0000 | 0.0000 | 13 | 13 | 0.0000 |

## 图表

### Accuracy 对比

![accuracy](C:/WorkSource/DSH_workspace/智能育种评估系统/results/figures/accuracy.png)

### Macro-F1 对比

![macro_f1](C:/WorkSource/DSH_workspace/智能育种评估系统/results/figures/macro_f1.png)

### JSON 有效率对比

![json_rate](C:/WorkSource/DSH_workspace/智能育种评估系统/results/figures/json_rate.png)

### 无法判断比例对比

![unknown_rate](C:/WorkSource/DSH_workspace/智能育种评估系统/results/figures/unknown_rate.png)

### 响应时间对比

![latency](C:/WorkSource/DSH_workspace/智能育种评估系统/results/figures/latency.png)

### 各类别 F1 对比

![per_class_f1](C:/WorkSource/DSH_workspace/智能育种评估系统/results/figures/per_class_f1.png)

### 混淆矩阵（方案 C）

![confusion_C](C:/WorkSource/DSH_workspace/智能育种评估系统/results/figures/confusion_C.png)
