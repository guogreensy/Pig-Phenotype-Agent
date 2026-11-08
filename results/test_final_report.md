# 实验结果（自动生成）

> 本文件由 `scripts/run_experiment.py` 依据原始 CSV 自动生成，
> 其中的每个数字都可以用 `scripts/plot_results.py` 从 `results/raw/` 重新计算得到。

## 运行元数据

| 项 | 值 |
|---|---|
| run_id | test_final |
| 来源文件 | results\raw\test_main_v2_raw.csv |
| 模型 | glm-4.1v-thinking-flash |
| 记录数 | 600 |
| 实验时间范围 | 2026-10-19T13:12:46 ~ 2026-10-19T13:38:47 |
| 提示词 SHA256 | B=914878c32ff2c99d9384861cec5ae33e7a0edfc38ee28bd78880397d7b2eb998；C=e5c513d404ddacb484c5743abcf5c2f34d32bd457ead29bea2dd40f95db89b2f；A=098f7653cb3ec2854bea8e91e27e1703cfdb68f0756460426c03dc6dfbfb58fe |
| 指标汇总 | results\metrics\test_final_metrics.csv |
| 分类明细 | results\metrics\test_final_per_class.csv |
| 生成时间 | 2026-10-19T13:44:16 |

## 指标汇总

| 方案 | 调用数 | 成功 | 失败率 | JSON 有效率 | 无法判断比例 | 有效分类数 | Accuracy(有效) | Accuracy(严格) | Macro-F1 | 平均响应(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 200 | 200 | 0.00% | 0.00% | 0.00% | 199 | 0.3769 | 0.3750 | 0.2325 | 4.21 |
| B | 200 | 200 | 0.00% | 0.00% | 0.00% | 200 | 0.3700 | 0.3700 | 0.2193 | 3.66 |
| C | 200 | 200 | 0.00% | 99.50% | 0.00% | 200 | 0.3300 | 0.3300 | 0.1672 | 3.59 |

## 各类别指标（有效分类子集）

| 方案 | 类别 | Precision | Recall | F1 | 分类子集样本 | 全部样本 | Recall(严格) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | standing(站立) | 0.6286 | 0.2821 | 0.3894 | 78 | 78 | 0.2821 |
| A | lying_lateral(侧卧) | 0.3208 | 0.8361 | 0.4636 | 61 | 62 | 0.8226 |
| A | lying_other(其他卧姿) | 0.4000 | 0.0426 | 0.0769 | 47 | 47 | 0.0426 |
| A | sitting(坐姿) | 0.0000 | 0.0000 | 0.0000 | 13 | 13 | 0.0000 |
| B | standing(站立) | 0.6000 | 0.2692 | 0.3717 | 78 | 78 | 0.2692 |
| B | lying_lateral(侧卧) | 0.3230 | 0.8387 | 0.4664 | 62 | 62 | 0.8387 |
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
