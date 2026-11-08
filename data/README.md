# 数据集说明

本目录存放实验用数据。**图片不提交到 git**（见 `.gitignore`），可通过
`python scripts/build_dataset.py` 完整复现（含来源、随机种子与筛选规则）。

## 1. 来源与许可

| 项 | 内容 |
|---|---|
| 数据集 | `anilbhujel/viewpoint-aware-pig-posture-recognition` |
| 平台 | HuggingFace Datasets |
| 许可 | **CC BY 4.0** —— 使用时必须署名 |
| 使用划分 | 官方 **seen-VP test**（`seenVP_test.csv` + `seenVP_test_images/`），**未使用**训练划分 |

引用：

```bibtex
@inproceedings{CV4Animals2026,
  title     = {Viewpoint-Aware Pig Posture Recognition and Benchmark Dataset},
  author    = {Bhujel, Anil, Bashar Mk, and Morris, Daniel},
  year      = {2026}
}
```

## 2. 处理流程（`scripts/build_dataset.py`）

1. 下载 `seenVP_test.csv`（300 张源图 / 2603 个实例标注）；
2. **贪心选图**：按"站立/侧卧/其他卧姿/坐姿"四类配额选择源图，只下载被选中的源图；
3. **逐实例裁剪**：按标注 bbox 裁出单只猪，并向外扩 8% 边距；
4. **质量过滤**：短边 ≥64px、面积 ≥8192px²、长宽比 ≤4:1，不合格记录原因并排除；
5. **按源图分组划分**：同一源图的全部裁切只进入同一个 split，避免近重复泄漏；
   随机种子 `20261019`，dev 比例 0.2；
6. 写出图片与 `manifest.csv` / `labels.csv`，并做"磁盘图片数 = 清单行数"一致性校验。

## 3. 实际结果（2026-10-19 构建）

| 项 | 数值 |
|---|---|
| 选中源图 | 53 张（下载 ≈9.8 MB） |
| 裁切候选 | 471 个 |
| 质量过滤排除 | 3 个（短边<64px 2 个、面积<8192px² 1 个） |
| 最终样本 | **309 个**（6.7 MB） |
| dev | 109 条 / 12 张源图 |
| test | **200 条 / 41 张源图** |
| `manifest.csv` SHA256 | `846AE422ABA702878B19D38832F1DFFDAB595EB8A760ED7C489A84CAE435A512` |

类别分布：

| split | standing | lying_lateral | lying_other | sitting |
|---|---:|---:|---:|---:|
| dev (109) | 44 | 24 | 36 | 5 |
| test (200) | 78 | 62 | 47 | **13** |

裁切尺寸（test）：短边最小 66px、中位 169px、最大 324px；
短边 <128px 占 27.5%。坐姿类中位短边 150px，样本最少且偏小，
**其指标置信区间较宽，报告中不得据此下强结论**。

## 4. 文件说明

| 文件/目录 | 说明 |
|---|---|
| `manifest.csv` | 权威样本清单：`sample_id, source_image, bbox, class_id, class_name, label, split, image_path, crop_w, crop_h, filter_reason, sha256, source_url` |
| `labels.csv` | 供人工复核的标签表（含 `reviewer`/`review_status` 列，默认 `unreviewed`） |
| `samples/dev/<label>/` | dev 划分裁切图 |
| `samples/test/<label>/` | test 划分裁切图（**正式实验只跑这个划分**） |
| `datasets/_cache/` | 源图缓存（可删除，重跑脚本会重新下载） |
| `datasets/` | 构建脚本与缓存目录 |

## 5. 类别映射

原始 5 类映射为规范 4 类（见 `src/schema.py`）：

| 原始 `class_id` | 原始名称 | 规范类别 |
|---|---|---|
| 0 | Lateral_lying_left | `lying_lateral` |
| 1 | Lateral_lying_right | `lying_lateral` |
| 2 | Sitting | `sitting` |
| 3 | Standing | `standing` |
| 4 | Sternal_lying | `lying_other` |

原始标签左/右侧卧不做区分（本任务不要求区分左右侧），这是**有意的标签合并**，
不是丢失信息。

## 6. 复现与校验

```bat
python scripts\build_dataset.py          REM 重建（已缓存则不再下载）
python scripts\check_env.py              REM 检查数据集就绪情况
```

构建是**确定性**的：同一脚本与参数多次运行会得到完全相同的 `manifest.csv`
（已验证在 `PYTHONHASHSEED=0/12345/999` 下 SHA256 一致）。

## 7. 使用限制

- 仅用于本课程项目的研究与教学，遵循 CC BY 4.0；
- 数据来自 2 个猪栏的 8 台固定相机，**不代表**其它猪场/品种/光照条件；
- 不得把裁切图或模型输出用于疾病诊断、背膘厚或肌内脂肪估计。
