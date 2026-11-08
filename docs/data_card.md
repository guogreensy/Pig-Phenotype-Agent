# 数据卡（Data Card）

## 基本信息

| 项 | 内容 |
|---|---|
| 数据集名称 | Viewpoint-Aware Pig Posture Recognition Dataset |
| 维护者 | Anil Bhujel（HuggingFace 账号 `anilbhujel`） |
| 获取地址 | https://huggingface.co/datasets/anilbhujel/viewpoint-aware-pig-posture-recognition |
| 许可 | **CC BY 4.0** |
| 本项目使用的划分 | `seenVP_test.csv` + `seenVP_test_images/`（官方测试划分之一） |
| 本项目构建日期 | 2026-10-19 |
| 构建脚本 | `scripts/build_dataset.py`（确定性，可用 SHA256 校验复现） |

## 采集背景（依据数据集 README）

- 采集自 **2 个真实猪栏**，每栏 4 台相机：2 台鱼眼顶装（1280×720）、
  2 台 Orbbec Femto RGB（1920×1080）；
- 标注为**逐实例**：`bbox`（XYWH 像素）+ `class_id` + 相机视角角（方位角/仰角，
  由 PnP 标定推算）；
- 相机坐标系与视角角定义见原 README（zenith 约定：0° 为正上方，90° 为水平）。

## 标签体系

| class_id | 原始标签 | 含义 | 本项目规范类别 |
|---|---|---|---|
| 0 | Lateral_lying_left | 左侧卧 | `lying_lateral` |
| 1 | Lateral_lying_right | 右侧卧 | `lying_lateral` |
| 2 | Sitting | 坐姿 | `sitting` |
| 3 | Standing | 站立 | `standing` |
| 4 | Sternal_lying | 胸腹着地俯卧 | `lying_other` |

合并说明：
- 左/右侧卧合并为 `lying_lateral`，因为本任务的目标是姿态识别而非朝向判定；
- `Sternal_lying` 映射为 `lying_other`，与提示词中"躯干着地但不符合侧卧定义"一致。

## 本项目的数据处理

1. **只取测试划分**：避免与数据集作者的监督训练集重叠，也避免我们自己"训练"任何模型
   （本项目不训练模型，仅用 API）。
2. **选择性下载**：先用 CSV 贪心选图（53 张），再只下载这些源图（≈9.8 MB），
   不下载 6000+ 文件的完整数据集。
3. **逐实例裁剪**：向外扩 8% 边距，使单只猪完整且带少量上下文。
4. **质量过滤**：短边 ≥64px、面积 ≥8192px²、长宽比 ≤4:1。共排除 3 个实例。
5. **按源图分组划分**：同源图的裁切强制同 split，避免近重复泄漏；
   种子 `20261019`，dev 0.2 / test 0.8。
6. **分层限量**：test 不超过 200 条，按类别比例抽样。

## 规模与分布（实测）

| split | 总数 | standing | lying_lateral | lying_other | sitting | 源图数 |
|---|---:|---:|---:|---:|---:|---:|
| dev | 109 | 44 | 24 | 36 | 5 | 12 |
| test | 200 | 78 | 62 | 47 | 13 | 41 |

裁切尺寸（test 集）：短边最小 66px、中位 169px、最大 324px；短边 <128px 占 27.5%。

## 已知偏差与限制

| 偏差 | 影响 | 缓解 |
|---|---|---|
| **类别不均衡**：sitting 在测试集仅 13 条 | 该类指标方差大 | 报告中逐类报告并注明样本量，不下强结论 |
| 裁切图偏小（27.5% 短边 <128px） | 细节不足，模型可能依赖轮廓与地面接触关系 | 上采样到 512px 再送入模型；在局限性中说明 |
| 同源图多裁切共享背景/光照 | 统计独立性弱于独立拍摄 | 按源图分组划分；报告源图数量 |
| 仅 2 个猪栏、8 台相机 | 跨场泛化未验证 | 明确声明适用范围 |
| 单一物种/日龄段的商品猪 | 不能外推到母猪、仔猪 | 声明 |

## 隐私与合规

- 数据为公开数据集，许可允许研究与再分发（需署名）；
- 不含人员面部或可识别个人信息；
- 裁切图会通过公网发送至模型服务商进行处理，使用前请确认该行为符合你的要求。

## 引用

```bibtex
@inproceedings{CV4Animals2026,
  title     = {Viewpoint-Aware Pig Posture Recognition and Benchmark Dataset},
  author    = {Bhujel, Anil, Bashar Mk, and Morris, Daniel},
  year      = {2026}
}
```
