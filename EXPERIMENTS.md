# 下一步：固定 BsiNet 输出，比较 SAM 自动提示策略

## 为什么先做这个实验

现有 `test-bsinet.py`、`test-bsinet1.py` 用于单张影像的人机交互，不能单凭几张修正图证明泛化改进。本地 BsiNet 项目目前还包含河流困难负样本、纯网络整图预测和独立河流 ROI 评估流程；这条语义误检实验应继续独立验证，SAM 不应被当作河流分类器。

参考论文：Li et al., *Achieving precise cropland parcel extraction from remote sensing images through integration of segment anything model and adaptive mask refinement*, Computers and Electronics in Agriculture 243 (2026), 111347，DOI: 10.1016/j.compag.2025.111347。

论文第 3.1 节用 SLIC 超像素内部的距离变换极大值作为提示点，第 3.2 节按稳定性过滤重叠并迭代补全，第 3.3 节处理 50% 重叠切片的接缝。这里先检验提示策略，之后再决定是否实现迭代补全和整图拼接。

这是**受论文启发的 BsiNet 引导实验，不是论文完整复现**：SLIC 在 BsiNet 预测的每个连通实例内部运行；每个提示点分别调用 SAM，同一次预测的多个候选也参与过滤。论文的 3 m 距离阈值不能直接等同于这里默认的 3 像素。参数需要在验证集确定后冻结。

| 实验 | 输入 / 提示 | 用途 |
| --- | --- | --- |
| E0 | 固定 BsiNet 实例，无 SAM | 基线 |
| E1 | BsiNet 外接框 | SAM 框提示对照 |
| E2 | 同样的框 + 实例内部最远点 | 单点提示对照 |
| E3 | 同样的框 + SLIC 自适应内部点 | 多提示拆分、漏分辨识 |

E1–E3 使用相同权重、输入实例、框扩展、稳定性阈值和过滤规则。E3 提示数量更多，因此同时比较运行时间和提示数；不能把差异全部归因于点的位置。如 E3 有效，后续增加等数量网格点对照，再做重叠过滤消融。

## 数据和评估要求

- 使用同一组固定 ID。先选择 10–20 张**验证图**，包括粘连田块、细长田块、完整独立田块、复杂背景和切片边缘情况。已经多次观察调参的 30、55 可以用于调试，不应再充当未见测试样本。
- 之后固定参数，在空间独立的测试区运行，不能用测试结果反复调阈值。
- 输入实例必须是单通道整数 ID 图（0 为背景），不能使用彩色可视化图。BsiNet `instance_pred/*_labels.tif` 使用 `--instance-suffix _labels`；旧 `save10/instance/*.png` 使用空后缀。
- 评估需要人工真值**实例 ID 图**。只有二值 mask 不能评价粘连；不要用简单连通域生成的“实例真值”替代人工田块实例。
- 图像和标签必须逐像素对齐，不会自动 resize。有双方地理参考时检查 CRS 和仿射网格。PNG 没有地理参考时只能确认尺寸，请另行确认来源与像素位置。
- 默认 RGB 波段为 1、2、3，要求 UInt8。其他动态范围必须明确传入 `--rgb-range LOW HIGH`；多光谱影像必须明确 `--rgb-bands R G B`。尺度由训练/验证数据或传感器约定确定，不在测试集上拟合。
- 栅格使用有效像元完整的小样本；影像存在 NoData 时拒绝运行。实例标签兼容 BsiNet 的 `NoData=0`：0 按背景评价，不按忽略区处理；其他 NoData 约定拒绝运行。

## Linux 运行

在 SAM 仓库根目录、Python >= 3.9 的已有 PyTorch 环境中执行（建议单独使用 Python 3.10/3.11 环境，避免影响旧 BsiNet 环境）。按实际 Linux 路径设置变量：

```bash
python -m pip install -e ./segment-anything
python -m pip install -r experiments/requirements.txt
python -m unittest discover -s tests -v

export IMAGE_DIR=/path/to/validation/image
export BSINET_INSTANCES=/path/to/bsinet/instance_pred
export GT_INSTANCES=/path/to/validation/instance
export CHECKPOINT=/path/to/sam_vit_b_01ec64.pth
export IDS=/path/to/fixed_validation_ids.txt
export RUN=outputs/validation_prompt_v1

# 固定列表的内容例子：一行一个实际存在的 ID，不含扩展名。
# 30
# 55
# 实际对照请使用完整的预先选择验证列表。

# E0：不改变 BsiNet 输出，直接评价。
python -m experiments.evaluate \
  --gt-dir "$GT_INSTANCES" --prediction-dir "$BSINET_INSTANCES" \
  --prediction-suffix _labels --ids "$IDS" --output-dir "$RUN/E0_eval"

# E1、E2、E3：每个模式一个独立目录。
for MODE in box interior slic; do
  python -m experiments.run_prompt_ablation \
    --image-dir "$IMAGE_DIR" --instance-dir "$BSINET_INSTANCES" \
    --instance-suffix _labels --ids "$IDS" --checkpoint "$CHECKPOINT" \
    --mode "$MODE" --device cuda:0 --output-dir "$RUN/$MODE"

  python -m experiments.evaluate \
    --gt-dir "$GT_INSTANCES" --prediction-dir "$RUN/$MODE/instance" \
    --ids "$IDS" --output-dir "$RUN/${MODE}_eval"
done
```

如果 BsiNet 文件名是 `30.png` 而非 `30_labels.tif`，删除运行和评估命令中的后缀参数。所有输出目录必须不存在，以免覆盖旧实验。CPU 冒烟测试可用 `--device cpu`，速度会较慢。

输出包括：

- `instance/<ID>.tif`：UInt32 实例图，继承原始影像 CRS 与仿射变换；0 为背景，**不保证沿用原实例 ID**。
- `<ID>_prompts.json`：框、提示坐标、候选/接受数量、稳定性、SAM 分数、覆盖率、保留残余面积和耗时。
- `manifest.json`：运行参数、样本 ID、代码提交号、权重 SHA256、依赖版本，以及 `ground_truth_used=false`。
- `*_eval/metrics.csv`：逐图分数与计数。
- `*_eval/summary.json`：累计计数后计算的总体分数，不是逐图分数简单平均。

## 过滤策略和限制

每个原始实例先按 8 邻域拆成连通组件，SAM 候选依次检查预测 IoU、稳定性、原始实例附近支持率和面积；有点提示时，只保留包含前景点的连通域，点落在候选背景时直接拒绝。稳定性按官方 SAM 约定计算：高阈值像素数量 / 低阈值像素数量，默认偏移为 1。

候选按稳定性、预测 IoU 排序。与已接受掩膜重叠超过候选面积 5% 的候选被丢弃，少量重叠裁掉；裁剪后保留最大连通域。候选不得覆盖其他 BsiNet 实例，扩展仅允许进入尚未占用的原始背景。

覆盖率不足时保留原始组件。通过检查的候选写入新 ID。未覆盖的原始像素不再生成独立实例：只有一个候选时归回该候选；多个候选时按欧氏最近距离归入同一原始组件内的已接受候选。日志记录 `residual_components_attached` 和 `output_instance_count`，方便区分残片与真正的多候选拆分。多候选的最近距离分配不保证每个 ID 的拓扑连通性，仍需检查空间结果。

这项修正保留原始前景像素，因此单候选实验主要验证保守扩展，不能缩小原有过大掩膜。当前方法不会主动清除 BsiNet 原有假阳性，不能用来证明河流误检已解决，也不能恢复 BsiNet 完全漏掉的田块。不要仅通过强制减少实例数量判断精度改善，必须重算 PQ 和匹配 IoU。

E1–E3 的输入连通组件拆分属于共同预处理，E0 使用原输出。如果基线一个 ID 对应多个断开的对象，需要另行确认该标签约定，不能将 ID 清理带来的差异全归因于 SAM。

这一阶段不实现跨图实例匹配、整图接缝融合或人工曲线切缝。实例评价在每张图内进行，跨 tile 的同一田块可能被多次计数；整图结论必须另做空间一致性评价。

## 指标与接受标准

### 残余像素处理对照

新增 `--residual-policy attach|replace`，默认 `attach` 维持已有实验行为。`replace` 模式只保存通过既有质量、面积、支持率和覆盖率检查的 SAM 候选，允许移除未覆盖的原始前景；候选为空、覆盖率不足或原组件面积太小时仍保留原组件。日志中的 `removed_pixels` 记录本组件被移除的原始前景像素，不能直接解释为纠正了多少误检。

先固定原有 `ids20.txt`、`--mode interior` 和所有阈值，仅增加 `--residual-policy replace`，输出到 `outputs/replace20`，评价输出到 `outputs/replace20_eval`，与 `outputs/point20_eval` 和 `outputs/base20_eval` 对照。此轮不加入自动负点，不读取真值生成提示。检查 Precision/PQ 是否改善，以及 Recall 是否下降，保留所有旧结果。不要仅根据一张图的边缘更平滑判断有效。

替换模式可能误删真实田块区域，默认覆盖率门槛仅为 0.5，并不保证语义正确。它也不能恢复完全漏掉的实例或自动删除没有合格 SAM 候选的假阳性。若实验支持进一步改进，再另做负点提示对照。

主要比较 Mask Precision/Recall/IoU/Dice、Instance Precision/Recall/F1、匹配实例平均 IoU、PQ，以及实例边界 F1。实例采用与本地 BsiNet `validate_best.py` 相同的 IoU >= 0.50 降序贪心一对一匹配。边界包含不同实例 ID 之间的内部边界，采用 4 邻域标签变化、前景一侧像素，默认容差为 2 像素（方形 Chebyshev 邻域），容差必须在所有组保持一致。图像外侧不构造虚拟边界。分母为零的分数记为 0，计数同时保留用于解释空图。

不要沿用旧 BsiNet `eval.py` 的 GOC/GUC/GTC 命名：那里实际计算 Accuracy/IoU/Dice；论文的 GTC 是对象级几何分类误差，Dice 不是论文 GTC。本实验没有实现论文 GTC，也不报告它。

先看 E2/E3 能否在固定样本上提高 Instance F1/PQ，并查看粘连是否减少、实例 FP 是否增加。Mask IoU 单独提高不足以证明田块拆分正确；当前保留原始像素的设计也会限制语义修正。只有改进在独立测试区成立，才继续扩展到迭代补全、边界/距离头引导提示和重叠整图拼接。

现有交互脚本和 `original` 分支保持可用；新实验位于 `codex/sam-prompt-ablation` 分支。Windows 合成测试不代表 Linux/CUDA 或真实遥感数据已验证。
