# 整图滑窗尺度实验

## 本阶段范围

固定 BsiNet 整图实例、SAM 权重和内部点提示规则，比较 SAM 窗口大小与步长。不改变 BsiNet 训练或其整图预测。默认采用先前的 shrink 配置：内部点、替换、support-pad=0；每个局部候选映射回其唯一的 BsiNet 全局 ID。

**这是边界精修与分块融合的第一阶段，不是新的田块发现或粘连拆分算法。** 同一全局 ID 的断开区域仍使用同一 ID，不按窗口重新编号，也不会用真值帮助匹配。此约束便于隔离尺度影响，但无法研究跨窗口的新实例匹配冲突。

默认融合为 `center`：同一像素采用距窗口边缘更远的预测；距离相同取按原图行列顺序先处理的窗口，背景预测同样参与竞争。它不是置信度加权平均。另支持 `union`、`intersection`、`majority`（平票为背景），均对有效覆盖的二值前景融合，再恢复固定源 ID。本阶段没有 SAM 概率平均或实例稳定性择优拼接，不能称为论文完整复现。

## 固定实验域

`--eval-mask` 使用已检查的 `outputs/check3/potential_validation.tif`。掩膜只用来确定读图范围和评分位置，不参与生成提示。推理读取其外接框向外固定扩展 `--context 1024` 像素的窗口，保留完整影像上下文；不同组必须使用同一掩膜、context、波段和归一化范围。

窗口以**原始影像左上角(0,0)**为统一网格起点，步长遍历到影像末端；最后窗口允许小于指定尺寸，不向内挪动来制造额外重叠。RGB不在脚本中缩放，SAM自身会执行输入尺寸变换，因此不同窗口仍存在细节分辨率差异。无效影像或覆盖区外像素置零，不能对这些像素评分。RGB默认1/2/3波段，需确认原影像波段含义；非UInt8必须显式提供固定 `--rgb-range LOW HIGH`。

所有索引来自同一原图、实际权重训练来源、标注完整性和 CRS 等价仍需核实。当前区域只能用于开发探索，不能称为独立测试。

## 先在 Linux 做一次小范围试跑

在 `sam-experiments` 的 PyCharm 底部 Terminal，使用 `sam-exp` 环境。同步新代码后执行：

```bash
cd ~/xy/sam-experiments
conda activate sam-exp

python -m experiments.run_tiles --image "$HOME/xy/data/塔城镇_49(1).tif" --prediction "$HOME/xy/bsinet/BsiNet-torch-main_ori/full_prediction_E_overlap/full_instance.tif" --coverage "$HOME/xy/bsinet/BsiNet-torch-main_ori/full_prediction_E_overlap/full_coverage.tif" --eval-mask outputs/check3/potential_validation.tif --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --tile-size 512 --stride 256 --max-windows 1 --allow-crs-mismatch --output-dir outputs/tile_test
```

试跑选择与候选区、预测前景都相交的首个窗口，保存实际窗口位置。它只检查运行流程，没有整区覆盖，不生成精度结果。`manifest.json` 的状态应为 `smoke_only`。不要把一个窗口的输出和整区基线比较。

## 完整运行一组

试跑通过后，移除 `--max-windows 1`，使用新目录 `outputs/tile512`；如需自动评价，追加 `--gt "$HOME/xy/data/qgis_label_result3/instance.tif"`。真值仅在推理、融合完成后读取。默认没有真值输入时不计算精度。

其他参数不变，分别运行以下9组，目录建议 `outputs/t256_s256` 等，避免覆盖：

| tile-size | stride |
|---|---|
|256|256|
|256|192|
|256|128|
|512|512|
|512|384|
|512|256|
|1024|1024|
|1024|768|
|1024|512|

先完成一组并检查结果，再提交九组任务。每次输出目录必须不存在，脚本不自动覆盖或清理旧产物。若中途出错，保留失败目录，换新名称重跑；尚不支持断点续跑。

## 输出内容

- `instance.tif`：融合后的源实例 ID，包含处理范围及上下文，可能比原图小。保留正确裁剪后的仿射变换，不应按数组原点当作整幅图。
- `tiles/00000.tif`等：逐窗口实例，保存其全局坐标和源 ID，可供后续重新融合。
- `windows.jsonl`：逐窗口坐标、耗时、有效面积、提示和候选日志；提示坐标是窗口局部坐标，加窗口x/y得到原图坐标。
- `count.tif`：每个像素的有效窗口观测次数。
- `conflict.tif`：至少一次预测前景、至少一次预测背景的像素。固定源ID规则下只检测前景分歧，不检测新实例拆分冲突；两次都错误且一致也不会标记。
- `manifest.json`：参数、固定提示设置、输入实例/掩膜/权重哈希、代码提交号、版本、网格、裁剪范围、窗口数、耗时、CRS警告。`complete`表示计算完成，不表示测试区独立。
- `work/`：磁盘映射累积数组（count/votes/weight/center，共约每处理范围像素13字节）；保留以便复核和再融合，逐窗口文件另占磁盘。整图计算前检查空间。
- `summary.json`：仅完整运行且提供真值时生成，同一范围内的基线、SAM指标和差值。

## 评价边界

评分必须覆盖固定掩膜所有像素；缺少观测时直接报错，不丢弃难点。像素统计在掩膜内计算。边界先在完整输出上求，再只统计距掩膜边缘大于容差的像素，避免把裁剪边缘当田块边界。

**当前实例F1/PQ按同一固定ROI裁剪后的全局ID计算。** 不按连通区域重新编号，但候选区边界仍可能切断田块并改变IoU；报告明确标记 `instance_scope`。这些是开发诊断，不是完整田块的整图发表指标。后续需要明确完整田块纳入/排除规则，并在独立区域评价后才能给出泛化结论。不同组在完全相同ROI内比较，不能依据某组效果挑选ROI。

完整运行也可稍后评价：

```bash
python -m experiments.evaluate_tiles --run-dir outputs/tile512 --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --eval-mask outputs/check3/potential_validation.tif --allow-crs-mismatch
```

这会写入该运行目录的 `summary.json`（重复评价会更新此报告），不修改原始数据。试跑目录会被拒绝评价。

Windows合成测试：网格边缘覆盖、固定编号、交并投票规则、完整读写/评价、试跑不评分。不等同于Linux/CUDA或塔城镇真实影像已通过验证。
