# 整图实验前检查

## 多候选训练划分

`--split-dir` 可以重复传入。多目录时，排除所有候选训练足迹的并集及缓冲区，仅保留所有验证足迹的交集，再与有效影像和覆盖区求交。候选掩膜仍输出 `potential_validation.tif`。`val_pixels` 是验证足迹并集大小；`common_val_pixels` 是交集大小；`potential_gt_foreground` 可用于检查剩余候选区是否包含田块。这不是实例数量或精度。

用户已确认整幅有效区域完整标注时，传入 `--complete-annotation`，报告将记录此项为用户声明；脚本不自动验证。此声明不确认训练来源、CRS等价，也不自动将 `evaluation_ready` 改为true。

所有索引必须来自同一原始影像和相同像素原点；`tile-size` 必须与这些索引真实窗口一致。候选列表不保证覆盖全部历史训练来源，得到共同区域也不能证明对未知训练数据独立。没有共同候选区或田块样本不足时，应使用来源明确的模型或另一幅独立影像，不放宽排除规则来挑选好结果。

## 显式允许 CRS 差异的像素诊断

默认仍严格检查 CRS。已核对数据来源、尺寸和像素网格，仍存在 CRS 表达差异时，可在命令末尾添加 `--allow-crs-mismatch`，并使用新输出目录 `outputs/check2`。该参数只将“双方都有 CRS 且仅 CRS 不同”降为警告；缺失 CRS、尺寸或仿射变换不同仍然报错。不重投影、不修改输入，不认定 CRS 等价，`evaluation_ready` 仍为 false。

JSON 记录 `allow_crs_mismatch`、`crs_mismatch_bypassed_for` 和 `crs_equivalence_confirmed=false`。预览和候选掩膜使用原始影像的坐标元数据，尚未确认地理参考等价前只作像素诊断，不应当作已校准的地理成果。

在 Linux 的 `sam-experiments` 项目 PyCharm 底部 Terminal 中运行，激活 `sam-exp`。不需要重新安装依赖、加载 SAM 权重或使用 GPU。

```bash
cd ~/xy/sam-experiments
conda activate sam-exp
python -m experiments.check_full --image '/home/xubo/xy/data/塔城镇_49(1).tif' --gt /home/xubo/xy/data/qgis_label_result3/instance.tif --prediction /home/xubo/xy/bsinet/BsiNet-torch-main_ori/full_prediction_E_overlap/full_instance.tif --coverage /home/xubo/xy/bsinet/BsiNet-torch-main_ori/full_prediction_E_overlap/full_coverage.tif --output-dir outputs/check
```

路径来自当前实验配置，缺少文件时脚本会报出具体路径，不会猜测替代数据。影像名称的括号必须保留引号。输出目录必须不存在；重跑请改为 `outputs/check2`，不覆盖旧报告。

输出：

- `report.txt`：首先阅读的检查报告。
- `report.json`：详细元数据、计数及原预测目录的 `full_inference_summary.json` 内容。
- `preview.tif`：降采样的地理参考诊断图，可在 QGIS 打开。红色表示有效覆盖之外的真值前景，蓝色为覆盖内背景，绿色为覆盖内真值前景，深灰为未覆盖且无真值前景；它不是精度图，也不是 RGB 原图。

脚本分块检查尺寸、CRS、仿射网格、波段、标签整数类型、负值、NoData 和覆盖值，并统计影像有效区、覆盖区、覆盖外真值/预测前景。不重投影、不重采样输入，不修改任何输入文件。元数据不对齐时仍生成失败报告并返回非零退出码，不继续逐像素分析。

标签的 `NoData=0` 按 BsiNet 约定作为背景编号统计；不能因此认定这些区域已完整标注。影像有效区取前三波段的掩膜与有限值交集；此阶段不推断 RGB 顺序或归一化范围，后续推理前另行确认。预览采用最近邻缩小，会遗漏很细的区域，数字统计以原始网格为准。

第一轮不传 `--split-dir`，报告会明确提示训练/验证区域未核实。不能把旧 `train_qgis_yn3` 的列表直接当作 `weight_train_overlap_yn` 权重的实际训练划分。

确定实际训练该权重所用的数据目录后，可追加 `--split-dir /实际数据目录`。需要同时包含 `tiles_index.csv`（filename,x,y）、`train_ids.txt`、`val_ids.txt`。`--tile-size` 是该索引对应的原始切片大小，默认256；不是将来 SAM 实验的窗口大小。`--train-buffer` 默认256像素，以矩形方式排除训练足迹附近区域。

指定划分后输出 `potential_validation.tif`：验证列表覆盖矩形 ∩ 影像有效区 ∩ 预测覆盖区 − 带缓冲的训练足迹。它不包含采样时被丢弃的验证背景区域，不自动外推为连续验证区。脚本检测空间像元重叠，即使训练和验证 ID 不同也会统计重叠并排除。

如果另有人工确认的完整标注区域，可追加 `--annotation-roi /path/to/roi.tif`，只保留其有效非零区域。它必须逐像素对齐。脚本无法证明用户提供的 ROI 已完整标注，也无法证明所提供的划分确实用于权重训练，所以 `evaluation_ready` 始终为 false。待检查报告、实际训练记录和标注范围人工核实后，才选择固定实验区。此次不计算精度，不自动选取最佳结果区域，不开始九组分块实验。

本地合成检查：

```bash
python -m unittest discover -s tests -v
```
