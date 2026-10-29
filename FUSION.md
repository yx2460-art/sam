# 固定窗口结果的融合对照

本阶段复用已完成的 `outputs/t256_s192`（256窗口/192步长/25%重叠），不重新运行 SAM，不加载 PyTorch 或模型权重。固定同一套窗口预测、BsiNet源实例编号、计分区域和边界容差，仅改变融合规则。

## Linux 操作

在 `sam-experiments` 项目的 PyCharm 底部终端，激活 `sam-exp` 并拉取新代码后执行：

```bash
python -m experiments.refuse_tiles --run-dir outputs/t256_s192 --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/fusion
```

默认一次生成四种方法：

| 目录 | 规则 |
|---|---|
| `center` | 原窗口中心优先结果，作为重融合一致性对照 |
| `union` | 至少一次有效观测为前景则保留 |
| `intersection` | 所有有效观测均为前景才保留 |
| `majority` | 超过半数有效观测为前景才保留；平票为背景 |

无效观测不投票；无有效观测处输出0，评分掩膜内不允许出现这类空洞。这里融合的是**二值掩膜，不是连续SAM概率**。每个保留像素继承原BsiNet全局实例ID，不新增实例拆分或跨实例匹配。求并只合并这些窗口的预测，不会恢复任何窗口都没有预测到的田块，也不会扩张到原BsiNet实例以外。

## 输出

- `outputs/fusion/comparison.csv`：基线与四种方法的指标表、融合耗时。
- `outputs/fusion/comparison.json`：源实验记录、缓存哈希、同一评分区的重叠/冲突面积及完整指标。
- `outputs/fusion/<方法>/instance.tif`：对应融合实例图。
- 各方法目录还保存 `count.tif`、`manifest.json`、`summary.json`，兼容已有评价工具。

所有新结果独立写入，不覆盖旧目录；已有 `outputs/fusion` 时需使用新名称。原实验的 `tiles`、`work`、记录和结果均只读。若不传 `--gt`，仅生成融合图，不计算精度；可用已有 `experiments.evaluate_tiles` 后续评价。

## 检查与限制

要求源 `manifest.json` 为complete且采用固定源编号协议；试跑、未完成记录、错误缓存大小、缺失窗口文件均拒绝。当前工作目录必须是项目根目录，因为旧记录可能保存了相对数据路径。脚本检查基线和掩膜与源记录哈希一致、窗口计数、缓存取值、`count.tif`网格与观测数，并要求缓存能够重现原融合结果。开始和结束时校验源记录与缓存哈希，防止比较期间改动。

旧版本未在缓存创建时保存哈希，因此这些检查不能证明源缓存从创建起从未改变；本次会记录当前哈希，不追溯宣称已验证历史完整性。缺少缓存时直接报错，不会自动重跑SAM。当前也不通过窗口TIFF重建缺失的缓存。

`scoring_overlap_pixels`表示固定评分区中有效观测次数大于1的像素；`scoring_conflict_pixels`表示这些观测中同时存在前景和背景的像素。它们与融合方法无关，是同一套输入的分歧统计，不等于错误像素数量。实例ID被固定，因此不包含拆分/合并冲突。影像、有效覆盖及原逐窗口结果应保持不变。

`fusion_seconds`只统计新融合图的生成，不能当作完整推理速度；不包括原SAM推理、预检、评价等开销。`source_inference_seconds`继承原实验记录，其计时包含原脚本的推理与输出等工作，不是纯GPU时间。

指标沿用固定ROI裁剪口径，仍是开发诊断，训练独立性未确认。主要对比Precision/Recall/IoU、PQ和边界F1的取舍，不因某一项提高就认定方法全面有效。下一阶段再用同一冲突区开展二次推理对照。
