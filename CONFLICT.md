# 冲突区局部二次分割与随机区域对照

## 目标和固定条件

主对照是已生成的求并融合 `outputs/fusion/union`。二次推理复用同一个SAM权重、BsiNet整图实例、内部点/替换/support-pad=0规则。区别是读取冲突附近的扩大上下文，不重新运行第一阶段滑窗。

三组比较：求并不处理、求并+冲突定位二次分割、求并+匹配随机区域二次分割。另报告原BsiNet基线。实验不根据真值挑选位置、生成提示或选择候选；真值在两组推理全部结束之后才读取评分。

仍固定BsiNet全局实例ID，不新增漏检田块或执行新实例拆分。二次候选由原BsiNet局部实例提示，可在该原实例范围内增减求并结果的像素。没有合格SAM候选的连通组件保留求并结果，不用BsiNet回退结果覆盖求并。仅在预先规划的回写掩膜内更新，扩大上下文内的其他像素不回写。

## 区域规划

从原滑窗run的缓存读count与votes，`0 < votes < count`为前景冲突。候选区之外不选择位置。冲突不是错误的同义词，也不包括所有窗口一致但都错的区域。

默认：冲突向外扩展8像素，合并相邻扩展区，按最多256像素边长拆分核心，每核心至少包含32个原始冲突像素；按冲突像素数降序、坐标稳定排序，最多选择16个核心。扩展区、分块大小、阈值和上限应在比较前固定。它只处理被选中的冲突子集，报告总冲突面积、可选核心数、选中数量、覆盖冲突面积和计划回写面积。不能把子集精度变化称为整幅冲突全部修复。

核心外再补充128像素影像上下文；影像边缘用无效零填充保持窗口尺寸一致，无效像素不作为前景。提示坐标为局部上下文窗口坐标，日志记录其原图左上角位置。

随机组按固定种子41，将每个核心的**同一形状回写掩膜**平移到有效评分区域中，避开原始冲突和选中冲突回写区，随机组之间也不重叠。匹配上下文尺寸、计划回写像素和区域数量，并限制BsiNet预测前景比例相差不超过20个百分点。这是预设的预测信息匹配，不使用真值。真实提示/解码调用数和耗时可能不同，报告如实列出，不能声称计算预算完全相等。一个种子只用于首轮运行，后续应预先指定多个种子报告分布，不按分数挑种子。

若无法放置面积相等的随机对照，脚本明确报错，且不开始SAM推理；不悄悄减少随机数量或允许两组预算不等。需要查看规划/数据或明确修改区域上限，再对两组一起重做，记录变化。

## Linux 运行

Windows推送代码后，在Linux的 `sam-experiments` 项目PyCharm底部Terminal、`sam-exp` 环境中拉取代码。

可以先仅规划，不加载模型，不读取真值：

```bash
python -m experiments.refine_conflicts --union-dir outputs/fusion/union --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --plan-only --output-dir outputs/region_plan
```

确认规划成功后运行两组局部推理与评价（使用新目录，不复用规划目录）：

```bash
python -m experiments.refine_conflicts --union-dir outputs/fusion/union --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/region
```

如重融合结果实际保存于 `outputs/fusion2`，将两条命令的 `--union-dir` 改为 `outputs/fusion2/union`。不要移动或删除原滑窗run的缓存，原manifest引用了它们。原模型/预测/ROI哈希不匹配、缓存改变、union与原缓存不一致、评分区域未覆盖等均报错。

本次新处理只加载一次SAM，两组依次运行。`--gt`可省略，此时不计算精度。原实验目录、影像和标签均只读，不覆盖已有输出。中途异常时会保留 `plan.json`及已产生的patch；没有complete报告时不能当作已完成结果，重跑需另取目录。

## 输出解释

- `plan.json`：固定位置、窗口大小和区域预算；`--plan-only`到此停止。
- `conflict.tif`：整个固定评分区内原始冲突域。
- `conflict_write.tif`、`random_write.tif`：两组等面积计划回写区，可在QGIS叠加核对。
- `conflict/instance.tif`、`random/instance.tif`：局部回写后的完整处理范围结果。未规划位置与原求并保持一致。
- 两个组内的 `patch_000.tif` 等：包含上下文的二次预测，不能直接当作回写范围。
- `patches.json`：窗口、提示、接受/回退情况、实际有资格回写的面积、耗时。
- `comparison.csv`：整个同一固定ROI内BsiNet、union、conflict、random四组指标。二次组时间不包括原第一阶段推理，不能直接当作端到端时间。
- `report.json`：参数、预算、提示调用数、增减像素、全部ROI以及固定原始冲突域/一致域的评价。

局部评价域对所有方法相同，不根据处理后的结果重选冲突像素。窄冲突域按边界容差腐蚀后可能没有可评分的内部边界，边界分母为0时分数为0；必须结合counts解释，不能称为边界精度下降到0。实例F1/PQ仍是ROI裁剪口径，尤其不宜拿窄局部域的PQ解释完整田块变化。

训练来源独立性和CRS等价仍未确认，本阶段只作开发对照。判断效果需同时看全局IoU/Recall/PQ、原冲突域变化、一致域是否受损，以及实际处理调用和时间；不只挑一项提高。随机组主要检验定位作用；由于两组空间内容、实例数量和实际可接受候选不同，首次比较不能单独证明因果或泛化提升。
# Offline diagnosis after inference

## Cached boundary-constrained update ablation

From the Linux project root in `sam-exp`, replay the completed two-arm region experiment:

```bash
python -m experiments.constrain_regions --run-dir outputs/region --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --radius 2 --output-dir outputs/boundary2
```

The radius is fixed at 2 pixels for this next development experiment; do not choose a radius by inspecting GT performance and then call that an independent test. No model or checkpoint is loaded. The cached final two-arm rasters encode the accepted second-stage candidates and fallback decisions; replay restricts those existing updates instead of reinterpreting raw patch predictions.

The allowed write mask is the intersection of the original planned arm mask, fixed evaluation ROI, and a band around **union instance-label boundaries** (four-neighbor transitions, Chebyshev radius). It includes touching-instance boundaries as well as foreground/background edges. The band is computed before ROI/write masking and is never recomputed from the updated raster. Since pixels outside the cached crop are unavailable, the outer radius+1 pixels of the crop are protected equally for both arms. Predicted interiors outside the band retain union values. The same radius is used for conflict and random arms; intersected write areas and actual modification counts can differ and are reported, not claimed to be equal budgets.

Outputs include `boundary_band.tif`, each arm's `allowed_write.tif` and `instance.tif`, `comparison.csv`, `report.json`, and a short `summary.txt`. Evaluation compares BsiNet, union, both unrestricted cached results, and both constrained results on the original fixed ROI. It reports IoU, boundary F1, instance F1, PQ, and gross harmful/correct deletions and additions relative to union. GT is opened only after both constrained outputs are saved; it never determines update masks. Source files remain unchanged, existing output directories are rejected, source hashes and original update counts are checked. Current candidate hashes are snapshots rather than historical seals. Use a new output directory for each run; a failed evaluation can leave saved outputs but no completed report.

This is a post-hoc development ablation, not a validated improvement or publication novelty claim. Reducing the number of permitted changes mechanically reduces both harmful and beneficial changes; judge final quality and retained corrections as well as deletion counts. Fixed BsiNet IDs still prevent discovering or splitting instances, and training independence remains unverified.

For an existing `outputs/diagnosis/changes.csv`, extract location counts without reopening rasters or running inference:

```bash
python -m experiments.summarize_regions --diagnosis-dir outputs/diagnosis
```

This writes `location_summary.txt` and prints its contents. Existing summaries are protected against overwrite. New `diagnose_regions` runs also generate this summary automatically. It reports both the share of all harmful deletions and harmful deletions divided by the entire location area; the latter is not a true-foreground-conditioned error rate. Compare near/away groups within each classification. Their areas may differ. Boundary/edge associations are descriptive and do not establish causation or an inference-time update rule.

Run from the Linux project root with the existing `sam-exp` environment:

```bash
python -m experiments.diagnose_regions --run-dir outputs/region --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/diagnosis
```

This does not load SAM or rerun inference. It leaves the region experiment unchanged and writes a short `summary.txt`, a `changes.csv`, and a provenance-bearing `diagnosis.json`. Use a new output directory on each run.

Changes are counted relative to union: removed true foreground (harmful deletion), removed false foreground (correct deletion), added true foreground, added false foreground, and foreground ID changes. These are gross transition counts, not net TP/FP differences. Masks distinguish all ROI, original conflict, selected original conflict, planned writes outside original conflict, and outside planned writes. For the conflict arm, writes outside original conflict are the selected buffer area; for random they are the random control area.

Each mask is also split by proximity to GT instance boundaries and first-stage window perimeters (default Chebyshev radius 2 pixels, configurable with `--radius`). Boundary and window-edge categories overlap: do not sum across these two classifications. GT is used only for post-hoc diagnosis, never inference or selection. These are foreground pixel diagnostics, not independent test or parcel-identity accuracy. Recorded hashes validate available source provenance; hashes for current region outputs are snapshots, not proof they were sealed at inference time.
