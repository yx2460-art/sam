# 新实验：固定区域融合评分，再进行必要的正点交互

本流程替代“原始BsiNet实例直接交互，然后只报整块开发ROI分数”的主实验。顺序为：**先固定区域，比较原始／求交／求并精度；记录融合后是否仍粘连；必要时从选定融合结果继续SAM交互；确认导出后，再在完全相同的区域评分。** 只比较已有缓存，不重跑全图SAM。既有脚本保留，可复核历史实验。

## 固定原则

- 区域在读取本轮GT前确定。可用源实例ID的完整包围矩形，或原图像素矩形 `--bounds x y width height`。矩形不等于源实例前景掩膜，背景补充区域也能参与评价。
- 评分／回写域是该矩形与旧实验有效标注ROI的交集；`experiment.json`记录矩形面积、实际评分像素和范围。未标注区不当作背景。不应把这称为未经裁剪的完整田块评价。
- 额外128像素上下文用于观察和SAM，不参与评分，也不会回写。标点页面的变暗区代表不属于本轮回写域。
- 第一次评分固定GT哈希。交互后必须使用同一GT、同一域和同一容差，表格保留所有融合结果。
- 求交／求并仍保留源实例ID，没有自动解决粘连的保证。IoU不提升与粘连不是同一概念；需结合实例F1、PQ和影像边界人工判断是否进入交互。
- 进入交互时明确选择融合起点并记录原因。先看过分数再选择起点属于开发决策，不能当作独立盲测。本流程不自动按GT分数逐块挑选候选，也不使用GT生成提示。

## 第一阶段：固定区域与融合对照

以下以源实例2的包围矩形为区域，目录全部使用新的名称：

```bash
python -m experiments.region_workflow prepare --run-dir outputs/t256_s192 --instance-id 2 --output-dir outputs/region_case2
```

prepare不读取GT。打开 `outputs/region_case2/index.html`，查看原图、原始实例、中心优先参考、求交、求并。它们来自同一批滑窗缓存。可用 `--bounds 1000 6800 1000 600` 替代 `--instance-id` 指定原图矩形（数字仅为格式示例，需改成真实目标区域）；区域须在缓存覆盖范围内，否则先生成覆盖该区域的滑窗结果。

```bash
python -m experiments.region_workflow score --experiment-dir outputs/region_case2 --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/region_case2_fusion_eval
```

查看 `outputs/region_case2_fusion_eval/summary.txt`、`comparison.csv`：表格有original、center、union、intersection，四项指标及counts都保存在 `scores.json`。original是当前源实例文件，不保证是未经人工编辑的纯BsiNet结果。此阶段若已满足要求，可结束并记录无需交互；不必为了走流程强制修改。

## 第二阶段：仍有问题时，从融合结果继续交互

这一阶段就是 **SAM交互后处理**。“从求并继续”表示修改已有的求并实例图，不是再运行求并。`start`只生成标点页面，`infer`才运行SAM；SAM接收原始RGB与正点，所选融合实例图用于确定当前目标、保护其他ID以及回写候选。

首次建议只修一个粘连田块：在希望单独提取的真实田块内部点一次，导出后运行推理。一个正点不保证成功；候选不符合真实影像边界时不要接受，保留本次输出并使用新目录重试。多个点表示多个不同田块，并不是给同一个田块增加多个正点。

例如判断求并结果仍粘连，选择union继续：

```bash
python -m experiments.region_workflow start --experiment-dir outputs/region_case2 --fusion-scores outputs/region_case2_fusion_eval --fusion union --reason "fusion still contains merged parcels" --output-dir outputs/region_case2_interaction
```

`--fusion intersection` 则从求交继续。选择应记录理由，不能把不同起点混称为同一实验。程序没有默认偷偷回到原始BsiNet，project.json的instances指向本区域所选融合图。评分尚未完成、分数属于其他区域或源文件已改变时拒绝开始。

打开 `outputs/region_case2_interaction/index.html`。每个不同实际田块内部各点一下，1至8点；想只修一个田块可点一个正点。使用仅正点模式，无粘连大框、无负点。默认目标为源区域对应ID（若该融合图中仍存在），否则为区域内面积最大的现有ID。其他已有ID受保护；如需修改其他ID，可start时加 `--target-id 实际ID`。求交删空整个区域时，目标ID为0，支持在预测背景内生成新实例。

导出 `region_points.json` 并放入 `outputs/region_case2_interaction/region_points.json`：

```bash
python -m experiments.region_workflow infer --project-dir outputs/region_case2_interaction --points outputs/region_case2_interaction/region_points.json --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --output-dir outputs/region_case2_proposals
```

打开 `outputs/region_case2_proposals/review.html`。逐点原始候选和失败原因在diagnostics.html。正点并不保证分割成功，可能需要进一步交互；不生成强制几何拆分。原父实例剩余像素保留，其他ID不覆盖。

只有候选合理才勾选接受，导出 `split_approvals.json`，放入 `outputs/region_case2_proposals/split_approvals.json`：

```bash
python -m experiments.region_workflow apply --proposals-dir outputs/region_case2_proposals --approvals outputs/region_case2_proposals/split_approvals.json --output-dir outputs/region_case2_result
```

主结果为 `outputs/region_case2_result/region_result.tif`，保留地理坐标但只是**区域及上下文裁块，不是整幅影像**。最终只回写固定region_mask内部，上下文与选定融合起点保持一致。changed_pixels.tif显示实际改动。unclipped_candidate是中间候选审计输出，可能包含上下文改动，不应用作最终成果或评分。

## 第三阶段：在同一区域重新评分

```bash
python -m experiments.region_workflow score --experiment-dir outputs/region_case2 --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --result-dir outputs/region_case2_result --output-dir outputs/region_case2_final_eval
```

最终表格保留 **original、center、union、intersection、interactive**。前四行应与首次评分完全相同，否则报错。interactive对应记录的start_method，分析其相对该融合起点的变化，而非只挑与最差方法比较。所有方法共用固定评分像素；该区域的分数不能直接与过去整个开发ROI的31.58%、47.62%等数字横比。

记录：区域位置和面积、初始融合精度、进入交互原因、正点数量与重试次数、是否接受、交互后精度及成本。看过GT分数后对同一区域反复调参应保留每次尝试，不称首次盲测。来源与训练独立性尚未核实，结果仍是开发实验。所有输出目录拒绝覆盖，重试请改新名称。
