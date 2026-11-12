# 正点交互细分与背景补充

## 已有标点如何继续

更新代码后，**不需要重新生成项目或重新标7个点**。原来的project.json和点文件可继续使用。例如点文件已放在项目子目录时：

```bash
python -m experiments.split_instances infer --project-dir outputs/split_project --points outputs/split_project/split_points.json --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --output-dir outputs/split_proposals_v2
```

打开 `outputs/split_proposals_v2/review.html`。终端和候选页面会逐点显示原预测归属，以及是否生成候选。点在预测背景内不再报错；旧标注页面的限制文字不会自动更新，新prepare生成的页面会在点击后直接显示归属。归属是原预测ID，不是人工标注正确性的判定。

这是新的实例拆分实验，不再要求用户在无法拆分的求交／求并结果中选择。先选完整的源实例，在原图上对不同实际田块各点一个正点，运行SAM生成子实例，再由用户确认是否接受。整个流程为离线页面配合命令行，页面本身不需要联网；不是实时点击即推理的服务器，也不是QGIS插件。

## 1. 准备完整实例标注页面

Linux项目根目录、sam-exp环境：

```bash
python -m experiments.split_instances prepare --run-dir outputs/t256_s192 --output-dir outputs/split_project
```

命令从旧实验manifest读取原始影像、**完整BsiNet源实例图**和ROI路径，挑选与ROI相交的实例，但读取其整个实例范围及128像素上下文，不把ROI或256网格当作实例裁剪边界。它不使用旧滑窗的裁块结果作完整实例输入。本次起点是完整源实例，不是之前人工融合页面的中心优先结果。初始预测可能经过人工编辑，仍只作为开发输入。

可以先在QGIS用“识别要素”查看一个粘连实例的栅格ID，然后添加 `--instance-ids 123` 仅准备该ID（将123替换为实际ID）。建议先选择1至3个影像上分界清楚的真实粘连案例。没有指定ID时只按ROI筛选位置，不使用GT。

单个上下文任一边超过4096像素会拒绝，不悄悄裁断；可改选具体实例，或在了解显存／分辨率影响后显式增加 `--max-side`。SAM仍会内部缩放输入，大上下文不保证更好。明确输入其他数据时，可替代 `--run-dir` 使用 `--image ... --instances ... --instance-ids ...`。

浏览器打开 `outputs/split_project/index.html`。黄色标出原预测实例边界，**不是人工标点的有效范围边界**。每个不同田块内部各点一下，每区域1至8点。序号对应不同田块，不是同一个田块的多个正点。可撤销最后一点或清空。不标点的实例保持不动。不要查看GT后选择提示。

点在当前实例内，参与细分；点在原预测背景内，参与补充候选；点在其他已有实例内，显示其ID并保护其结果，不中断其他点。需要修改另一个已有ID时，应通过 `--instance-ids` 单独打开对应实例。只有超出当前影像裁块、落在无有效影像数据内或坐标格式错误时才拒绝输入，不要求人工服从原预测前景。

点击导出 `split_points.json`，将实际下载文件复制到项目 `outputs/split_points.json`。页面刷新会丢失未导出的点，请先完成一小批并导出。

## 2. 生成SAM拆分候选

```bash
python -m experiments.split_instances infer --project-dir outputs/split_project --points outputs/split_points.json --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --output-dir outputs/split_proposals
```

使用SAM ViT-B，每个田块一个正点。当前实例内正点使用源实例包围框，背景正点不使用限制它的源实例框。每次只输入该田块的正点，**没有把其他正点作为负点**。候选允许覆盖当前实例及背景，但其他已有实例和无效影像区始终受保护。候选必须包含自己的点、不能同时包含其他田块的点，并满足默认SAM质量分数0.8及最小30像素。每个点选择合格候选中质量分数最高者。重叠部分按SAM掩膜logit归属，再仅保留与各自正点连通的子区域。

若SAM对某个点没有合格候选，则记录该点原因，继续处理其余点；不同点仍返回同一大片区域时，不通过距离／直线／强制分水岭伪造拆分。可调整正点后另开新输出目录重试，保留尝试记录。阈值是开发默认值，不保证经过独立验证。候选有多个不代表实际田块识别正确，必须人工检查。局部候选限于现有影像上下文；疑似被上下文裁断时应扩大prepare的context后另建项目，不能把截断候选当完整田块。

**前景可增加，但不删除原前景**：允许确认后将原预测背景补成新田块，不侵入其他已有实例；未分配的原实例像素保留原父ID。因此可能输出多个子ID加一个剩余父ID，并不保证完全覆盖为期望的几个田块。候选页显示剩余像素和新增前景数量，应检查是否仍粘连或误加道路、河流等。此阶段没有自动画分割线功能，也不能修复所有粘连。

## 3. 预览、确认与导出

### 先诊断未生成候选的正点

新版本每次infer自动保存 `diagnostics.html`、`diagnostics.json`、`diagnostics.csv` 和 `diagnostics/`。在review页面点击“查看逐点SAM原始候选、分数及具体拒绝原因”，或直接用浏览器打开 `diagnostics.html`。

每个正点展示SAM实际返回的所有多掩膜候选，不只展示通过者。每个候选包含：

- SAM预测质量分数及当前固定阈值，分数不是与GT计算的真实IoU。
- 原始阈值化掩膜、保护其他实例及无效影像后的掩膜、与当前正点连通的区域，三图均叠加全部正点编号。
- 各阶段面积、被保护规则移除的像素、自己的点是否在掩膜内、还包含哪些正点、是否触及上下文边缘。
- 低于质量阈值、自己的点缺失、包含其他正点、面积过小、非有限数值等全部适用原因。
- 是否通过筛选、是否被选中进入重叠处理，以及该点最终是否保留。通过但分数不是最高的候选会明确标注，不能误称为模型没输出。

归属其他已有实例的点显示“受保护、未调用SAM”，不伪造原始候选。NPZ保存全分辨率的三个二值阶段掩膜，PNG只做显示降采样（最长边约1000像素），不影响推理或评分。诊断会额外占用磁盘和写图时间，保存目录后可离线查看，无需重跑推理。没有保存原始浮点logits，本轮诊断不支持直接重新计算任意logit阈值。

**筛选规则及默认阈值没有因增加诊断而改变**。原始掩膜可能覆盖其他已有实例，只用于观察，不允许从诊断页直接接受或导出。诊断也没有读取GT。旧输出只保留了筛选后的结果，无法恢复被丢弃的原始候选，需要使用原标点重新infer一次、新建输出目录，例如：

```bash
python -m experiments.split_instances infer --project-dir outputs/split_project --points outputs/split_project/split_points.json --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --output-dir outputs/split_diagnostic_v1
```

完成后查看 `outputs/split_diagnostic_v1/diagnostics.html`。不需要重新prepare，也不需要重新标点。先判断是SAM掩膜包含多个田块、质量筛选、保护范围裁剪，还是重叠处理导致失败，再设计下一项对照，不盲目降低阈值。

打开 `outputs/split_proposals/review.html`，对照原边界与彩色子实例。只有合理的候选才勾选“确认接受此拆分”，默认全部不接受。导出 `split_approvals.json`，复制至 `outputs/split_approvals.json`。

```bash
python -m experiments.split_instances apply --proposals-dir outputs/split_proposals --approvals outputs/split_approvals.json --output-dir outputs/split_result
```

输出 `instance.tif` 是与源实例图同尺寸、同坐标的**整图**。新ID从全图现有最大ID之后分配，记录于 `split_result.json`，被拒绝或未处理实例保持原样。只写实际改变像素，重叠上下文不会回退之前确认的其他实例。若两组已接受候选要写同一片背景，导出前拒绝，不能静默采用最后一组；先接受其中一组，再以其导出结果为源建立下一轮。加载QGIS后可用唯一值着色，点击查询不同子田块ID；核对它们不是仅仅视觉上划了线。

点文件、模型、源数据和候选均记录哈希。文件改变会拒绝后续操作。输出目录必须新建，不自动覆盖。中断后的部分目录不能当作完成结果；换新目录重跑。

## 4. 决策完成后单独评分

```bash
python -m experiments.split_instances evaluate --result-dir outputs/split_result --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --eval-mask outputs/check3/potential_validation.tif --allow-crs-mismatch --output-dir outputs/split_eval
```

CRS覆盖选项仅沿用先前已知的CRS声明差异处理，不重投影或证明CRS等价。查看 `outputs/split_eval/summary.txt` 中before/after四项指标。全程标点、生成候选、确认导出不读取GT。纯ID细分不会改变前景IoU；接受背景补充后IoU可能提高或降低，须同时看实例F1、PQ、实例边界及误加前景。ROI裁剪口径、训练独立性未知等限制继续适用，不能将小批开发结果称为独立测试提升。

建议先记录小批粘连案例的拆分成功／拒绝情况、正点数和重复尝试次数，判断SAM是否能生成有用候选，再开展较大规模效率评价。点文件中的浏览器时间包含暂停，属于自报记录，不能单独证明比人工画线更高效。
