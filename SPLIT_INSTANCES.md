# 正点交互拆分粘连实例

这是新的实例拆分实验，不再要求用户在无法拆分的求交／求并结果中选择。先选完整的源实例，在原图上对不同实际田块各点一个正点，运行SAM生成子实例，再由用户确认是否接受。整个流程为离线页面配合命令行，页面本身不需要联网；不是实时点击即推理的服务器，也不是QGIS插件。

## 1. 准备完整实例标注页面

Linux项目根目录、sam-exp环境：

```bash
python -m experiments.split_instances prepare --run-dir outputs/t256_s192 --output-dir outputs/split_project
```

命令从旧实验manifest读取原始影像、**完整BsiNet源实例图**和ROI路径，挑选与ROI相交的实例，但读取其整个实例范围及128像素上下文，不把ROI或256网格当作实例裁剪边界。它不使用旧滑窗的裁块结果作完整实例输入。本次起点是完整源实例，不是之前人工融合页面的中心优先结果。初始预测可能经过人工编辑，仍只作为开发输入。

可以先在QGIS用“识别要素”查看一个粘连实例的栅格ID，然后添加 `--instance-ids 123` 仅准备该ID（将123替换为实际ID）。建议先选择1至3个影像上分界清楚的真实粘连案例。没有指定ID时只按ROI筛选位置，不使用GT。

单个上下文任一边超过4096像素会拒绝，不悄悄裁断；可改选具体实例，或在了解显存／分辨率影响后显式增加 `--max-side`。SAM仍会内部缩放输入，大上下文不保证更好。明确输入其他数据时，可替代 `--run-dir` 使用 `--image ... --instances ... --instance-ids ...`。

浏览器打开 `outputs/split_project/index.html`。黄色标出完整待拆实例边界。每个不同田块内部各点一下，2至8点，点应落在所选源实例前景内。序号对应不同子田块，**不是同一个田块的多个正点**。可撤销最后一点或清空。不标点的实例保持不动。不要查看GT后选择提示。

点击导出 `split_points.json`，将实际下载文件复制到项目 `outputs/split_points.json`。页面刷新会丢失未导出的点，请先完成一小批并导出。

## 2. 生成SAM拆分候选

```bash
python -m experiments.split_instances infer --project-dir outputs/split_project --points outputs/split_points.json --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --output-dir outputs/split_proposals
```

使用SAM ViT-B，每个子田块一个正点，结合完整源实例包围框。每次只输入该子田块的正点，**没有把其他正点作为负点**。候选必须包含自己的点、不能同时包含其他子田块的点，并满足默认SAM质量分数0.8及子实例最小30像素。每个点选择合格候选中质量分数最高者。重叠部分按SAM掩膜logit归属，再仅保留与各自正点连通的子区域。

若SAM对不同点仍返回同一大片区域，则标记无法可靠区分，不通过距离／直线／强制分水岭伪造拆分。可调整正点后另开新输出目录重试，保留尝试记录。阈值是开发默认值，不保证经过独立验证。候选有多个不代表实际田块识别正确，必须人工检查。

**前景保持不变**：仅在原实例内分配新ID，不侵入其他实例，不增加外部前景；未分配像素保留原父实例ID。因此可能输出多个子ID加一个剩余父ID，并不保证完全覆盖为期望的几个田块。候选页显示剩余像素数量，应检查剩余部分是否仍构成粘连或多余实例。此阶段没有自动画分割线功能，也不能修复所有粘连。

## 3. 预览、确认与导出

打开 `outputs/split_proposals/review.html`，对照原边界与彩色子实例。只有合理的候选才勾选“确认接受此拆分”，默认全部不接受。导出 `split_approvals.json`，复制至 `outputs/split_approvals.json`。

```bash
python -m experiments.split_instances apply --proposals-dir outputs/split_proposals --approvals outputs/split_approvals.json --output-dir outputs/split_result
```

输出 `instance.tif` 是与源实例图同尺寸、同坐标的**整图**。新ID从全图现有最大ID之后分配，记录于 `split_result.json`，被拒绝或未处理实例保持原样。重叠上下文不会覆盖之前确认的其他实例。加载QGIS后可用唯一值着色，点击查询不同子田块ID；核对它们不是仅仅视觉上划了线。

点文件、模型、源数据和候选均记录哈希。文件改变会拒绝后续操作。输出目录必须新建，不自动覆盖。中断后的部分目录不能当作完成结果；换新目录重跑。

## 4. 决策完成后单独评分

```bash
python -m experiments.split_instances evaluate --result-dir outputs/split_result --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --eval-mask outputs/check3/potential_validation.tif --allow-crs-mismatch --output-dir outputs/split_eval
```

CRS覆盖选项仅沿用先前已知的CRS声明差异处理，不重投影或证明CRS等价。查看 `outputs/split_eval/summary.txt` 中before/after四项指标。全程标点、生成候选、确认导出不读取GT。此实验不改变前景，所以前景IoU应保持一致；重点看实例F1、PQ、实例边界以及实际子ID与GT匹配。ROI裁剪口径、训练独立性未知等限制继续适用，不能将小批开发结果称为独立测试提升。

建议先记录小批粘连案例的拆分成功／拒绝情况、正点数和重复尝试次数，判断SAM是否能生成有用候选，再开展较大规模效率评价。点文件中的浏览器时间包含暂停，属于自报记录，不能单独证明比人工画线更高效。
