# 中心区域扩大上下文对照

本阶段比较原始影像中同一位置的 256×256 与 512×512 裁块。512 的边长加倍、面积为四倍，不是把 256 图片插值放大。SAM 自身仍执行其输入预处理。

沿用已验证的求并缓存作为共同起点 `outputs/fusion/union`，从缓存投票分歧选择问题区域，同时生成匹配随机对照。两尺度使用完全相同的位置、回写掩膜、原始 BsiNet 提示来源和候选接受规则。每个回写掩膜不超过 256×256，裁块围绕其中心放置；掩膜较小时，256 裁块本身也包含周围影像。中心坐标可在 `patches.json` 中核对。不会重新运行全图滑窗，也不会处理上下文中回写范围以外的像素。

这次保留 union 作为起点，以衔接之前的局部处理实验；它不代表全局融合实验的最佳选择。四个推理组为 conflict_s256、random_s256、conflict_s512、random_s512。没有接受候选的组件保留共同起点。固定 BsiNet ID，不发现新实例或拆分田块。此次不叠加边界约束，以便直接检验上下文本身的作用。

先在 Linux 项目根目录、sam-exp 环境中规划（不加载 SAM、不读取 GT）：

```bash
python -m experiments.compare_contexts --union-dir outputs/fusion/union --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --plan-only --output-dir outputs/context_plan
```

确认成功后使用另一个新目录运行：

```bash
python -m experiments.compare_contexts --union-dir outputs/fusion/union --checkpoint "$HOME/xy/sam/segment-anything/sam_vit_b_01ec64.pth" --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/contexts
```

可显式指定 `--context-sizes 256 512`，最小尺寸必须不小于 `--core-size`（默认256），所有尺寸须同奇偶以保证中心一致。裁块越过影像边缘时无效填充，不将填充值当作有效前景。保留来源目录和 work 缓存；输出目录已存在会拒绝覆盖，中断后保留部分结果，重跑须另取目录。

输出 `plan.json`、两个共享的 `*_write.tif`、四组完整范围的 `instance.tif`、各组原图裁块预测 `patch_*.tif`、含原图位置和接受日志的 `patches.json`，以及 `report.json`、`comparison.csv`、`summary.txt`。GT 在四组推理完成后才读取。summary 比较 BsiNet、原始 union、四个处理组的全 ROI 指标及相对 union 的有害／有益删除量。report 的 write_evaluation 在同一固定冲突回写区和随机回写区分别评价所有方法，注意窄掩膜的边界分母可能为零。

判断扩大上下文是否有用，应成对比较 conflict_s512 与 conflict_s256，同时查看随机组、全 ROI、固定回写区、删除／新增像素及耗时。上下文可能改变连通组件、提示数和接受数，因此这是局部处理流程的上下文对照，并非严格相等的解码预算，也不能单独归因于某一个 SAM 提示。扩大上下文不保证提升，可能引入相邻田块干扰。

当前为已反复使用的开发 ROI，训练来源独立性未确认，实例指标按 ROI 裁剪计算。本实验不是独立测试或显著性检验；不根据 GT 选位置、提示或逐块挑选较高分尺度。为大田块专门设计的完整实例多尺度处理仍需后续实验。
