# 人工自主选择融合实验（离线页面）

固定使用 `outputs/t256_s192` 同一批已保存的 SAM 滑窗预测。准备阶段验证缓存，生成中心优先、求交、求并三组候选，不读取GT、不重新运行SAM。当前结果固定为中心优先；“保留”与“求并”因此是不同操作。这个起点与前面局部上下文实验的 union 起点不同，汇报中须写清楚。

这是人工辅助融合演示，暂不是QGIS插件。浏览器可点击原图总览的网格定位，查看原图、当前结果、求交、求并，再逐区域选择。黄色是实例边界，掩膜外变暗。256像素网格覆盖整个开发ROI，没有按GT或误差挑选样例。每块只回写ROI内部，块间不重叠；分歧像素数仅供查看，不决定人工选择。未查看的区域默认保留，可明确确认“已查看”而仍选择保留。

## Linux 操作步骤

在项目根目录、sam-exp环境中准备：

```bash
python -m experiments.manual_fusion prepare --run-dir outputs/t256_s192 --output-dir outputs/manual_fusion
```

用文件管理器找到 `outputs/manual_fusion/index.html`，右键用 Firefox/Chrome 打开（不要用PyCharm源码编辑器）。它不需要服务器。保留同目录的PNG，复制演示材料时复制整个目录。点击网格或输入区域ID定位，依据原始影像选择“保留当前结果／求交／求并”。不要打开GT对照选择，也不要先评分再改选择以挑高分。

页面会尝试在浏览器本地保存进度，但不同浏览器对本地文件存储的支持可能不同，建议经常点击“导出 decisions.json”备份。导出的JSON包含全部区域、是否查看、最终选择、选择／确认事件及浏览器会话时间。时间是自报的页面停留时间，含暂停，不等于净操作时间，也不等于系统验证的点击成本。导出多次可能产生带编号文件，请明确使用最后一次文件。

将下载的文件复制为项目中的 `outputs/manual_decisions.json`。可用文件管理器复制；默认下载位置若是 `~/Downloads/decisions.json`，也可执行：

```bash
cp "$HOME/Downloads/decisions.json" outputs/manual_decisions.json
```

先固化人工选择，**本步骤没有GT参数**：

```bash
python -m experiments.manual_fusion apply --project-dir outputs/manual_fusion --decisions outputs/manual_decisions.json --output-dir outputs/manual_result
```

选择固化后单独评分：

```bash
python -m experiments.manual_fusion evaluate --result-dir outputs/manual_result --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/manual_eval
```

最终查看 `outputs/manual_eval/summary.txt`。比较中心优先、全局求交、全局求并、人工局部选择的同ROI IoU、边界F1、实例F1、PQ，同时查看已查看区域数、修改区域数、修改像素、计划回写面积及操作记录。prepare/apply/evaluate的输出目录都必须是新目录。候选／输入被修改、决策来自其他项目、区域重复或遗漏会拒绝应用；评分前检查固化决策和结果哈希。该流程记录选择，但无法从技术上证明操作者没有在外部查看GT，须遵守实验协议。

## QGIS定位与展示

将原始影像与 `outputs/manual_fusion/regions.geojson` 一同加入QGIS。GeoJSON保留来源影像坐标并声明该CRS，是面向QGIS的原生坐标导出，不是强制转成WGS84的通用网络GeoJSON。区域属性包含id、原图像素坐标和分歧像素数。用“识别要素”点网格查看id，再在浏览器输入该id。确认QGIS识别的图层CRS与原图一致；源数据CRS等价性仍需核实，本工具不重投影。

还可叠加三个 `candidates/*/instance.tif` 以及最终 `outputs/manual_result/instance.tif` 检查。此版不会从QGIS点击自动向浏览器发送区域或写入结果；地图定位与页面选择是两步操作。

## 汇报口径

说明“同一组自动候选上增加人工区域选择”，而不是自动算法超过基线。图可直接使用选择页面四列对照，加上最终指标表和人工成本。固定BsiNet实例ID，不新增、拆分或直接剔除整个实例；求交可能删除真实前景，求并可能保留错误区域，没有必然精度提升。只有256×256局部融合选择，本轮不混入裁块尺度选择，以免无法区分改进来源。

当前BsiNet输入可能混有人工修改、训练来源未确认，因此结果属于既有输入上的开发演示，不是纯自动BsiNet的独立测试或论文最终证据。重新推理和数据划分核实留待后续处理。
