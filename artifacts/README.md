# artifacts 目录说明

本目录保存实验报告会直接引用的图表与图表生成脚本。

## 文件列表

### 图表生成脚本

- `generate_report_figures.py`

### 静态图表

- `report_fig_milestone_compare.png`
- `report_fig_checkpoint_trend.png`
- `report_fig_metric_heatmap.png`
- `report_fig_closedloop_validation.png`
- `report_fig_sft1000_loss.png`

### 交互式页面

- `report_interactive_dashboard.html`

## 作用

这些文件用于支撑实验报告中的几个核心判断：

1. `V2.4 -> V2.5` 的里程碑提升
2. `V2.5` 不同 checkpoint 的趋势比较
3. `750-step` 与闭环训练 best checkpoint 的差异
4. 训练 loss 曲线与 early stopping 现象

## 为什么把图表单独放出来

把图和生成脚本单独放在这里，可以更快地看到：

- 结果确实有图表支撑
- 图表不是手工拼的
- 指标比较有独立证据链
