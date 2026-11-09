# LLM Post Training SFT/DPO

本仓库记录一次面向代码补全任务的 `Qwen2.5-Coder-1.5B` 后训练项目。项目主线是把源码构造成 `FIM`（Fill-in-the-Middle）样本，先做 `SFT`，再探索 `DPO` 是否能在已训练模型基础上继续提升。

最终结论很直接：

- 最终采用模型：`V2.5 fn-only filtered / 750-step`
- 主要有效增益来自：更干净、更贴近 FIM 任务的数据构造 + checkpoint 选择
- `DPO`：早期几轮（合成 / mimic rejected）没有超过 SFT；最终一轮改用 SFT 模型自身错误输出（self-modelgen）构造偏好对，重复、越界与长度失控明显减少，效果优于 SFT 基线
- 最终一轮 DPO 的数据集、日志和 checkpoint 随训练服务器到期丢失，因此本仓库公开的分数均为 SFT `750-step`，DPO 只记录方法与效果方向，不给具体分数
- 训练硬件：华为昇腾 `Ascend 910B3` NPU，64GB HBM 级别
- 训练框架：`MindSpeed-LLM + PyTorch + torch_npu + HCCL`

## 项目做了什么

这个项目不是从零预训练模型，而是在已有代码模型上做任务型后训练：

```text
Qwen2.5-Coder-1.5B
  -> FIM 数据构造
  -> 数据过滤与质量审计
  -> SFT 训练
  -> 多 checkpoint 固定评测
  -> 选择 750-step checkpoint
  -> DPO 偏好学习实验
  -> MCore checkpoint 导出为 HuggingFace 格式
```

FIM 任务形式：

```text
input  = prefix + suffix
target = middle
goal   = 在给定前后文时补全中间代码
```

相比普通 left-to-right 续写，FIM 更接近真实代码编辑场景，但也更容易出现边界错位、重复补全、结构不完整、输出过长等问题。

## 最终结果

| 项目 | 结果 |
| --- | --- |
| Base model | `Qwen2.5-Coder-1.5B` |
| Final SFT data | `fim_v2_5_fnonly_filtered` |
| Final checkpoint | `750-step` |
| Local fixed eval | `0.3449` |
| Export format | `HuggingFace` |
| Hardware | Huawei Ascend `910B3`, 64GB HBM class |

公开 benchmark-style 结果快照：

| Metric | Value |
| --- | ---: |
| Score | `84.9073` |
| EM | `73.953013` |
| ES | `86.778345` |
| TLE pass rate | `87.7426` |
| Non repetition rate | `99.6936` |

私有评测入口、个人账号、服务器绝对路径、完整日志和 checkpoint 不放入公开仓库。

## 训练过程

训练不是一次长训到底，而是先跑通闭环，再逐步扩大实验：

1. 打通 `HF -> MCore` 模型转换。
2. 打通 FIM 数据转换、训练、checkpoint 保存。
3. 用小步数训练确认环境、日志、评测都可用。
4. 构造 `V2.1 clean-path`，验证清洗数据是否有效。
5. 扩展到 `V2.4 XXL clean-path`，测试更大规模 clean 数据。
6. 收紧到 `V2.5 fn-only filtered`，重点过滤重复风险、边界续写风险和过长 middle。
7. 对 `250 / 500 / 750 / 1000 step` 做同口径 fixed eval。
8. 选择 fixed eval 最好的 `750-step`。
9. 以 `750-step` 为起点做 `DPO`：早期几轮验证链路但没有提分；最终一轮改用 self-modelgen rejected，效果优于 SFT 基线。
10. 将最终 MCore checkpoint 导出为 HuggingFace 格式。

代表性 SFT 参数：

| 参数 | 值 |
| --- | --- |
| Tensor Parallel | `1` |
| Pipeline Parallel | `1` |
| Micro Batch Size | `1` |
| Global Batch Size | `8` |
| Sequence Length | `4096` |
| Learning Rate | `7.5e-7` |
| Min LR | `7.5e-8` |
| LR Decay | `cosine` |
| Precision | `bf16` |

## 数据集

最终 SFT 数据集：

```text
fim_v2_5_fnonly_filtered
train: 69931
valid: 8729
test:  8610
full:  87270
```

关键过滤规则：

- 只保留 `function_body` / `method_body`
- 去掉 `middle_len > 899`
- 去掉 `repetition_risk`
- 去掉 `boundary_continuation`

这样做的目的不是追求样本越多越好，而是让训练样本更像“真实可学习的代码补全任务”。

详见：

- [dataset/final_dataset_manifest.md](dataset/final_dataset_manifest.md)
- [dataset/data_construction_pipeline.md](dataset/data_construction_pipeline.md)

## DPO 实验记录

DPO 的目标是让模型学会区分同一个 FIM prompt 下更好和更差的补全：

```text
same prefix + suffix
chosen   = better middle
rejected = worse middle
```

本项目尝试过几类 DPO 数据：

- `DPO smoke 20-step`：验证训练链路能否跑通
- `MiMo` synthetic rejected：早期合成偏好对
- `Codex/manual` hard negative：小批量人工/辅助构造负样本
- `modelgen` rejected：用当前 SFT 模型真实生成错误构造偏好对
- `modelgen + GPT mimic`：用真实错误做锚点，再扩量 mimic 数据

关键结论：

- DPO 训练、回拉评测、scorer 链路已经跑通。
- 早期 `DPO smoke 20-step` 与 `V2.5 750-step` 打平，没有带来独立提升；扩量后的 MiMo / mimic 数据也没有稳定超过 SFT。
- 根因主要在偏好对质量和训练强度：`rejected` 如果不够贴近真实模型错误，DPO 很容易学不到对 fixed eval 有用的边界控制；默认弱学习率下输出几乎不变。
- 最终一轮据此调整：以 `750-step` 为初始化和 reference，对 held-out FIM prompt 多采样生成候选，按相似度区间、结构完整性、后缀越界、重复窗口和长度比例筛出“看似合理但错误”的输出作为 `rejected`，gold middle 作为 `chosen`，构建约 5k–6k 对偏好数据，并对比了更强的更新设置（`lr=5e-7`）。
- 最终一轮 DPO 后，重复、越界抄后缀和长度失控明显减少，效果优于 SFT 基线。

说明：DPO 部分当时更偏“快速验证和冲刺实验”。早期几轮的中间程序和评测产物保留了；最终一轮的数据集、日志和 checkpoint 留在训练服务器上，随服务器到期丢失。因此公开仓库采用诚实记录方式：保留可解释的实验脉络、代表脚本、最终结论和失败原因，最终一轮只写方法和效果方向，不补造分数或日志。

详见：

- [docs/DPO_EXPERIMENT_RECORD.md](docs/DPO_EXPERIMENT_RECORD.md)
- [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)

## 仓库结构

```text
.
├── README.md
├── EXPERIMENT_REPORT.md
├── RESULTS.md
├── PUBLICATION_CHECKLIST.md
├── dataset/
│   ├── README.md
│   ├── final_dataset_manifest.md
│   └── data_construction_pipeline.md
├── docs/
│   ├── ASSET_INDEX.md
│   ├── DPO_EXPERIMENT_RECORD.md
│   ├── REPO_STRUCTURE.md
│   └── REPRO_GUIDE.md
├── scripts/
│   ├── README.md
│   ├── run_fim_v2_5_fnonly_filtered_sft.sh
│   ├── eval_official_mixed_set2_v2_5_750.sh
│   ├── prepare_submit_v2_5_750_container.sh
│   └── score_fim_eval_predictions.py
└── artifacts/
    ├── README.md
    ├── generate_report_figures.py
    ├── report_fig_*.png
    └── report_interactive_dashboard.html
```

## 环境

原始实验使用本地 + 远端分层环境：

```text
Local orchestration: Windows / PowerShell
Remote training: Linux container on Huawei Ascend NPU
Hardware: Huawei Ascend 910B3, 64GB HBM class
Framework: MindSpeed-LLM + PyTorch + torch_npu + HCCL
Base model: Qwen2.5-Coder-1.5B
```

脚本中使用公开占位路径和环境变量。迁移到新机器时，需要替换：

```text
WORK_ROOT
RUNTIME_REPO
FIM_TOOL_DIR
EVAL_ROOT
SOURCE_DATA
MANIFEST
ASCEND_RT_VISIBLE_DEVICES
MASTER_PORT
```

## 不包含的内容

公开仓库不包含：

- 完整 checkpoint
- 完整原始训练数据
- 完整训练日志
- 私有服务器路径
- 个人账号和私有运行记录

这个仓库保留的是公开可读层：方法、流程、关键结果、代表脚本、数据构建说明和图表。
