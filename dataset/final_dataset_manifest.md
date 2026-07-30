# 最终采用数据集 Manifest

## 一句话结论

本项目最终采用的数据主线是：

- SFT：`fim_v2_5_fnonly_filtered`
- DPO：`fim_v2_5_dpo_modelgen110_gptmimic9758`

推荐 adopted 数据目录：

```text
${WORK_ROOT}/datasets/_FINAL_ADOPTED_20260524/
```

## SFT 数据集

数据集 ID：

```text
fim_v2_5_fnonly_filtered
```

推荐目录：

```text
${WORK_ROOT}/datasets/_FINAL_ADOPTED_20260524/sft/v2_5_fnonly_filtered_20260519/
```

训练入口文件：

```text
${WORK_ROOT}/datasets/_FINAL_ADOPTED_20260524/sft/v2_5_fnonly_filtered_20260519/sft_source/fim_v2_5_fnonly_filtered_train.source.parquet
```

切分与规模：

```text
train: 69931
valid: 8729
test:  8610
full:  87270
```

核心过滤原则：

- 只保留 `function_body` 与 `method_body`
- 去掉 `middle_len > 899`
- 去掉 `repetition_risk`
- 去掉 `boundary_continuation`

选择原因：

- 相比更早版本，目标更聚焦在真实函数体补全
- 明显压缩了重复续写和结构越界风险
- 作为后续 DPO 的底座更稳定

## DPO 数据集

数据集 ID：

```text
fim_v2_5_dpo_modelgen110_gptmimic9758
```

推荐目录：

```text
${WORK_ROOT}/datasets/_FINAL_ADOPTED_20260524/dpo/fim_v2_5_dpo_modelgen110_gptmimic9758/
```

训练入口文件：

```text
${WORK_ROOT}/datasets/_FINAL_ADOPTED_20260524/dpo/fim_v2_5_dpo_modelgen110_gptmimic9758/fim_v2_5_dpo_modelgen110_gptmimic9758_9868_alpaca_pairwise.json
```

规模：

```text
total pairs:     9868
modelgen_strict: 110
gpt_mimic:       9758
```

质量检查结果：

```text
prompt+chosen duplicate: 0
full triple duplicate:   0
chosen == rejected:      0
FIM token / markdown leak in rejected: 0
```

选择原因：

- `chosen` 统一使用真实 held-out FIM 样本的 gold middle
- `rejected` 以真实模型生成错误为基底，再用 GPT mimic 做受控扩量
- 相比早期 MiMo / pilot 数据，更接近正式 DPO 训练口径

## 非最终主线但保留的历史数据

以下目录作为实验记录保留，但不属于当前最终主训练入口：

- 早期 `V1` / `V2.1` / `V2.2` / `V2.4` SFT 分支
- `MiMo` synthetic rejected 分支
- `Codex manual` hard-negative 小批次分支
- 早期 modelgen pilot / style mimic 目录

它们用于：

- 保留实验演化轨迹
- 支撑报告中的消融与方法比较
- 为后续复现实验提供历史证据
