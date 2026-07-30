# Reproduction Guide

## 1. Objective

This project fine-tunes `Qwen2.5-Coder-1.5B` for `FIM` code completion and evaluates whether filtered SFT data and a later DPO branch improve completion quality.

## 2. Final Selected Version

- Dataset branch: `V2.5 fn-only filtered`
- Selected checkpoint: `750-step`
- Export format: `HuggingFace`

## 3. Runtime Environment

The original run used Huawei Ascend NPU hardware rather than an NVIDIA CUDA GPU:

```text
Hardware: Huawei Ascend 910B3
HBM: 64GB class
Container/runtime: Linux container with Ascend toolkit
Framework: MindSpeed-LLM + PyTorch + torch_npu + HCCL
```

`torch.cuda.is_available() == False` is expected in this environment and does not imply that the NPU runtime is unavailable. For a fresh machine, verify `torch_npu`, HCCL communication, and the Ascend toolkit before launching training.

## 4. Key Steps

The practical workflow has four layers:

1. Data preparation and filtering
2. SFT training
3. Fixed evaluation
4. HuggingFace-format export

## 5. Script Entrypoints

### SFT Training

[scripts/run_fim_v2_5_fnonly_filtered_sft.sh](../scripts/run_fim_v2_5_fnonly_filtered_sft.sh)

This script shows the representative SFT launch flow:

- load the filtered FIM source parquet
- convert it into the training format expected by MindSpeed-LLM
- launch SFT with configurable iteration and save intervals
- record logs, checkpoint paths, and a run manifest

### Fixed Evaluation

[scripts/eval_official_mixed_set2_v2_5_750.sh](../scripts/eval_official_mixed_set2_v2_5_750.sh)

This script demonstrates the evaluation path for the selected 750-step checkpoint. Replace the evaluation root and input JSONL path with the benchmark you are allowed to use.

### Local Scorer

[scripts/score_fim_eval_predictions.py](../scripts/score_fim_eval_predictions.py)

Scores prediction JSONL files with exact match, similarity, syntax, boundary, repetition, and length-related signals.

### HF Export

[scripts/prepare_submit_v2_5_750_container.sh](../scripts/prepare_submit_v2_5_750_container.sh)

Exports the selected MindSpeed/MCore checkpoint into a HuggingFace-style directory and copies required tokenizer/config files.

## 6. Required Path Changes

Before running the scripts on another machine, set or edit:

```text
WORK_ROOT
RUNTIME_REPO
FIM_TOOL_DIR
EVAL_ROOT
SOURCE_DATA
MANIFEST
BASE_HF_DIR
ASCEND_RT_VISIBLE_DEVICES
MASTER_PORT
```

The scripts intentionally avoid hard-coded personal account names in this public version.

## 7. What This Repository Can Reproduce

This repository is a lightweight public record. It can reproduce the structure of the training/evaluation/export workflow, but it does not include full private artifacts such as raw datasets, checkpoints, or full training logs.
