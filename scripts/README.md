# Scripts

This directory keeps representative scripts for the public project record.

## `run_fim_v2_5_fnonly_filtered_sft.sh`

Representative SFT training entrypoint:

- reads the filtered FIM source dataset
- converts it into the MindSpeed-LLM training format
- records a run manifest
- writes checkpoints and logs under `WORK_ROOT`

## `eval_official_mixed_set2_v2_5_750.sh`

Representative fixed-evaluation entrypoint:

- creates a checkpoint view for the selected 750-step checkpoint
- points the evaluator to the base tokenizer and MCore checkpoint
- writes result files under `WORK_ROOT/eval`

Replace `EVAL_ROOT` and the input JSONL path with a benchmark you are allowed to use.

## `prepare_submit_v2_5_750_container.sh`

Representative MCore-to-HuggingFace export script:

- creates a checkpoint view
- runs `convert_ckpt.py`
- copies tokenizer/config files into the exported HF directory
- checks that model weight files exist after conversion

## `score_fim_eval_predictions.py`

Local scorer for prediction JSONL files. It computes exact-match, similarity, syntax, boundary, repetition, and length-related metrics.

## Required Environment Variables

The shell scripts use public placeholder defaults. Set these for your own machine:

```text
WORK_ROOT=/path/to/project-workspace
RUNTIME_REPO=/path/to/MindSpeed-LLM
FIM_TOOL_DIR=/path/to/fim-tools/fim
EVAL_ROOT=/path/to/eval-benchmark
ASCEND_RT_VISIBLE_DEVICES=0
MASTER_PORT=16864
```

The scripts assume an Ascend NPU runtime with `torch_npu` and HCCL available.
