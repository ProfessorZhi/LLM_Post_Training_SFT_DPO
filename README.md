# LLM FIM Post-Training with SFT and DPO

This repository documents a post-training project for code completion with `Qwen2.5-Coder-1.5B`.

The project focuses on `FIM` (Fill-in-the-Middle) style code completion. It builds a filtered FIM dataset, runs supervised fine-tuning, compares checkpoints with a fixed evaluation pipeline, and records a DPO exploration branch that did not outperform the final SFT checkpoint.

## Highlights

- Base model: `Qwen2.5-Coder-1.5B`
- Task: FIM code completion
- Main method: SFT on a filtered FIM corpus
- Preference-learning exploration: DPO with model-generated and synthetic preference pairs
- Final selected checkpoint: `V2.5 fn-only filtered / 750-step`
- Training hardware: Huawei Ascend `910B3` NPU, 64GB HBM class
- Training stack: `MindSpeed-LLM`, `PyTorch`, `torch_npu`, `HCCL`

## What To Read First

1. [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)
2. [RESULTS.md](RESULTS.md)
3. [dataset/final_dataset_manifest.md](dataset/final_dataset_manifest.md)
4. [docs/REPRO_GUIDE.md](docs/REPRO_GUIDE.md)
5. [scripts/README.md](scripts/README.md)
6. [artifacts/README.md](artifacts/README.md)

## Repository Structure

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

## Project Flow

```text
Source-code corpus
  -> FIM sample construction
  -> quality filtering and manifest audit
  -> SFT training with MindSpeed-LLM
  -> fixed checkpoint evaluation
  -> 750-step checkpoint selection
  -> DPO branch analysis
  -> HuggingFace-format export
```

## Environment

The original experiments used a split local/remote workflow:

```text
Local orchestration: Windows / PowerShell
Remote training: Linux container on Huawei Ascend NPU hardware
Hardware: Huawei Ascend 910B3, 64GB HBM class
Framework: MindSpeed-LLM + PyTorch + torch_npu + HCCL
Base model: Qwen2.5-Coder-1.5B
```

The shell scripts in [scripts/](scripts/) use environment variables for runtime paths. Replace `WORK_ROOT`, `RUNTIME_REPO`, `FIM_TOOL_DIR`, checkpoint paths, and NPU visibility settings for a new machine.

## What Is Not Included

This public repository intentionally excludes large or private artifacts:

- full model checkpoints
- complete raw training data
- full training logs
- server-specific runtime directories
- personal account names, local absolute paths, and private submission records

The included files are the public, reviewable layer: methodology, selected results, dataset construction notes, representative scripts, and generated figures.
