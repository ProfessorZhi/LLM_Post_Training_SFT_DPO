# Repository Structure

This repository is a public, lightweight project record. It keeps the parts that help readers understand and reuse the workflow, while excluding private or oversized artifacts.

## Root Files

- [README.md](../README.md): project overview and reading order
- [EXPERIMENT_REPORT.md](../EXPERIMENT_REPORT.md): main experiment report
- [RESULTS.md](../RESULTS.md): selected results and checkpoint comparison
- [PUBLICATION_CHECKLIST.md](../PUBLICATION_CHECKLIST.md): public-release sanity checklist

## `dataset/`

Dataset notes and construction logic:

- [dataset/README.md](../dataset/README.md): why the repository includes dataset documentation instead of full raw data
- [dataset/final_dataset_manifest.md](../dataset/final_dataset_manifest.md): final SFT and DPO data branches
- [dataset/data_construction_pipeline.md](../dataset/data_construction_pipeline.md): how source-code snippets become FIM training samples

## `docs/`

Supporting documentation:

- [REPRO_GUIDE.md](REPRO_GUIDE.md): runtime, script entrypoints, and reproduction notes
- [ASSET_INDEX.md](ASSET_INDEX.md): public artifact map and excluded artifact policy
- [REPO_STRUCTURE.md](REPO_STRUCTURE.md): this file

## `scripts/`

Representative scripts:

- `run_fim_v2_5_fnonly_filtered_sft.sh`: SFT training entrypoint
- `eval_official_mixed_set2_v2_5_750.sh`: fixed evaluation entrypoint
- `prepare_submit_v2_5_750_container.sh`: MCore-to-HuggingFace export entrypoint
- `score_fim_eval_predictions.py`: local FIM prediction scorer

The scripts are meant as reproducible references. They still require a configured Ascend/MindSpeed-LLM runtime and local dataset/checkpoint paths.

## `artifacts/`

Generated report figures and a static dashboard export:

- checkpoint trend figure
- milestone comparison figure
- metric heatmap
- closed-loop validation figure
- loss curve figure
- `generate_report_figures.py`
- `report_interactive_dashboard.html`

## Environment Assumptions

```text
Local orchestration: Windows / PowerShell or equivalent shell environment
Remote training: Linux container
Hardware: Huawei Ascend 910B3 NPU, 64GB HBM class
Framework: MindSpeed-LLM + PyTorch + torch_npu + HCCL
```

Migrating to a different Ascend machine mainly requires replacing path variables and device visibility settings. Migrating to NVIDIA GPUs would require rewriting the launch path because this project uses `torch_npu` and HCCL rather than CUDA/NCCL.

## Excluded From Public Git

- full checkpoints
- raw and intermediate large datasets
- full training logs
- private local paths
- personal account names
- private submission or leaderboard records
