# Dataset Notes

This directory explains the dataset branches, construction rules, and quality checks used by the project.

The public repository does not include full training data files because raw and converted corpora can be large and may carry license or privacy constraints. Instead, it keeps:

- final dataset manifest
- data construction pipeline
- filtering rules
- key statistics and quality checks

## Final Dataset Branches

- SFT: `fim_v2_5_fnonly_filtered`
- DPO: `fim_v2_5_dpo_modelgen110_gptmimic9758`

Read:

- [final_dataset_manifest.md](final_dataset_manifest.md)
- [data_construction_pipeline.md](data_construction_pipeline.md)

## Expected Reproduction Layout

For reproduction, place the adopted dataset artifacts under a workspace-specific data root, for example:

```text
${WORK_ROOT}/datasets/fim_sft/v2_5_fnonly_filtered_20260519/
${WORK_ROOT}/datasets/fim_dpo/fim_v2_5_dpo_modelgen110_gptmimic9758/
```

Then update the script variables that point to `SOURCE_DATA`, `MANIFEST`, and DPO training files.
