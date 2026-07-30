# Asset Index

This repository keeps the public documentation layer of the project. Large runtime assets are intentionally excluded.

## Included

- experiment report
- result summary
- dataset construction notes
- representative training/evaluation/export scripts
- generated figures
- static dashboard export

## Excluded

- raw source-code corpora
- converted parquet/json training datasets
- full checkpoints
- full training logs
- private server paths
- personal account names
- private submission records

## Expected Local Layout For Reproduction

A reproduction workspace can use any equivalent layout. The scripts assume these logical locations:

```text
WORK_ROOT/
  datasets/
  runs/
  eval/
  archives/

RUNTIME_REPO/
  MindSpeed-LLM runtime files
  model_from_hf/
  model_weights/

FIM_TOOL_DIR/
  data conversion and launch helper scripts
```

Set those paths with environment variables before running the scripts.
