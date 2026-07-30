# Results

## Selected Version

The final selected model branch is:

```text
V2.5 fn-only filtered / 750-step
```

## Fixed Evaluation

The 750-step checkpoint was selected because it gave the best fixed-evaluation result among the compared SFT checkpoints.

| Branch | Fixed eval score | Notes |
| --- | ---: | --- |
| `V2.5 750-step` | `0.3449` | selected SFT checkpoint |
| DPO branches | below selected SFT | kept as diagnostic experiments |

## Public Benchmark Snapshot

A public benchmark-style evaluation snapshot from the project record is:

| Metric | Value |
| --- | ---: |
| Score | `84.9073` |
| EM | `73.953013` |
| ES | `86.778345` |
| TLE pass rate | `87.7426` |
| Non repetition rate | `99.6936` |

The benchmark identity and private submission metadata are intentionally omitted from this public repository.

## Interpretation

The main gain came from improving FIM data quality and selecting the right SFT checkpoint. DPO completed the training and evaluation path, but the available preference data did not produce a stable improvement over the selected SFT checkpoint.
