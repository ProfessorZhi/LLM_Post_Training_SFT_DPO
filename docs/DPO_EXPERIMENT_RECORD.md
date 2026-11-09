# DPO Experiment Record

## 1. Why DPO Was Tried

After SFT, the model could usually generate code that was close to the target, but several failure modes remained:

- repeated fragments
- boundary continuation into the suffix
- incomplete multiline structure
- overly long or overly short middle completions
- semantically close but reference-mismatched code

DPO was introduced to test whether preference learning could push the model away from these common wrong completions.

## 2. Preference Data Definition

Each DPO sample follows the same FIM task:

```text
prompt   = prefix + suffix
chosen   = preferred middle
rejected = worse middle
```

The key constraint is that `chosen` and `rejected` must answer the same prompt. If the rejected sample is random, unrelated, or too easy to distinguish, DPO may reduce loss without improving the actual FIM evaluation.

## 3. Data Branches Tried

### Smoke Test

Purpose:

- verify DPO training could launch
- verify checkpoint save/load
- verify local evaluation after DPO

Representative result:

| Model | Fixed eval score | Result |
| --- | ---: | --- |
| `V2.5 750-step SFT` | `0.3449` | baseline |
| `DPO smoke 20-step` | `0.3449` | tied baseline |

Conclusion: the DPO pipeline worked, but this first run did not improve the selected SFT checkpoint.

### Synthetic Rejected Branch

Purpose:

- create quick rejected answers without waiting for large model-generation jobs
- test whether obvious bad completions can provide a useful preference signal

Observed issue:

- synthetic rejected samples can be too far from the model's real error distribution
- DPO can learn a superficial distinction without fixing boundary/repetition problems in fixed eval

### Manual / Assisted Hard Negatives

Purpose:

- construct smaller but more realistic bad completions
- focus on boundary, repetition, and structure errors

Observed issue:

- quality is better
- scale is too small for a stable training signal

### Model-Generated Rejected Branch

Purpose:

- use the SFT model's own wrong outputs as rejected samples
- align DPO data with actual inference-time failures

Observed issue:

- real model-generated rejected samples are valuable but expensive to collect and filter
- strict filtering leaves a relatively small usable set

### Modelgen + GPT Mimic Branch

Purpose:

- keep real model errors as anchors
- expand preference pairs with mimic-style rejected completions

Final adopted DPO data branch:

```text
fim_v2_5_dpo_modelgen110_gptmimic9758
total pairs:     9868
modelgen_strict: 110
gpt_mimic:       9758
```

Quality checks:

```text
prompt+chosen duplicate: 0
full triple duplicate:   0
chosen == rejected:      0
FIM token / markdown leak in rejected: 0
```

## 4. Why the Early DPO Rounds Did Not Beat SFT

The early DPO rounds recorded above did not produce a stable fixed-eval improvement over `V2.5 fn-only filtered / 750-step`.

Likely reasons:

- The strongest rejected samples were too few.
- Synthetic/mimic rejected samples may not match the actual model error distribution closely enough.
- Some preference pairs improve surface preference loss but do not directly improve exact middle reconstruction.
- FIM fixed eval is sensitive to boundary length and syntax; DPO data needs to target those errors very precisely.
- Training strength was kept conservative to avoid damaging the already useful SFT checkpoint.

## 4.1 Final Round: Self-Modelgen DPO

Based on the diagnosis above, a final round changed how rejected samples were built:

- Initialization and reference: `V2.5 fn-only filtered / 750-step`.
- Rejected: multiple samples from the SFT model on held-out FIM prompts, filtered by similarity band, structural completeness, suffix overlap, repetition window and length ratio to keep plausible-but-wrong outputs. Chosen: the gold middle.
- Scale: roughly 5k–6k preference pairs; a stronger update setting (`lr=5e-7`) was compared against the default weak update.
- Result: repetition, suffix copying and length blow-ups dropped clearly, and the model outperformed the SFT baseline on the fixed FIM eval.

The final dataset, logs and checkpoint were stored on the training server and were lost when it expired. This record therefore states the method and the direction of the effect only, without scores.

## 5. Record-Keeping Status

The DPO work was done as a late-stage experimental branch. Representative scripts, dataset manifests, quality checks, and final conclusions were preserved, but not every intermediate interactive operation was logged as a formal experiment record.

This repository therefore does not pretend that the DPO branch has a perfect audit trail. It records:

- what data branches were tried
- what each branch was meant to test
- what result changed model-selection decisions
- why the final model stayed with SFT

## 6. What Would Be Improved Next

The next DPO iteration should prioritize:

- collecting more real model-generated rejected completions
- filtering for boundary/repetition/structure errors specifically
- comparing prediction diffs before relying on loss
- testing stronger but controlled DPO learning rates
- evaluating on a larger fixed set before replacing the SFT checkpoint

The shortest next path is not "more synthetic pairs"; it is "more real, targeted rejected samples from the current model."
