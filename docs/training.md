# Longer matched training comparison

**Status: running. No results from this batch are claimed yet.** The initial feasibility study remains available in [the results guide](experiments.md). This next comparison uses longer exposure, validation-only learning-rate selection, and the complete official BANKING77 test set.

## Frozen schedule

| Setting | Value |
|---|---|
| Sizes | 0.8B, 4B, 9B |
| Tuning seed | 101, separate from main-comparison seeds |
| Learning rates per method | 0.00003 and 0.0001 |
| Tuning exposure | 1,000 updates per trial |
| Selection data | All 997 validation examples |
| Main seeds | 11, 22, 33 |
| Initial supervised exposure | 4,000 updates |
| Each continuation | 4,000 additional updates |
| Main methods | Supervised, continued supervision, exact RL, sampled RL |
| Final evaluation | All 3,080 test examples and 1,000 reserved calibration examples |

Every method gets the same number of learning-rate trials. The fixed selection rule is highest validation accuracy, then lowest selection negative log-likelihood, then lower learning rate. This prioritizes selection quality; confidence and accepted-case error remain separately reported outcomes.

The selected supervised tuning artifact is shared by all continuation trials. The main comparison then starts fresh at each main seed, with a common supervised artifact for the three continuation methods. Continuations replay the seed-matched data RNG and skip the initial exposure, giving each method identical subsequent examples and candidate order. Initial plus continuation exposure covers approximately one pass through the 7,999 training rows; convergence is not guaranteed.

All hyperparameter decisions are frozen before test/calibration evaluation begins. The validation loader checks its partition filename, hash, and training-group isolation. Each runner also verifies hashes of all data partitions and refuses changes to its frozen plan.

## One optimizer, several training approaches

An optimizer is the rule that adjusts trainable parameters using gradients. These runs all use **AdamW**. A training step, also called an optimizer update, adjusts the LoRA adapters and confidence heads while the backbone weights stay frozen. With accumulation set to one in this schedule, each step consumes one training example.

The four approaches are supervised learning, continued supervision, exact RL, and sampled RL. Exact versus sampled describes how the RL objective and gradient are computed. Both still use AdamW to apply parameter updates. Continued supervision is the control that tests whether extra training alone explains a gain attributed to RL.

## Why 168,000 training steps?

The count covers every planned run at all three sizes; it is not the number of optimizers or the training length of a single model.

| Phase | Runs per size | Steps per run | Total steps per size |
|---|---:|---:|---:|
| Tuning: four approaches, two learning rates each | 8 | 1,000 | 8,000 |
| Main comparison: four approaches, three seeds each | 12 | 4,000 | 48,000 |
| Total per size | 20 | Varies | 56,000 |
| All three sizes | 60 | Varies | 168,000 |

Two learning-rate trials give each approach an equal tuning opportunity. Three main seeds expose run-to-run variation. The supervised artifact is trained once per main seed and reused as the starting point for its three continuation branches. Its creation is not counted three times.

This budget is a declared experimental choice, not evidence that this amount of training is optimal or sufficient for convergence. Larger tuning grids or more seeds could improve the study at additional cost. The present schedule keeps those costs fixed and visible.

## What progress and completion time mean

The exact training percentage is completed optimizer steps divided by 168,000. It excludes validation, test evaluation, and post-hoc calibration work. A training counter reaching its end therefore does not mean the entire batch is finished.

An approximate batch percentage can weight the remaining training and evaluation tasks by measured time. We use observed update and inference speeds where available. Until an RL timing is observed, the estimate assumes an RL update takes 1.3 times the supervised update time. A broad allowance around that estimate accounts for uncertainty; it is a planning range, not a statistical confidence interval. Estimates can change as more tasks finish.

The three GPUs run independently. The overall finish estimate is the longest remaining per-GPU duration, not the sum of all three durations. A failed or stopped job makes its completion estimate unavailable until the problem is resolved.

Batch completion means its scheduled training, validation selection, full-test evaluations, and calibration controls are finished. Analysis, broader transfer studies, archive adaptation, artifact releases, and blog publication remain separate project work. The [roadmap](roadmap.md) keeps those boundaries explicit; there is no invented whole-project completion percentage.

## Run it

First complete the [installation and BANKING77 preparation](quickstart.md). The configurations share the original pinned backbone cache. Each command below assigns one independent study to one GPU; only run the assignments your machine supports.

```bash
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=4
export MYJEV_TRACKING_MODE=disabled
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_longer_study.py 0.8b
# In separate terminals on a three-GPU machine:
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_longer_study.py 4b
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/run_longer_study.py 9b
```

There are eight tuning runs and twelve main runs per size, plus validation and full evaluation passes. This is a substantial local workload expected to take many hours. The reference setup uses three 24 GB A30s; it does not rent cloud hardware.

## Check progress and resume

```bash
.venv/bin/python scripts/monitor_longer_study.py --once
.venv/bin/python scripts/monitor_longer_study.py --interval 60
```

The three-size monitor reports each stage, current training step, completed validation trials, and full evaluations. With only one size launched, the other sizes remain marked as starting; use `--once` for a snapshot instead of waiting for all three.

Per-size logs and state live under `results/longer-v1/`; adapters and resumable state live under `artifacts/longer-v1/`. Restart the same runner command to resume. A per-size process lock prevents duplicate runners. Completed stages are skipped; interrupted updates after the most recent checkpoint are preserved separately before replay. A new stage refuses to start below 8 GiB free disk. Do not change the plan in place after starting a study.

## What this batch does not cover

This batch focuses on the four main methods. It does not complete the remaining correctness-only/Brier ablations at longer exposure, replicated precision controls, broad transfer, archive adaptation, or final release selection. Those remain on the [roadmap](roadmap.md). New results should be reported with the actual completed seeds, data exposure, precision, and evaluation scope.
