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
