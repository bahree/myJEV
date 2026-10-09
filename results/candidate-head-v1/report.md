# Candidate-attention head on frozen Qwen

One original 216,193-parameter head reads the frozen Qwen3.5-0.8B text representations. This is a one-seed teaching experiment with fixed settings. It has a different prompt, runtime and training setup from the matched alias/Clef comparison, so differences cannot be attributed to the head alone.

## Checks before the main run

The disposable head fit 16 training rows in 27 updates: 100.00% training accuracy and loss 0.04116. These rows were used only for a memorization check; the head was discarded.

The 100-update pilot used 1.935 GiB peak PyTorch allocation and took 70.3 seconds. Its saved/reloaded fixture was identical. The backbone remained frozen.

## Main run and held-out decisions

Seed 11 trained for 1,000 updates and 8,000 example exposures. AdamW learning rate was 0.001, without tuning or validation-selected checkpoints. The head has 216,193 trainable parameters and the backbone has 752,393,024 frozen parameters. Mean rendered training length was 1009.2 tokens.

The training session, including checkpoint writes, logging shutdown and its saved fixture check, took 674.3 seconds, with 1.938 GiB peak allocated memory. These measurements exclude later evaluation. Descriptive validation accuracy was 56.97% on 997 examples.

| Confidence | Test accuracy | Macro-F1 | Correctness Brier | Multiclass Brier | ECE | Correctness AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Raw selected probability | 58.02% | 0.5856 | 0.1741 | 0.5599 | 0.0549 | 0.8203 |
| Temperature-scaled selected probability | 58.02% | 0.5856 | 0.1713 | 0.5571 | 0.0387 | 0.8217 |

All 3,080 official BANKING77 test examples are included. Temperature 0.919910 was fitted only on the 1,000 calibration examples. ECE uses 15 equal-width bins. The accuracy group-bootstrap interval [56.45%, 59.82%] conditions on this one trained checkpoint; it does not include training-seed variation.

## Acceptance under the frozen calibration thresholds

Each threshold was selected on calibration data and applied unchanged to test. The empirical error targets do not certify deployment risk. Empty accepted sets have undefined error.

| Calibration target | Test coverage | Accepted | Accepted-case error | One-sided 95% error upper bound |
|---|---:|---:|---:|---:|
| coverage_0.5 | 50.97% | 1570 | 18.66% | 20.35% |
| coverage_0.8 | 79.97% | 2463 | 33.01% | 34.60% |
| coverage_0.9 | 89.74% | 2764 | 37.05% | 38.59% |
| empirical_error_0.01 | 1.98% | 61 | 1.64% | 7.54% |
| empirical_error_0.05 | 16.72% | 515 | 2.33% | 3.75% |

## Seven authored requests

Seven pre-existing authored diagnostics, without example selection. BANKING calibration does not establish calibration on these tasks.

| Request | Expected | Selected | Confidence | Correct |
|---|---|---|---:|---|
| billing | billing | other | 0.9993 | False |
| technical | technical | other | 0.9961 | False |
| none-of-these | other | other | 0.9998 | True |
| refund-day-13 | approve | approve | 0.5684 | True |
| refund-day-14 | deny | deny | 0.5693 | True |
| quoted-instruction | billing | other | 0.9972 | False |
| synthetic-blog-format | tutorial | tutorial | 0.9811 | True |

Every calibrated response matched after a fresh artifact reload. The implementation checks one backbone invocation for each scoring batch. The expected demo answers are never included in model inputs.

## Warm scoring on the A30

The original three-candidate request measured 49.65/50.04 ms p50/p95 over 100 calls after 10 warmups. Peak allocated memory was 1.452 GiB. Warm three-candidate scoring including tokenization, no HTTP. One loaded model on the measured GPU. PyTorch memory excludes CUDA context memory.

The small saved head still needs the Qwen backbone. These local research artifacts use `CandidateHeadModel`; they are separate from the six published adapter releases and the Docker service.

## Reproduce

Follow [the candidate-head guide](../../docs/candidate-head.md) for training and scoring. Run `python scripts/summarize_candidate_head.py` to regenerate this report and both figures from the tracked JSON and training log. Recomputing metrics from saved logits uses the optional evidence bundle and `scripts/replay_head_evaluation.py`.
