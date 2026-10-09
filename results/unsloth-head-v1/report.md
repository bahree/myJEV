# Alias readout and Clef head on the same Qwen backbone

The separate answer-only supervised experiment uses Qwen3.5-0.8B. Both arms use the same BF16 backbone revision and rank-8 adapters, but their prompts and answer readouts differ. The completed supervision/RL study and released default are unchanged.

## Feasibility pilot

Each pilot completed 100 updates with eight examples per update. Memory is the PyTorch peak allocated on an A30; it excludes CUDA context memory. Time includes first-step kernel preparation and optimizer checkpointing.

| Readout | Trainable parameters | Peak allocated GiB | Training seconds | Mean input tokens |
|---|---:|---:|---:|---:|
| alias | 540,672 | 2.241 | 194.1 | 875.8 |
| clef | 27,865,604 | 3.014 | 331.5 | 1908.1 |

The pilots establish runtime feasibility. Initial adapter hashes and example/order hashes agree across the paired pilots. A prior compiled attempt failed on its first backward pass. Both reported pilots use eager execution, with no silent input truncation.

## Validation-only learning-rate selection

| Readout | Learning rate | Validation accuracy | Selection NLL |
|---|---:|---:|---:|
| alias | 3e-05 | 60.88% | 1.9408 |
| alias | 0.0001 | 67.30% | 1.5628 |
| clef | 3e-05 | 65.50% | 1.2303 |
| clef | 0.0001 | 70.81% | 1.0538 |

Selected rates: alias 0.0001, clef 0.0001. Each trial received 250 updates on seed 101. The frozen rule used validation accuracy, then NLL, then the lower rate. No test data selected these settings.

## All three seeds

Temperatures were fitted on calibration only, after separate learning-rate selection on validation. Confidence here is the selected option probability. Lower Brier is better; ECE uses fifteen equal-width bins.

| Seed | Readout | Accuracy % | Raw correctness Brier | Calibrated correctness Brier | Calibrated multiclass Brier | Calibrated ECE | Correctness AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| 11 | alias | 81.88 | 0.1086 | 0.1042 | 0.2600 | 0.0141 | 0.8674 |
| 11 | clef | 83.73 | 0.1088 | 0.0989 | 0.2374 | 0.0181 | 0.8720 |
| 22 | alias | 80.42 | 0.1182 | 0.1182 | 0.2885 | 0.0184 | 0.8441 |
| 22 | clef | 83.41 | 0.0966 | 0.0967 | 0.2394 | 0.0168 | 0.8758 |
| 33 | alias | 81.72 | 0.1130 | 0.1131 | 0.2787 | 0.0269 | 0.8474 |
| 33 | clef | 83.34 | 0.1177 | 0.1021 | 0.2463 | 0.0255 | 0.8600 |

## Paired accuracy difference

Clef minus alias, in percentage points, for seeds 11 / 22 / 33: +1.85 / +2.99 / +1.62.

The mean difference is +2.15 points. The conditional test-group interval is [+1.40, +2.84] points; the sample seed standard deviation is 0.73 points. The interval holds these three trained seed pairs fixed and does not include training variance.

## Accepting decisions or asking for review

Thresholds are selected on calibration, then applied unchanged to test. The operating points describe observed errors and coverage without certifying deployment risk. Zero accepted cases have undefined error.

| Seed | Readout | Test coverage at calibration 80% threshold | Accepted-case error | Accepted cases | Group-bootstrap error interval |
|---|---|---:|---:|---:|---|
| 11 | alias | 83.70% | 10.32% | 2578 | [9.15%, 11.44%] |
| 11 | clef | 83.90% | 9.29% | 2584 | [8.24%, 10.26%] |
| 22 | alias | 82.60% | 12.46% | 2544 | [11.16%, 13.72%] |
| 22 | clef | 82.11% | 8.15% | 2529 | [7.12%, 9.21%] |
| 33 | alias | 82.99% | 11.07% | 2556 | [9.82%, 12.32%] |
| 33 | clef | 82.79% | 9.18% | 2550 | [8.00%, 10.28%] |

## Training time and memory

Each main run received 1,000 updates and 8,000 examples. Session time includes training, checkpoint work, logging shutdown and the saved fixture check; it is elapsed time on its assigned GPU rather than a kernel-only measurement. The three seed workers ran on separate A30s.

| Seed | Readout | Training seconds | Peak allocated GiB | Mean training tokens |
|---|---|---:|---:|---:|
| 11 | alias | 1027.9 | 2.091 | 875.5 |
| 11 | clef | 2451.8 | 3.024 | 1907.7 |
| 22 | alias | 1012.4 | 2.091 | 875.5 |
| 22 | clef | 2405.6 | 3.024 | 1907.7 |
| 33 | alias | 1003.4 | 2.091 | 875.5 |
| 33 | clef | 2376.2 | 3.024 | 1907.7 |

## Candidate order with calibration held fixed

Seed 11 checkpoints scored three deterministic per-request shuffles of all 3,080 test requests. Temperature and every acceptance threshold stayed at their original calibration values. Each row describes one permutation; the repeated requests are not pooled as independent evidence.

| Readout | Permutation seed | Accuracy | Changed selected ID | Coverage at original 80% threshold | Accepted-case error |
|---|---:|---:|---:|---:|---:|
| alias | 101 | 83.12% | 10.23% | 83.77% | 9.73% |
| alias | 202 | 83.44% | 10.42% | 82.99% | 9.51% |
| alias | 303 | 81.75% | 11.10% | 83.90% | 10.53% |
| clef | 101 | 84.74% | 7.56% | 84.71% | 8.39% |
| clef | 202 | 84.22% | 8.25% | 84.45% | 8.54% |
| clef | 303 | 84.38% | 8.15% | 84.64% | 8.71% |

## Warm scoring

One request at a time, ten warmups and 100 measured calls on an otherwise idle target A30. Other GPUs may run other seeds. Timings include tokenization and result conversion, exclude HTTP, and use eager execution with the reference causal-convolution implementation.

| Readout | Candidates | Tokens | p50 ms | p95 ms | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| alias | 3 | 92 | 44.05 | 47.76 | 1.609 |
| alias | 77 | 869 | 44.96 | 46.76 | 1.660 |
| clef | 3 | 169 | 50.25 | 50.90 | 1.716 |
| clef | 77 | 1901 | 85.57 | 86.15 | 1.829 |

## Three questions sharing one Clef pass

Original qualitative demo, outside BANKING training. Raw scores; BANKING calibration does not transfer to these questions.

| Question | Expected | Selected | Raw selected score |
|---|---|---|---:|
| route | billing | billing | 0.9996 |
| refund_requested | yes | yes | 0.7778 |
| urgency | routine | routine | 0.9617 |

The probe made one backbone call. All three questions use the Choice type. Other field types, images and general multi-question accuracy remain outside this probe.


## Reproduce

Use the [comparison guide](../../docs/unsloth.md) for dependencies, training commands and scope. Regenerate this report with `python scripts/summarize_head_comparison.py`. The configuration fixes two learning-rate trials per arm and three fresh main seeds. Candidate-order tests keep calibration fixed and are reported separately.
