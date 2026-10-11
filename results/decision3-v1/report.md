# Decision 3.0 and myJEV on the same A30 hardware

All six pinned releases completed the full 3,080-row BANKING77 test, 1,000 calibration rows, six CLINC150 cohorts, five perturbations of the same 64 cases and seven authored demos. These are released-system comparisons with different training exposure. No checkpoint, prompt or temperature was selected from test results.

## Original BANKING77 requests

| Model | Confidence calculation | Accuracy | Correctness Brier | ECE | Test coverage at 80% calibration coverage | Accepted-case error |
|---|---|---:|---:|---:|---:|---:|
| d3-lite | native-selection | 89.81% | 0.0645 | 0.0181 | 82.01% | 3.44% |
| d3-lite | temperature | 89.81% | 0.0640 | 0.0103 | 82.34% | 3.43% |
| d3-mini | native-selection | 92.27% | 0.0569 | 0.0319 | 80.71% | 2.41% |
| d3-mini | temperature | 92.27% | 0.0553 | 0.0136 | 80.84% | 2.45% |
| myjev-0.8b | native-selection | 83.90% | 0.1062 | 0.0325 | 83.28% | 8.77% |
| myjev-0.8b | temperature | 83.90% | 0.1062 | 0.0325 | 83.28% | 8.77% |
| myjev-0.8b-rl | native-selection | 82.14% | 0.1518 | 0.1521 | 82.56% | 11.05% |
| myjev-0.8b-rl | temperature | 82.14% | 0.1216 | 0.0421 | 82.89% | 10.93% |
| myjev-4b | native-selection | 89.94% | 0.0709 | 0.0361 | 81.40% | 3.71% |
| myjev-4b | temperature | 89.94% | 0.0709 | 0.0361 | 81.40% | 3.71% |
| myjev-4b-rl | native-selection | 90.55% | 0.0850 | 0.0857 | 79.42% | 5.19% |
| myjev-4b-rl | temperature | 90.55% | 0.0695 | 0.0190 | 80.84% | 5.14% |

Temperature fits use calibration labels only, with a 1e-12 floor before taking log probabilities. Native myJEV correctness-head results, macro-F1, multiclass Brier, AUROC, 15-bin reliability, every threshold and accepted count are in [summary.json](summary.json). D3 native confidence measures normalized entropy and is reported separately as a ranking signal. Calibration thresholds are empirical rules; their selected error targets do not certify future error.

## Paired accuracy differences

| Pair | Difference in percentage points | Conditional 95% group interval |
|---|---:|---:|
| d3-lite-minus-myjev-0.8b | +5.91 | [+4.80, +7.08] |
| d3-lite-minus-myjev-0.8b-rl | +7.66 | [+6.53, +8.86] |
| d3-mini-minus-myjev-4b | +2.34 | [+1.46, +3.28] |
| d3-mini-minus-myjev-4b-rl | +1.72 | [+0.94, +2.50] |

These intervals resample test groups for the published checkpoints. They exclude variation from retraining and cannot isolate an architecture or training-method effect.

## Transfer with the same BANKING thresholds

| Model | CLINC cohort | Rows | Accuracy | Coverage at BANKING 80% threshold | Accepted-case error |
|---|---|---:|---:|---:|---:|
| d3-lite | near | 256 | 94.14% | 95.31% | 3.69% |
| d3-lite | distant | 256 | 95.70% | 94.14% | 1.24% |
| d3-lite | unsupported-near-none | 150 | 32.67% | 29.33% | 45.45% |
| d3-lite | unsupported-near-deferral | 150 | 0.00% | 34.00% | 100.00% |
| d3-lite | oos-none | 256 | 69.53% | 66.80% | 21.05% |
| d3-lite | oos-deferral | 256 | 0.00% | 33.98% | 100.00% |
| d3-mini | near | 256 | 94.14% | 93.36% | 2.09% |
| d3-mini | distant | 256 | 97.66% | 94.14% | 0.41% |
| d3-mini | unsupported-near-none | 150 | 62.00% | 27.33% | 46.34% |
| d3-mini | unsupported-near-deferral | 150 | 0.00% | 36.67% | 100.00% |
| d3-mini | oos-none | 256 | 82.03% | 71.48% | 6.56% |
| d3-mini | oos-deferral | 256 | 0.00% | 21.09% | 100.00% |
| myjev-0.8b | near | 256 | 81.64% | 67.97% | 6.32% |
| myjev-0.8b | distant | 256 | 69.53% | 55.86% | 13.29% |
| myjev-0.8b | unsupported-near-none | 150 | 0.00% | 11.33% | 100.00% |
| myjev-0.8b | unsupported-near-deferral | 150 | 0.00% | 12.00% | 100.00% |
| myjev-0.8b | oos-none | 256 | 0.00% | 11.33% | 100.00% |
| myjev-0.8b | oos-deferral | 256 | 0.00% | 12.50% | 100.00% |
| myjev-0.8b-rl | near | 256 | 80.08% | 39.06% | 14.00% |
| myjev-0.8b-rl | distant | 256 | 67.19% | 29.30% | 14.67% |
| myjev-0.8b-rl | unsupported-near-none | 150 | 0.00% | 0.00% | n/a (no accepted requests) |
| myjev-0.8b-rl | unsupported-near-deferral | 150 | 0.00% | 1.33% | 100.00% |
| myjev-0.8b-rl | oos-none | 256 | 0.00% | 7.42% | 100.00% |
| myjev-0.8b-rl | oos-deferral | 256 | 0.00% | 7.81% | 100.00% |
| myjev-4b | near | 256 | 89.45% | 68.36% | 4.00% |
| myjev-4b | distant | 256 | 81.25% | 52.34% | 5.97% |
| myjev-4b | unsupported-near-none | 150 | 47.33% | 4.00% | 100.00% |
| myjev-4b | unsupported-near-deferral | 150 | 0.00% | 6.00% | 100.00% |
| myjev-4b | oos-none | 256 | 82.42% | 1.56% | 25.00% |
| myjev-4b | oos-deferral | 256 | 0.00% | 3.91% | 100.00% |
| myjev-4b-rl | near | 256 | 85.94% | 50.39% | 11.63% |
| myjev-4b-rl | distant | 256 | 81.64% | 36.33% | 8.60% |
| myjev-4b-rl | unsupported-near-none | 150 | 43.33% | 14.67% | 100.00% |
| myjev-4b-rl | unsupported-near-deferral | 150 | 0.00% | 16.00% | 100.00% |
| myjev-4b-rl | oos-none | 256 | 74.61% | 3.52% | 88.89% |
| myjev-4b-rl | oos-deferral | 256 | 0.00% | 6.64% | 100.00% |

This table uses the equally fitted selection temperatures. Deferral cohorts omit the correct option, so acceptance is a false acceptance. Explicit none-of-these cohorts supply a selectable answer. No CLINC labels were used for fitting.

## Candidate and wording changes

| Model | View | Changed answers / 64 | Accuracy |
|---|---|---:|---:|
| d3-lite | reverse | 5 | 93.75% |
| d3-lite | shuffle | 3 | 96.88% |
| d3-lite | renamed-ids | 1 | 96.88% |
| d3-lite | whitespace | 1 | 96.88% |
| d3-lite | instruction-paraphrase | 0 | 95.31% |
| d3-lite | original | 0 | 95.31% |
| d3-mini | reverse | 0 | 96.88% |
| d3-mini | shuffle | 0 | 96.88% |
| d3-mini | renamed-ids | 0 | 96.88% |
| d3-mini | whitespace | 0 | 96.88% |
| d3-mini | instruction-paraphrase | 0 | 96.88% |
| d3-mini | original | 0 | 96.88% |
| myjev-0.8b | reverse | 4 | 89.06% |
| myjev-0.8b | shuffle | 3 | 89.06% |
| myjev-0.8b | renamed-ids | 0 | 92.19% |
| myjev-0.8b | whitespace | 3 | 87.50% |
| myjev-0.8b | instruction-paraphrase | 2 | 90.62% |
| myjev-0.8b | original | 0 | 92.19% |
| myjev-0.8b-rl | reverse | 7 | 79.69% |
| myjev-0.8b-rl | shuffle | 4 | 84.38% |
| myjev-0.8b-rl | renamed-ids | 0 | 79.69% |
| myjev-0.8b-rl | whitespace | 6 | 79.69% |
| myjev-0.8b-rl | instruction-paraphrase | 3 | 81.25% |
| myjev-0.8b-rl | original | 0 | 79.69% |
| myjev-4b | reverse | 3 | 93.75% |
| myjev-4b | shuffle | 3 | 96.88% |
| myjev-4b | renamed-ids | 0 | 93.75% |
| myjev-4b | whitespace | 2 | 96.88% |
| myjev-4b | instruction-paraphrase | 1 | 95.31% |
| myjev-4b | original | 0 | 93.75% |
| myjev-4b-rl | reverse | 1 | 93.75% |
| myjev-4b-rl | shuffle | 1 | 96.88% |
| myjev-4b-rl | renamed-ids | 0 | 95.31% |
| myjev-4b-rl | whitespace | 1 | 93.75% |
| myjev-4b-rl | instruction-paraphrase | 0 | 95.31% |
| myjev-4b-rl | original | 0 | 95.31% |

The views share requests and are never pooled as independent samples. Each shuffle is determined by the request ID. Temperatures and thresholds remain those fitted for original requests.

## Local scoring latency and memory

| Model | BANKING p50 / p95 ms | Short-demo p50 / p95 ms | Peak allocated GiB | Cached load s |
|---|---:|---:|---:|---:|
| d3-lite | 112.47 / 122.87 | 57.29 / 61.00 | 1.710 | 3.52 |
| d3-mini | 223.99 / 231.55 | 71.01 / 72.46 | 8.701 | 9.27 |
| myjev-0.8b | 103.37 / 104.60 | 50.46 / 54.91 | 1.535 | 1.00 |
| myjev-0.8b-rl | 110.56 / 112.75 | 55.66 / 60.28 | 1.535 | 1.39 |
| myjev-4b | 211.98 / 214.01 | 69.40 / 75.95 | 8.153 | 2.12 |
| myjev-4b-rl | 211.55 / 213.25 | 68.60 / 74.37 | 8.153 | 2.14 |

The separate benchmark runs one model at a time on GPU 0. Timing includes tokenization, scoring and response conversion, with CUDA synchronization, and excludes HTTP and downloading. Each run has five initial warmups, then five more at the start of each benchmark workload. D3 retains its vision tower and verifies the full package during loading. myJEV uses its release loader. Backbone CUDA-event timings and reserved memory are also saved. Quality-run timings may overlap other GPU work and are not used for this table.

## Seven authored demonstrations

| Model | Case | Expected | Selected | Selected probability | Native confidence |
|---|---|---|---|---:|---:|
| d3-lite | billing | billing | billing | 0.9304 | 0.7453 |
| d3-lite | technical | technical | technical | 0.8548 | 0.6199 |
| d3-lite | none-of-these | other | other | 0.9966 | 0.9774 |
| d3-lite | refund-day-13 | approve | approve | 0.9998 | 0.9967 |
| d3-lite | refund-day-14 | deny | deny | 0.9973 | 0.9727 |
| d3-lite | quoted-instruction | billing | billing | 0.9246 | 0.7118 |
| d3-lite | synthetic-blog-format | tutorial | tutorial | 0.9998 | 0.9983 |
| d3-mini | billing | billing | billing | 0.9863 | 0.9266 |
| d3-mini | technical | technical | technical | 0.9385 | 0.7881 |
| d3-mini | none-of-these | other | other | 0.9997 | 0.9971 |
| d3-mini | refund-day-13 | approve | approve | 0.9999 | 0.9989 |
| d3-mini | refund-day-14 | deny | deny | 0.9999 | 0.9992 |
| d3-mini | quoted-instruction | billing | billing | 0.9818 | 0.9079 |
| d3-mini | synthetic-blog-format | tutorial | tutorial | 0.9999 | 0.9987 |
| myjev-0.8b | billing | billing | billing | 0.9764 | 0.9764 |
| myjev-0.8b | technical | technical | technical | 0.9835 | 0.9835 |
| myjev-0.8b | none-of-these | other | other | 0.8359 | 0.8359 |
| myjev-0.8b | refund-day-13 | approve | approve | 0.9891 | 0.9891 |
| myjev-0.8b | refund-day-14 | deny | approve | 0.9892 | 0.9892 |
| myjev-0.8b | quoted-instruction | billing | billing | 0.9688 | 0.9688 |
| myjev-0.8b | synthetic-blog-format | tutorial | tutorial | 0.9499 | 0.9499 |
| myjev-0.8b-rl | billing | billing | billing | 1.0000 | 0.8727 |
| myjev-0.8b-rl | technical | technical | technical | 1.0000 | 0.8635 |
| myjev-0.8b-rl | none-of-these | other | other | 0.9636 | 0.6848 |
| myjev-0.8b-rl | refund-day-13 | approve | approve | 0.9991 | 0.7750 |
| myjev-0.8b-rl | refund-day-14 | deny | approve | 0.7138 | 0.7394 |
| myjev-0.8b-rl | quoted-instruction | billing | billing | 1.0000 | 0.8612 |
| myjev-0.8b-rl | synthetic-blog-format | tutorial | tutorial | 0.5971 | 0.1530 |
| myjev-4b | billing | billing | billing | 0.9834 | 0.9834 |
| myjev-4b | technical | technical | technical | 0.9669 | 0.9669 |
| myjev-4b | none-of-these | other | other | 0.9828 | 0.9828 |
| myjev-4b | refund-day-13 | approve | approve | 0.9672 | 0.9672 |
| myjev-4b | refund-day-14 | deny | deny | 0.9835 | 0.9835 |
| myjev-4b | quoted-instruction | billing | billing | 0.9383 | 0.9383 |
| myjev-4b | synthetic-blog-format | tutorial | tutorial | 0.9975 | 0.9975 |
| myjev-4b-rl | billing | billing | billing | 1.0000 | 0.8591 |
| myjev-4b-rl | technical | technical | technical | 1.0000 | 0.8556 |
| myjev-4b-rl | none-of-these | other | other | 1.0000 | 0.8301 |
| myjev-4b-rl | refund-day-13 | approve | approve | 1.0000 | 0.8556 |
| myjev-4b-rl | refund-day-14 | deny | deny | 1.0000 | 0.8565 |
| myjev-4b-rl | quoted-instruction | billing | billing | 1.0000 | 0.8572 |
| myjev-4b-rl | synthetic-blog-format | tutorial | tutorial | 1.0000 | 0.8592 |

The two confidence columns have different meanings for D3 and RL releases. These examples were written for teaching and do not estimate population accuracy.

## Reproduce

Run `python scripts/summarize_decision3.py --require-benchmark` after restoring the prediction files. [The guide](../../docs/decision3.md) gives the model pins, setup, commands, architecture comparison and evidence locations.
