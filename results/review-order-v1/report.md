# Candidate-order sensitivity of the released models

Six released seed-11 checkpoints, three fixed permutations of the same 3080 examples. No calibration refit, retraining, or test-driven order selection. Repeated permutations are not independent test examples.

Training randomizes candidates, but the Qwen readout binds their current positions to token aliases. Order changes both placement and alias assignment. This diagnostic measures their combined sensitivity; it does not isolate position from token identity.

| Size | Release | Original accuracy | Shuffled accuracy, seeds 101 / 202 / 303 | Changed selected ID, same order seeds | Brier range |
|---|---|---:|---|---|---|
| 0.8b | continued_sft | 83.90% | 84.81% / 84.55% / 85.00% | 7.89% / 8.05% / 8.21% | 0.0996-0.1026 |
| 0.8b | exact | 82.14% | 82.89% / 82.24% / 81.98% | 8.67% / 9.19% / 9.71% | 0.1203-0.1337 |
| 4b | continued_sft | 89.94% | 89.97% / 89.58% / 90.10% | 3.99% / 4.22% / 3.99% | 0.0711-0.0732 |
| 4b | exact | 90.55% | 90.62% / 90.39% / 90.65% | 3.64% / 3.67% / 3.77% | 0.0799-0.0836 |
| 9b | continued_sft | 89.35% | 89.42% / 89.45% / 89.81% | 3.73% / 3.64% / 4.06% | 0.0767-0.0775 |
| 9b | exact | 89.06% | 89.03% / 89.19% / 89.64% | 4.94% / 4.97% / 4.74% | 0.0788-0.0872 |

## The original 80% calibration-coverage threshold stays fixed

| Size | Release | Original test coverage / error | Order 101 coverage / error | Order 202 coverage / error | Order 303 coverage / error |
|---|---|---|---|---|---|
| 0.8b | continued_sft | 83.28% / 8.77% | 81.27% / 7.75% | 80.84% / 7.95% | 81.49% / 8.05% |
| 0.8b | exact | 83.83% / 10.53% | 82.34% / 9.15% | 83.21% / 10.11% | 82.31% / 9.66% |
| 4b | continued_sft | 81.40% / 3.71% | 80.71% / 3.74% | 80.91% / 3.89% | 80.49% / 3.91% |
| 4b | exact | 82.60% / 3.73% | 83.86% / 4.18% | 83.93% / 4.33% | 83.80% / 4.15% |
| 9b | continued_sft | 82.89% / 4.66% | 84.55% / 4.95% | 83.80% / 5.04% | 84.22% / 5.36% |
| 9b | exact | 83.18% / 4.10% | 83.21% / 3.78% | 83.38% / 4.01% | 82.66% / 3.77% |

These are achieved test coverages and accepted-case errors at a threshold previously selected for 80% calibration coverage. They are not equal-coverage comparisons or certified error targets. Per-file reports retain accepted counts and uncertainty.

All 18 metrics files preserve reliability bins, group-bootstrap accuracy intervals and achieved coverage/error at the original calibration thresholds. Those thresholds and each release's confidence source stay fixed. Different orders can exchange correct and incorrect cases even when aggregate accuracy barely changes. Do not interpret a small mean change as individual prediction invariance.

Reproduce the summary with `python scripts/summarize_review_order.py`. For new inference, prepare the pinned BANKING77 data, then run `python scripts/review_candidate_order.py --size 0.8b --hub --output results/my-order-check`; repeat for 4b and 9b with an appropriate GPU. The released revisions are in `model_cards/facts.json`; public text-free calibration inputs supply the original-order reference. Preserve the published evidence directory when collecting new runs. These six-checkpoint tests do not establish robustness over new training seeds or arbitrary candidate descriptions.
