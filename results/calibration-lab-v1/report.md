# Calibration lab: synthetic results

Fixed nonlinear map, four classes, separate training/validation/calibration/test. Three matched initialization/order seeds. No test tuning. This is a teaching experiment, not evidence about natural-data model quality.

Uniform replacement occurs with probability 8%; a replacement redraws the original class one quarter of the time. Expected changed labels: 6%. The true observed distribution is `(1 - 0.08) * p_clean + 0.08 / 4`.

| Seed | Objective | Temperature | Accuracy | Multiclass Brier raw / scaled | Selected Brier raw / scaled | ECE raw / scaled | Oracle MSE raw / scaled |
|---|---|---|---|---|---|---|---|
| 11 | ce | 1.717 | 0.5400 | 0.6248 / 0.5912 | 0.2569 / 0.2329 | 0.1459 / 0.0254 | 0.0278 / 0.0209 |
| 11 | brier | 2.568 | 0.5479 | 0.6656 / 0.5850 | 0.2905 / 0.2330 | 0.2225 / 0.0289 | 0.0386 / 0.0201 |
| 22 | ce | 1.696 | 0.5488 | 0.6121 / 0.5835 | 0.2509 / 0.2298 | 0.1356 / 0.0189 | 0.0249 / 0.0191 |
| 22 | brier | 2.651 | 0.5498 | 0.6666 / 0.5804 | 0.2918 / 0.2303 | 0.2314 / 0.0199 | 0.0389 / 0.0192 |
| 33 | ce | 1.897 | 0.5479 | 0.6165 / 0.5838 | 0.2535 / 0.2298 | 0.1497 / 0.0199 | 0.0277 / 0.0205 |
| 33 | brier | 2.530 | 0.5542 | 0.6543 / 0.5816 | 0.2822 / 0.2301 | 0.2105 / 0.0233 | 0.0363 / 0.0194 |

Multiclass Brier sums squared errors across the four classes. Selected-correctness Brier compares the largest selection probability with whether its selected class is correct; it is not a separately learned confidence head. Oracle MSE averages across classes against the known observed probability vector. ECE uses 10 equal-width bins, left closed and right open except the last bin. Full bins and NLL appear in JSON.

Positive scalar temperature preserves argmax accuracy. It is fitted by calibration NLL, so it does not guarantee improved test Brier or ECE. CE and Brier receive equal exposure and hyperparameters, not individually optimized tuning budgets. Neither loss is guaranteed to win.

CPU runtime: 5.57 seconds; one Torch thread. Regenerate with `.venv/bin/python scripts/calibration_lab.py`. Protocol is written before fitting; training/validation logs and dataset/code hashes are retained.
