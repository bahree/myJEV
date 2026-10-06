# Replicated 4B precision study

Completed seed pairs: 3/3. Final descriptive table.

4B SFT, same seeds, 4000 examples, LR, adapter targets and readout. BF16 versus NF4 with FP32 nonquantized modules. Training configuration comparison, not an inference-only quantization toggle.

Three seeds; final-checkpoint test. No new tuning. Does not identify the precision effect on RL, continued supervision or 9B. Capacity-only conclusions still require care.

| Seed | BF16 accuracy | NF4 accuracy | NF4 minus BF16 | BF16 Brier | NF4 Brier |
|---|---:|---:|---:|---:|---:|
| 11 | 87.44% | 86.23% | -1.20 pp | 0.1185 | 0.1347 |
| 22 | 87.18% | 87.95% | +0.78 pp | 0.1243 | 0.1192 |
| 33 | 84.84% | 87.69% | +2.86 pp | 0.1307 | 0.1200 |

Regenerate: `python scripts/summarize_precision_study.py`. The frozen plan and source hashes identify the exact controls. Further paired uncertainty analysis should precede strong claims from small differences.
