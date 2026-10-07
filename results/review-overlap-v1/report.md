# BANKING77 punctuation-overlap sensitivity

Post-review punctuation-only overlap check and descriptive test-row exclusion. Official test and trained models unchanged. Does not estimate the causal training effect of overlap; semantic overlap may remain.

4 of 3,080 official test rows match training rows under the extra normalization. The sensitivity view contains 3076 test rows. Every flagged pair has the same intent label. No request text is redistributed.

| Run | Official accuracy | Excluding flagged rows | Change (pp) |
|---|---:|---:|---:|
| 0.8b/main/seed-11/continued_sft | 83.90% | 83.88% | -0.0209 |
| 0.8b/main/seed-11/exact | 82.14% | 82.15% | +0.0093 |
| 0.8b/main/seed-11/sampled | 81.23% | 81.24% | +0.0081 |
| 0.8b/main/seed-11/sft | 79.19% | 79.19% | +0.0054 |
| 0.8b/main/seed-22/continued_sft | 83.02% | 83.03% | +0.0104 |
| 0.8b/main/seed-22/exact | 80.16% | 80.14% | -0.0258 |
| 0.8b/main/seed-22/sampled | 80.49% | 80.46% | -0.0254 |
| 0.8b/main/seed-22/sft | 78.54% | 78.51% | -0.0279 |
| 0.8b/main/seed-33/continued_sft | 81.88% | 81.86% | -0.0236 |
| 0.8b/main/seed-33/exact | 81.79% | 81.76% | -0.0237 |
| 0.8b/main/seed-33/sampled | 79.87% | 79.84% | -0.0262 |
| 0.8b/main/seed-33/sft | 79.45% | 79.42% | -0.0267 |
| 4b/main/seed-11/continued_sft | 89.94% | 89.92% | -0.0131 |
| 4b/main/seed-11/exact | 90.55% | 90.54% | -0.0123 |
| 4b/main/seed-11/sampled | 89.38% | 89.37% | -0.0138 |
| 4b/main/seed-11/sft | 87.44% | 87.42% | -0.0163 |
| 4b/main/seed-22/continued_sft | 90.16% | 90.15% | -0.0128 |
| 4b/main/seed-22/exact | 89.97% | 89.95% | -0.0130 |
| 4b/main/seed-22/sampled | 89.58% | 89.56% | -0.0136 |
| 4b/main/seed-22/sft | 87.18% | 87.16% | -0.0167 |
| 4b/main/seed-33/continued_sft | 87.60% | 87.58% | -0.0161 |
| 4b/main/seed-33/exact | 90.29% | 90.28% | -0.0126 |
| 4b/main/seed-33/sampled | 85.65% | 85.63% | -0.0187 |
| 4b/main/seed-33/sft | 84.84% | 84.82% | -0.0197 |
| 9b/main/seed-11/continued_sft | 89.35% | 89.34% | -0.0138 |
| 9b/main/seed-11/exact | 89.06% | 89.04% | -0.0142 |
| 9b/main/seed-11/sampled | 88.77% | 88.75% | -0.0146 |
| 9b/main/seed-11/sft | 86.72% | 86.70% | -0.0173 |
| 9b/main/seed-22/continued_sft | 89.38% | 89.40% | +0.0187 |
| 9b/main/seed-22/exact | 89.61% | 89.60% | -0.0135 |
| 9b/main/seed-22/sampled | 89.03% | 89.01% | -0.0143 |
| 9b/main/seed-22/sft | 87.27% | 87.26% | -0.0166 |
| 9b/main/seed-33/continued_sft | 88.70% | 88.69% | -0.0147 |
| 9b/main/seed-33/exact | 89.35% | 89.34% | -0.0138 |
| 9b/main/seed-33/sampled | 87.82% | 87.81% | -0.0158 |
| 9b/main/seed-33/sft | 87.24% | 87.22% | -0.0166 |

The official score remains primary. Excluding four evaluated rows is a sensitivity check, not a correction to the trained weights or proof of a maximum causal contamination effect. Any retraining with different exclusions would be a new experiment.

Reproduce after preparing the pinned data; uses original predictions when present, otherwise the exported text-free calibration inputs: `python scripts/review_banking_overlap.py`.
