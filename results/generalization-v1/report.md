# Generalization extension

Completed jobs: **8/36**. Partial progress only; no final cross-size ranking.

Frozen subsample diagnostics with native scalar/policy confidence and unchanged BANKING77 thresholds. Near/distant use CLINC descriptions. None-option classification and confidence-based deferral are separate tasks. No transfer tuning.

| Size | Seed | Method | Task | Cohort | N | Accuracy | Coverage at BANKING 80% threshold | Accepted error | Rejected |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| 0.8b | 11 | continued_sft | transfer | distant | 256 | 69.53% | 67.97% | 16.09% | 0 |
| 0.8b | 11 | continued_sft | transfer | near | 256 | 81.64% | 78.52% | 7.96% | 0 |
| 0.8b | 11 | continued_sft | transfer | oos-deferral | 256 | 0.00% | 19.14% | 100.00% | 0 |
| 0.8b | 11 | continued_sft | transfer | oos-none | 256 | 0.00% | 19.53% | 100.00% | 0 |
| 0.8b | 11 | continued_sft | transfer | unsupported-near-deferral | 150 | 0.00% | 14.67% | 100.00% | 0 |
| 0.8b | 11 | continued_sft | transfer | unsupported-near-none | 150 | 0.00% | 15.33% | 100.00% | 0 |
| 0.8b | 11 | continued_sft | robustness | missing-correct | 64 | 0.00% | 31.25% | 100.00% | 0 |
| 0.8b | 11 | continued_sft | robustness | original | 64 | 92.19% | 87.50% | 5.36% | 0 |
| 0.8b | 11 | continued_sft | robustness | quoted-instruction | 64 | 90.62% | 78.12% | 4.00% | 0 |
| 0.8b | 11 | continued_sft | robustness | reverse | 64 | 89.06% | 84.38% | 7.41% | 0 |
| 0.8b | 11 | continued_sft | robustness | length | 64 | 90.62% | 81.25% | 3.85% | 0 |
| 0.8b | 11 | continued_sft | robustness | irrelevant | 64 | 92.19% | 84.38% | 5.56% | 0 |
| 0.8b | 11 | continued_sft | robustness | partial-description-paraphrases | 64 | 87.50% | 81.25% | 3.85% | 0 |
| 0.8b | 11 | exact | transfer | distant | 256 | 67.19% | 78.91% | 22.28% | 0 |
| 0.8b | 11 | exact | transfer | near | 256 | 80.08% | 80.86% | 11.11% | 0 |
| 0.8b | 11 | exact | transfer | oos-deferral | 256 | 0.00% | 21.09% | 100.00% | 0 |
| 0.8b | 11 | exact | transfer | oos-none | 256 | 0.00% | 20.31% | 100.00% | 0 |
| 0.8b | 11 | exact | transfer | unsupported-near-deferral | 150 | 0.00% | 13.33% | 100.00% | 0 |
| 0.8b | 11 | exact | transfer | unsupported-near-none | 150 | 0.00% | 10.67% | 100.00% | 0 |
| 0.8b | 11 | exact | robustness | partial-description-paraphrases | 64 | 82.81% | 82.81% | 7.55% | 0 |
| 0.8b | 11 | exact | robustness | irrelevant | 64 | 82.81% | 85.94% | 9.09% | 0 |
| 0.8b | 11 | exact | robustness | missing-correct | 64 | 0.00% | 34.38% | 100.00% | 0 |
| 0.8b | 11 | exact | robustness | quoted-instruction | 64 | 79.69% | 75.00% | 6.25% | 0 |
| 0.8b | 11 | exact | robustness | original | 64 | 79.69% | 87.50% | 12.50% | 0 |
| 0.8b | 11 | exact | robustness | length | 64 | 85.94% | 82.81% | 5.66% | 0 |
| 0.8b | 11 | exact | robustness | reverse | 64 | 79.69% | 82.81% | 13.21% | 0 |
| 0.8b | 22 | continued_sft | transfer | distant | 256 | 67.19% | 60.16% | 12.34% | 0 |
| 0.8b | 22 | continued_sft | transfer | near | 256 | 81.25% | 71.48% | 9.29% | 0 |
| 0.8b | 22 | continued_sft | transfer | oos-deferral | 256 | 0.00% | 6.64% | 100.00% | 0 |
| 0.8b | 22 | continued_sft | transfer | oos-none | 256 | 1.17% | 7.03% | 100.00% | 0 |
| 0.8b | 22 | continued_sft | transfer | unsupported-near-deferral | 150 | 0.00% | 14.00% | 100.00% | 0 |
| 0.8b | 22 | continued_sft | transfer | unsupported-near-none | 150 | 2.00% | 14.67% | 100.00% | 0 |
| 4b | 11 | continued_sft | transfer | distant | 256 | 81.25% | 89.45% | 13.10% | 0 |
| 4b | 11 | continued_sft | transfer | near | 256 | 89.45% | 91.80% | 6.38% | 0 |
| 4b | 11 | continued_sft | transfer | oos-deferral | 256 | 0.00% | 19.53% | 100.00% | 0 |
| 4b | 11 | continued_sft | transfer | oos-none | 256 | 82.42% | 68.75% | 13.07% | 0 |
| 4b | 11 | continued_sft | transfer | unsupported-near-deferral | 150 | 0.00% | 34.00% | 100.00% | 0 |
| 4b | 11 | continued_sft | transfer | unsupported-near-none | 150 | 47.33% | 44.67% | 50.75% | 0 |
| 4b | 11 | continued_sft | robustness | partial-description-paraphrases | 64 | 92.19% | 82.81% | 1.89% | 0 |
| 4b | 11 | continued_sft | robustness | missing-correct | 64 | 0.00% | 31.25% | 100.00% | 0 |
| 4b | 11 | continued_sft | robustness | length | 64 | 95.31% | 81.25% | 0.00% | 0 |
| 4b | 11 | continued_sft | robustness | original | 64 | 93.75% | 84.38% | 1.85% | 0 |
| 4b | 11 | continued_sft | robustness | reverse | 64 | 93.75% | 87.50% | 3.57% | 0 |
| 4b | 11 | continued_sft | robustness | quoted-instruction | 64 | 93.75% | 84.38% | 3.70% | 0 |
| 4b | 11 | continued_sft | robustness | irrelevant | 64 | 93.75% | 84.38% | 1.85% | 0 |
| 9b | 11 | continued_sft | transfer | distant | 256 | 81.25% | 93.36% | 15.90% | 0 |
| 9b | 11 | continued_sft | transfer | near | 256 | 89.45% | 90.23% | 8.23% | 0 |
| 9b | 11 | continued_sft | transfer | oos-deferral | 256 | 0.00% | 18.75% | 100.00% | 0 |
| 9b | 11 | continued_sft | transfer | oos-none | 256 | 63.28% | 51.17% | 39.69% | 0 |
| 9b | 11 | continued_sft | transfer | unsupported-near-deferral | 150 | 0.00% | 14.00% | 100.00% | 0 |
| 9b | 11 | continued_sft | transfer | unsupported-near-none | 150 | 32.00% | 16.00% | 100.00% | 0 |

For deferral-only cohorts the true answer is absent, so selection accuracy is zero by construction. Lower acceptance is desirable there; this must not be mixed with explicit none-option classification accuracy. Fixed thresholds can yield very different coverage after distribution shift.

Limits: Fixed subsamples, not full CLINC evaluation; Paraphrases cover eight labels and are not human validated; Sampled RL not included in this bounded transfer extension; No temperature control in this first extension.

Regenerate from recorded metrics: `python scripts/summarize_generalization.py`. Per-run predictions and failure logs are retained privately; aggregate metrics and source hashes are public evidence.
