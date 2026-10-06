# Generalization extension

Completed jobs: **2/36**. Partial progress only; no final cross-size ranking.

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

For deferral-only cohorts the true answer is absent, so selection accuracy is zero by construction. Lower acceptance is desirable there; this must not be mixed with explicit none-option classification accuracy. Fixed thresholds can yield very different coverage after distribution shift.

Limits: Fixed subsamples, not full CLINC evaluation; Paraphrases cover eight labels and are not human validated; Sampled RL not included in this bounded transfer extension; No temperature control in this first extension.

Regenerate from recorded metrics: `python scripts/summarize_generalization.py`. Per-run predictions and failure logs are retained privately; aggregate metrics and source hashes are public evidence.
