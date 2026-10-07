# Paired archive and forgetting changes

Post-hoc paired group bootstrap, 5,000 draws, RNG42, after minus before. Resamples original post groups for archive and recorded example groups for public tasks. Conditional on this one trained seed and available sample; no multiplicity adjustment or uncertainty over teacher correctness. Archive agreement is with unreviewed machine labels. Brier uses each deployed scalar correctness head. Deferral accuracy is zero by construction and not a performance measure.

| Cohort | N / groups | Agreement / accuracy change, pp | 95% paired group interval, pp | Brier change | 95% paired group interval |
|---|---:|---:|---|---:|---|
| archive | 90 / 32 | +18.89 | [+5.55, +32.26] | -0.3053 | [-0.4336, -0.1804] |
| banking | 3080 / 3079 | +0.16 | [-0.39, +0.71] | -0.0096 | [-0.0153, -0.0040] |
| clinc/near | 256 / 256 | +1.17 | [-1.17, +3.52] | -0.0033 | [-0.0245, +0.0180] |
| clinc/distant | 256 / 256 | +1.95 | [-1.17, +5.08] | -0.0401 | [-0.0732, -0.0053] |
| clinc/oos-none | 256 / 256 | -10.94 | [-14.84, -7.03] | +0.0620 | [+0.0280, +0.0968] |
| clinc/unsupported-near-none | 150 / 150 | -12.67 | [-18.00, -7.33] | -0.0793 | [-0.1364, -0.0250] |
| clinc/oos-deferral | 256 / 256 | +0.00 | [+0.00, +0.00] | -0.5186 | [-0.5599, -0.4748] |
| clinc/unsupported-near-deferral | 150 / 150 | +0.00 | [+0.00, +0.00] | -0.4279 | [-0.4808, -0.3753] |

Positive agreement change is better; positive Brier change is worse. These are changes in outcomes, not causal attribution: the study has one adaptation run and no seed population inference. Raw archive predictions/text are private. Regenerate intervals publicly with `python scripts/analyze_archive_changes.py` from compact hashed IDs/groups, correctness events and confidence values. `--extract` requires original private predictions and recreates that compact input; no model inference is needed.
