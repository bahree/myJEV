# CLINC forgetting after exploratory archive adaptation

Public CLINC labels measure classification; archive training labels remain unreviewed machine references. Single-seed exploratory forgetting.

| Cohort | N | Before accuracy | After accuracy | Change, pp |
|---|---:|---:|---:|---:|
| distant | 256 | 81.25% | 83.20% | +1.95 |
| near | 256 | 89.45% | 90.62% | +1.17 |
| oos-deferral | 256 | 0.00% | 0.00% | +0.00 |
| oos-none | 256 | 82.42% | 71.48% | -10.94 |
| unsupported-near-deferral | 150 | 0.00% | 0.00% | +0.00 |
| unsupported-near-none | 150 | 47.33% | 34.67% | -12.67 |

Deferral cohorts deliberately omit a correct option, so zero selection accuracy is by construction. JSON reports acceptance with both fixed original and freshly BANKING-calibrated thresholds. No threshold is fitted on CLINC. Group intervals are conditional on the sampled items and this one adaptation seed.
