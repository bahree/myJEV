# Paired longer-study contrasts

Paired group bootstrap, 2000 draws, RNG 42; averages the three observed seeds. Conditional on those seeds, not a population interval over initializations. Exploratory contrasts, no multiple-comparison adjustment. Positive accuracy delta is better; positive Brier delta is worse. Confidence sources differ in Brier contrasts.

| Size | Candidate minus reference | Metric | Mean delta | 95% group interval |
|---|---|---|---:|---|
| 0.8b | exact/deployed minus continued_sft/deployed | correct | -1.5693 pp | [-2.1970, -0.9524] pp |
| 0.8b | sampled/deployed minus continued_sft/deployed | correct | -2.4026 pp | [-3.0392, -1.7530] pp |
| 0.8b | sampled/deployed minus exact/deployed | correct | -0.8333 pp | [-1.3204, -0.3573] pp |
| 0.8b | exact/deployed minus continued_sft/temperature | brier | 0.0248 | [0.0201, 0.0297] |
| 0.8b | sampled/deployed minus continued_sft/temperature | brier | 0.0275 | [0.0226, 0.0323] |
| 4b | exact/deployed minus continued_sft/deployed | correct | 1.0390 pp | [0.5628, 1.4940] pp |
| 4b | sampled/deployed minus continued_sft/deployed | correct | -1.0281 pp | [-1.5801, -0.5088] pp |
| 4b | sampled/deployed minus exact/deployed | correct | -2.0671 pp | [-2.5010, -1.6545] pp |
| 4b | exact/deployed minus continued_sft/temperature | brier | 0.0078 | [0.0047, 0.0109] |
| 4b | sampled/deployed minus continued_sft/temperature | brier | 0.0267 | [0.0231, 0.0302] |
| 9b | exact/deployed minus continued_sft/deployed | correct | 0.1948 pp | [-0.3575, 0.7143] pp |
| 9b | sampled/deployed minus continued_sft/deployed | correct | -0.6061 pp | [-1.1692, -0.0650] pp |
| 9b | sampled/deployed minus exact/deployed | correct | -0.8009 pp | [-1.1909, -0.4006] pp |
| 9b | exact/deployed minus continued_sft/temperature | brier | 0.0200 | [0.0152, 0.0248] |
| 9b | sampled/deployed minus continued_sft/temperature | brier | 0.0290 | [0.0246, 0.0338] |

Read the [paired seed deltas and seed spread](../review-seeds-v1/report.md) beside these conditional test-resampling intervals. A positive conditional interval is not evidence that every training seed improves.

Reproduce: `python3 scripts/analyze_longer_study.py`. Compact paired inputs retain example/group IDs, gold label IDs, correctness and squared confidence error. They omit request text. Source prediction hashes are recorded separately; original full predictions are retained in the private evidence archive.
