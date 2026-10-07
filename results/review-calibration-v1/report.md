# Exploratory matched post-hoc confidence controls

All four views were specified before this analysis. Original test results had already been inspected; this is an exploratory follow-up, not a preregistered experiment. Fits use only the original calibration partition. No trained weights or released artifacts were changed.

Native confidence is the SFT scalar head or RL policy expectation. Selection temperature calibrates the answer distribution; learned temperature applies the same one-parameter binary log-odds map to the trained scalar/policy correctness estimate. Positive temperature preserves answer argmax. No view is selected using test results.

| Size | Method | Confidence view | Accuracy | Correctness Brier | ECE (15 bins) | AUROC |
|---|---|---|---:|---:|---:|---:|
| 0.8b | sft | native | 79.06% | 0.1897 | 0.1918 | 0.8156 |
| 0.8b | sft | selection_raw | 79.06% | 0.1654 | 0.1616 | 0.7986 |
| 0.8b | sft | selection_temperature | 79.06% | 0.1256 | 0.0307 | 0.8048 |
| 0.8b | sft | learned_temperature | 79.06% | 0.1467 | 0.1035 | 0.8156 |
| 0.8b | continued_sft | native | 82.93% | 0.1445 | 0.1378 | 0.8401 |
| 0.8b | continued_sft | selection_raw | 82.93% | 0.1366 | 0.1316 | 0.8451 |
| 0.8b | continued_sft | selection_temperature | 82.93% | 0.1073 | 0.0454 | 0.8424 |
| 0.8b | continued_sft | learned_temperature | 82.93% | 0.1202 | 0.0652 | 0.8401 |
| 0.8b | exact | native | 81.36% | 0.1321 | 0.0827 | 0.8449 |
| 0.8b | exact | selection_raw | 81.36% | 0.1521 | 0.1518 | 0.7680 |
| 0.8b | exact | selection_temperature | 81.36% | 0.1184 | 0.0424 | 0.7797 |
| 0.8b | exact | learned_temperature | 81.36% | 0.1425 | 0.1443 | 0.8449 |
| 0.8b | sampled | native | 80.53% | 0.1347 | 0.0994 | 0.8338 |
| 0.8b | sampled | selection_raw | 80.53% | 0.1581 | 0.1559 | 0.7851 |
| 0.8b | sampled | selection_temperature | 80.53% | 0.1207 | 0.0415 | 0.7940 |
| 0.8b | sampled | learned_temperature | 80.53% | 0.1453 | 0.1365 | 0.8338 |
| 4b | sft | native | 86.48% | 0.1245 | 0.1246 | 0.8709 |
| 4b | sft | selection_raw | 86.48% | 0.1125 | 0.1092 | 0.8459 |
| 4b | sft | selection_temperature | 86.48% | 0.0892 | 0.0280 | 0.8476 |
| 4b | sft | learned_temperature | 86.48% | 0.0985 | 0.0890 | 0.8709 |
| 4b | continued_sft | native | 89.23% | 0.1038 | 0.1040 | 0.8652 |
| 4b | continued_sft | selection_raw | 89.23% | 0.0940 | 0.0933 | 0.8725 |
| 4b | continued_sft | selection_temperature | 89.23% | 0.0740 | 0.0370 | 0.8698 |
| 4b | continued_sft | learned_temperature | 89.23% | 0.0848 | 0.0557 | 0.8652 |
| 4b | exact | native | 90.27% | 0.0818 | 0.0922 | 0.8685 |
| 4b | exact | selection_raw | 90.27% | 0.0905 | 0.0910 | 0.7546 |
| 4b | exact | selection_temperature | 90.27% | 0.0733 | 0.0183 | 0.7617 |
| 4b | exact | learned_temperature | 90.27% | 0.0796 | 0.0644 | 0.8685 |
| 4b | sampled | native | 88.20% | 0.1007 | 0.0698 | 0.7673 |
| 4b | sampled | selection_raw | 88.20% | 0.1111 | 0.1115 | 0.7767 |
| 4b | sampled | selection_temperature | 88.20% | 0.0852 | 0.0247 | 0.7820 |
| 4b | sampled | learned_temperature | 88.20% | 0.0983 | 0.0488 | 0.7673 |
| 9b | sft | native | 87.08% | 0.1176 | 0.1154 | 0.8658 |
| 9b | sft | selection_raw | 87.08% | 0.1028 | 0.0978 | 0.8789 |
| 9b | sft | selection_temperature | 87.08% | 0.0821 | 0.0335 | 0.8779 |
| 9b | sft | learned_temperature | 87.08% | 0.0943 | 0.0775 | 0.8658 |
| 9b | continued_sft | native | 89.15% | 0.1054 | 0.1051 | 0.8637 |
| 9b | continued_sft | selection_raw | 89.15% | 0.0969 | 0.0955 | 0.8810 |
| 9b | continued_sft | selection_temperature | 89.15% | 0.0742 | 0.0420 | 0.8834 |
| 9b | continued_sft | learned_temperature | 89.15% | 0.0877 | 0.0662 | 0.8637 |
| 9b | exact | native | 89.34% | 0.0942 | 0.1255 | 0.8882 |
| 9b | exact | selection_raw | 89.34% | 0.0905 | 0.0894 | 0.8666 |
| 9b | exact | selection_temperature | 89.34% | 0.0681 | 0.0250 | 0.8747 |
| 9b | exact | learned_temperature | 89.34% | 0.0933 | 0.1157 | 0.8882 |
| 9b | sampled | native | 88.54% | 0.1032 | 0.1309 | 0.8656 |
| 9b | sampled | selection_raw | 88.54% | 0.1006 | 0.1008 | 0.8243 |
| 9b | sampled | selection_temperature | 88.54% | 0.0772 | 0.0181 | 0.8263 |
| 9b | sampled | learned_temperature | 88.54% | 0.1027 | 0.1197 | 0.8656 |

Means describe seeds 11, 22, 33. Per-seed metrics, reliability bins and empirical operating points remain alongside the summary. Error targets do not provide guarantees. The one-parameter confidence map cannot correct every calibration defect.

Reproduce on exported text-free inputs: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/review_calibration.py --from-compact`. Full-source mode requires original saved predictions. Public inputs include labels, scores and IDs, but no request text.
