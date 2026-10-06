# Completed longer study: descriptive results

All 24 tuning runs and 36 main runs completed. The frozen schedule totals 168,000 optimizer updates. Each main evaluation uses the full 3,080-example official BANKING77 test set; post-hoc calibration uses 1,000 reserved examples. Three seeds: 11, 22, 33.

Initial SFT receives 4,000 updates; continued SFT, exact RL and sampled RL each receive another 4,000 from the same seed-matched SFT artifact. Thus continuation methods share exposure; SFT alone has less exposure. 0.8B and 4B use BF16 LoRA; 9B uses NF4 QLoRA.

| Size | Method | Accuracy mean | Seed SD | Deployed Brier | Test coverage at calibration 80% target | Accepted-case error |
|---|---|---:|---:|---:|---:|---:|
| 0.8b | sft | 79.06% | 0.47% | 0.1897 | 81.94% | 12.56% |
| 0.8b | continued_sft | 82.93% | 1.01% | 0.1445 | 82.63% | 9.84% |
| 0.8b | exact | 81.36% | 1.06% | 0.1321 | 82.65% | 10.39% |
| 0.8b | sampled | 80.53% | 0.68% | 0.1347 | 82.21% | 10.96% |
| 4b | sft | 86.48% | 1.43% | 0.1245 | 82.66% | 6.45% |
| 4b | continued_sft | 89.23% | 1.42% | 0.1038 | 83.84% | 4.67% |
| 4b | exact | 90.27% | 0.29% | 0.0818 | 83.14% | 3.81% |
| 4b | sampled | 88.20% | 2.21% | 0.1007 | 82.94% | 7.47% |
| 9b | sft | 87.08% | 0.31% | 0.1176 | 83.34% | 6.02% |
| 9b | continued_sft | 89.15% | 0.38% | 0.1054 | 81.87% | 4.25% |
| 9b | exact | 89.34% | 0.28% | 0.0942 | 82.28% | 3.65% |
| 9b | sampled | 88.54% | 0.63% | 0.1032 | 82.25% | 4.79% |

Deployed confidence uses the supervised scalar head for SFT/continued SFT and the candidate-conditioned confidence policy for RL. These differ. The JSON also retains the same policy readout across all methods. Seed SD is not a confidence interval; these are descriptive averages, not significance tests. Calibration-selected thresholds do not force exactly 80% test coverage.

| Size | Supervised method | Temperature Brier | Test coverage at calibration 80% target | Accepted-case error |
|---|---|---:|---:|---:|
| 0.8b | sft | 0.1256 | 82.10% | 12.68% |
| 0.8b | continued_sft | 0.1073 | 82.82% | 9.29% |
| 4b | sft | 0.0892 | 82.76% | 6.65% |
| 4b | continued_sft | 0.0740 | 82.24% | 4.22% |
| 9b | sft | 0.0821 | 83.59% | 5.97% |
| 9b | continued_sft | 0.0742 | 81.74% | 4.18% |

Temperature confidence is the calibrated selected-option probability, not a calibrated version of the scalar correctness head. This control tests a different confidence source. Accuracy is unchanged by temperature scaling.

## First findings and limits

- Continued supervision has the highest mean accuracy at 0.8B. Exact RL has the highest mean accuracy at 4B and 9B; sampled RL has lower mean accuracy than exact RL at every size.
- The highest observed mean accuracy is 4B exact RL. This does not establish a significant win over every alternative or an intrinsic advantage of 4B over 9B; precision differs.
- Continued SFT with temperature scaling has lower mean correctness Brier than either RL method at all three sizes. Confidence-aware RL therefore has not demonstrated a general advantage over simple calibration.
- Brier, discrimination and accepted-case error answer different questions. The operational rows retain achieved coverage so calibration gains are not mistaken for universal deferral gains.
- Paired uncertainty analysis, broader transfer/robustness, remaining ablations, replicated precision checks and isolated serving benchmarks remain pending. No production default is selected.
- The short pilot used different exposure and a 256-example test subset. Do not interpret differences from it as a controlled estimate of longer-training benefit.

Regenerate with `python3 scripts/summarize_longer_study.py`. The companion JSON records every source metric/manifest hash, per-seed values and aggregate values. Original per-run uncertainty and predictions remain in their source directories.
