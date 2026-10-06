# What the completed Qwen study taught us

The frozen batch completed on October 6 at 02:18 UTC (October 5, 7:18 p.m. Pacific). It contains 24 tuning runs, 36 main runs and 168,000 optimizer updates. All main evaluations use the 3,080 official BANKING77 test examples; calibration uses 1,000 reserved examples. This closes the scheduled training batch, not all transfer, adaptation or release work.

## The result depends on the control

| Size | Initial SFT | Continued SFT | Exact RL | Sampled RL |
|---|---:|---:|---:|---:|
| 0.8B | 79.06% | 82.93% | 81.36% | 80.53% |
| 4B | 86.48% | 89.23% | 90.27% | 88.20% |
| 9B | 87.08% | 89.15% | 89.34% | 88.54% |

These are three-seed mean accuracies. Initial SFT uses 4,000 updates; each continuation adds 4,000 from the same seed-matched initial model. The fair test of an RL benefit is therefore against continued SFT, not just the initial checkpoint. At 0.8B, comparing exact RL only with initial SFT would suggest an improvement, while continued SFT does better with the same extra exposure.

![Completed accuracy comparison](../results/longer-v1/figures/accuracy.png)

The compact [paired evidence](../results/longer-v1/paired-analysis.md) resamples test groups, pairing examples across methods and averaging the three observed seeds:

| Exact RL minus continued SFT | Accuracy difference | 95% paired group interval |
|---|---:|---:|
| 0.8B | -1.57 percentage points | [-2.20, -0.95] |
| 4B | +1.04 percentage points | [+0.56, +1.49] |
| 9B | +0.19 percentage points | [-0.36, +0.71] |

These exploratory intervals are conditional on the three seeds and unadjusted for multiple comparisons. They are not uncertainty over every possible initialization. The 4B contrast provides evidence of improvement under this protocol; the 9B interval does not distinguish its small mean gain from zero. More capacity did not establish a monotonic advantage, and 9B's NF4 precision differs from BF16 at the smaller sizes.

## Calibration is not settled by accuracy

| Size | Continued SFT, temperature confidence | Exact RL, policy confidence | Sampled RL, policy confidence |
|---|---:|---:|---:|
| 0.8B | 0.1073 | 0.1321 | 0.1347 |
| 4B | 0.0740 | 0.0818 | 0.1007 |
| 9B | 0.0742 | 0.0942 | 0.1032 |

Mean correctness Brier, lower is better. Temperature scaling fits the selection logits on calibration data and uses the selected-option probability as confidence. RL reports the expected value of its candidate-conditioned confidence policy. These are distinct confidence sources, not two calibration transforms of the same scalar head.

![Completed confidence comparison](../results/longer-v1/figures/confidence.png)

Temperature-scaled continued supervision has lower Brier at all three sizes. The paired Brier contrasts also favor it within their conditional intervals. Confidence-aware RL therefore has not shown a general advantage over this simple control. Improving an initially weak learned head would have been an insufficient success criterion.

Operational ranking can differ from Brier. At the calibration-selected 80% coverage target, 4B exact RL achieved mean test coverage of 83.14% with 3.81% accepted-case error; temperature-scaled continued SFT achieved 82.24% with 4.22% error. These achieved coverages differ and this is a descriptive operating-point comparison, not an equal-coverage significance result. The full summary preserves thresholds and per-run uncertainty through its source files. An empirical calibration error target is not a production risk guarantee.

## Sampling did not help this finite-action experiment

Sampled RL had lower mean accuracy than exact RL at every size, by 0.83, 2.07 and 0.80 percentage points respectively. The paired conditional intervals favor exact optimization. At 4B, sampled accuracy also varied more across seeds: SD 2.21 percentage points versus 0.29 for exact RL.

That pattern is consistent with noisy estimation being a possible contributor, but it does not isolate the cause. Eight samples, confidence-policy initialization, learning-rate selection and the fixed schedule are also part of this experiment. The result does not imply that sampling is unnecessary when actions cannot be enumerated. Here the finite action space lets us compute the expectation directly, which makes exact optimization a particularly important control.

## What we learned about scale, data and resources

- The completed 0.8B model is a useful result, not a failed implementation: continued SFT reaches 82.93%. The short pilot and longer study changed exposure, tuning, continuation handling and evaluation size, so their difference is not a clean causal estimate of training duration alone.
- The best observed mean accuracy, 90.27%, belongs to 4B exact RL. That is a candidate for further evaluation, not an automatically selected production default. We did not perform a paired cross-size superiority test here, and precision complicates the 9B comparison.
- TF-IDF/logistic regression reached 88.28% on the same official test split using the full training partition. Exposure and tuning differ, so it is not a matched neural training control, but this inexpensive fixed-taxonomy baseline remains operationally relevant.
- The approximately one-epoch initial-plus-continuation budget supports a matched study; it does not prove convergence. Endpoint evaluations cannot reconstruct a validation learning curve or a principled early-stop decision.
- LoRA adapters are compact updates, not self-contained inference engines. Final comparisons must include the pinned backbone, heads, precision, input lengths and HTTP overhead. Final-checkpoint measurements now exist for six packaged candidates; they still do not establish Jev-equivalent performance.

## Completed transfer and precision follow-ups

All 36 frozen transfer/robustness jobs completed, covering continued supervision and exact RL at three sizes and three seeds. CLINC diagnostics use fixed subsets, not the complete benchmark. Temperature controls reuse BANKING77 calibration without fitting on transfer outcomes.

| Size | Continued SFT near / distant accuracy | Exact RL near / distant accuracy | Continued SFT / exact explicit OOS-none accuracy |
|---|---:|---:|---:|
| 0.8B | 81.51% / 68.62% | 81.64% / 67.45% | 0.39% / 3.52% |
| 4B | 87.50% / 80.86% | 86.72% / 79.56% | 76.30% / 59.38% |
| 9B | 88.15% / 83.46% | 86.59% / 81.90% | 60.16% / 58.07% |

The BANKING accuracy benefit does not establish an RL transfer benefit. The 4B near/distant paired intervals include zero; its exact-minus-supervised explicit OOS-none difference is -16.93 pp with a conditional 95% group interval [-19.53, -14.19]. The exact-RL OOS-none accuracies are 74.61%, 21.88% and 81.64% for seeds 11/22/33: this substantial initialization sensitivity is not represented by a bootstrap conditional on those seeds. None-option classification is distinct from deferral. On the OOS-deferral cohort, calibrated 4B supervision accepts 4.17% of requests at its original BANKING threshold, versus 7.68% for exact RL; all accepted answers there are wrong by construction. Changing coverage under shift prevents treating this as a matched-coverage guarantee.

The replicated 4B NF4 SFT control also completed. NF4-minus-BF16 accuracy differs by -1.20, +0.78 and +2.86 pp across seeds. The mean is +0.81 pp, with conditional group interval [+0.28, +1.35]; Brier difference is +0.0002 [-0.0049, +0.0053]. This is evidence about these three trained checkpoints, not a universal quantization advantage. Nonquantized modules also changed dtype. It does not isolate precision effects on 9B or RL.

The [paired extension report](../results/extension-analysis-v1/report.md) includes compact text-free inputs, source hashes and regeneration commands. Intervals average the observed seeds before resampling shared example groups; they are exploratory and do not include uncertainty over new training seeds or new task families.

## Local default and alternatives

The recommended starting artifact is **4B continued supervision with temperature confidence**, using seed 11 as the fixed packaging convention. It balances 89.23% mean BANKING accuracy, 0.0740 mean correctness Brier, stronger explicit unsupported-option transfer than 4B exact, and approximately 82 ms short-request HTTP p50 on the A30. This is a judgment across observed trade-offs, not a preregistered optimization or a production guarantee. Its confidence is a calibration-only selection-score proxy, not the supervised scalar head.

Keep 4B exact RL as the higher in-domain-accuracy alternative (90.27% mean), and 0.8B as the lower-resource option. The measured 9B continued model is slower at approximately 115 ms HTTP p50, has essentially equal BANKING accuracy and better distant-CLINC accuracy; workload-specific priorities can therefore change the choice. Final local measurements use packaged seed-11 checkpoints; quality summaries use all three seeds. They are not measurements of an average model.

The scratch model is already implemented and evaluated. Its unstable synthetic study and 1.30% BANKING diagnostic establish useful failure cases, not a useful natural-language replacement. Archive annotation/adaptation and external publication remain separate gates. See the [release tracker](models.md) and [scratch study](scratch-plan.md).

## Reproduce and inspect

```bash
python3 scripts/summarize_longer_study.py
python3 scripts/analyze_longer_study.py
```

The second command requires NumPy and Matplotlib from the research environment. Its public compressed inputs contain only IDs, groups, label IDs, correctness and squared confidence error. Full original predictions remain in the private evidence archive, with hashes linking the compact extract to its sources. `--extract` rebuilds that extract where the original files are available. The summary, paired intervals and plots are locally regenerated rather than copied from an external dashboard.

## Operational uncertainty, not just point estimates

The [per-seed operating-point report](../results/longer-v1/operational/report.md) retains 135 fixed-threshold measurements from 27 model/confidence combinations. For 4B exact RL, the calibration 1% empirical-error target gave test coverage of 16.75-34.42%, error of 0.39-1.32%, and one-sided 95% binomial upper bounds of 1.22-2.06%. This does not establish a 1% production guarantee. Seeds share test examples and are not pooled as independent observations. Group intervals and the independence assumptions of binomial bounds are explicit in the report.
