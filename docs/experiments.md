# Experiments and evidence

Read the [completed Qwen findings and lessons](qwen-findings.md), including paired uncertainty, confidence controls and what changes next.


The first milestone asks whether the implementation works and whether the proposed comparisons are worth scaling. It does not establish that confidence-aware RL is better than supervision or post-hoc calibration.

## Completed longer comparison

All three sizes completed the 168,000-update schedule, 24 tuning evaluations and 36 main evaluations on all 3,080 official test examples. The [regenerable summary](../results/longer-v1/summary.md) reports three-seed means, seed SD, deployed confidence and temperature controls. Continued supervision leads mean accuracy at 0.8B; exact RL leads at 4B and 9B. Continued supervision with temperature scaling has lower mean correctness Brier than both RL methods at every size. These are descriptive findings, not paired significance claims or a release selection.

The short pilot below remains separate evidence with different exposure and test size.

## Scope of the short comparison

Three seeds (11, 22, 33) are used at each size. The initial supervised run receives 100 updates, with one example per update. Continued supervision and each RL branch receive another 100 updates from the seed-matched supervised checkpoint. All use the same fixed random 256-example BANKING77 official-test subset and 256 reserved calibration examples. The 0.8B and 4B runs use BF16 LoRA; 9B uses NF4 QLoRA.

The 45 pilot evaluations comprise 15 per size: seven seed-11 methods/ablations and four main methods for each of the other two seeds. The table below covers the replicated main comparison, not all ablations.

## Accuracy and policy confidence

| Size | Method | Mean accuracy | Seed SD | Policy-confidence Brier |
|---|---|---:|---:|---:|
| 0.8B | sft | 51.30% | 5.88% | 0.3309 |
| 0.8B | continued_sft | 57.29% | 3.91% | 0.4555 |
| 0.8B | exact | 55.73% | 4.10% | 0.2718 |
| 0.8B | sampled | 47.53% | 2.39% | 0.2537 |
| 4B | sft | 70.70% | 0.00% | 0.5076 |
| 4B | continued_sft | 71.88% | 3.73% | 0.6050 |
| 4B | exact | 71.61% | 3.63% | 0.2040 |
| 4B | sampled | 69.40% | 3.63% | 0.2139 |
| 9B | sft | 73.70% | 3.54% | 0.5626 |
| 9B | continued_sft | 75.00% | 2.56% | 0.6695 |
| 9B | exact | 72.14% | 2.00% | 0.1964 |
| 9B | sampled | 69.66% | 2.35% | 0.2132 |

Seed SD measures variation across three runs; it is not a confidence interval. Lower Brier is better. The policy column holds the confidence parameterization fixed across methods, but supervised artifacts normally deploy their scalar head. It is therefore only one confidence comparison.

![Main pilot results](../results/figures/three-seed-study.png)

## Compare against calibration, too

Continued supervision has the highest observed mean accuracy at every size. Exact RL is closer than sampled RL to that supervised control. These observations are conditional on short exposure, three seeds, and a small fixed subset.

RL improves the weak supervised confidence policy. However, temperature-scaled supervised selection logits achieve lower mean correctness Brier than either RL policy in this pilot. Improving only the weak policy baseline would be an incomplete argument for RL.

![Continued-supervision confidence controls](../results/figures/confidence-controls.png)

Constant base-rate confidence is included because a low Brier score alone does not imply useful discrimination or high accuracy. Operational evaluation also needs accepted-case error and coverage, with thresholds chosen on calibration data only.

## Other evidence

TF-IDF/logistic regression reaches **88.28% accuracy** on all 3,080 official test examples. Temperature scaling reduces correctness Brier from **0.1246 to 0.0703**, without changing argmax decisions. It trains on the complete training partition, so exposure and test-set size differ from the neural pilot.

The available scripts also support an untouched-backbone control, GLiClass, CLINC transfer cohorts, and paired robustness diagnostics. The initial 0.8B transfer/robustness work is limited; the roadmap calls for broader checks before drawing generalization conclusions. One-seed precision diagnostics do not isolate quantization effects robustly.

## Find the underlying evidence

| Evidence | Location |
|---|---|
| Main comparison summary | [final-study-table.json](../results/final-study-table.json) |
| Paired contrasts against continued supervision | [paired-comparisons.json](../results/paired-comparisons.json) |
| Seed-11 metrics and predictions | `results/pilot-{size}-{method}-evaluation/` |
| Seed-22/33 metrics and predictions | `results/three-seed-{size}/seed-{seed}/{method}/` |
| Temperature and constant controls | Corresponding `*-posthoc/` directories |
| Training manifests and update records | [artifact-manifests](../results/artifact-manifests/) |
| HTTP benchmark measurements | [docker-size-validation](../results/docker-size-validation/) |
| Full observed GPU activity | [telemetry chart](../results/gpu-telemetry-20261004T205942Z/gpu-activity.png) |

Per-run reports contain macro-F1, correctness/selection Brier metrics, confidence discrimination, disclosed ECE bins, reliability values, and calibration-selected coverage/error results. Paired bootstrap intervals are conditional on the observed seeds and are not adjusted for multiple comparisons. Consult [the protocol](protocol.md) before treating pilot numbers as production error guarantees.

## Reproduce figures or train new runs

From the repository root after installing the locked environment:

```bash
.venv/bin/python scripts/plot_study.py
.venv/bin/python scripts/summarize_study.py
.venv/bin/python scripts/paired_comparisons.py
```

These commands use the included evaluation evidence and require no model download. Run them in a writable checkout; they regenerate output files.

For new training, begin with the supervised 0.8B artifact in the [quick start](quickstart.md). Then, in a separate experiment checkout if you want to preserve the published results unchanged:

```bash
mkdir -p results
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/pilot_suite.py 0.8b
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/three_seed_pilot.py 0.8b
```

These runners skip completed artifacts/evaluations. An existing published `results/` tree will therefore cause matching evaluations to be skipped: move the published results aside to a backup before collecting fresh measurements. Keep original evidence and new runs distinct. Adapt the size argument and supervised artifact for 4B or 9B.

## Serving measurements

Local GPU container checks passed at all three sizes. Warm concurrency-one HTTP p50/p95 were 60.1/63.7 ms (0.8B), 78.6/81.0 ms (4B), and 101.2/103.7 ms (9B). Each check used 40 short three-candidate requests, cached backbones, and an idle target GPU while another GPU was still training. These are implementation measurements, not an isolated-host service-level guarantee.

Read [hosting](hosting.md) for limits, readiness, queues, precision, and deployment details. Longer training, full-test neural comparison, replicated precision controls, and representative serving workloads remain on the [roadmap](roadmap.md).
