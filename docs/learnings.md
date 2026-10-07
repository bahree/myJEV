# Engineering and experimental lessons

Read the [completed Qwen findings and lessons](qwen-findings.md), including paired uncertainty, confidence controls and what changes next.


These notes distinguish measured observations from choices and unresolved questions. The [experiment guide](experiments.md) contains the current pilot results; the longer study is complete with [descriptive results](../results/longer-v1/summary.md).

## Count work across experiments, not as one model

The 168,000-step budget comprises 24,000 tuning steps and 144,000 main-comparison steps. It spans three model sizes, four training approaches, two tuning learning rates and three main seeds. All methods use AdamW; exact and sampled describe objective estimation, not additional optimizers. The [training guide](training.md) gives the full arithmetic.

One initial supervised run consumes 4,000 examples. Each continuation consumes another 4,000, starting from that supervised artifact. With 7,999 training rows, each continuation model has roughly one epoch in its ancestry. Reusing the supervised artifact does not mean adding all three continuation branches into one model's history.

## A fixed budget does not establish convergence

The budget supports a matched comparison under equal exposure and tuning opportunity. It does not establish each model's best achievable result. Training loss can fall while held-out performance worsens, and losses from supervised and RL objectives are not interchangeable.

Improvement should be assessed using held-out accuracy/F1, correctness Brier, calibration and accepted-case error at fixed coverage. A future convergence study needs predefined periodic validation, a meaningful improvement threshold and patience across several checks. Its stopping decisions must not use the test set. Current endpoint-only evaluations cannot establish a reliable plateau.

Numerical failure is a different question: NaN/infinite loss, fatal process errors and disk exhaustion warrant stopping an affected job. A health check must not interpret a negative RL loss or ordinary fluctuations as failure. Our unattended monitoring does not claim autonomous scientific judgment between observations.

## Hardware is only part of elapsed time

The reference host has three 24 GB A30 GPUs, one independent size per GPU. The schedule uses one example per update, reference-policy computation for RL, frequent checkpointing and full evaluation passes. These choices also contribute to duration. Device utilization is not a measure of kernel efficiency, and no measured newer-GPU speedup is claimed. Batch-size or backend improvements should be benchmarked separately before changing a frozen comparison.

## Small adapters do not imply a small serving model

LoRA reduces the learned update and training state, while inference still needs the backbone. The size study controls the model family to investigate capacity, not to establish an optimal architecture. One-pass scoring avoids the output decoding loop, but still processes the full input. Fitting a training pilot on one 24 GiB A30 does not establish maximum-context serving capacity or Jev-equivalent latency.

The [hardware and model-size rationale](training.md#choosing-model-size-around-the-hardware) distinguishes per-device fit from aggregate GPU memory. The [inference cost explanation](inference.md#what-the-adapter-saves-and-what-inference-still-costs) separates adapter files, backbone requirements, measured HTTP latency and untested optimizations. Final model selection must weigh quality against these costs; 9B is not automatically the default.

## Keep observability separate from training

Local logs remain authoritative even when a dashboard is unavailable. An independent W&B bridge can show live progress and import completed histories without restarting training. Record observation time, original step/time axes and historical-import labels explicitly. Keep secrets, weights and archive text outside telemetry uploads. See [W&B setup and usage](tracking.md).

## Test inference through the real deployment path

A successful image build and CUDA tensor operation did not prove that the first model request would work: the pilot container needed a C compiler for runtime kernel startup. A wrapper that resolved a virtualenv Python symlink also lost its environment's dependencies. Both failures were recorded and fixed. Python, CLI, HTTP and Docker checks must exercise the actual artifact and custom confidence head.

The pilot passed these interface checks at all three sizes. Its short latency samples are not final service-level guarantees. Representative final-checkpoint inputs, local serving benchmarks, calibration and public artifact/image releases subsequently completed. Empty-model-cache and review follow-ups are recorded separately in the hosting guide. See [inference and Docker](inference.md) and [the hosting protocol](hosting.md).

## Scratch models exposed two different failure modes

A small encoder fit the initial routing template but failed a reordered layout. Restoring that layout recovered some accuracy; varied training layouts and more exposure improved validation, but the final three-seed study remained unstable. See the [scratch evidence](scratch.md) rather than a single successful example.

The BANKING77 scratch diagnostic collapsed to one class at 1.30% accuracy. Its low correctness Brier was a consequence of low confidence in usually wrong predictions, not useful classification. A fast model and a low calibration loss can coexist with an unusable system. This is why the teaching tracks retain quality, uncertainty, data exposure and serving costs together.
