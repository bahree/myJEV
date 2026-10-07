# Roadmap and completed work

The two main tracks are implemented and evaluated: adapting Qwen at three sizes and building a small decision network from random initialization. Useful accuracy, confidence and deployment cost remain separate questions. The [findings](qwen-findings.md) and [decision lessons](decision-lessons.md) explain what the measurements support.

## Completed main study

- [x] Single-pass candidate selection, separate confidence estimation, exact finite-action and sampled objectives, reward arithmetic and gradient checks.
- [x] Grouped BANKING77 training/validation/calibration splits with the official test untouched; TF-IDF, untouched readout, supervised/Brier and post-hoc confidence controls.
- [x] Three-seed comparisons at 0.8B, 4B and 9B, with matched continued-supervision controls, retained training logs and validation-based selection.
- [x] Three-seed 4B precision controls and 36 transfer/robustness jobs with fixed calibration thresholds.
- [x] Paired uncertainty, accepted-case error/coverage, confidence controls, telemetry, plots and reproducible evidence.
- [x] Original policy-edit fixtures testing boundaries, exceptions and quoted instructions without new training.

See [experiments](experiments.md), [protocol](protocol.md), [tracking](tracking.md) and the [completed comparison](../results/longer-v1/summary.md).

## Completed scratch teaching track

- [x] Byte tokenizer, shared Transformer encoder and candidate scorer trained from random weights; no autoregressive generation.
- [x] Synthetic uncertainty fixtures, three-seed supervised/exact/sampled study and calibration controls, preserving instability and failed layout transfer.
- [x] One-seed BANKING77 diagnostic: 1.30% accuracy and one-class collapse. This is a failed quality result.
- [x] Saved teaching package, model card, checksum manifest, CPU/A30 measurements and Python/CLI/HTTP/Docker equivalence.
- [x] Corrected controlled learning ladder: all three seeds learned the deterministic rule on held-out nonces. The earlier shortcut-prone diagnostic is retained separately.

The ladder success does not repair the earlier natural-language failure or establish deployment quality. See [scratch findings](scratch.md) and the [implementation gates](scratch-plan.md). Wider workload scaling and a new frozen protocol for layout generalization are prospective extensions.

## Completed exploratory transfer and simpler controls

- [x] Freeze provisional archive groups before adaptation; compare unadapted/adapted artifacts and measure forgetting on public tasks.
- [x] Publish compact paired change estimates without redistributing private archive annotations or per-post predictions.
- [x] ModernBERT fixed-taxonomy control, fresh-process serving measurement and TF-IDF CPU serving baseline.
- [x] SST-2 transfer on the official validation set with unchanged BANKING calibration; no task training or threshold tuning.
- [x] Failure casebook, seven original runnable demos, a CPU calibration lab and teaching diagrams.
- [x] Source-based System One/OpenJev review, separating external claims and datasets from experiments actually run here.

These are bounded follow-ups, not retroactive changes to the frozen main comparison. ModernBERT uses one seed and a different training budget; SST-2 is familiar sentiment classification and possible pretraining exposure remains. See [decision lessons](decision-lessons.md), [research review](system-one-research.md), and [peer review](system-one-peer-review.md).

The archive study measures agreement with machine-generated references. Independent human review of rubric reliability, labels and semantic grouping remains necessary before interpreting it as human correctness. The [archive card](datasets/blog-archive.md) documents this limitation. A prepared audit does not count as a completed audit.

## Completed reusable releases

- [x] Six public adapter/head releases on Hugging Face with immutable revisions, meaningful model cards, separate licenses and verified downloaded hashes.
- [x] Select calibrated 4B continued supervision as the default within the published myJEV family. A fixed-taxonomy encoder remains a distinct, useful alternative.
- [x] Python, CLI, HTTP, local GPU Docker and clean-install checks; six-candidate benchmark evidence and output equivalence.
- [x] Versioned local `myjev:0.1.1-hub` image with direct pinned Hub load and host/container equality.
- [x] Published `amitbahree/myjev:0.1.1` on [Docker Hub](https://hub.docker.com/r/amitbahree/myjev), verified an anonymous digest pull and exact GPU HTTP response equality; retained logs and the [publication receipt](../results/container-registry-v1/publication.json).
- [x] Explicit input limits, overload behavior, managed endpoint custom-container recipe and request-content logging disabled by default.
- [x] MIT source license, repeatable curated publication, reader documentation and authentic logs.

See [models](models.md), [inference](inference.md) and [hosting](hosting.md). Publishing weights does not create a running endpoint. The managed recipe is documented but unexecuted; paid cloud deployment is not a required deliverable.

## Independent-review follow-up

- [x] Evaluate all six released checkpoints on three frozen candidate permutations, with original calibration thresholds; retain every prediction and operational result.
- [x] Display seed-level paired deltas and their spread beside conditional test-group intervals.
- [x] Apply identical post-hoc selection and learned-correctness temperature controls to all 36 runs; reproduce 144 metrics and 36 fits exactly from public inputs.
- [x] Disclose the punctuation-overlap sensitivity and preserve the official test results as primary.
- [x] Add aggregate UTF-8 and HTTP body limits while retaining exact token checks; measure representative maximal-byte rejection.
- [x] Rebuild through the public root Dockerfile, verify an empty model-cache startup, publish 0.1.2 and check anonymous pull plus exact GPU outputs.
- [x] Refresh six model cards, measured training/reliability figures, scratch and serving transcripts, all-seed CPU results, and actual-theme rendering.

These are bounded post-review checks. [Matched calibration results](../results/review-calibration-v1/report.md) qualify the earlier training-method interpretation; released weights and confidence settings remain unchanged. The [current container receipt](../results/container-registry-v2/publication.json) records the new input caps and its distinct cached and empty-cache conditions.

## Remaining owner decisions and optional extensions

- [ ] Independently human-audit the archive labels and grouping before human-correctness claims.
- [ ] Publish the blog articles and replace draft-series placeholders with actual article links. The private authoring tree contains four main articles plus a bonus calibration lab; article Markdown is excluded from this repository.
- [ ] If pursuing production use, evaluate the actual workload, accepted-error requirements, memory headroom and monitoring with independent data.

Phi, SmolLM and MAI fine-tuning remain deferred. The bounded ModernBERT control does not imply these alternatives were evaluated. Additional backbone studies, scratch capacity/layout experiments, distillation, optimized serving backends and broader policy tasks should begin with a new frozen protocol and resource pilot. No matched TypeSafe Jev latency comparison or reconstruction of its proprietary network is claimed.

Additional training seeds and longer Brier/correctness-only ablations remain optional research extensions. Certified selective-risk procedures, a non-root container and workload-specific authenticated deployment hardening are separate production work, not guarantees supplied by this research release.
