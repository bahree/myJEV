# Milestone history

## 2026-10-07: Docker Hub release

- Published `amitbahree/myjev:0.1.1`; recorded the immutable manifest digest and verified an anonymous pull followed by GPU response equivalence.
- Added direct run commands, a Docker Hub overview, publication logs and explicit cached-validation conditions.
- Replaced vendor-specific decision-head discussion with general backbone, head, training and confidence principles.

Earlier entries below describe their checkpoint at the time; current availability is documented in the model and inference guides.

## Research and release preparation

- License source code under MIT and expand the teaching outline to four posts.
- Run the frozen all-size transfer/robustness extension; queue matched precision controls and isolated serving checks.
- Prepare six local candidate packages, without declaring a default or uploading weights.

## Qwen lessons and implemented scratch teaching track

- Completed the Qwen reporting gate with paired analysis, public lessons and figures for both private Hugo drafts.
- Added a random-initialized 201K-parameter scorer, byte tokenizer, synthetic generator, training and calibration controls.
- Retained the failed template pilot, unstable three-seed synthetic study and one-class BANKING77 failure diagnostic.
- Verified shared Python/CLI/HTTP loading and GPU Docker behavior; recorded CPU/A30 and HTTP timings without implying matched Jev performance.
- Preserved local logs and imported twelve completed scratch histories into W&B as historical runs.

## Longer comparison completed; scratch track planned

- Retained all 60 completed training artifacts locally and recorded 168,000 updates across the frozen schedule.
- Published full-test descriptive results for 36 main evaluations, source metric hashes, manifests and training histories, with a summary script.
- Added the scratch model implementation plan, milestone gates, synthetic-data controls and two-track teaching scope. Alternative pretrained backbones remain deferred.
- Kept statistical significance, transfer and final release selection explicitly pending.

## Tracking and reader guides

- Added reusable W&B live monitoring, historical imports, evidence snapshots and a credential-free environment template.
- Explained the 168,000-step budget, epoch counters, convergence limitations and retained local logs.
- Added a Python/CLI/HTTP/Docker walkthrough and troubleshooting, separating pilot checks from pending release validation.

This history describes what each reader-facing snapshot contains. It records the state of the project at publication; upcoming work is tracked separately in the [roadmap](docs/roadmap.md).

## Longer comparison started: 2026-10-04

- Documented AdamW versus training steps, the 168,000-step budget, and batch progress/ETA semantics.
- Added validation-only learning-rate selection with equal trial budgets.
- Started the 0.8B, 4B, and 9B runners on the existing A30s.
- Defined 4,000 initial and 4,000 continuation updates with full official-test evaluation after tuning.
- Continuations now see matched subsequent examples rather than restarting the initial data subset.
- Added resume-safe orchestration, frozen data/configuration checks, and progress monitoring.
- Removed em dashes from project prose and GPU figure titles.

Results from this batch are pending; no new quality claim is made.

## Inference pilot: prepared 2026-10-04

### Available

- Single-pass candidate-selection model, separate confidence heads, and exact/sampled RL objectives.
- Pinned 0.8B, 4B, and 9B configurations with a completed short three-seed comparison.
- BANKING77 preparation, classical and generalist controls, and calibration/evaluation tools.
- Python, CLI, HTTP, Docker, and a managed-hosting recipe.
- Saved pilot predictions, numerical metrics, manifests, and regenerable charts.
- Reader guides covering architecture, training, evaluation, hosting, and planned releases.

### Findings and scope

Continued supervision has the highest observed mean accuracy at each size in this short pilot. Temperature scaling is a stronger confidence control than the RL policies in the mean Brier comparison. Neural evaluations use a fixed 256-example test subset; the training schedule does not establish convergence.

### Still planned

Longer matched training, broader transfer/adaptation studies, released adapters, a registry image, source-license selection, and published blog articles. These entries will be filled in as work is completed; this milestone does not imply their availability.
