# Roadmap

This page makes incomplete work visible as the project develops. Checked items describe the current milestone; unchecked items are planned work, not measured results or release dates.

## Milestone 1: Working implementation and feasibility

- [x] Single-pass candidate selection and separate confidence estimation.
- [x] Supervised, exact expected-reward, and sampled optimization implementations.
- [x] Objective arithmetic and gradient checks.
- [x] BANKING77 splits, TF-IDF baseline, and post-hoc confidence controls.
- [x] Short three-seed main comparison at 0.8B, 4B, and 9B; 45 pilot evaluations including ablations.
- [x] Saved predictions, manifests, update logs, GPU telemetry, and regenerable figures.
- [x] Python, CLI, HTTP, and GPU Docker checks at all three sizes.
- [x] Synthetic input-boundary checks at 160 candidates and 4,096 tokens.

## Milestone 2: Stronger training and evaluation

- [ ] Longer matched training with equal validation-based tuning opportunities. [This batch is running](training.md); results are pending.
- [ ] Frozen main neural comparison on the full official BANKING77 test split.
- [ ] Replicated precision controls to separate capacity from quantization effects.
- [ ] Transfer and robustness studies at every size, extending the initial 0.8B diagnostics.
- [ ] More extensive uncertainty reporting for rare accepted-case errors.

## Follow-up: Phi and architecture efficiency

The next backbone pilot will evaluate Microsoft Phi after the frozen Qwen batch. Start with [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct), a 3.8B candidate, subject to a pinned revision, license/data review and local loader compatibility. This is scheduled follow-up work, not a launched job or a demonstrated improvement.

- [ ] Verify token aliases, context limits, hidden-state extraction and adapter targets; run a 100-update memory/throughput pilot on one A30 before a larger commitment.
- [ ] Compare untouched readout and supervised adaptation with matched data exposure and tuning opportunities. Reuse frozen partitions and reserve calibration data for calibration.
- [ ] Benchmark against Qwen 4B for similar size and against the smallest useful Qwen model for deployment cost. Measure model-only and HTTP latency, memory, accuracy and accepted-case error/coverage across input lengths and candidate counts.
- [ ] Treat a dedicated encoder/candidate head as a separate architecture comparison. A Phi swap alone does not test whether token-alias scoring is the right design.
- [ ] Consider MAI only after identifying a specific locally downloadable checkpoint with suitable licensing, size and task support. API availability does not establish local fine-tuning feasibility.

No Jev-equivalent latency is assumed. A useful comparison must disclose hardware and serving differences; generating fewer tokens alone does not prove a better decision architecture. Expand to RL or more seeds only after the pilot establishes a useful comparison and its resource cost.

## Milestone 3: New tasks and adaptation

- [x] Publish the [blog-archive study card](datasets/blog-archive.md), with preparation and annotation status.
- [ ] Review and freeze the archive rubrics, grouping and evaluation protocol.
- [ ] Audit machine-assisted labels against independent human judgments.
- [ ] Freeze evaluation groups before adaptation.
- [ ] Compare unadapted and adapted artifacts and measure forgetting on public tasks.

This part remains exploratory. Annotation material and draft articles are not distributed here. Machine-label agreement must be distinguished from human-audited correctness.

## Milestone 4: Reusable releases

- [ ] Select a default artifact using quality, deferral, latency, and memory measurements.
- [ ] Publish adapters, heads, calibration, manifests, and model cards on Hugging Face.
- [ ] Publish a versioned container image with verified load and output behavior.
- [ ] Expand serving benchmarks to isolated-host runs and representative document lengths.
- [ ] Compare direct scoring, constrained one-token generation, minimal JSON, and answer plus explanation under matched serving conditions. See the [benchmark protocol](hosting.md#planned-scoring-versus-generation-benchmark) and its related-work reference.
- [ ] Finalize source licensing and release packaging.
- [ ] Add links to the two published blog articles.

A paid cloud endpoint is optional and is not a required project deliverable. The documented managed-hosting recipe can be used independently once release artifacts exist.

## Reader experience after the longer batch

The existing guides are available now. The next presentation pass will make them easier to discover from the landing page, drawing on [helloLondon's documentation structure](https://github.com/bahree/helloLondon).

- [ ] Add a prominent README documentation table linking the full index and each main reader task, including W&B, lessons and inference/Docker.
- [ ] Clarify read-results, train-your-own and released-model entry paths; add an annotated directory map and experiment-lifecycle diagram.
- [ ] Add an actual artifact-linked request/response example explaining scores, confidence and unsupported inputs.
- [ ] Explain deployable artifacts versus resumable checkpoints and consolidate troubleshooting.
- [ ] Refresh results and authentic training/GPU visuals after analysis, with source evidence, uncertainty and regeneration commands.
- [ ] Validate the release inference/Docker walkthrough from a clean environment and replace pending model/image references with tested releases.
- [ ] Add the two published blog links, mutual navigation and documentation link checks.

These tasks follow the running training batch and do not change its frozen protocol. Research and release requirements above remain separate dependencies.
