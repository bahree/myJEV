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

## Milestone 3: New tasks and adaptation

- [ ] Publish a reviewed new-task protocol and dataset card.
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
