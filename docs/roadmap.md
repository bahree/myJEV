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

- [x] Longer matched training with equal validation-based tuning opportunities; [descriptive results](../results/longer-v1/summary.md) recorded.
- [x] Frozen main neural comparison on the full official BANKING77 test split, all three seeds and sizes. Paired group analysis is recorded; broader uncertainty and release checks remain.
- [ ] Replicated precision controls: three 4B NF4 SFT runs at matched 4,000-example exposure are frozen and queued after GPU 0 completes its transfer jobs; completed BF16 controls are reused.
- [ ] Transfer and robustness studies at every size, extending the initial 0.8B diagnostics.
- [x] Per-seed operational report with accepted counts, fixed calibration thresholds, group intervals and conditional binomial upper bounds; production guarantees remain unsupported.

## Deferred: alternative pretrained backbones

Phi, SmolLM and MAI exploration is parked to focus on the two active tracks: pretrained Qwen adaptation and a decision model built from scratch. No alternative-backbone run is scheduled. If this work is revisited, consider [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct), a 3.8B candidate, subject to a pinned revision, license/data review and local loader compatibility. The checklist below is retained for a future decision, not as a requirement for the current two-track study.

- [ ] Verify token aliases, context limits, hidden-state extraction and adapter targets; run a 100-update memory/throughput pilot on one A30 before a larger commitment.
- [ ] Compare untouched readout and supervised adaptation with matched data exposure and tuning opportunities. Reuse frozen partitions and reserve calibration data for calibration.
- [ ] Benchmark against Qwen 4B for similar size and against the smallest useful Qwen model for deployment cost. Measure model-only and HTTP latency, memory, accuracy and accepted-case error/coverage across input lengths and candidate counts.
- [ ] Treat a dedicated encoder/candidate head as a separate architecture comparison. A Phi swap alone does not test whether token-alias scoring is the right design.
- [ ] Consider MAI only after identifying a specific locally downloadable checkpoint with suitable licensing, size and task support. API availability does not establish local fine-tuning feasibility.

No Jev-equivalent latency is assumed. A useful comparison must disclose hardware and serving differences; generating fewer tokens alone does not prove a better decision architecture. Expand to RL or more seeds only after the pilot establishes a useful comparison and its resource cost.

## Teaching extension: a decision model from random initialization

See the [implementation plan](scratch-plan.md) for architecture, milestones S0-S6, test gates, data generation, resource budgeting and completion criteria.

This implemented teaching checkpoint returns to the build-and-explain approach of helloLondon. It is an original educational model, not a reconstruction of TypeSafe Jev's undisclosed internals. The official announcement describes architecture, parallel sampling and RLCD at a high level; that is insufficient to reproduce its network and training recipe.

For the deferred pretrained-backbone comparison, consider [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) as a genuinely smaller pretrained decoder control. [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) is an optional closer-size comparison with published training materials. Start with small pilots rather than repeating the entire main study for every backbone. These checkpoints are not yet integrated or evaluated.

The first prototype uses a deterministic byte tokenizer and a randomly initialized shared Transformer encoder with 201,175 parameters. The BANKING77 configuration has 217,559 parameters because its positional embedding budget is larger. The original 5-20M design remains a possible extension, not an implemented model. Candidate IDs remain outside the network. Shared candidate scoring, scalar confidence and the experimental confidence policy are implemented.

- [x] Implement the byte tokenizer, encoder, masking and candidate scorer without pretrained weights.
- [x] Generate versioned tasks with known uncertainty and held-out templates/pairings; retain the failed layout-transfer pilot.
- [x] Verify padding, candidate-order equivariance, renamed IDs, absence of generation and finite-action reward/gradient arithmetic.
- [x] Complete the three-seed supervised, temperature/constant calibration and matched exact/sampled teaching comparison. Retain all seed results, including instability.
- [x] Run the one-seed BANKING77 diagnostic on isolated splits. It collapsed to one class at 1.30% accuracy; this is a failed quality result.
- [x] Measure serialized weights, CPU/A30 model latency and GPU Docker HTTP latency; verify Python/CLI/HTTP equivalence on the short synthetic request.
- [x] Publish runnable stages, teaching explanations and reproducible evidence; update both private blog drafts.
- [ ] Extend workload/candidate scaling, robustness and uncertainty measurements before recommending a deployment artifact.
- [ ] Investigate seed and layout sensitivity under a newly frozen validation protocol if further scratch development is pursued. Do not retune against the existing test results.

Foundations: [Deep Sets](https://arxiv.org/abs/1703.06114) for set symmetry, [Set Transformer](https://arxiv.org/abs/1810.00825) for attention over sets, and [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) for post-hoc calibration. These motivate components; none describes TypeSafe's proprietary architecture. [OpenJev-RLCD](https://arxiv.org/abs/2609.38850) is a separate implementation involving sampled rationales, so it must not be presented as our one-pass design or an official Jev architecture disclosure.

## Milestone 3: New tasks and adaptation

- [x] Publish the [blog-archive study card](datasets/blog-archive.md), with preparation and annotation status.
- [x] Freeze an exploratory machine-label protocol with existing rubrics/provisional groups and explicit limits.
- [ ] Human-review archive rubric reliability and semantic grouping.
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
- [x] License the source code under MIT.
- [ ] Finalize artifact licensing review and release packaging.
- [ ] Add links to the published blog-series articles.

A paid cloud endpoint is optional and is not a required project deliverable. The documented managed-hosting recipe can be used independently once release artifacts exist.

## Reader experience after the longer batch

The existing guides are available now. The next presentation pass will make them easier to discover from the landing page, drawing on [helloLondon's documentation structure](https://github.com/bahree/helloLondon).

- [x] Add a prominent README documentation table linking the full index and each main reader task, including W&B, lessons and inference/Docker.
- [x] Clarify read-results, train-your-own and released-model entry paths; add an annotated directory map and experiment-lifecycle diagram.
- [ ] Add an actual artifact-linked request/response example explaining scores, confidence and unsupported inputs.
- [x] Explain deployable artifacts versus resumable checkpoints and consolidate troubleshooting, including shared-cache/disk failures.
- [ ] Refresh results and authentic training/GPU visuals after analysis, with source evidence, uncertainty and regeneration commands.
- [ ] Validate the release inference/Docker walkthrough from a clean environment and replace pending model/image references with tested releases.
- [ ] Add the published blog-series links, mutual navigation and documentation link checks.

These tasks follow the completed training batch and do not change its frozen protocol. Research and release requirements above remain separate dependencies.

## Sequence the work without changing the current experiment

1. Finish the Qwen reporting gate: audit, paired analysis, lessons in both blog drafts and the public repo, regenerable figures and pushed evidence. Complete this before scratch implementation.
2. Audit completed artifacts and aggregate the Qwen results before selecting further large runs.
3. Implement the scratch model progressively: supervised synthetic task first, correctness confidence and calibration second, exact/sampled training after behavioral tests pass.
4. Evaluate the scratch model and Qwen artifacts on appropriate shared tasks, recording unequal pretraining exposure and complete serving costs. Phi, SmolLM and MAI remain deferred and are not release dependencies.
5. Present both build tracks in the articles, preserving distinct learning goals and honest shared-task comparisons. Different purposes do not prevent one model from winning a measured metric; unequal pretraining prevents attributing that difference solely to architecture.

The scratch extension has completed its first controlled study, one-seed natural-language failure diagnostic and pilot serving checks; see the [walkthrough](scratch.md) for completed checks and remaining evidence. It complements the existing adaptation study and does not delay its result audit or silently expand the frozen training budget.

Current execution details and failed quality gates are recorded in the [scratch plan checkpoint](scratch-plan.md#execution-checkpoint). Implementation checks passing does not establish useful model quality.

The expanded transfer/robustness study is frozen and running across 18 checkpoints (three sizes, three seeds, continued supervision and exact RL). Results will be reported after validation; this is not additional tuning. The blog scope now has four drafts: scratch/overview, Qwen training, evaluation/transfer, and inference/hosting.
