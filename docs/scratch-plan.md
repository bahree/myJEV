# Build a decision model from scratch: implementation plan

Status: planned implementation, no scratch weights or results yet. The companion Qwen longer study has completed; its [descriptive results](../results/longer-v1/summary.md) are a separate evidence track. Phi, SmolLM and MAI are parked.

## Purpose and boundaries

Build a small, inspectable decision network from random initialization, then explain what pretrained Qwen adaptation adds. Both tracks accept context, instructions and request-supplied candidate descriptions, select a candidate and estimate its correctness. This is an original teaching model, not a reproduction of TypeSafe Jev's undisclosed architecture. Broad language competence and Jev-equivalent speed are not promised.

Synthetic tasks teach mechanisms; BANKING77 tests limited natural-language learning. Shared-task results can favor one model, but unequal pretraining exposure prevents attributing differences solely to architecture. A decoder-family sweep and large-scale language pretraining are outside this phase.

## Proposed architecture

Start with four bidirectional Transformer encoder blocks, hidden size 256, four attention heads and feed-forward size 1,024, with shared weights across context and candidate text. Use an 8,192-token tokenizer trained only on training text; use a deterministic byte tokenizer for initial fixtures. Count actual parameters rather than treating the provisional 5-20M range as measured.

Encode context/instructions once and candidates as a padded batch. Candidate representations cross-attend to encoded context, then optionally pass through one attention block across candidates. No candidate-index positional embeddings: token positions inside each description still matter. A shared scalar head produces one logit per candidate. Mask padding before attention and softmax, map the selected index back to the caller's ID outside the network, and reject over-limit inputs.

Begin with selection only. Add a candidate-conditioned scalar correctness head using detached predictions as supervised correctness targets. After that works, add the experimental 21-bin confidence policy so exact and sampled reward training can reuse the existing objective definitions. Document which heads train in each phase, the source of correctness labels and whether confidence gradients update the encoder. Use a frozen supervised reference for RL.

There is no vocabulary projection for answer aliases and no generation loop. The computation includes multiple encoder operations and candidate attention; calling it non-autoregressive does not make work independent of candidate count. Exact candidate permutation equivariance is a testable design goal in evaluation mode, not an assumed result in every stochastic training step.

## Milestones and acceptance gates

| Stage | Implementation and evidence | Gate before moving on |
|---|---|---|
| S0: close Qwen batch | Validate manifests/counts, regenerate per-seed summary, retain source hashes and logs; then paired uncertainty analysis | Distinguish descriptive means from significant effects and deployed from policy confidence |
| S1: minimal scorer | Add `src/myjev/scratch/` encoder/scorer/config and tiny deterministic fixtures; use random weights and a CPU training smoke run | Overfit a small unambiguous fixture; candidate padding has no effect; reordering maps scores correctly; save/reload agrees |
| S2: controlled learning | Add versioned synthetic generator and dataset card; train-only tokenizer; grouped train/validation/calibration/test split | Disjoint templates/compositions, exact labels and known uncertainty verified; baselines run on identical groups |
| S3: confidence | Scalar confidence and temperature/constant controls; then 21-bin policy and exact/eight-sample leave-one-out RL with KL | Reward/gradient checks, same initialization/exposure, deterministic deployment evaluated separately; thresholds fit on calibration only |
| S4: language diagnostic | Train on existing BANKING77 partitions with descriptions and random candidate order; reuse TF-IDF and Qwen evidence with exposure disclosed | Official test untouched during tuning; validate configuration before test; report failures and candidate-length constraints |
| S5: deployment | Versioned scratch manifest, dedicated loader behind shared score contract, Python/CLI/HTTP then Docker | Artifact type cannot be mistaken for Qwen; one result contract, proper masking/input errors, equivalence and CPU/A30 benchmarks |
| S6: teaching release | Worked tensor-shape example, training curves, calibration and resource figures, runnable commands and checkpoint card | Every plotted result has provenance and a regeneration command; proposed features clearly separate from measured ones |

The Qwen result audit and CPU scratch implementation can proceed independently. Use one idle A30 only after the CPU gates pass. Do not rerun the 168,000-update schedule for the scratch model by default.

## Synthetic data and leakage controls

Start with structured routing rules: attributes in a context determine which request-defined option applies. Generate clean cases, distractors, missing-correct-option cases and label-preserving candidate reorderings. Group variants from a source scenario together. Hold out compositions and templates, not merely random rows, to test transfer beyond memorized strings.

For known uncertainty, generate a latent outcome from an explicit distribution and reveal only controlled evidence. Keep its conditional probabilities available to the evaluator; never expose hidden outcomes or probability metadata to model input. Separate irreducible ambiguity from corrupted labels and from missing information. Compare predicted option probabilities and selected-answer confidence to their appropriate targets.

Freeze generator version, seeds, group IDs and hashes before model selection. Fit tokenizer and preprocessing only on training text. Maintain a dataset card explaining why synthetic evidence does not establish natural-language generalization.

## Training and resource budget

Run a 100-update memory/throughput pilot for each new configuration. Start at 256 context tokens, 64 tokens per candidate and at most eight candidates. These are initial experiment limits, not release guarantees. Test candidate scaling separately at 2, 8 and 32; larger counts and long documents require fresh fit/latency checks.

Use at most two validation-selected learning rates in the first pilot. If selection fails to learn an easy rule, inspect data, masking and optimization before spending on RL. If the pilot is viable, freeze a main budget with three seeds and equal exposure across continued supervision, exact RL and sampled RL. Choose that budget from validation learning curves and measured cost, never test results. Record the decision before main runs. Do not quote a total duration before throughput is measured.

Stop affected runs for nonfinite values or resource exhaustion. Predefine validation patience and minimum improvement for exploratory training; main matched runs either retain a fixed budget or use a disclosed matched stopping protocol. Do not conflate a health guard with scientific convergence assessment.

## Evaluation and fair interpretation

Measure selection accuracy/macro-F1, correctness Brier, option-distribution Brier, confidence discrimination, reliability and accepted-case error/coverage. Choose thresholds only on calibration data; show achieved test coverage and uncertainty, including when an error target fails. Use paired groups where tasks match and keep seed variability distinct from sampling uncertainty.

Ablate candidate interaction to test whether it adds value. Verify candidate reordering, renamed IDs, description paraphrases, distractors, absent answers and longer inputs. Report synthetic and natural-language results separately. Distillation would be a third training condition requiring an explicit future decision and teacher/data provenance.

Measure total scratch weights against Qwen backbone plus adapters and heads. Report CPU and A30 model latency separately from HTTP, cold start, warm p50/p95, concurrency, peak memory and batch/candidate sizes. A small parameter count is not itself measured speed. A Jev comparison requires equivalent tasks and disclosure of service/hardware differences.

## Blog integration and completion

Part 1 builds the scratch scorer with visible tensors and a worked decision, then explains how Qwen pretraining, LoRA/QLoRA and confidence objectives change the problem. Part 2 evaluates both tracks, examines where each fails, and serves verified artifacts. Retain two posts unless runnable explanations become too long, in which case split rather than remove depth.

The scratch track is complete when a fresh checkout can train the small controlled example, reproduce the reported evaluation, load its saved artifact and make the same decision through supported interfaces. Competitive broad-language accuracy is a research outcome, not a prerequisite for an honest teaching release.

Foundations: [Deep Sets](https://arxiv.org/abs/1703.06114), [Set Transformer](https://arxiv.org/abs/1810.00825), and [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). See [attribution](attribution.md) for the separate architectures of public Jev-inspired implementations.
