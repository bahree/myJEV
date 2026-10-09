# Scratch-model design and validation

The scratch scorer combines a byte encoder, candidate attention and confidence heads. Read the [walkthrough](scratch.md) for runnable commands and measured results. The [Qwen study](qwen-findings.md) provides a separate pretrained-backbone comparison.

## Purpose and boundaries

Build a small, inspectable decision network from random initialization, then explain what pretrained Qwen adaptation adds. Both tracks accept context, instructions and request-supplied candidate descriptions, select a candidate and estimate its correctness. The architecture was designed for this teaching experiment. TypeSafe Jev’s undisclosed architecture cannot be reproduced from its announcement. Broad language competence and Jev-equivalent speed are not promised.

Synthetic tasks teach mechanisms; BANKING77 tests limited natural-language learning. Shared-task results can favor one model, but unequal pretraining exposure prevents attributing differences solely to architecture. A decoder-family sweep and large-scale language pretraining are outside this study.

## Initial design and implemented prototype

The initial larger design proposed four bidirectional Transformer encoder blocks, hidden size 256, four attention heads and feed-forward size 1,024, with shared weights across context and candidate text. The implemented first checkpoint instead uses two blocks, hidden size 64 and deterministic bytes, totaling 201,175 parameters for the synthetic configuration. The larger design remains untested. Use an 8,192-token tokenizer trained only on training text; use a deterministic byte tokenizer for initial fixtures. Count actual parameters rather than treating the provisional 5-20M range as measured.

Encode context/instructions once and candidates as a padded batch. Candidate representations cross-attend to encoded context, then optionally pass through one attention block across candidates. No candidate-index positional embeddings: token positions inside each description still matter. A shared scalar head produces one logit per candidate. Mask padding before attention and softmax, map the selected index back to the caller's ID outside the network, and reject over-limit inputs.

Begin with selection only. Add a candidate-conditioned scalar correctness head using detached predictions as supervised correctness targets. After that works, add the experimental 21-bin confidence policy so exact and sampled reward training can reuse the existing objective definitions. Document which heads train in each phase, the source of correctness labels and whether confidence gradients update the encoder. Use a frozen supervised reference for RL.

There is no vocabulary projection for answer aliases and no generation loop. The computation includes multiple encoder operations and candidate attention; calling it non-autoregressive does not make work independent of candidate count. Exact candidate permutation equivariance is a testable design goal in evaluation mode, not an assumed result in every stochastic training step.

## Validation requirements

| Component | Evidence to inspect |
|---|---|
| Minimal scorer | Tiny-fixture overfit, padding/order behavior, save/reload equality |
| Controlled data | Disjoint groups, exact labels, known uncertainty and matched baselines |
| Confidence | Scalar/temperature/constant controls, reward arithmetic and gradients, frozen reference |
| Language diagnostic | Preserved official test, disclosed exposure, reported failures |
| Deployment | Dedicated artifact type, shared response contract, Python/CLI/HTTP/Docker equality |

These checks distinguish an implementation that works from a model that is useful on natural language. Neither synthetic success nor a successful HTTP response supplies the missing natural-language quality evidence.

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

## What the prototype teaches

The implemented scorer has 201,175 randomly initialized parameters. Synthetic diagnostics establish controlled-rule learning, padding/order behavior and inference equivalence. Its one-seed BANKING77 diagnostic collapsed to one class at 1.30% accuracy. That quality failure remains part of the evidence.

No scratch artifact is recommended for production. A larger encoder, a trained tokenizer, broader language data or a different rule layout would be a new experiment, with its own frozen protocol. The current prototype does not establish how a 5-20M-parameter model would perform.
