# Experimental protocol

The committed configurations describe **100-update feasibility pilots**, not converged models or a completed scaling study. Freeze a main comparison's updates, exposure, learning rates, seeds, candidate descriptions and validation selection rule before examining its test outputs. The orchestration command defaults to 1,000 updates and seeds 11, 22, 33; this is an execution setting, not evidence that 1,000 steps is sufficient.

## Semantics and objectives

The v2 protocol uses the native Qwen chat framing with thinking disabled. A single decoder forward produces a final hidden state. Projection against verified alias output embeddings yields selection logits. A shared 128-dimensional interaction between hidden state and alias embedding produces a scalar correctness logit and 21 confidence-policy logits per candidate. Aliases are verified distinct single tokens at creation and reload. The initial alphabet/digit pool has only 62 usable aliases for this tokenizer; additional verified CJK symbols extend it to 160. They are identifiers, not translations. Their semantic associations are a limitation; order randomization reduces but does not eliminate alias bias.

The policy factorizes as `p(answer) p(confidence | answer)`. Confidence actions are 0, .05, …, 1. Exact training sums over all actions. Sampled training draws eight joint actions with replacement, using a leave-one-out baseline and a detached advantage. Both maximize expected `c - (q-c)^2` minus forward KL to the frozen supervised joint policy. A frozen backbone is shared by independent current/reference adapters and separate confidence heads; a test checks reference equivalence to an independently copied network. Correctness-only ablations use `c` with the same KL and architecture. At deployment, the answer is the argmax; reported policy confidence is its conditional expected q. This deterministic behavior must be evaluated separately from sampled reward.

The squared-error term has its maximum at the correctness probability when that probability is fixed. Answer optimization, discretization, shared parameters and KL change this setting. No calibration guarantee is claimed. For the supervised head, the selected answer's detached correctness trains a scalar BCE loss. The confidence policy is pretrained with expected squared error over every candidate. Supervised+Brier adds multiclass selection Brier with coefficient .1; continued-supervised includes the same scalar/policy head training as initial supervised training.

## Matched controls

Use identical initialization and seed order for SFT and SFT+Brier. Exact, sampled and continued-SFT all start from the **same saved SFT artifact** and receive equal additional examples and optimizer steps. Compare SFT+Brier both at initial exposure and, when drawing continued-method conclusions, at matched total exposure. Report SFT initialization cost as well as continuation cost. This implementation supports microbatch one with configurable gradient accumulation; it does not implement distributed training. Steps and examples appear in every log, with elapsed synchronized wall time and peak allocated VRAM. Wall time on a shared GPU is not isolated GPU compute time.

The untouched backbone uses the identical myJEV prompt, alias mapping and selection readout. Its max selection probability is an explicitly named proxy confidence. Scalar, policy, max-score, calibrated max-score, empirical base-rate and constant .5 confidence are distinct controls. Temperature fitting uses calibration labels only and cannot improve top-1 accuracy.

The 4B pilot uses BF16 LoRA; 9B uses NF4 QLoRA with BF16 computation and FP32 unquantized layers, consistently during training and reload. Before interpreting scale, repeat 4B with NF4 using the same exposure to separate precision effects from capacity. Three seeds are required for final comparisons. A single pilot does not meet this gate. Use the shared Hub cache and retain one adapter/head artifact and one resumable state per run. Do not delete selected checkpoints or their manifests.

## Evaluation

Select confidence thresholds on reserved calibration data. On test data, report the coverage actually obtained at those thresholds, not nominal calibration coverage. Ties can make them differ greatly. Error-target thresholds use empirical calibration error and are **not certified production error bounds**. Test reports include a one-sided binomial error bound and a group-bootstrap interval; use the group interval for correlated archive examples. Bins are fifteen equal-width intervals, with confidence 1 in the final bin. Empty acceptance sets have null error, never zero.

Accuracy, macro-F1, selection multiclass Brier, selected-correctness Brier, correctness AUROC, ECE and reliability bins are separate fields. For missing-answer cohorts, all selections are incorrect; selection multiclass Brier over an absent target is not a meaningful closed-set score and should be omitted in reports. Deferral is measured through coverage and accepted error; selecting an explicit none option is a separate classification task.

Archive resampling units are original post families, not excerpts. Paired method differences should use the same sampled groups and be aggregated across seeds. No perfect-reviewer cascade simulation is presented as a measured human workflow.

## Gates still needed for a research release

Full three-seed matched comparisons, same-size precision controls, untouched and GLiClass benchmarks, independent calibration, CLINC transfer, human-reviewed archive labels/adaptation/forgetting, all robustness cohorts, and deployment benchmarks at released precision. No default model is selected solely by parameter count. A 100-update pilot is insufficient to name a winning training method or model size.
