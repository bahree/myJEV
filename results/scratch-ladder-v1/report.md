# Scratch diagnostic ladder

Diagnostic, not an optimization comparison. Tiny stage evaluates its training examples intentionally. Deterministic stage uses new record IDs but same template and colors, so it tests elementary rule recovery, not unfamiliar-language transfer. No uncertainty/noise, missing options, or test-selected hyperparameters. All three seeds reported; no additional tuning on failure.

| Stage | Seed | N | Accuracy | Predeclared criterion | Pass |
|---|---:|---:|---:|---:|---|
| tiny-overfit | 11 | 8 | 100.00% | 100% | True |
| tiny-overfit | 22 | 8 | 100.00% | 100% | True |
| tiny-overfit | 33 | 8 | 100.00% | 100% | True |
| deterministic-rule | 11 | 64 | 75.00% | 95% | False |
| deterministic-rule | 22 | 64 | 100.00% | 95% | True |
| deterministic-rule | 33 | 64 | 68.75% | 95% | False |

Budget: six independent random initializations, 300 updates each, batch eight, fixed learning rate 0.001. No pretrained weights. Passing a rung demonstrates learnability in that narrow setting; failing under this budget does not establish impossibility. The original noisy/template-transfer and BANKING studies are unchanged.

Reproduce in a fresh checkout without this local output: `python scripts/run_scratch_ladder.py --device cuda:0`. The source script refuses to overwrite a frozen run. Training logs and all fixture predictions are retained.

## Subsequent design review

The deterministic rung confounds cyclic numeric record IDs with color labels. Its successful seed cannot establish signal-rule recovery. Tiny-set evaluation also contains four unique contexts repeated twice. All original results remain preserved; see `design-review.json` and the separately frozen corrected `results/scratch-ladder-v2/report.md`. This is a fixture correction, not evidence of which mechanism any model learned.
