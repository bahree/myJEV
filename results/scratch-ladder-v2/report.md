# Scratch deterministic-rule rung, corrected nuisance design

Post-v1 structural correction, not an independent untouched scientific test. Every random nuisance nonce appears with each of four colors; train/test nonces disjoint. Only color cue distinguishes labels within nonce. Same-template deterministic recovery; no general-language/noisy-task claim. All results retained, no retuning.

V1 assigned colors cyclically by numeric case index, so its success could reflect a shortcut. That confounded run is preserved. V2 pairs each random nonce with all four colors, making nonce-only accuracy exactly 25%; evaluation nonces are unseen. This correction was frozen before V2 inference and did not retune any training setting.

| Seed | Test decisions / nonce groups | Accuracy | Predeclared criterion | Pass |
|---|---:|---:|---:|---|
| 11 | 64 / 16 | 100.00% | 95% | True |
| 22 | 64 / 16 | 100.00% | 95% | True |
| 33 | 64 / 16 | 100.00% | 95% | True |

Additional bounded budget: 900 updates total (three runs of 300). Including v1, the entire diagnostic extension uses 2,700 updates. The original 13,000-update scratch study and its failures are unchanged. Passing this rung is a basic rule-recovery sanity check, not proof of transfer to noisy or natural-language tasks.

Reproduce in a fresh output checkout with `python scripts/run_scratch_ladder_v2.py --device cuda:0`. All generated fixtures, hashes, loss logs and predictions are retained.
