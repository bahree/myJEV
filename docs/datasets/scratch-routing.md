# Original synthetic routing fixtures

Purpose: learn and inspect direct candidate scoring, template sensitivity and known uncertainty with randomly initialized weights. These are diagnostic teaching data, not a benchmark of general language ability.

Provenance: generated locally by `src/myjev/scratch/data.py`, with deterministic Python RNG seeds and original templates. The generated fixtures are dedicated under CC0-1.0; model/code licensing is separate. No external text or model-generated labels are used.

Rows contain a color signal or two equally likely signals, three color options and an explicit `other` option. The hidden outcome is sampled from the declared distribution. If its color is absent, the correct decision is `other`. Evaluation retains the conditional distribution; model input includes only context, instructions and descriptions. Labels are exact under this artificial rule, while ambiguity is irreducible given the visible two-signal input.

Partitions contain 2,048 training, 256 validation, 256 calibration and 512 test rows. Scenarios have unique group IDs; candidate reorderings and other derived variants must retain the source group. Surface templates differ across partitions. Test ambiguity pairings differ from training/validation/calibration. Clean color rules and vocabulary are shared. Distinct IDs alone do not establish semantic novelty; the held-out transformations are intentionally limited and disclosed.

Version 1 uses one training layout and exposed severe layout sensitivity. Version 2 expands training layouts and nonce lengths after validation diagnostics. Test labels were not used to choose those changes. Nonces identify examples and contain no label signal by construction; version 1 and non-training partitions include split names, which introduce a visible distribution difference. This is not a natural dataset. No claim of broad transfer follows from fitting these templates.

The byte tokenizer requires no corpus fitting. Files receive SHA-256 hashes in the dataset manifest. A study freeze records those hashes and source revision before main evaluation. Rebuild with `python scripts/run_scratch.py data --version 2 --output <new-directory>` and compare the recorded hashes.

Excluded claims: production routing quality, robust free-form instruction following, natural uncertainty calibration, human-label agreement, adversarial safety, low deployment error rates and Jev reproduction. The initial toy task has an expected optimal accuracy of 87.5% because a quarter of its rows have two equally likely distinct outcomes. A finite sampled test may vary around that expectation. Report probability quality against both sampled outcomes and known conditional probabilities.
