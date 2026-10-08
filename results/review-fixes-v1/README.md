# Verification receipt scope

`completion.json` is a historical receipt for the reviewed study snapshot at public commit `87e36f7c634be0f983a5d9cc3119fb06f4ae6a12`. Its `final-public-tests.log` belongs to that snapshot. The earlier `final-public-order-replay.json` records the worktree base before export; the replay included the exported files subsequently committed at `87e36f7`, and its recorded output hashes identify them.

Two entries under `evidence_sha256` are private blog-validation evidence: `results/hugo-validation-v3/render-checks.json` and `results/hugo-validation-v3/math-checks.json`. They are not public reproduction inputs and are intentionally absent from this repository. All other listed evidence is public. The original receipt is retained; later corrections and validation live in `results/followup-fixes-v1/`.

An empty captured stdout log does not itself establish success or failure. Inspect its associated structured receipt, exit status where recorded, actual response and source hashes.
