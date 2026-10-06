# Model and container releases

**Release tracker: placeholders, not download links.** The pilot checkpoints exist on the training machine, but no trained myJEV artifact or container image is publicly released yet. You can train a local artifact with the [quick start](quickstart.md).

| Planned artifact | Current state | Public model card / download |
|---|---|---|
| myJEV 0.8B | Training and serving pilot measured | Pending selection and release |
| myJEV 4B | Training and serving pilot measured | Pending selection and release |
| myJEV 9B | NF4 training and serving pilot measured | Pending selection and release |
| Reference GPU container | Built and tested locally | Registry image and digest pending |
| Default recommended artifact | Not selected | Pending quality/resource comparison |

## What a release will contain

Each model release should include a pinned backbone/tokenizer revision, LoRA adapters, confidence-head weights, prompt and alias semantics, precision settings, calibration parameters, and a checksummed manifest. Its model card should state training exposure, data provenance, evaluation scope, tested input limits, limitations, and license requirements.

Weights alone do not reproduce this interface: the custom confidence logic requires the shared myJEV loader. Merged weights will only be offered after save/reload and output-equivalence checks. Quantization or backend changes require fresh calibration evaluation.

## When this page changes

A release entry becomes usable only when its immutable revision, load command, and evaluation evidence are populated. Until then, there is no hosted demo, no published Hub identifier to copy, and no production recommendation. The 9B model is not automatically the default.

The [hosting guide](hosting.md) already describes local execution and the unexecuted managed-endpoint recipe. Publishing a model on the Hub will not itself create a running endpoint.

## Local candidate packaging checkpoint

Six seed-11 candidates are now packaged locally: continued supervision with calibration-only temperature scaling, and exact RL, at 0.8B, 4B and 9B. Seed 11 is a fixed packaging convention, not the best test seed. Each directory contains copied, checksummed adapters/heads/manifest plus a draft model card; no backbone, training text or optimizer state is included. No default is selected and nothing has been uploaded.

The [inventory](../results/release-candidates-v1/inventory.json) records source artifacts, evaluation hashes and file checksums. `scripts/prepare_release_candidates.py` reproduces this packaging from completed local runs. Final representative benchmark/equivalence checks are frozen and queued until other GPU jobs finish. Namespace/visibility can be supplied when uploading is ready; it does not block these local checks.
