# Model and container releases

**Release tracker: placeholders, not download links.** The six final candidate checkpoints exist on the training machine, but no trained myJEV artifact or container image is publicly released yet. You can train a local artifact with the [quick start](quickstart.md).

| Planned artifact | Current state | Public model card / download |
|---|---|---|
| myJEV 0.8B | Final local candidate checks completed | Pending selection and release |
| myJEV 4B | Final local candidate checks completed | Pending selection and release |
| myJEV 9B | NF4 final local candidate checks completed | Pending selection and release |
| Reference GPU container | Built and tested locally | Registry image and digest pending |
| Default local starting artifact | 4B continued SFT, temperature confidence, seed 11 | Public upload pending |

## What a release will contain

Each model release should include a pinned backbone/tokenizer revision, LoRA adapters, confidence-head weights, prompt and alias semantics, precision settings, calibration parameters, and a checksummed manifest. Its model card should state training exposure, data provenance, evaluation scope, tested input limits, limitations, and license requirements.

Weights alone do not reproduce this interface: the custom confidence logic requires the shared myJEV loader. Merged weights will only be offered after save/reload and output-equivalence checks. Quantization or backend changes require fresh calibration evaluation.

## When this page changes

A release entry becomes usable only when its immutable revision, load command, and evaluation evidence are populated. Until then, there is no hosted demo, no published Hub identifier to copy, and no production recommendation. The 9B model is not automatically the default.

The [hosting guide](hosting.md) already describes local execution and the unexecuted managed-endpoint recipe. Publishing a model on the Hub will not itself create a running endpoint.

## Local candidate packaging checkpoint

Six seed-11 candidates are now packaged locally: continued supervision with calibration-only temperature scaling, and exact RL, at 0.8B, 4B and 9B. Seed 11 is a fixed packaging convention, not the best test seed. Each directory contains copied, checksummed adapters/heads/manifest plus a draft model card; no backbone, training text or optimizer state is included. The local starting recommendation is the 4B calibrated supervised candidate; nothing has been uploaded.

The [inventory](../results/release-candidates-v1/inventory.json) records source artifacts, evaluation hashes and file checksums. `scripts/prepare_release_candidates.py` reproduces this packaging from completed local runs. Final representative benchmark/equivalence checks passed for all six candidates. Namespace/visibility can be supplied when uploading is ready; it does not block these local checks.

## Why this local default

Choose `artifacts/release-candidates-v1/myjev-4b-continued_sft-seed11` for the first local deployment. Seed 11 was fixed for packaging, not selected by test performance. The three-seed quality study favors its confidence Brier and explicit unsupported-option transfer over 4B exact RL, while the final seed-11 short-request HTTP p50 is 82.18 ms. Keep 4B exact available for its higher BANKING accuracy. This recommendation is exploratory and should be revisited against your task and calibration data.

| Packaged candidate | Short HTTP p50 / p95 | Maximum direct-path allocated VRAM across three measured workloads |
|---|---:|---:|
| 0.8B continued SFT | 57.48 / 61.15 ms | 1.55 GiB |
| 0.8B exact RL | 60.88 / 61.91 ms | 1.55 GiB |
| 4B continued SFT | 82.18 / 87.33 ms | 8.18 GiB |
| 4B exact RL | 81.85 / 83.30 ms | 8.18 GiB |
| 9B continued SFT | 115.09 / 118.64 ms | 11.24 GiB |
| 9B exact RL | 117.55 / 123.42 ms | 11.24 GiB |

HTTP figures use 100 warm three-candidate requests at concurrency one. Direct-path memory covers three/32 candidates with short/longer repeated context, not every request up to 4,096 tokens. CUDA allocated VRAM excludes some driver/runtime allocations. All six passed Python/CLI/HTTP equality, single-pass, oversized-rejection and short 160-candidate checks; actual Docker responses also matched their saved fixture. These are local engineering tests, not model-quality guarantees or a public image release.
