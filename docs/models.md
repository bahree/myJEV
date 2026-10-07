# Model and container releases

Six seed-11 adapter/head releases are public on Hugging Face. **Start with [myJEV-4B](https://huggingface.co/bahree/myJEV-4B)**, continued supervised training with temperature calibration. The `-RL` repositories retain the exact expected-reward comparison. None is a hosted inference endpoint.

| Release | Training and confidence | Immutable revision |
|---|---|---|
| [bahree/myJEV-0.8B](https://huggingface.co/bahree/myJEV-0.8B) | Continued SFT; temperature calibration | [`1c956c89d21c0ab136e98ffe66a16752fa37d823`](https://huggingface.co/bahree/myJEV-0.8B/tree/1c956c89d21c0ab136e98ffe66a16752fa37d823) |
| [bahree/myJEV-0.8B-RL](https://huggingface.co/bahree/myJEV-0.8B-RL) | Exact RL; expected confidence grid | [`44b2ab8e78cb176ed01a63c6fee2146b2d91cc95`](https://huggingface.co/bahree/myJEV-0.8B-RL/tree/44b2ab8e78cb176ed01a63c6fee2146b2d91cc95) |
| [bahree/myJEV-4B](https://huggingface.co/bahree/myJEV-4B) | Continued SFT; temperature calibration | [`38f7cca5a8530483309f576b0c3dd1756bc27c33`](https://huggingface.co/bahree/myJEV-4B/tree/38f7cca5a8530483309f576b0c3dd1756bc27c33) |
| [bahree/myJEV-4B-RL](https://huggingface.co/bahree/myJEV-4B-RL) | Exact RL; expected confidence grid | [`0a9413105fc84cb500850059b641755890c4201b`](https://huggingface.co/bahree/myJEV-4B-RL/tree/0a9413105fc84cb500850059b641755890c4201b) |
| [bahree/myJEV-9B](https://huggingface.co/bahree/myJEV-9B) | Continued SFT; temperature calibration | [`31de42741a9388d07c32465ae8704bafe272b3ae`](https://huggingface.co/bahree/myJEV-9B/tree/31de42741a9388d07c32465ae8704bafe272b3ae) |
| [bahree/myJEV-9B-RL](https://huggingface.co/bahree/myJEV-9B-RL) | Exact RL; expected confidence grid | [`08dc92307b02fad8c333d41b6f4e1ccfacf16c39`](https://huggingface.co/bahree/myJEV-9B-RL/tree/08dc92307b02fad8c333d41b6f4e1ccfacf16c39) |

## What each release contains

These are **adapter/head packages**, not merged or self-contained backbone weights. They include LoRA adapters, custom confidence heads, calibration settings, prompt/token-alias semantics, precision settings, license notices and a checksummed manifest. The manifest pins the separately downloaded Qwen backbone and tokenizer. The 0.8B/4B releases use BF16 LoRA; 9B uses NF4 QLoRA.

Use the shared **myJEV custom loader**. Ordinary text generation, a generic model widget or loading only the LoRA adapter does not reproduce the selection/confidence interface. Merged weights are not offered. Backend or quantization changes need new equivalence and calibration checks. Original adapter/head contributions carry MIT terms; the Qwen backbone's Apache-2.0 terms remain separate.

```python
from myjev import DecisionModel

model = DecisionModel.load(
    "bahree/myJEV-4B",
    revision="38f7cca5a8530483309f576b0c3dd1756bc27c33",
)
```

See the [inference guide](inference.md) for complete requests, CLI, HTTP and Docker commands. All six immutable downloads were checked against the uploaded package hashes. [Upload receipts](../results/release-readiness-v1/hub/) distinguish publication evidence from the earlier local candidate checks. The tested GPU container is published on [Docker Hub](https://hub.docker.com/r/amitbahree/myjev); the [publication receipt](../results/container-registry-v1/publication.json) records its digest and GPU equality check. No paid managed endpoint is running. [Hosting](hosting.md) includes the unexecuted managed-endpoint recipe.

## Packaging provenance

Seed 11 is the fixed packaging convention, not the best test seed. The releases contain no training text, optimizer state or backbone weights. The earlier [candidate inventory](../results/release-candidates-v1/inventory.json) preserves the frozen experiment artifacts. The final publication package separately adds license and provenance files. Six local candidates passed representative benchmark and output-equivalence checks before upload.

## Why this default within the myJEV family

Choose the pinned `bahree/myJEV-4B` release for the first local deployment. Seed 11 was fixed for packaging, not selected by test performance. The three-seed quality study favors its confidence Brier and explicit unsupported-option transfer over 4B exact RL, while the final seed-11 short-request HTTP p50 is 82.18 ms. Keep 4B exact available for its higher BANKING accuracy. This recommendation is exploratory and should be revisited against your task and calibration data.

| Packaged candidate | Short HTTP p50 / p95 | Maximum direct-path allocated VRAM across three measured workloads |
|---|---:|---:|
| 0.8B continued SFT | 57.48 / 61.15 ms | 1.55 GiB |
| 0.8B exact RL | 60.88 / 61.91 ms | 1.55 GiB |
| 4B continued SFT | 82.18 / 87.33 ms | 8.18 GiB |
| 4B exact RL | 81.85 / 83.30 ms | 8.18 GiB |
| 9B continued SFT | 115.09 / 118.64 ms | 11.24 GiB |
| 9B exact RL | 117.55 / 123.42 ms | 11.24 GiB |

HTTP figures use 100 warm three-candidate requests at concurrency one. Direct-path memory covers three/32 candidates with short/longer repeated context, not every request up to 4,096 tokens. CUDA allocated VRAM excludes some driver/runtime allocations. All six passed Python/CLI/HTTP equality, single-pass, oversized-rejection and short 160-candidate checks; actual Docker responses also matched their saved fixture. These are local engineering tests, not model-quality guarantees or a public image release.

## Read and reproduce the release evidence

The [reader model cards](../model_cards/README.md) explain each variant, installation and per-artifact versus three-seed results. The [card publication manifest](../results/model-card-refresh-v2/publication-manifest.json) records the revisions above; card improvements did not change runtime weights.

This recommendation is limited to the published candidate-description interface. The separate [ModernBERT fixed-taxonomy control](../results/encoder-control-v1/report.md) reached 90.78% BANKING77 accuracy with a smaller encoder, but used one seed and 23,997 training examples versus Qwen's 8,000. It is a useful operational alternative when the 77 labels are fixed, not a matched architecture experiment. See [decision lessons](decision-lessons.md) before choosing a larger model.

The rebuilt local image `myjev:0.1.1-hub` passed a pinned Hub load and exact host/container response comparison. Its [provenance](../results/release-container-v2/provenance.json) and [HTTP evidence](../results/release-container-v2/hub-http.json) document the source-only update over the clean-tested dependency image. This exact image is now published as `amitbahree/myjev:0.1.1`; see the [publication receipt](../results/container-registry-v1/publication.json) for its immutable reference and pull verification.
