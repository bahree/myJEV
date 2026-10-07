# Model and container releases

Six seed-11 adapter/head releases are public on Hugging Face. **Start with [myJEV-4B](https://huggingface.co/bahree/myJEV-4B)**, continued supervised training with temperature calibration. The `-RL` repositories retain the exact expected-reward comparison. None is a hosted inference endpoint.

| Release | Training and confidence | Immutable revision |
|---|---|---|
| [bahree/myJEV-0.8B](https://huggingface.co/bahree/myJEV-0.8B) | Continued SFT; temperature calibration | [`a56043bc190b4ab66a704dea771388b28edcb53d`](https://huggingface.co/bahree/myJEV-0.8B/tree/a56043bc190b4ab66a704dea771388b28edcb53d) |
| [bahree/myJEV-0.8B-RL](https://huggingface.co/bahree/myJEV-0.8B-RL) | Exact RL; expected confidence grid | [`535d20d095dcc3577a9c155703bdf4d417134573`](https://huggingface.co/bahree/myJEV-0.8B-RL/tree/535d20d095dcc3577a9c155703bdf4d417134573) |
| [bahree/myJEV-4B](https://huggingface.co/bahree/myJEV-4B) | Continued SFT; temperature calibration | [`ca23134ff8d223927d545a30c594e27d68db4400`](https://huggingface.co/bahree/myJEV-4B/tree/ca23134ff8d223927d545a30c594e27d68db4400) |
| [bahree/myJEV-4B-RL](https://huggingface.co/bahree/myJEV-4B-RL) | Exact RL; expected confidence grid | [`40d140f2b29c15b8ba252551aba745e9d546f5dc`](https://huggingface.co/bahree/myJEV-4B-RL/tree/40d140f2b29c15b8ba252551aba745e9d546f5dc) |
| [bahree/myJEV-9B](https://huggingface.co/bahree/myJEV-9B) | Continued SFT; temperature calibration | [`bd9e54ff1222732209963aeb003c92c54c4260cb`](https://huggingface.co/bahree/myJEV-9B/tree/bd9e54ff1222732209963aeb003c92c54c4260cb) |
| [bahree/myJEV-9B-RL](https://huggingface.co/bahree/myJEV-9B-RL) | Exact RL; expected confidence grid | [`50cf88fc60d9cb755c1aa62530c6fde03cf47335`](https://huggingface.co/bahree/myJEV-9B-RL/tree/50cf88fc60d9cb755c1aa62530c6fde03cf47335) |

## What each release contains

These are **adapter/head packages**, not merged or self-contained backbone weights. They include LoRA adapters, custom confidence heads, calibration settings, prompt/token-alias semantics, precision settings, license notices and a checksummed manifest. The manifest pins the separately downloaded Qwen backbone and tokenizer. The 0.8B/4B releases use BF16 LoRA; 9B uses NF4 QLoRA.

Use the shared **myJEV custom loader**. Ordinary text generation, a generic model widget or loading only the LoRA adapter does not reproduce the selection/confidence interface. Merged weights are not offered. Backend or quantization changes need new equivalence and calibration checks. Original adapter/head contributions carry MIT terms; the Qwen backbone's Apache-2.0 terms remain separate.

```python
from myjev import DecisionModel

model = DecisionModel.load(
    "bahree/myJEV-4B",
    revision="ca23134ff8d223927d545a30c594e27d68db4400",
)
```

See the [inference guide](inference.md) for complete requests, CLI, HTTP and Docker commands. All six immutable downloads were checked against the uploaded package hashes. [Upload receipts](../results/release-readiness-v1/hub/) distinguish publication evidence from the earlier local candidate checks. The tested GPU container is built locally; no public registry image or paid managed endpoint is claimed. [Hosting](hosting.md) includes the unexecuted managed-endpoint recipe.

## Packaging provenance

Seed 11 is the fixed packaging convention, not the best test seed. The releases contain no training text, optimizer state or backbone weights. The earlier [candidate inventory](../results/release-candidates-v1/inventory.json) preserves the frozen experiment artifacts. The final publication package separately adds license and provenance files. Six local candidates passed representative benchmark and output-equivalence checks before upload.

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
