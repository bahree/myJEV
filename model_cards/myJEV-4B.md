---
license: mit
language:
- en
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
datasets:
- PolyAI/banking77
tags:
- myjev
- intent-classification
- single-pass
- decision-model
- confidence-estimation
- custom-inference
- lora
- temperature-scaling
model-index:
- name: myJEV-4B
  results:
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      type: PolyAI/banking77
      name: BANKING77 official test
      split: test
    metrics:
    - type: accuracy
      name: Accuracy (%) for released seed-11 checkpoint
      value: 89.9351
---

# myJEV-4B

The recommended starting point within the published myJEV family: choose a route and return confidence.

Give it a request and descriptions of the allowed choices. It returns the chosen ID, a score for each choice and an estimate of correctness, without generating an answer paragraph. The main training task is banking support routing. Possible workflows include choosing a support queue, selecting the next workflow branch and handing uncertain cases to a reviewer after setting a threshold on your own calibration data.

By [Amit Bahree](https://huggingface.co/bahree). [Source and study](https://github.com/bahree/myJEV) | [Training findings](https://github.com/bahree/myJEV/blob/main/docs/qwen-findings.md) | [Inference and hosting](https://github.com/bahree/myJEV/blob/main/docs/inference.md)

## Why choose this version?

Choose this version for the best measured starting balance of banking-intent quality, confidence error and local serving cost within the published myJEV family. It uses continued supervised training followed by temperature calibration. The 4B-RL variant scores slightly higher on banking accuracy; this standard release has lower confidence Brier error and stronger explicit unsupported-option transfer in the study.

**If your labels are fixed, compare a smaller classifier too.** A separate one-seed ModernBERT-base control reached 90.78% BANKING77 accuracy and 0.0555 correctness Brier after temperature calibration. It saw 23,997 training examples over three epochs, versus 8,000 example presentations in these myJEV runs, so this is a practical control with a different budget, not a matched architecture comparison. Its output head fixes the 77 labels; myJEV accepts candidate descriptions with each request. That flexibility does not establish accuracy on an unfamiliar taxonomy. See [the decision guide](https://github.com/bahree/myJEV/blob/main/docs/decision-lessons.md) and [encoder control](https://github.com/bahree/myJEV/blob/main/results/encoder-control-v1/report.md).

This is a **Qwen/Qwen3.5-4B backbone plus a small adapter, confidence heads and calibration settings**. The download here contains the learned additions; the loader also downloads the separately pinned backbone. A small adapter file does not eliminate the backbone’s memory or compute cost. Serving reads the input once and performs no autoregressive generation.

## Results you can compare

The released checkpoint uses seed 11 by a fixed packaging convention. It was not selected for having the best test result. Accuracy and macro-F1 below use all 3,080 examples in BANKING77’s official test split. Brier measures squared error of reported correctness confidence, where lower is better.

| Evaluation | Accuracy | Macro-F1 | Correctness Brier |
|---|---:|---:|---:|
| This released checkpoint, seed 11 | 89.94% | 0.8979 | 0.0709 |
| Mean of seeds 11, 22 and 33 | 89.23% | 0.8904 | 0.0740 |

Accuracy’s sample standard deviation across those three seeds is 1.42 percentage points. The [paired analysis](https://github.com/bahree/myJEV/blob/main/results/longer-v1/paired-analysis.md) reports uncertainty for method contrasts. Three observed seeds do not establish performance across all future runs or user tasks.

For this seed, warm HTTP latency was **82.18 ms p50 / 87.33 ms p95**, using 100 requests, concurrency one and a short three-candidate request on one NVIDIA A30. The maximum allocated GPU memory observed for direct scoring across the three benchmark workloads was **8.18 GiB**. That excludes some driver/runtime allocations and is not a maximum-context memory guarantee. Startup to readiness was 15.39 seconds with cached weights, measured once. All three sizes were validated on 24 GB A30 hardware; precision here is **BF16 LoRA**. [Full serving conditions and evidence](https://github.com/bahree/myJEV/blob/main/docs/hosting.md#completed-candidate-validation-and-local-default).

## Run it

The reference environment is Linux, Python 3.12 and an NVIDIA GPU. The pinned requirements include PyTorch, Transformers, PEFT and the four-bit runtime. A working NVIDIA driver and a host C compiler are needed for the tested GPU path. On Debian/Ubuntu, install `gcc` and `libc6-dev` if missing.

```bash
git clone https://github.com/bahree/myJEV.git
cd myJEV
git checkout ac51018d8e679f0cc444e559bb7a5c12709ae8c2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps .
```

Use the custom `DecisionModel` loader. Loading only the adapter, calling ordinary `generate()`, or using a generic classification widget does not execute this model’s complete decision interface.

```python
from myjev import DecisionModel

model = DecisionModel.load(
    "bahree/myJEV-4B",
    revision="a1b9e3b1181293220012cfb15587cdba0767ae8e",
    device="cuda:0",
)
request = {
    "context": "I was charged twice.",
    "instructions": "Select the appropriate support route.",
    "candidates": [
        {
            "id": "billing",
            "description": "Charges, invoices, and refunds"
        },
        {
            "id": "technical",
            "description": "Errors and configuration"
        },
        {
            "id": "other",
            "description": "Neither listed route applies"
        }
    ]
}
result = model.score(request)
print(result["selected_id"], result["confidence"])
```

The recorded pinned-runtime check selected `billing` with confidence approximately `0.9834` on that exact fixture. It demonstrates a working call, not a guarantee on a new support taxonomy. The immutable revision above contains the tested runtime files; subsequent card-only edits leave those files unchanged.

The same request is checked into the source repository as `examples/request.json`:

```bash
myjev score --artifact bahree/myJEV-4B \
  --revision a1b9e3b1181293220012cfb15587cdba0767ae8e \
  --input examples/request.json

# Run the local HTTP service in a separate terminal.
myjev serve --artifact bahree/myJEV-4B \
  --revision a1b9e3b1181293220012cfb15587cdba0767ae8e

# Once /readyz succeeds:
curl -fsS http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Python, CLI and HTTP return the same response fields. Keep your artifact revision pinned in deployment configuration. Publishing this package does not start a hosted endpoint. The source repository includes tested local Docker instructions and a separate, unexecuted managed-hosting recipe.

## What the scores mean

**Selection scores** are normalized values used to rank the supplied candidates. The selected ID is the highest-scoring choice.

**Reported confidence** is the selected option’s probability after temperature scaling on reserved BANKING77 calibration data. In this standard release it is derived from the selection scores, rather than the separate RL confidence policy. Calibration on banking intents does not establish calibration for arbitrary new tasks or candidate descriptions.

The API also returns `artifact_revision` and `calibration_revision` so callers can identify the exact behavior they used. Set acceptance/deferral thresholds using representative calibration data. An explicit `other` candidate is a classification option; confidence-based deferral is a separate decision by your application.

## What the matched calibration follow-up changed

The [exploratory controls](https://github.com/bahree/myJEV/blob/main/results/review-calibration-v1/report.md) apply identical selection-temperature fitting to every method, and separately apply one binary log-odds temperature to each trained correctness estimate. All fits use calibration only. With selection temperature, exact RL has slightly lower mean 4B Brier (mixed seed directions) and lower 9B Brier on all three seeds; continued supervision leads at 0.8B. This qualifies the earlier released-configuration comparison. Brier and error ranking can move differently, so it does not automatically choose a new deferral policy.

The card tables still describe this released artifact and its unchanged confidence settings. None of the alternative fits was selected for deployment using test results. Three-seed bootstrap intervals hold those trained checkpoints fixed; seed spread is a separate uncertainty source. [Per-seed contrasts](https://github.com/bahree/myJEV/blob/main/results/review-seeds-v1/report.md) show that exact RL beats continued supervision at 4B in two of three seeds, while sampled RL trails exact in eight of nine size/seed pairs.

## How it was trained

This model received 4,000 initial supervised updates and another 4,000 supervised updates. Continuing supervised training is a control for the extra optimization used by the reinforcement-learning variants. A single temperature was then fitted on reserved calibration data to adjust the selected-option probability. Calibration examples were not used to fit the adapter, and the official test partition was not used to choose the temperature.

The matched runs use one example per optimizer update, so the initial and continuation stages each present 4,000 training examples. Training uses the grouped training portion of BANKING77, with separate validation and calibration partitions and the official test split preserved. Candidate order is randomized. The backbone remains frozen while low-rank adapter updates and custom heads learn the task. That reduces training storage; it does not turn the large pretrained backbone into a tiny inference model.

BANKING77 supplies 77 closely related banking intents. It is one task family, not a generalist instruction mixture. CLINC150 was kept outside training and tuning for transfer and unsupported-request evaluation. These released weights have no archive adaptation. The separate blog-archive and scratch-model experiments are documented in the project rather than folded into these results.

## The model family

Quality columns are three-seed BANKING77 means. Runtime columns are measured on the released seed-11 artifacts under the same short-request conditions described above.

| Model | Accuracy | Confidence Brier | HTTP p50 | Allocated VRAM |
|---|---:|---:|---:|---:|
| [myJEV-0.8B](https://huggingface.co/bahree/myJEV-0.8B) | 82.93% | 0.1073 | 57.48 ms | 1.55 GiB |
| [myJEV-0.8B-RL](https://huggingface.co/bahree/myJEV-0.8B-RL) | 81.36% | 0.1321 | 60.88 ms | 1.55 GiB |
| [myJEV-4B](https://huggingface.co/bahree/myJEV-4B) | 89.23% | 0.0740 | 82.18 ms | 8.18 GiB |
| [myJEV-4B-RL](https://huggingface.co/bahree/myJEV-4B-RL) | 90.27% | 0.0818 | 81.85 ms | 8.18 GiB |
| [myJEV-9B](https://huggingface.co/bahree/myJEV-9B) | 89.15% | 0.0742 | 115.09 ms | 11.24 GiB |
| [myJEV-9B-RL](https://huggingface.co/bahree/myJEV-9B-RL) | 89.34% | 0.0942 | 117.55 ms | 11.24 GiB |

The standard releases use continued supervised training and temperature scaling. `-RL` releases use exact confidence-aware reward training. The 9B models also change backbone precision to four-bit NF4, so differences cannot be attributed solely to parameter count. The study includes a separate 4B precision control.

## Tested scope and limits

The manifest accepts 2 to 160 candidates and up to 4,096 tokenizer tokens for the entire rendered prompt. Duplicate IDs and oversized requests are rejected instead of silently truncated. Short 160-candidate smoke checks passed for every release; that is not evidence of equally good accuracy or latency at 160 choices. Candidate wording, order, missing correct options and quoted instructions can change decisions. Check the transfer and robustness results before choosing this model for a new workflow.

The reference backend has been checked for artifact reloads and Python/CLI/HTTP/Docker consistency, including pinned Hub download checks. New quantization, merged weights or optimized backends need their own equivalence and calibration measurements. No matched speed comparison with the proprietary Jev service has been performed.

## Files, licenses and provenance

The package includes `adapter/`, `heads.safetensors`, `manifest.json`, `LICENSE`, `BACKBONE_LICENSE` and `NOTICE.md`. It does not include backbone weights, training text or optimizer state.

- Original myJEV code, adapters and heads: MIT, copyright Amit Bahree.
- Qwen backbone: Apache-2.0, with its original terms retained in `BACKBONE_LICENSE`.
- [BANKING77](https://huggingface.co/datasets/PolyAI/banking77): CC BY 4.0; Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020).
- Backbone revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Artifact revision: `3a7979eeff5cfb0bd2cee2fe2ea559c314ad9f5eef2de4ef4cb44759dbd73f02`.
- Calibration revision: `f0d2ddcaf6650e1f44415d49b1cf8b303d4d6fef5f73bb6198ee91600560e8d1`.

[Source repository](https://github.com/bahree/myJEV) | [Dataset rationale](https://github.com/bahree/myJEV/blob/main/docs/datasets/banking77.md) | [All releases and deployment notes](https://github.com/bahree/myJEV/blob/main/docs/models.md)
