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

Give it a request and descriptions of the allowed choices. For “I was charged twice,” supply `billing`, `technical` and `other` with descriptions of those routes. The model returns a chosen ID, scores for the choices and confidence in its answer, without generating a reply. Banking support routing was its training task; trying new instructions and choices needs evaluation on those requests.

By [Amit Bahree](https://huggingface.co/bahree). [Source and study](https://github.com/bahree/myJEV) | [Training findings](https://github.com/bahree/myJEV/blob/main/docs/qwen-findings.md) | [Inference and hosting](https://github.com/bahree/myJEV/blob/main/docs/inference.md)

Start with [Python](#run-it) or the [Docker container](#run-with-docker). The [seven examples](https://github.com/bahree/myJEV/blob/main/results/demos-v1/report.md#what-each-request-asks) explain support routing, a refund cutoff, a misleading quoted instruction and post-format classification, with recorded outputs and mistakes. They are authored demonstrations, separate from the held-out evaluations below.

## Why choose this version?

Choose this version for the best measured starting balance of banking-intent quality, confidence error and local serving cost within the published myJEV family. It uses continued supervised training followed by temperature calibration. The 4B-RL variant has higher banking accuracy on two of three seeds and in the mean; this standard release has lower confidence Brier error and stronger explicit unsupported-option transfer in the study.

For fixed labels, a smaller classifier is also worth comparing. A separate one-seed ModernBERT-base control reached 90.78% BANKING77 accuracy and 0.0555 correctness Brier after temperature calibration. It saw 23,997 training examples over three epochs, versus 8,000 example presentations in these myJEV runs. The different budgets prevent a matched architecture comparison. Its output head fixes the 77 labels; myJEV accepts candidate descriptions with each request. That flexibility does not establish accuracy on an unfamiliar taxonomy. See [the decision guide](https://github.com/bahree/myJEV/blob/main/docs/decision-lessons.md) and [encoder control](https://github.com/bahree/myJEV/blob/main/results/encoder-control-v1/report.md).

myJEV-4B uses **Qwen/Qwen3.5-4B**, the pretrained network that reads the text. This repository supplies the small learned updates (LoRA adapters), output layers for confidence and calibration settings. The myJEV loader downloads the matching backbone as well. Serving needs the full network's memory and computation even though it returns scores in one pass.

Each size has a supervised release and an `-RL` release so we can compare two ways of continuing the same supervised starting model. Supervised training learns from labelled answers. Reinforcement learning (RL) rewards correct decisions and appropriate confidence. Both use the BANKING77 labels and receive the same additional training examples. The [training guide](https://github.com/bahree/myJEV/blob/main/docs/training.md) explains those controls; [Reinforcement Learning - An Introduction](https://blog.desigeek.com/post/2021/07/reinforcement-learning-an-introduction/) introduces the reward approach.

## Run it

The reference environment is Linux, Python 3.12 and an NVIDIA GPU. The pinned requirements include PyTorch, Transformers, PEFT and the four-bit runtime. A working NVIDIA driver and a host C compiler are needed for the tested GPU path. On Debian/Ubuntu, install `gcc` and `libc6-dev` if missing.

```bash
git clone https://github.com/bahree/myJEV.git
cd myJEV
git checkout a242d9cf4194a591f922e04abfc6075ee82c38cc
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
    device="auto",
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

The recorded GPU check selected `billing` with confidence approximately `0.9834` on this fixture. That demonstrates the call on one request. The immutable revision above contains the tested model files; later edits to this card leave those files unchanged. The default device setting selects CUDA when available. CPU behavior and the 9B restriction are described below.

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

Python, CLI and HTTP return the same response fields. Keep your artifact revision pinned in deployment configuration. These commands run locally; the separate [managed-hosting recipe](https://github.com/bahree/myJEV/blob/main/docs/hosting.md#hugging-face-inference-endpoints-recipe) has not been deployed or tested in the cloud.

## Run with Docker

The [published container](https://hub.docker.com/r/amitbahree/myjev) supplies Python and the model-loading code. It downloads this release and the pinned Qwen backbone from Hugging Face when they are absent from the mounted cache. Subsequent starts reuse those files. Running it locally requires no hosted Hugging Face endpoint.

This Bash example uses NVIDIA GPU access on Linux or Windows WSL 2. The [Docker guide](https://github.com/bahree/myJEV/blob/main/docs/docker.md) includes Windows PowerShell commands, macOS CPU instructions, memory notes and the tested platform limits.

```bash
docker run --rm --name myjev --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v myjev-hf-cache:/cache/huggingface \
  -e MYJEV_ARTIFACT=bahree/myJEV-4B \
  -e MYJEV_REVISION=a1b9e3b1181293220012cfb15587cdba0767ae8e \
  amitbahree/myjev@sha256:abcc30c2ee218950b00086e3bc06d4909ad1d4a5834a8478410f39e6da2a1ba3
```

Once the log says `Application startup complete`, open another terminal:

```bash
docker cp myjev:/app/examples/request.json ./request.json
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Use `curl.exe` in Windows PowerShell. Stop the service with `docker stop myjev`; the named volume retains the downloads.

For CPU use, replace the image with `amitbahree/myjev@sha256:cdee2c8ec6d6c953b1b18a9b0506ad79b589e5a60127d3a5df11dc8e7316cc07` and omit `--gpus all`. That CPU image includes x86-64 and ARM64 packages. ARM64 passed import/arithmetic checks under emulation, but full-model warmup exceeded 15 minutes; native Mac and Windows execution remain unverified. Start with the 0.8B model for a CPU trial; 4B needs more RAM and time. CPU scores can differ from the recorded GPU scores, and the GPU calibration study has not been repeated on CPU.

Device selection defaults to `auto`: use visible CUDA, otherwise CPU for supported releases, with a warning in the logs. An explicit `MYJEV_DEVICE=cuda:0` requires CUDA. Docker itself rejects `--gpus` on an unsupported host before the application starts, so CPU commands omit that flag. The research container runs as root; host authentication and remote HTTPS setup are covered in the [hosting guide](https://github.com/bahree/myJEV/blob/main/docs/hosting.md).



## What the scores mean

**Selection scores** are normalized values used to rank the supplied candidates. The selected ID is the highest-scoring choice.

**Reported confidence** is the selected option’s probability after temperature scaling on reserved BANKING77 calibration data. In this standard release it is derived from the selection scores, rather than the separate RL confidence policy. Calibration on banking intents does not establish calibration for arbitrary new tasks or candidate descriptions.

The API also returns `artifact_revision` and `calibration_revision` so callers can identify the exact behavior they used. Set acceptance/deferral thresholds using representative calibration data. The [threshold protocol](https://github.com/bahree/myJEV/blob/main/docs/protocol.md) explains why the searched empirical operating points are not certified risk guarantees. An explicit `other` candidate is a classification option; confidence-based deferral is a separate decision by your application.

## Results you can compare

The released checkpoint uses seed 11 by a fixed packaging convention. It was not selected for having the best test result. Accuracy and macro-F1 below use all 3,080 examples in BANKING77’s official test split. Brier measures squared error of reported correctness confidence, where lower is better.

| Evaluation | Accuracy | Macro-F1 | Correctness Brier |
|---|---:|---:|---:|
| This released checkpoint, seed 11 | 89.94% | 0.8979 | 0.0709 |
| Mean of seeds 11, 22 and 33 | 89.23% | 0.8904 | 0.0740 |

Accuracy’s sample standard deviation across those three seeds is 1.42 percentage points. The [paired analysis](https://github.com/bahree/myJEV/blob/main/results/longer-v1/paired-analysis.md) reports uncertainty for method contrasts. Three observed seeds do not establish performance across all future runs or user tasks.

For this seed, warm HTTP latency was **82.18 ms p50 / 87.33 ms p95**, using 100 requests, concurrency one and a short three-candidate request on one NVIDIA A30. The maximum allocated GPU memory observed for direct scoring across the three benchmark workloads was **8.18 GiB**. That excludes some driver/runtime allocations and is not a maximum-context memory guarantee. Startup to readiness was 15.39 seconds with cached weights, measured once. All three sizes were validated on 24 GB A30 hardware; precision here is **BF16 LoRA**. [Full serving conditions and evidence](https://github.com/bahree/myJEV/blob/main/docs/hosting.md#completed-candidate-validation-and-local-default).

## What the matched calibration follow-up changed

The [exploratory controls](https://github.com/bahree/myJEV/blob/main/results/review-calibration-v1/report.md) apply identical selection-temperature fitting to every method, and separately apply one binary log-odds temperature to each trained correctness estimate. All fits use calibration only. With selection temperature, exact RL has slightly lower mean 4B Brier (mixed seed directions) and lower 9B Brier on all three seeds; continued supervision leads at 0.8B. For the separate binary-temperature correctness estimate, the lower 4B exact-RL mean is driven by seed 33: exact-minus-continued Brier deltas are +0.0018 / +0.0004 / -0.0179. This qualifies the earlier released-configuration comparison. Brier and error ranking can move differently, so it does not automatically choose a new deferral policy.

The card tables still describe this released artifact and its unchanged confidence settings. None of the alternative fits was selected for deployment using test results. Three-seed bootstrap intervals hold those trained checkpoints fixed; seed spread is a separate uncertainty source. [Per-seed contrasts](https://github.com/bahree/myJEV/blob/main/results/review-seeds-v1/report.md) show that exact RL beats continued supervision at 4B in two of three seeds, while sampled RL trails exact in eight of nine size/seed pairs.

## How it was trained

This model received 4,000 initial supervised updates and another 4,000 supervised updates. Continuing supervised training is a control for the extra optimization used by the reinforcement-learning variants. A single temperature was then fitted on reserved calibration data to adjust the selected-option probability. Calibration examples were not used to fit the adapter, and the official test partition was not used to choose the temperature.

The matched runs use one example per optimizer update, so the initial and continuation stages each present 4,000 training examples. Training uses the grouped training portion of BANKING77, with separate validation and calibration partitions and the official test split preserved. Candidate order is randomized. The backbone remains frozen while low-rank adapter updates and custom heads learn the task. That reduces training storage; it does not turn the large pretrained backbone into a tiny inference model.

[BANKING77](https://github.com/bahree/myJEV/blob/main/docs/datasets/banking77.md), published by PolyAI, pairs customer banking messages with 77 intent labels. We chose it because closely related requests make routing errors possible to inspect while keeping the expected label defined. It covers one task family. [CLINC150](https://github.com/bahree/myJEV/blob/main/docs/datasets/clinc150.md), introduced by Larson and colleagues, adds 150 intents across ten domains and unsupported requests. We kept it outside training and tuning to test transfer.

These weights have no blog-archive adaptation. The archive study, model built from random weights and new decision-head experiments have their own data and results. The [project walkthrough](https://github.com/bahree/myJEV/blob/main/docs/walkthrough.md) connects those experiments without treating them as properties of this release.

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

The 9B models also change backbone precision to four-bit NF4, so differences cannot be attributed solely to parameter count. The study includes a separate 4B precision control.

## Tested scope and limits

The manifest accepts 2 to 160 candidates and up to 4,096 tokenizer tokens for the entire rendered prompt. Duplicate IDs and oversized requests are rejected instead of silently truncated. Short 160-candidate smoke checks passed for every release; that is not evidence of equally good accuracy or latency at 160 choices. Candidate wording, order, missing correct options and quoted instructions can change decisions. In the [frozen order study](https://github.com/bahree/myJEV/blob/main/results/review-order-v1/report.md), changed selected IDs ranged from 7.89-9.71% per permutation at 0.8B and 3.64-4.97% at larger sizes. Each request received its own seeded shuffle; aggregate accuracy can mask these changes. Check the transfer and robustness results before choosing this model for a new workflow.

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
