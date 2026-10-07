# myJEV: single-pass decision inference

Run the reference myJEV Python/HTTP service on an NVIDIA GPU. It scores a supplied set of candidates in one backbone forward pass, returning the selected ID, normalized selection scores and correctness confidence. It does not generate answer text. Selection scores and reported confidence have different meanings; the artifact records the confidence mode and calibration.

[Source and experiments](https://github.com/bahree/myJEV) · [Inference guide](https://github.com/bahree/myJEV/blob/main/docs/inference.md) · [Six model releases](https://github.com/bahree/myJEV/blob/main/docs/models.md)

## What is in the image

- Linux `amd64`, Python, pinned CUDA-enabled PyTorch dependencies and the custom myJEV loader/service.
- Adapter, confidence-head and calibration support for the released 0.8B, 4B and 9B artifacts.
- No model weights, training data, blog drafts or credentials. The first model load downloads the selected release and its separately pinned backbone/tokenizer.

Image tag `0.1.1` identifies the tested Hub-loader container revision. The installed Python package metadata remains `0.1.0`; immutable image and model digests identify the exact runtime. Docker on the validation host reports about 10.5 GB of image storage; the compressed Linux amd64 registry layers total 3.48 GB. These figures exclude downloaded weights. Keep a persistent cache and allow additional disk space for the selected backbone.

## Run the default 4B model

The tested host has an NVIDIA A30 with 24 GB VRAM, a compatible driver and NVIDIA Container Toolkit. This release is not an ARM/CPU deployment benchmark. LoRA reduces adapter storage and training state; inference still executes the full backbone.

```bash
docker pull amitbahree/myjev:0.1.1
mkdir -p .cache/huggingface
docker run --rm --name myjev --gpus device=0 \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/.cache/huggingface:/cache/huggingface" \
  -e MYJEV_ARTIFACT=bahree/myJEV-4B \
  -e MYJEV_REVISION=38f7cca5a8530483309f576b0c3dd1756bc27c33 \
  -e MYJEV_QUEUE_SIZE=8 -e MYJEV_TIMEOUT=30 \
  amitbahree/myjev:0.1.1
```

Use the immutable image reference in the [release receipt](https://github.com/bahree/myJEV/blob/main/results/container-registry-v1/publication.json) when reproducing results. The first startup includes downloads; later startups reuse the mounted cache. Set `HF_HUB_OFFLINE=1` only after all pinned files are cached. One process loads one model on one GPU. Select another released model by changing both artifact ID and its matching pinned revision.

In another terminal, wait for readiness and submit a request:

```bash
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' \
  --data '{"context":"I was charged twice.","instructions":"Select the appropriate support route.","candidates":[{"id":"billing","description":"Charges, invoices, and refunds"},{"id":"technical","description":"Errors and configuration"},{"id":"other","description":"Neither listed route applies"}]}'
```

The saved default response selects `billing` with confidence approximately `0.9834`, in `selection` mode. This is a demonstration, not a guarantee for other requests. [Full saved outputs](https://github.com/bahree/myJEV/blob/main/results/demos-v1/report.md) include failures as well as successes. Stop with `docker stop myjev` from another terminal.

## Service behavior and limits

`GET /healthz` reports liveness; `GET /readyz` succeeds after model loading and warmup. `POST /score` is the main route. `/health` and `/generate` are compatibility aliases; `/generate` still performs scoring, not token generation.

Released manifests cap requests at 160 candidates and 4,096 rendered input tokens, including candidate descriptions. Oversized inputs and duplicate IDs are rejected. Synthetic boundary checks establish input handling, not accuracy at those limits. A full queue returns 429, and a request deadline returns 504. A timed-out GPU operation keeps its capacity slot until computation finishes.

The supplied command binds to host loopback. For remote access, add an authenticated HTTPS reverse proxy with body/rate limits. Request-content and access logging are disabled in the supplied service. Logs still contain operational startup/error messages. The [hosting guide](https://github.com/bahree/myJEV/blob/main/docs/hosting.md) explains concurrency, memory, calibration and the unexecuted Hugging Face managed-container recipe. No hosted endpoint is included with this image.

## Reproducibility and licensing

The repository retains build provenance, push/pull records and GPU response-equivalence checks. Pull validation on the build host can reuse image layers; it is not an empty-cache download benchmark. Published latency measurements state their workload, cache state, hardware and precision. Containerization does not establish calibration on a new task.

Original source is MIT licensed. Third-party runtime packages retain their respective licenses. Downloaded backbones, adapters and datasets have separate terms recorded in their model/data cards. This is a research release, with known limits on transfer and unsupported-request handling.
