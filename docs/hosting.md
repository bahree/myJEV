# Local and managed hosting

The same `DecisionModel.score()` implementation is used by evaluation, Python, CLI, HTTP and Docker. It loads immutable backbone/tokenizer revisions, an adapter, custom heads and checked artifact files. For a Hub release, call `DecisionModel.load("namespace/repository", revision="<40-character commit>")`; an unpinned Hub branch is rejected. There is no published myJEV Hub release yet.

## Local installation

Follow the root README's locked installation. Select one model at process startup. Default binding is `127.0.0.1:8000`, with one worker per GPU. `GET /healthz` is liveness; `GET /readyz` is readiness after actual model loading and a warmup score. `POST /score` accepts the documented request and returns selected ID, normalized selection scores, confidence mode/value and artifact/calibration revisions. `/health` and `/generate` are managed-container compatibility aliases.

Duplicate IDs, fewer than two candidates, more than the artifact's candidate limit, unexpected fields and overlong inputs are rejected. Token counting includes the entire rendered prompt. No silent truncation occurs. Pilot manifests enforce at most 160 candidates and 4,096 input tokens. The 0.8B, 4B and 9B artifacts passed synthetic 4,096-token requests with both 2 and 160 candidates and rejected 4,097 tokens (see `results/limits-*.json`). These synthetic boundary checks do not establish accuracy across document lengths. Limits are acceptance caps, not calibration guarantees.

`MYJEV_QUEUE_SIZE=8` bounds total running plus waiting jobs. A full queue returns 429 with Retry-After. `MYJEV_TIMEOUT=30` returns 504 on deadline. Timed-out GPU jobs retain capacity until they actually finish, because Python cancellation cannot stop an executing CUDA kernel. The model is serialized on one worker thread. Graceful shutdown stops readiness and drains existing work. Request-body logging is off; Uvicorn access logging is disabled in the supplied commands.

## Docker

```bash
docker build -t myjev:0.1.0 .
docker run --rm --gpus all myjev:0.1.0 python -c 'import torch; print(torch.cuda.is_available())'
export MYJEV_ARTIFACT_HOST="$PWD/artifacts/pilot-0.8b/artifact"
export MYJEV_CACHE_HOST="$PWD/.cache/huggingface"
docker compose -f deploy/compose.yaml up --build
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/score -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Use a tested image digest for releases. The CUDA libraries come from the pinned PyTorch dependencies; the host supplies the GPU driver and NVIDIA container runtime. Containers bind all interfaces internally, but Compose publishes only to host loopback. See raw `results/docker-*` logs for the exact validation status; building an image alone does not prove GPU access or serving equivalence.

The real 0.8B GPU container passed Python/CLI/HTTP fixture equivalence and invalid-request rejection (`results/docker-equivalence.json`). Its initial startup failed because the minimal base image lacked a C compiler required by a runtime kernel; the Dockerfile now installs `gcc` and `libc6-dev`. Both the failed startup and successful validation logs are retained. Existing HTTP concurrency measurements were collected during other training jobs, so they are not isolated deployment benchmarks.

For remote use, place an authenticated HTTPS reverse proxy in front of the loopback service. Terminate TLS there, enforce bearer authentication or an identity provider, limit body size and request rate, and keep request-content logs disabled. Do not expose the unauthenticated backend directly. One process per GPU avoids duplicating model memory unexpectedly.

Benchmark with `scripts/benchmark.py --output results/http-c1.json --concurrency 1`, then repeat with concurrency 2, 4 and 8. Report success rates alongside p50/p95; a fast overload response is not fast inference. Isolate the GPU for final latency measurements. Model-loading time and warm request latency are separate measures.

## Hugging Face Inference Endpoints recipe

**Recipe only; no paid endpoint was created.** [Official custom-container documentation](https://huggingface.co/docs/inference-endpoints/en/engines/custom_container) permits custom inference logic and mounts the selected model repository at `/repository`. [Configuration guidance](https://huggingface.co/docs/inference-endpoints/guides/configuration) describes the container port and health-route settings. Hosting adapter weights on the Hub is not an active endpoint.

1. Prepare an artifact-only Hub repository with adapter files, `heads.safetensors`, `manifest.json`, model card and provenance. Upload only after deciding the repository name and visibility. Pin the resulting commit. Verify a clean download produces identical outputs. No optimizer state or training text belongs in this release.
2. Push the tested Docker image to a registry the endpoint can access; record its immutable digest. Configure a custom container with that image, port 8000, health route `/readyz`, and environment `MYJEV_ARTIFACT=/repository`, `MYJEV_DEVICE=cuda:0`, `MYJEV_QUEUE_SIZE=8`, `MYJEV_TIMEOUT=30`. POST to `/score`; `/generate` supports platforms expecting that route.
3. The manifest currently refers to the separately pinned backbone on the Hub. The container must be allowed to download that backbone into its cache during cold start, with access credentials if the backbone requires them. An adapter-only repository is not fully self-contained. For offline deployment, prewarm the same revision in the image/cache and test with `HF_HUB_OFFLINE=1`; do not assume the endpoint automatically bundles referenced backbones.
4. Choose GPU RAM from **measured serving peaks** plus headroom at the tested maximum prompt length. Training peaks are not an endpoint sizing benchmark. Test QLoRA/NF4 compatibility on the endpoint GPU family. Do not infer 9B requirements from 0.8B. Start with one replica, one worker and authenticated access; avoid autoscaling until cold-start, timeout and concurrency behavior is measured.
5. Smoke test with an authenticated request (supply URL and token in your own environment):

```bash
curl -fsS "$ENDPOINT_URL/readyz" -H "Authorization: Bearer $HF_TOKEN"
curl -fsS "$ENDPOINT_URL/score" -H "Authorization: Bearer $HF_TOKEN" \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

6. Measure cold start, warm p50/p95, overload rates and behavior during scale-to-zero. Set the platform request deadline longer than the model timeout and include startup download time in readiness allowances. Scale-to-zero saves idle compute but adds cold starts. After the experiment, delete the endpoint in its dashboard and verify deletion; uploading or deleting Hub artifacts is a separate action.

Neither generic `vllm serve` nor a standard classification widget executes this custom confidence head. Any optimized backend, merged checkpoint or new precision requires save/reload equivalence and fresh calibration evaluation before release.

## Recorded per-size container check

The `myjev:study-checkpoint` image passed exact response equivalence at all three sizes. Each size served 40 requests at concurrency 1 and 40 at concurrency 4, all HTTP 200. These use the short three-candidate `examples/request.json`, a pre-cached backbone and a fresh container on GPU 0. Other GPUs were running study work; these are not isolated-host or maximum-context benchmarks. Source: `results/docker-size-validation/`.

| Size | Startup to ready (s) | Warm p50 / p95, concurrency 1 (ms) | Throughput, concurrency 1 (requests/s) | p95, concurrency 4 (ms) |
|---|---:|---:|---:|---:|
| 0.8B | 19.4 | 60.1 / 63.7 | 16.8 | 249.6 |
| 4B | 15.4 | 78.6 / 81.0 | 12.7 | 315.7 |
| 9B | 14.4 | 101.2 / 103.7 | 9.9 | 406.4 |

Image digest: `sha256:2f21b45a0df75eead5d0b935b7158211c67388075b76cea23b187e7a0259d0a3`. This image is local; it has not been pushed to a public registry. A failed benchmark-launcher attempt is retained separately; the successful run uses the correct virtualenv interpreter.

Startup is one observation per size in the order 0.8B → 4B → 9B. Shared filesystem/driver cache and prior GPU activity were not reset between sizes; the startup numbers are not a controlled model-size comparison.
