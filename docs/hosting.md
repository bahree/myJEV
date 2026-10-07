# Local and managed hosting

The same `DecisionModel.score()` implementation is used by evaluation, Python, CLI, HTTP and Docker. It loads immutable backbone/tokenizer revisions, an adapter, custom heads and checked artifact files. For a Hub release, call `DecisionModel.load("namespace/repository", revision="<40-character commit>")`; an unpinned Hub branch is rejected. The public default is [myJEV-4B](https://huggingface.co/bahree/myJEV-4B), pinned at `a1b9e3b1181293220012cfb15587cdba0767ae8e`; see the [release list](https://github.com/bahree/myJEV/blob/main/docs/models.md).

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

<a id="planned-scoring-versus-generation-benchmark"></a>

## Scoring-versus-generation benchmark

Status: completed for six seed-11 release candidates in `results/release-validation-v1/`. All preceding GPU studies had terminated and no GPU compute processes were present at the start. The image and backbone caches were warm. This addition did not change training configurations.

Compare four paths on the same pinned checkpoint, tokenizer, precision, candidate aliases and decision cases:

1. Direct single-pass candidate scoring.
2. Constrained generation of exactly one candidate token.
3. Minimal structured JSON containing the selected candidate ID.
4. Answer plus short explanation, explicitly measured as additional output work.

Use the same rendered input where possible; retain and disclose any prompt/template differences needed by each output format. Record actual input/output token counts, decoding constraints, output caps, backend versions, hardware, cache state and warmup. Separate the readout-only comparison from myJEV's complete response, which also computes correctness confidence. Generation paths returning only an answer do not provide equivalent confidence outputs.

Measure each path independently before mixed traffic. Repeat across representative lengths, candidate counts and concurrency levels, reporting accuracy, parsing/constraint failures, warm p50/p95, throughput, peak VRAM and cold starts. Save raw per-request timing and correctness records plus reproducible commands. Distinguish kernel/model time from end-to-end HTTP latency. If SGLang is used, first verify candidate-logit equivalence; its ordinary scoring endpoint does not implement myJEV's custom confidence heads. Backend or precision changes also require calibration checks.

Related work: Avi Chawla, [Build your own Jev (100% local)](https://blog.dailydoseofds.com/p/build-your-own-jev-100-local), September 22, 2026. The tutorial demonstrates SGLang scoring and distinguishes inference mechanics from training and calibration. Its illustrated generation comparison requests an explanation with up to 32 output tokens while scoring and generation share a server. Our one-token and minimal-JSON controls help separate output-work differences from serving overhead. This is related work, not independent validation of myJEV, and its timings are not our results.

## Hugging Face Inference Endpoints recipe

**Recipe only; no paid endpoint was created.** [Official custom-container documentation](https://huggingface.co/docs/inference-endpoints/en/engines/custom_container) permits custom inference logic and mounts the selected model repository at `/repository`. [Configuration guidance](https://huggingface.co/docs/inference-endpoints/guides/configuration) describes the container port and health-route settings. Hosting adapter weights on the Hub is not an active endpoint.

1. Prepare an artifact-only Hub repository with adapter files, `heads.safetensors`, `manifest.json`, model card and provenance. The six public `bahree/myJEV-*` releases provide this layout. Pin the selected commit. Verify a clean download produces identical outputs. No optimizer state or training text belongs in this release.
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

## Completed candidate validation and local default

All six seed-11 candidates passed Python/CLI/HTTP equality, one backbone call, a short 160-candidate smoke check, oversized-input rejection and exact Docker/Python response equality. Each container completed 100 HTTP requests at concurrency 1 and 4 for each of three workloads. The local default recommendation is **4B continued supervised training with temperature calibration**, preserving seed 11 by the packaging convention. The 4B exact-reward artifact remains the BANKING77 accuracy-oriented alternative. See [model selection](https://github.com/bahree/myJEV/blob/main/docs/models.md) for the quality trade-off; the default is a local research choice, not a published or production-validated service.

| Candidate | Short HTTP p50 / p95 (ms) | Startup to ready (s) | Largest allocated VRAM across measured scoring/generation workloads (GiB) |
|---|---:|---:|---:|
| 0.8B continued SFT + temperature | 57.48 / 61.15 | 20.49 | 1.58 |
| 0.8B exact | 60.88 / 61.91 | 13.42 | 1.58 |
| 4B continued SFT + temperature | 82.18 / 87.33 | 15.39 | 8.26 |
| 4B exact | 81.85 / 83.30 | 15.40 | 8.26 |
| 9B continued SFT + temperature | 115.09 / 118.64 | 19.40 | 11.35 |
| 9B exact | 117.55 / 123.42 | 18.42 | 11.35 |

HTTP figures use 100 warm concurrency-1 requests with three candidates on an A30. Startup is one observation with cached weights, not cold download time. Allocated VRAM excludes allocator reserve and driver/runtime allocations, and these workloads do not reach the 4,096-token acceptance cap. A 24 GB GPU is the validated local class; smaller managed GPU recommendations require new maximum-context measurements and runtime headroom. These figures supersede the earlier concurrent-study timings above without erasing that evidence.

The four output paths executed, but runtime completion does not establish usable generation. In the short workload, direct and constrained one-token paths produced valid outputs on all 20 repeats for every candidate. JSON was invalid on all 20 repeats for both 0.8B candidates and 4B exact. The 0.8B exact explanation path also failed format checks on every repeat. The repeated request is a timing/format diagnostic, not an accuracy benchmark. Only direct scoring returns our trained correctness confidence. Do not present a fast invalid JSON response as an equivalent inference result.

Regenerate the compact machine-readable summary with `python scripts/summarize_release_validation.py`. Raw requests, outputs, timing samples, container logs and source/image identities remain in `results/release-validation-v1/`; the summary is `results/release-readiness-v1/serving-summary.json`.

## Publish the prepared packages

The adapter-only `artifacts/hub-ready-v3/` snapshot adds MIT terms for the original adapter/head contributions, the pinned Qwen Apache-2.0 license and attribution, and BANKING77 provenance. Frozen study candidates are preserved separately. `results/release-readiness-v1/publication-manifest-v3.json` records every upload file's checksum. The six Qwen adapter/head releases use public repositories under `bahree`; an image registry is a separate decision.

For a new release to your own empty repository, review a package locally (the published `bahree` repositories are already populated):

```bash
.venv/bin/python scripts/publish_hub_artifact.py \
  --manifest results/release-readiness-v1/publication-manifest-v3.json \
  --candidate myjev-4b-continued_sft-seed11 \
  --repo-id YOUR_NAMESPACE/myjev-4b \
  --visibility private
```

This command checks local files only. Add `--apply` to create/upload the selected repository using your configured Hugging Face credentials. The helper refuses altered packages, symlinks, nonempty destinations and visibility mismatches. It downloads the immutable uploaded commit and checks every file hash. Then load that pinned commit using `DecisionModel.load("YOUR_NAMESPACE/myjev-4b", revision="COMMIT")` and compare with the local response. The six `bahree` releases have separate upload receipts; a dry-run report alone is not an upload record.

The local image tag is `myjev:0.1.0-release-candidate`. A registry destination is a separate owner choice:

```bash
# Choose and authenticate to your registry first.
export MYJEV_REGISTRY_IMAGE=YOUR_REGISTRY/YOUR_NAMESPACE/myjev:0.1.0
# Run these only when ready to publish the tested image.
docker tag myjev:0.1.0-release-candidate "$MYJEV_REGISTRY_IMAGE"
docker push "$MYJEV_REGISTRY_IMAGE"
docker image inspect "$MYJEV_REGISTRY_IMAGE" --format '{{json .RepoDigests}}'
```

A local image ID is not a pullable registry digest. Record the registry digest from the push/inspect result and use that immutable reference in the endpoint configuration. The cloud recipe remains unexecuted and requires no paid deployment to reproduce local results.

The recommended 4B temperature package additionally passed exact 4,096-token requests with 2 and 160 candidates, and rejected 4,097 tokens. Peak allocated VRAM was 9,538,996,736 bytes (8.88 GiB) in both synthetic cases, recorded in `results/release-readiness-v1/default-limits.json`. Other GPUs were active, so those elapsed times are not new isolated latency measurements. Continue using the tested 24 GB GPU class until total-process memory and startup headroom are measured on a smaller target.

## Clean release image verification

The rebuilt local `myjev:0.1.0-release-candidate` image installs the pinned dependencies into the base image and now includes the MIT license file in the installed wheel. Its local image ID is `sha256:238c4bb857950fcca869c4d8d6eb87ee179b4592d91ca2a082dbe5618de31643`; this is not yet a pullable registry reference. GPU access, Python/CLI/HTTP equivalence, the single-forward contract, 160-candidate smoke input, oversize rejection, real HTTP response equality and duplicate-ID rejection passed for the selected 4B temperature package. The managed `/health` and `/generate` aliases also passed. These checks used a read-only pre-cached backbone; they do not measure cold downloading or cloud deployment.

The evidence is under `results/release-readiness-v1/`, including the image build log, source checksums, package metadata, contract checks and container log. Reproduce the live-container check with:

```bash
.venv/bin/python scripts/verify_release_container.py \
  --artifact artifacts/hub-ready-v3/myjev-4b-continued_sft-seed11 \
  --image myjev:0.1.0-release-candidate --cache .cache/huggingface \
  --expected results/release-validation-v1/myjev-4b-continued_sft-seed11/equivalence.json
```

The `hub-ready-v3` cards pin public source commit `048afa79f43d6f0e84fc203cf43f5602372320c0`. The earlier packaging snapshots are retained for provenance. Version 1 predated the public source pin; version 2 predated the clarification that these individual artifacts have no archive adaptation. Adapter weights and inference manifests are unchanged between those packaging snapshots.

A separate fresh host virtual environment also installed `requirements.lock` and the built package successfully, passed `pip check`, and reproduced the previous Python/CLI/HTTP response exactly. Its response matched the clean Docker installation as well. Installation logs, the full package freeze and `clean-host-equivalence.json` are retained; the disposable verification environment was removed afterward to return disk space to training. This completes local direct-install and container checks for the selected checkpoint. It does not replace the remaining pinned Hub-download check after publication.
