# Run inference locally and in Docker

The pilot implementation has been checked through Python, CLI, HTTP and a GPU container at 0.8B, 4B and 9B. Final release artifacts and a registry image are still pending. There are no downloadable myJEV weights yet: train a pilot using [the quick start](quickstart.md), then substitute your artifact path below.

## One interface, four entry points

An artifact includes pinned backbone/tokenizer revisions, LoRA adapters, confidence heads, calibration and a manifest. Every entry point uses the same loader and one backbone forward pass. The returned selection scores rank candidates; correctness confidence estimates whether the selected answer is right. They have different meanings.

Python:

```python
import json
from pathlib import Path
from myjev import DecisionModel

model = DecisionModel.load("artifacts/pilot-0.8b/artifact")
request = json.loads(Path("examples/request.json").read_text())
print(model.score(request))
```

CLI, for one request or a JSONL batch:

```bash
.venv/bin/myjev score --artifact artifacts/pilot-0.8b/artifact --input examples/request.json
.venv/bin/myjev score --artifact artifacts/pilot-0.8b/artifact --input requests.jsonl --jsonl
```

HTTP, in a separate terminal:

```bash
.venv/bin/myjev serve --artifact artifacts/pilot-0.8b/artifact
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Readiness requires model loading and warmup. Health is available at `/healthz`. The service binds locally by default, rejects oversized inputs instead of truncating, and uses a bounded queue with overload/timeouts. See [hosting](hosting.md) for remote HTTPS authentication and queue behavior.

## GPU Docker

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. Build the pinned Dockerfile and verify GPU access:

```bash
docker build -t myjev:local .
docker run --rm --gpus all myjev:local \
  python -c 'import torch; print(torch.cuda.is_available())'
```

Start the repository's Compose service using the artifact and shared backbone cache:

```bash
export MYJEV_ARTIFACT_HOST="$PWD/artifacts/pilot-0.8b/artifact"
export MYJEV_CACHE_HOST="$PWD/.cache/huggingface"
docker compose -f deploy/compose.yaml up --build
```

Compose owns its configured image/build settings; the `myjev:local` tag above is for the standalone GPU smoke check. Once `/readyz` succeeds, use the same HTTP request shown above. Stop the service with:

```bash
docker compose -f deploy/compose.yaml down
```

One process loads one selected model. Preserve the pinned backbone cache and mount a complete artifact. Never assume an adapter directory alone includes its backbone weights. The container's loopback host binding is intended for local access; authenticated remote access needs the proxy setup in the hosting guide.

## Troubleshooting and release checks

| Symptom | Check |
|---|---|
| GPU is unavailable in the container | Host driver, NVIDIA Container Toolkit and `--gpus` access |
| Readiness does not succeed | Model-loading/warmup logs, artifact path, backbone cache and free VRAM |
| First model call needs a compiler | Use the repository image, which includes `gcc` and `libc6-dev` |
| Python dependencies disappear in a wrapper | Invoke `.venv/bin/python` directly; do not resolve its symlink to system Python |
| HTTP 429 or 504 | Queue capacity, request length and timeouts; failed requests are not fast inference |
| Backend output differs | Recheck custom heads, precision, pinned revisions and calibration |

Before releasing a checkpoint, repeat save/reload and Python/CLI/HTTP/Docker equivalence on that checkpoint and image. Measure warm p50/p95, throughput, peak VRAM, cold start and overload behavior on representative inputs. The proposed scoring-versus-generation controls are described in [the benchmark protocol](hosting.md#planned-scoring-versus-generation-benchmark); they are not completed results.

The [Hugging Face custom-container recipe](hosting.md#hugging-face-inference-endpoints-recipe) remains unexecuted. Publishing weights does not create a running endpoint. Replace local build tags with tested immutable image digests when registry releases become available.
