# Run myJEV with Docker

Give myJEV a request and candidate descriptions. It returns a selected ID, scores for those choices and an estimate of correctness, in one backbone pass. The container supplies Python and the custom loader. At first startup it downloads the selected adapters, heads and calibration from Hugging Face, then their pinned Qwen backbone. A named volume keeps those files for later starts. No hosted endpoint or training run is required.

[Source and experiments](https://github.com/bahree/myJEV) | [Windows, macOS and Linux instructions](https://github.com/bahree/myJEV/blob/main/docs/docker.md) | [Six model releases](https://github.com/bahree/myJEV/blob/main/docs/models.md)

## Pick an image

| Tag | Platforms | Use |
|---|---|---|
| `0.1.4` | Linux x86-64 | NVIDIA CUDA, or automatic CPU fallback for BF16 releases |
| `0.1.4-cpu` | Linux x86-64 and ARM64 | CPU, including Docker Desktop on Intel and Apple Silicon Macs |

Linux NVIDIA execution needs the host driver and Container Toolkit. Docker Desktop GPU access uses Windows WSL 2. The CPU image uses no Apple GPU acceleration. Linux is the validation host; ARM64 emulation and native Windows/Mac limits are recorded in the [checks](https://github.com/bahree/myJEV/blob/main/results/cpu-inference-v1/report.md).

## GPU example

```bash
docker run --rm --name myjev --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v myjev-hf-cache:/cache/huggingface \
  -e MYJEV_ARTIFACT=bahree/myJEV-4B \
  -e MYJEV_REVISION=38f7cca5a8530483309f576b0c3dd1756bc27c33 \
  amitbahree/myjev:0.1.4
```

Wait for `Application startup complete`. In another terminal:

```bash
docker cp myjev:/app/examples/request.json ./request.json
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' --data-binary @request.json
```

The supplied request routes “I was charged twice.” The recorded GPU output selects `billing`, with confidence about `0.9834`. The response also contains selection scores and revision hashes. Selection and correctness confidence have different meanings; the [walkthrough](https://github.com/bahree/myJEV/blob/main/docs/walkthrough.md) explains them.

## CPU example

```bash
docker run --rm --name myjev \
  -p 127.0.0.1:8000:8000 \
  -v myjev-hf-cache:/cache/huggingface \
  -e MYJEV_ARTIFACT=bahree/myJEV-0.8B \
  -e MYJEV_REVISION=1c956c89d21c0ab136e98ffe66a16752fa37d823 \
  amitbahree/myjev:0.1.4-cpu
```

Use the same copy/request commands after readiness. Start with 0.8B for lower memory use; 4B takes more RAM and time. The 9B NF4 releases require CUDA and fail clearly on CPU. CPU scores can differ from GPU scores. Published GPU calibration and accuracy have not been remeasured on CPU.

Windows PowerShell uses backticks for line continuation and `curl.exe` for requests. The [platform guide](https://github.com/bahree/myJEV/blob/main/docs/docker.md#windows-powershell) gives complete commands with a named volume, so no drive-letter bind mount is needed.

## Runtime behavior

`MYJEV_DEVICE=auto` selects visible CUDA, otherwise CPU with a warning. Set `MYJEV_DEVICE=cuda:0` to require CUDA. Docker rejects an unsupported `--gpus` flag before the application can start; omit it for CPU use. The loader keeps the requested model and precision and does not fall back after a GPU out-of-memory error.

Both images default to one CPU thread. The CPU image uses a 120-second request timeout; the GPU image uses 30 seconds. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `MYJEV_TIMEOUT` and `MYJEV_QUEUE_SIZE` as needed. A full queue returns 429; timeout returns 504 while an executing model call retains its capacity slot. Both images have a 15-minute health-start grace and run as root. The installed Python distribution remains version 0.1.0; image and model digests identify the deployed files.

The loader rejects oversized inputs instead of truncating them. HTTP bodies above 1 MiB return 413; invalid schema, non-finite numbers and aggregate text above 256 KiB return 422. The exact token cap is 4,096 across the rendered request. Maximum-length quality is outside the smoke checks.

Stop with `docker stop myjev`. The named cache remains. The examples expose port 8000 only on loopback and disable access logs. Remote access needs authenticated HTTPS; see [hosting](https://github.com/bahree/myJEV/blob/main/docs/hosting.md).

## Reproduce the image

The [release receipt](https://github.com/bahree/myJEV/blob/main/results/container-registry-v4/publication.json) records digests, source revision and checks. The GPU release applies `deploy/Dockerfile.patch` over immutable 0.1.3 dependencies. The root `Dockerfile` provides the full build path. `Dockerfile.cpu` and `requirements.cpu.lock` build the CPU image. Neither image includes model weights or credentials.

The historical 0.1.2 empty-model-cache check downloaded 9,341,722,221 bytes and reached readiness in 203.6 seconds using a disposable RAM-backed cache and preinstalled Docker layers. That is separate from the populated-cache 0.1.4 checks and does not predict fresh-machine download time. [Historical first-use record](https://github.com/bahree/myJEV/blob/main/results/review-container-v1/first-use.json).
