# CPU and GPU container checks

Version 0.1.4 adds automatic device selection. Visible CUDA is selected first; otherwise supported BF16 releases load on CPU with a warning. An explicit CUDA request remains strict. NF4 9B CPU loading fails before the backbone loads. Model weights, saved precision and calibration files are unchanged.

The [publication receipt](../container-registry-v4/publication.json) pins both Docker Hub images and public source `a242d9cf4194a591f922e04abfc6075ee82c38cc`. Both are pullable without credentials. These checks ran on Linux with an AMD EPYC 7413 CPU (eight logical processors available), 62 GiB host RAM and three 24 GB A30 GPUs. No native Windows or Mac host was available.

## Native x86-64 CPU requests

Each extended run scored the included duplicate-charge fixture plus the seven authored demos, with one CPU thread, cached weights, and concurrent host activity. These are observed durations, not isolated latency benchmarks. The helper uses a 600-second request timeout for observation; the published CPU default is 120 seconds. Input length and model size can exceed that default.

| Model | Scored requests | Cached readiness | Observed request range | Fixture selection | Fixture confidence |
|---|---:|---:|---:|---|---:|
| [myJEV-0.8B-RL](cpu-one-thread/myJEV-0.8B-RL.json) | 8 | 22.0 s | 8.49-15.10 s | billing | 0.872612 |
| [myJEV-0.8B](cpu-one-thread/myJEV-0.8B.json) | 8 | 24.1 s | 8.89-15.58 s | billing | 0.975705 |
| [myJEV-4B-RL](cpu-one-thread/myJEV-4B-RL.json) | 8 | 58.6 s | 55.63-101.82 s | billing | 0.859048 |
| [myJEV-4B](cpu-one-thread/myJEV-4B.json) | 8 | 60.9 s | 57.15-102.61 s | billing | 0.984125 |

The [final x86-64 CPU image check](final-cpu-amd64.json) additionally verifies its included request file, CLI/HTTP equality, the managed route, non-finite 422 responses and body-limit 413 responses. The [GPU image without GPU access](final-gpu-image-cpu-fallback.json) confirms automatic CPU selection on the same 0.8B fixture. These final-image checks used one thread and the same runtime source as the extended runs; image IDs remain in each record.

CPU scores differ slightly from the GPU fixtures. The 0.8B GPU supervised fixture reports 0.976432 confidence; the CPU observations preserve their actual values above. The published quality and calibration study has not been repeated on CPU. A short successful request does not establish maximum-input memory, throughput or calibration.

## GPU regression

All six release fixtures matched their saved GPU responses exactly in the final GPU image. The [installed-file check](../container-registry-v4/gpu-check.json) also matched every Python source checksum to the public source, verified CLI/HTTP equality, both managed routes, aggregate-text limits and non-finite rejection. The per-model records live in `final-gpu/`.

## Thread settings and earlier CPU feasibility

Eight threads caused minutes of warmup in the grouped convolution on this host. Those test containers were stopped. A standalone CPU grouped-convolution diagnostic with one thread completed in 0.0271 seconds; the one-thread model checks then completed. Both images default to one thread. Thread counts should be measured on the target machine rather than increased automatically.

The earlier [direct Python feasibility observations](feasibility.jsonl) used the CUDA-enabled PyTorch build on CPU with eight threads. They recorded first short-score times of 2.88 seconds at 0.8B and 12.02 seconds at 4B, with peak process RSS around 2.3 and 8.0 GiB respectively. Loading included downloads. Those single-process observations differ from the CPU-only wheel, one-thread Docker checks and simultaneous host activity above; they are not Docker latency predictions. The 8 GiB and 16 GiB Docker Desktop starting budgets in the reader guide leave headroom over those short-input observations and are not maximum-input guarantees.

## ARM64 scope

The CPU image index includes Linux amd64 and arm64. ARM64 imported PyTorch 2.14.1+cpu and completed a BF16 matrix multiplication under QEMU 8.2.2, using a read-only mounted emulator. No host binfmt or privileged emulator was installed. The initial eight-thread attempt was stopped while still warming up. The one-thread retry exceeded the 900-second readiness deadline. [One-thread attempt](arm64-one-thread.log), [final-image import/arithmetic log](final-arm64-import.log).

The Apple Silicon instructions therefore describe an experimental ARM64 build. They do not claim native Mac execution or full ARM64 output equivalence. QEMU timing cannot predict native Apple Silicon performance. Windows instructions use Docker Desktop's documented Linux-container/WSL 2 behavior but have no native Windows test receipt.

## Reproduce a check

```bash
python scripts/verify_device_container.py \
  --image amitbahree/myjev:0.1.4-cpu \
  --repo bahree/myJEV-0.8B \
  --revision 01fed6001aa675bbfcddbf2ebb3574b28c6da9b6 \
  --cache .cache/huggingface --threads 1 --cli --demos \
  --output results/my-cpu-check.json
```

This helper uses a read-only, already populated Hugging Face cache and offline mode. The [reader commands](../../docs/docker.md) use a writable named volume so missing files can download. The existing 0.1.2 empty-cache observation remains historical; no new first-download timing is claimed.
