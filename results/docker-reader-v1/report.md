# Docker quick-start checks

The published 0.1.4 GPU and 0.1.4-cpu images completed the included duplicate-charge request and the seven-request JSONL batch. The 4B GPU outputs matched the saved demo responses in every field. These are execution and response checks, not a new model-quality comparison.

| Route | Requests completed | Selected IDs in batch order |
|---|---:|---|
| 0.8B CPU | 1 single + 7 batch | billing, technical, other, approve, approve, billing, tutorial |
| 4B NVIDIA | 1 single + 7 batch | billing, technical, other, approve, deny, billing, tutorial |

The requests and expectation file were downloaded from public source revision `82f0ee064cd13d8e2f8cee6d468fab065f27a743` and matched the repository bytes. A read-only bind mount exposed a temporary folder containing spaces in its name. The expected answers were not passed to the model.

The runs used existing model caches mounted read-only, with `HF_HUB_OFFLINE=1`; they did not measure first-download time. CPU and GPU checks ran concurrently. Durations in [commands.json](commands.json) include model loading and are completion records, not isolated latency benchmarks. Host-specific paths use the documented placeholders.

Bash and PowerShell examples normalize to the same Docker argument sequences. The new automated runs took place on Linux; no Windows batch or NVIDIA passthrough check was performed. Output files are the original CLI stdout, and the adjacent stderr logs retain loading messages and warnings.

The [owner-provided screenshot](../../assets/windows-cpu-inference.png) separately records one completed 0.8B CPU request from Windows PowerShell, selecting `billing` with confidence `0.9767757654190063`. [Its receipt](owner-screenshot.json) identifies the image and model revisions and the unchanged image hash. No Windows hardware inventory, memory or timing was supplied. This evidence does not establish NPU or Mac support, or full CPU calibration.
