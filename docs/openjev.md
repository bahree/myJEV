# OpenJev as an external comparison

Decision: include it as related work and an optional separately reported baseline. It is not a replacement for the controlled 0.8B → 4B → 9B training study, and no OpenJev weights or hosted calls were used in the current results.

Inspected release: `openjev/openjev` at `ac97900fd034fdd7e7e536f3d4c21b836cae0750`. The [model card](https://huggingface.co/openjev/openjev/tree/ac97900fd034fdd7e7e536f3d4c21b836cae0750) identifies a 27B model, a 52-option single-pass limit, and larger candidate sets handled through multiple passes. Its primary serving measurements use H100/FP8. Those numbers are publisher-reported, not local A30 measurements. The weights carry CC BY-NC 4.0; helper/serving code is Apache 2.0. Do not bundle its weights into a myJEV release or assume the same reuse terms as the backbone.

The pinned `NOTICE` names Qwen3.8-27B as its base, which also makes it a different model generation from this study. The pinned `helper/shim.py` exposes calibrated selection probabilities. For choice questions its confidence is `(max(p) - 1/K) / (1 - 1/K)`, clamped at zero. That normalization above uniform is not a separately learned probability of selected-answer correctness. A comparison should retain it as `upstream_confidence` while evaluating the selected-option probability as an explicitly named confidence proxy, with the helper's four-decimal probability rounding disclosed.

For 77 candidates, its helper makes two group reads and a final winners read. Therefore a full BANKING77 comparison uses three backbone reads under that recipe. It must not be reported as a one-pass latency comparison. Alternatively, freeze a shared ≤52-candidate diagnostic task for all models and label that as a changed task, not the original 77-way benchmark.

`scripts/openjev_baseline.py` translates the shared dataset format into the helper API and evaluates the returned predictions with the same calibration-only operating thresholds. It refuses multi-pass candidate counts unless explicitly enabled. Run it only against a separately provisioned, reviewed endpoint:

```bash
.venv/bin/python scripts/openjev_baseline.py --url http://127.0.0.1:3000 \
  --revision ac97900fd034fdd7e7e536f3d4c21b836cae0750 --allow-multipass \
  --output results/openjev-local
```

The revision argument records provenance; the client cannot independently prove the remote server loaded those weights. Record the server's image, configuration, actual model revision and precision alongside results. Tokens are read from `OPENJEV_TOKEN` if authentication is configured and are not written to logs. Existing publisher benchmarks are not imported into the local results table. A GGUF route may be feasible on the local cards, but it would add a backend/precision comparison and requires its own validation; it has not been executed here.
