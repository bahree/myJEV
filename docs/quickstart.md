# Quick start

The current milestone provides code and recorded evidence. The public [myJEV-4B](https://huggingface.co/bahree/myJEV-4B) adapter/head release is the default starting point; training a local pilot is optional. Linux with Python 3.12 is the tested setup. GPU commands require compatible CUDA hardware; 9B also requires the locked quantization dependencies.

## Install and test

```bash
git clone https://github.com/bahree/myJEV.git
cd myJEV
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m pytest -q
export HF_HOME="$PWD/.cache/huggingface"
```

The tests do not download a backbone. The dependency lock includes a large CUDA-enabled PyTorch installation. For reading results only, browse [the experiment guide](experiments.md) without installing anything.

## Download and score the default

```bash
.venv/bin/myjev score --artifact bahree/myJEV-4B \
  --revision a1b9e3b1181293220012cfb15587cdba0767ae8e \
  --input examples/request.json
```

The default [myJEV-4B](https://huggingface.co/bahree/myJEV-4B) is available on Hugging Face. These releases contain adapters, custom heads, calibration and manifests, not merged backbones. Use the myJEV loader, which separately loads the manifest-pinned Qwen backbone and tokenizer. Publishing these files does not create a hosted endpoint. See [all six model releases](models.md).

## Train the 0.8B pilot

```bash
.venv/bin/python scripts/prepare_banking.py --revision 57ec275d8078af65b7731c2a98be812d844a6d6b
CUDA_VISIBLE_DEVICES=0 .venv/bin/myjev-train \
  --config configs/0.8b.json \
  --data data/banking77/train.jsonl \
  --output artifacts/pilot-0.8b
```

This downloads the pinned backbone and runs 100 supervised updates. The resulting artifact is at `artifacts/pilot-0.8b/artifact`. It includes adapters, confidence-head weights, and a manifest. Its short training does not make it a production classifier.

Use `configs/4b.json` or `configs/9b.json` with a distinct output directory for larger models. The pilot used one 24 GB A30 per independent run. Keep the shared backbone cache rather than duplicating model downloads.

## Score and serve

```python
from myjev import DecisionModel

model = DecisionModel.load("artifacts/pilot-0.8b/artifact")
print(model.score({
    "context": "I was charged twice.",
    "instructions": "Select the appropriate support route.",
    "candidates": [
        {"id": "billing", "description": "Charges, invoices, and refunds"},
        {"id": "technical", "description": "Errors and configuration"},
        {"id": "other", "description": "Neither listed route applies"}
    ]
}))
```

```bash
.venv/bin/myjev score --artifact artifacts/pilot-0.8b/artifact --input examples/request.json
.venv/bin/myjev serve --artifact artifacts/pilot-0.8b/artifact
```

From another terminal:

```bash
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

The result includes the selected ID, candidate scores, confidence, and artifact/calibration revisions. Duplicate candidate IDs and oversized input are rejected. Readiness follows warmup; the service has bounded queueing and timeouts. See [hosting](hosting.md) before exposing it remotely.

## Evaluate your artifact

```bash
.venv/bin/myjev-evaluate \
  --artifact artifacts/pilot-0.8b/artifact \
  --data data/banking77/test.jsonl \
  --calibration data/banking77/calibration.jsonl \
  --limit 256 --output results/my-local-evaluation
```

The limit reproduces the pilot's evaluation scope. Calibration labels select thresholds; test labels measure the resulting performance. Read saved accuracy together with coverage, accepted-case error, and uncertainty.

## Docker

After local training, on a host with Docker and NVIDIA Container Toolkit configured:

```bash
export MYJEV_ARTIFACT_HOST="$PWD/artifacts/pilot-0.8b/artifact"
export MYJEV_CACHE_HOST="$PWD/.cache/huggingface"
docker compose -f deploy/compose.yaml up --build
```

Compose publishes the service on host loopback port 8000. Use the same curl request as above. Stop it with `docker compose -f deploy/compose.yaml down`.

The image is built locally; no public container registry release is available yet. [Hosting](hosting.md) documents the measured image, limits, benchmark conditions, and Hugging Face custom-container recipe. The cloud recipe has not been executed.

For dashboard setup and a safe environment template, see [W&B tracking](tracking.md). For all inference interfaces, container startup and troubleshooting, see [inference and Docker](inference.md).
