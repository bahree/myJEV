# Quick start

This guide takes you from installation to one decision, then to a batch and a local HTTP service. It uses the published [myJEV-4B](https://huggingface.co/bahree/myJEV-4B), so you can follow it without training. If you want to build a model yourself, the optional pilot instructions follow the serving examples.

The tested setup is Linux, Python 3.12 and an NVIDIA A30 with 24 GB of GPU memory. The model's GPU runtime also needs a host C compiler; on Debian/Ubuntu, install `gcc` and `libc6-dev` if absent. The [Docker path](#docker) supplies the Python environment and compiler, but still needs the host NVIDIA driver and Container Toolkit. For reading outputs without a GPU or installation, open the [saved demo walkthrough](../results/demos-v1/report.md).

## Install and test

```bash
git clone --depth 1 https://github.com/bahree/myJEV.git
cd myJEV
python3.12 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m pytest -q
export HF_HOME="$PWD/.cache/huggingface"
```

The tests do not download model weights. The dependency lock includes a large CUDA-enabled PyTorch installation. Activating `.venv` makes the `python` commands in the other guides use these installed packages; repeat the activation when opening a new terminal. `HF_HOME` sets the download cache so later runs can reuse the same files.

## Download and score the default

```bash
.venv/bin/myjev score --artifact bahree/myJEV-4B \
  --revision 38f7cca5a8530483309f576b0c3dd1756bc27c33 \
  --input examples/request.json
```

The first call downloads files from the Hugging Face Hub: the small updates trained for myJEV and the original Qwen model they need. That original model is the **backbone**. The release records its required version, and the loader fetches it automatically. Keep the long `--revision` value to reproduce this example's file selection. See [all six model releases](models.md) for the other sizes and training variants.

The input file asks where to route “I was charged twice,” with billing, technical support and other as the choices. The recorded response selects `billing` with confidence approximately `0.9834`. It also contains a score for every choice and hashes identifying the model and calibration. The [README example](../README.md#an-actual-local-request-and-response) shows the complete request and saved output; the [walkthrough](walkthrough.md#2-follow-one-decision) explains the fields.

## Try more requests

```bash
python scripts/run_demos.py > demo-results.jsonl
head -n 1 demo-results.jsonl | python -m json.tool
```

The runner loads the same 4B model once, then scores seven requests. JSONL stores one request or response per line, and the line order is preserved. The second command prints the first response with indentation. The [seven-request walkthrough](../results/demos-v1/report.md#what-each-request-asks) explains why the expected answers change across support issues, refund dates, a misleading quote and a short post.

## Score and serve

Keep the model loaded when scoring several requests from Python. Run this in the environment installed above:

```python
from myjev import DecisionModel

model = DecisionModel.load(
    "bahree/myJEV-4B",
    revision="38f7cca5a8530483309f576b0c3dd1756bc27c33",
)
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
.venv/bin/myjev serve --artifact bahree/myJEV-4B \
  --revision 38f7cca5a8530483309f576b0c3dd1756bc27c33
```

From another terminal:

```bash
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

The server runs in the first terminal until you stop it. `/readyz` succeeds after the model has loaded and completed a warmup request. The `/score` call returns the same fields as the CLI. Duplicate candidate IDs and oversized input are rejected. The service limits queued work and applies timeouts; [hosting](hosting.md) explains those choices and authenticated remote access.

## Docker

The public [Docker Hub image](https://hub.docker.com/r/amitbahree/myjev) serves the default model without local training or a Python installation. The host needs Docker, an NVIDIA driver and NVIDIA Container Toolkit; the validated GPU class is an A30 with 24 GB VRAM.

```bash
export MYJEV_IMAGE=amitbahree/myjev@sha256:55c78ef13329ab35712e27c9815eff94c17ba1f5fc145bd58eb705d0f16e1f6d
docker pull "$MYJEV_IMAGE"
mkdir -p .cache/huggingface
docker run --rm --name myjev --gpus device=0 \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/.cache/huggingface:/cache/huggingface" \
  -e MYJEV_ARTIFACT=bahree/myJEV-4B \
  -e MYJEV_REVISION=38f7cca5a8530483309f576b0c3dd1756bc27c33 \
  "$MYJEV_IMAGE"
```

Use the same curl request above after `/readyz` succeeds. Stop with `docker stop myjev`. First startup downloads the pinned release and its backbone into the mounted cache. The image contains no model weights. The [inference guide](inference.md#gpu-docker) also covers source builds and Compose for locally trained artifacts. [Hosting](hosting.md) links the registry receipt, GPU equality check, limits and benchmark conditions. Its managed cloud recipe remains unexecuted.

For dashboard setup and a safe environment template, see [W&B tracking](tracking.md). For all inference interfaces, container startup and troubleshooting, see [inference and Docker](inference.md).

The optional [Decision-1 comparison](decision-1.md) evaluates an external hosted model. It is independent of the local commands above and needs an OpenRouter account only when you choose to make its API calls.

## Train the 0.8B pilot

To learn how the model is trained, run this optional small experiment. It prepares BANKING77 and trains a new local model; it does not modify a downloaded release.

```bash
.venv/bin/python scripts/prepare_banking.py --revision 57ec275d8078af65b7731c2a98be812d844a6d6b
CUDA_VISIBLE_DEVICES=0 .venv/bin/myjev-train \
  --config configs/0.8b.json \
  --data data/banking77/train.jsonl \
  --output artifacts/pilot-0.8b
```

This downloads the pinned backbone and runs 100 supervised updates. The resulting artifact is at `artifacts/pilot-0.8b/artifact`. It includes adapters, confidence-head weights, and a manifest. Its short training does not make it a production classifier.

Use `configs/4b.json` or `configs/9b.json` with a distinct output directory for larger models. The pilot used one 24 GB A30 per independent run. Keep the shared backbone cache rather than duplicating model downloads.

Score that newly trained package with `.venv/bin/myjev score --artifact artifacts/pilot-0.8b/artifact --input examples/request.json`. Its short training run will not reproduce the released 4B model's output.

## Evaluate your artifact

```bash
.venv/bin/myjev-evaluate \
  --artifact artifacts/pilot-0.8b/artifact \
  --data data/banking77/test.jsonl \
  --calibration data/banking77/calibration.jsonl \
  --limit 256 --output results/my-local-evaluation
```

The limit reproduces the pilot's evaluation scope. Calibration labels select thresholds; test labels measure the resulting performance. Read saved accuracy together with coverage, accepted-case error, and uncertainty.
