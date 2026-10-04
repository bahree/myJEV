# Model and container releases

**Release tracker — placeholders, not download links.** The pilot checkpoints exist on the training machine, but no trained myJEV artifact or container image is publicly released yet. You can train a local artifact with the [quick start](quickstart.md).

| Planned artifact | Current state | Public model card / download |
|---|---|---|
| myJEV 0.8B | Training and serving pilot measured | Pending selection and release |
| myJEV 4B | Training and serving pilot measured | Pending selection and release |
| myJEV 9B | NF4 training and serving pilot measured | Pending selection and release |
| Reference GPU container | Built and tested locally | Registry image and digest pending |
| Default recommended artifact | Not selected | Pending quality/resource comparison |

## What a release will contain

Each model release should include a pinned backbone/tokenizer revision, LoRA adapters, confidence-head weights, prompt and alias semantics, precision settings, calibration parameters, and a checksummed manifest. Its model card should state training exposure, data provenance, evaluation scope, tested input limits, limitations, and license requirements.

Weights alone do not reproduce this interface: the custom confidence logic requires the shared myJEV loader. Merged weights will only be offered after save/reload and output-equivalence checks. Quantization or backend changes require fresh calibration evaluation.

## When this page changes

A release entry becomes usable only when its immutable revision, load command, and evaluation evidence are populated. Until then, there is no hosted demo, no published Hub identifier to copy, and no production recommendation. The 9B model is not automatically the default.

The [hosting guide](hosting.md) already describes local execution and the unexecuted managed-endpoint recipe. Publishing a model on the Hub will not itself create a running endpoint.
