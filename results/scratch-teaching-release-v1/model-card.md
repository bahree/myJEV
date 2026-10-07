---
license: mit
tags: [myjev, decision-model, from-scratch, teaching, research]
---

# myJEV scratch teaching checkpoint

This is a complete **201,175-parameter**, randomly initialized decision network trained on original synthetic routing fixtures. It is a teaching artifact, not a general-language classifier, a Jev reproduction or a production model. No pretrained backbone, LoRA adapter, external embedding model or text-generation loop is needed. This local package has not been uploaded to Hugging Face.

## Architecture and inference

The network uses a deterministic UTF-8 byte tokenizer, two shared encoder blocks at width 64 with four attention heads, candidate-to-context attention and candidate-set interaction. It produces candidate-selection scores and a separate scalar correctness-confidence estimate. These have different meanings. One network forward invokes the shared text encoder twice: once for context and once for the candidate batch. There is no autoregressive generation.

The package contains full FP32 network weights, not an adapter. The weight file occupies 809,244 bytes. The manifest pins tokenizer behavior, architecture, confidence mode and weight checksum. Confidence mode is `scalar`, temperature is 1.0 and calibration revision is `none`; no temperature calibration has been applied to this checkpoint.

## Training and provenance

This preserves the frozen study's initial supervised checkpoint for **seed 11 by packaging convention**, not a seed selected for its test result. Training used 1,000 supervised optimizer updates at batch size 8, giving 8,000 example presentations, and learning rate 0.001 selected on validation. All weights began randomly. The selected checkpoint is not an RL continuation.

Original version-2 fixtures were generated deterministically with seed 42 by `src/myjev/scratch/data.py`, using 2,048 training, 256 validation, 256 calibration and 512 test rows. They ask for a color signal, include an explicit `other` option, and include two-signal cases with known equal probabilities. Hidden probabilities are evaluator-only metadata. Surface layouts and some ambiguous color pairs differ across partitions. The fixtures are CC0-1.0; code and this model's original weights are MIT, copyright Amit Bahree. No outside training text or pretrained weights are included. Dataset file hashes are preserved in `training-data-provenance.json`.

## Measured quality and limitations

This seed scored **86.33% accuracy** on the 512-row synthetic test, with correctness Brier **0.1187**. That single result must be read alongside the three-seed supervised mean of **60.74% accuracy with 30.66 percentage-point standard deviation**. Continued SFT, exact RL and sampled RL means were 66.34%, 51.43% and 40.62%, respectively. These large seed differences show an unstable training recipe. The toy task has irreducible ambiguity and does not measure broad language understanding.

A **separate** 217,559-parameter BANKING77 diagnostic, with a different 512-byte configuration, collapsed to one class and achieved 1.30% accuracy. Those are not test results from this packaged synthetic checkpoint. The diagnostic provides no basis to call this checkpoint a useful natural-language classifier. Its separate low Brier result also does not establish useful decisions. See the [scratch report](https://github.com/bahree/myJEV/blob/main/docs/scratch.md) for both experiments and their limits.

## Tested interface limits and serving evidence

The manifest caps instructions plus newline plus context at **256 UTF-8 bytes**, each candidate description at **64 bytes**, and the candidate list at **32 options**; at least two options are required. These are byte counts, not Qwen token counts. The teaching release check accepts the exact three upper bounds together and rejects each bound plus one. Bounds establish accepted input shape, not quality on long inputs or 32 candidates. Quality and historical timing evidence use four candidates.

On the short four-candidate fixture, the previous deployment check measured approximately 1.97 ms CPU p50 with two threads and 2.94 ms on an A30. This is not a matched Qwen/Jev comparison. The original checkpoint was checked through Python, CLI, HTTP and Docker. This packaging snapshot is checked for exact CPU reload equality with that original artifact and byte-identical weight/manifest files. Full evidence is under `results/scratch-deployment/` and `results/scratch-teaching-release-v1/` in the source project.

## Load locally

Use the custom myJEV loader from public source commit `048afa79f43d6f0e84fc203cf43f5602372320c0` in https://github.com/bahree/myJEV. A generic text-generation widget does not implement this network.

```python
from myjev import DecisionModel
model = DecisionModel.load("artifacts/scratch-teaching-release-v1", device="cpu")
result = model.score({
    "context": "signal: red; case demo",
    "instructions": "Choose the signal color. Choose other if the color has no option.",
    "candidates": [
        {"id": "red", "description": "red"},
        {"id": "blue", "description": "blue"},
        {"id": "green", "description": "green"},
        {"id": "other", "description": "other"}
    ]
})
print(result)
```

The supported HTTP/CLI interfaces also use this loader. This checkpoint belongs to the separate scratch teaching track, not the six Qwen adapter release candidates. No managed endpoint is running.
