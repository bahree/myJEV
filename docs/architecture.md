# Architecture and objectives

myJEV receives context, task instructions, and 2–160 candidate IDs and descriptions. Candidate IDs belong to the calling application; they are not a fixed global label vocabulary.

## One forward pass

1. Map candidates to verified single-token aliases and render the pinned prompt template.
2. Tokenize the complete input. Reject inputs beyond the artifact's configured limit rather than silently truncating.
3. Run the backbone once and use its final hidden state for candidate scores and confidence.
4. Select the highest-scoring candidate and return its original ID, normalized scores, and correctness confidence.

Training randomizes candidate order. That does not guarantee order invariance; order changes require evaluation.

The reference implementation is in [model.py](../src/myjev/model.py), [prompt.py](../src/myjev/prompt.py), and [inference.py](../src/myjev/inference.py). Custom heads and adapters require this loader; a generic text-generation server does not automatically reproduce the outputs.

## What fine-tuning changes

The backbone weights stay frozen. Training updates attention LoRA adapters and the custom confidence heads. The 0.8B/4B backbones use BF16; 9B uses NF4 QLoRA for memory fit. The pretrained model already supports token scoring. Adaptation tests whether task decisions and learned correctness estimates improve over the untouched model and simpler calibration controls. See [why we fine-tune](training.md#why-fine-tune-an-already-pretrained-model) for the rationale, resource trade-offs and limits.

## Confidence has its own meaning

The supervised implementation includes a scalar correctness head. The RL experiment uses a candidate-conditioned policy over 21 confidence values: 0.00, 0.05, …, 1.00. At inference, an RL artifact reports the expected confidence for its deterministically selected answer. The grid is an experimental parameterization, not a necessary property of decision models.

Selection scores sum to one over the supplied options. Correctness confidence estimates whether the selected answer is right. Missing the correct option can still produce a high selection score, so unsupported requests need explicit evaluation.

## Same objective, exact and sampled gradients

For correctness `c` in `{0, 1}` and confidence `q`, the confidence-aware reward is:

```text
R(c, q) = c - (q - c)^2
```

Exact optimization sums over the finite joint answer/confidence action space. Sampled REINFORCE draws eight actions per example and uses a leave-one-out baseline. Both use AdamW to update the same trainable components, with the same policy architecture and KL regularization against a frozen supervised reference. Correctness-only reward, continued supervised training, Brier auxiliary supervision, temperature scaling, and constant-confidence controls test alternative explanations.

The scoring component alone does not guarantee calibration of the combined, regularized objective. [Objective tests](../tests/test_objectives.py) check arithmetic, finite-difference gradients, and sampled-gradient agreement. [The protocol](protocol.md) explains matched initialization and exposure; [the results](experiments.md) distinguish sampled training behavior from deployed argmax decisions.
