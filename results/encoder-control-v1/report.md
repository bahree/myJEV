# Adapted fixed-taxonomy encoder control

A single ModernBERT-base checkpoint was fully fine-tuned for the fixed 77 BANKING intents. This is a practical operational control, not a generalist candidate-conditioned replacement for myJEV, and not a matched architecture or objective comparison.

- Pinned model: `answerdotai/ModernBERT-base` at `8949b909ec900327062f0ebf497f51aef5e6f0c8`; 149,664,077 classifier parameters.
- Three epochs, 750 updates, 23,997 example exposures; batch 32, AdamW learning rate 2e-5, weight decay 0.01, linear decay with 10% warmup, seed 11. Best validation-NLL checkpoint: epoch 3.
- Existing grouped BANKING splits retained. Temperature fitted only on the reserved calibration split: 0.9215. Thresholds also fitted only there. Full official test evaluated after model selection.
- FP32 parameters/optimizer states, BF16 autocast, SDPA attention on one A30. Train/validation/checkpoint elapsed: 62.78 seconds. Peak allocated training memory: 3.73 GiB.

| Confidence | Test accuracy | Macro-F1 | Correctness Brier | ECE |
|---|---:|---:|---:|---:|
| raw | 90.78% | 0.9074 | 0.0574 | 0.0365 |
| temperature | 90.78% | 0.9074 | 0.0555 | 0.0198 |

Model-only latency on one frozen short BANKING request: p50 **13.87ms**, p95 **14.02ms**, 9 input tokens, 10 warmups and 100 measurements. Peak inference allocated memory: 0.59 GiB. The saved full encoder/classifier artifact occupies 574.4 MiB. These are not HTTP measurements or a worst-case 512-token benchmark. The final report uses the separate fresh-process benchmark when present; the original training-process memory measurement is preserved but can include retained model/gradient references. The fixed taxonomy is encoded in the classifier head, so its input contains only the utterance. Qwen reads instructions and supplied candidate descriptions as well; equal numbers of output classes do not imply equal input work.

The Qwen main continued-supervision arms saw 8,000 examples, while this control saw 23,997. This one-seed run also has a different optimizer schedule, pretraining history and fixed-label output contract. Compare practical task quality and resource cost, not causal superiority. It cannot classify a newly supplied taxonomy without replacing/fitting its head. The frozen 100-update pilot is retained as part of the training run, not additional exposure. No learning rate, seed or epoch was selected from test performance.

## Reproduce and provenance

Run `CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/.cache/huggingface" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/run_encoder_control.py` in a fresh output directory matching the script constants, after preparing the frozen BANKING splits. Original results intentionally refuse overwrite. `scripts/summarize_bounded_controls.py` regenerates this report from saved metrics.

The original model is Apache-2.0: [model card](https://huggingface.co/answerdotai/ModernBERT-base), [paper](https://arxiv.org/abs/2412.13663). BANKING provenance and licenses remain those of the existing dataset card. Local full-model artifacts are excluded from Git; no new model release is claimed. Protocol/source hashes, startup compatibility failure, per-step losses, validation checkpoints, calibration/test predictions and timings are retained. One rejected constructor option was removed before any training; the execution amendment records this without changing the scientific budget.
