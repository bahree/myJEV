# Replicated 4B precision study

Completed seed pairs: 0/3. Partial evidence; do not rank configurations yet.

4B SFT, same seeds, 4000 examples, LR, adapter targets and readout. BF16 versus NF4 with FP32 nonquantized modules. Training configuration comparison, not an inference-only quantization toggle.

Three seeds; final-checkpoint test. No new tuning. Does not identify the precision effect on RL, continued supervision or 9B. Capacity-only conclusions still require care.

| Seed | BF16 accuracy | NF4 accuracy | NF4 minus BF16 | BF16 Brier | NF4 Brier |
|---|---:|---:|---:|---:|---:|

Regenerate: `python scripts/summarize_precision_study.py`. The frozen plan and source hashes identify the exact controls. Further paired uncertainty analysis should precede strong claims from small differences.
