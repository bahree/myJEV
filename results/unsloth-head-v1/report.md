# Alias readout and Clef head on the same Qwen backbone

This is a separate answer-only supervised experiment on Qwen3.5-0.8B. Both arms use the same BF16 backbone revision and rank-8 adapters, but their prompts and answer readouts differ. The completed supervision/RL study and released default are unchanged.

## Feasibility pilot

Each pilot completed 100 updates with eight examples per update. Memory is the PyTorch peak allocated on an A30; it excludes CUDA context memory. Time includes first-step kernel preparation and optimizer checkpointing.

| Readout | Trainable parameters | Peak allocated GiB | Training seconds | Mean input tokens |
|---|---:|---:|---:|---:|
| alias | 540,672 | 2.241 | 194.1 | 875.8 |
| clef | 27,865,604 | 3.014 | 331.5 | 1908.1 |

These are runtime feasibility observations, not accuracy results. Initial adapter hashes and example/order hashes agree across the paired pilots. A prior compiled attempt failed on its first backward pass. Both reported pilots use eager execution, with no silent input truncation.

## Validation-only learning-rate selection

| Readout | Learning rate | Validation accuracy | Selection NLL |
|---|---:|---:|---:|
| alias | 3e-05 | 60.88% | 1.9408 |
| alias | 0.0001 | 67.30% | 1.5628 |
| clef | 3e-05 | 65.50% | 1.2303 |
| clef | 0.0001 | 70.81% | 1.0538 |

Selected rates: alias 0.0001, clef 0.0001. Each trial received 250 updates on seed 101. The frozen rule used validation accuracy, then NLL, then the lower rate. No test data selected these settings.

## Accuracy and calibration

The matched main comparison is not yet complete. No test-based architecture recommendation is made from the feasibility pilot.


## Reproduce

Use the [comparison guide](../../docs/unsloth.md) for dependencies, training commands and scope. Regenerate this report with `python scripts/summarize_head_comparison.py`. The configuration fixes two learning-rate trials per arm and three fresh main seeds. Candidate-order tests keep calibration fixed and are reported separately.
