# Completed extension contrasts

Paired group bootstrap, 2000 draws, RNG 42; average the three observed seeds before resampling groups. Conditional on these seeds, exploratory, no multiplicity adjustment. Positive correctness delta favors the candidate; positive Brier delta favors the reference. Precision compares training configurations including nonquantized dtype. Transfer is a fixed subsample and uses accuracy only.

| Study | Size | Cohort | Contrast | Metric | Difference | 95% interval |
|---|---|---|---|---|---:|---|
| precision | 4b | test | nf4 minus bf16 | correct | +0.8117 pp | [+0.2810, +1.3528] pp |
| precision | 4b | test | nf4 minus bf16 | brier | +0.0002 | [-0.0049, +0.0053] |
| transfer | 0.8b | distant | exact minus continued_sft | correct | -1.1719 pp | [-3.7760, +1.4323] pp |
| transfer | 0.8b | near | exact minus continued_sft | correct | +0.1302 pp | [-1.8229, +2.3438] pp |
| transfer | 0.8b | oos-none | exact minus continued_sft | correct | +3.1250 pp | [+1.8229, +4.4303] pp |
| transfer | 0.8b | unsupported-near-none | exact minus continued_sft | correct | +4.0000 pp | [+2.2222, +6.2222] pp |
| transfer | 4b | distant | exact minus continued_sft | correct | -1.3021 pp | [-3.7760, +1.1719] pp |
| transfer | 4b | near | exact minus continued_sft | correct | -0.7812 pp | [-2.4740, +0.7812] pp |
| transfer | 4b | oos-none | exact minus continued_sft | correct | -16.9271 pp | [-19.5312, -14.1927] pp |
| transfer | 4b | unsupported-near-none | exact minus continued_sft | correct | -5.1111 pp | [-8.8889, -1.5556] pp |
| transfer | 9b | distant | exact minus continued_sft | correct | -1.5625 pp | [-3.9062, +0.3906] pp |
| transfer | 9b | near | exact minus continued_sft | correct | -1.5625 pp | [-3.1250, -0.0000] pp |
| transfer | 9b | oos-none | exact minus continued_sft | correct | -2.0833 pp | [-4.2969, +0.2604] pp |
| transfer | 9b | unsupported-near-none | exact minus continued_sft | correct | +1.3333 pp | [-1.5556, +4.2222] pp |

Reproduce: `python scripts/analyze_extensions.py`. Use `--extract` only where private original predictions are present. Compact inputs contain IDs, groups, labels and numeric outcomes, never request text. The intervals do not establish a precision effect for 9B or RL, and transfer uncertainty does not represent all unfamiliar tasks.
