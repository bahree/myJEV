# Test-sample uncertainty and training-seed spread

The same test examples were evaluated for each seed. The group-bootstrap intervals below hold the three trained runs fixed; they do not include uncertainty over future training seeds. Seed deltas and their sample SD show that separate axis. Three runs are insufficient for a dependable distributional model of training variability. No new training was performed.

| Study / cohort | Size | Contrast | Metric | Seed 11 | Seed 22 | Seed 33 | Mean | Seed SD | Conditional group 95% interval |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| main / BANKING77 | 0.8b | exact/deployed minus continued_sft/deployed | correct | -1.7532 pp | -2.8571 pp | -0.0974 pp | -1.5693 pp | 1.3890 pp | [-2.1970, -0.9524] pp |
| main / BANKING77 | 0.8b | sampled/deployed minus continued_sft/deployed | correct | -2.6623 pp | -2.5325 pp | -2.0130 pp | -2.4026 pp | 0.3436 pp | [-3.0392, -1.7530] pp |
| main / BANKING77 | 0.8b | sampled/deployed minus exact/deployed | correct | -0.9091 pp | +0.3247 pp | -1.9156 pp | -0.8333 pp | 1.1220 pp | [-1.3204, -0.3573] pp |
| main / BANKING77 | 0.8b | exact/deployed minus continued_sft/temperature | brier | +0.0205 | +0.0376 | +0.0163 | +0.0248 | 0.0113 | [+0.0201, +0.0297] |
| main / BANKING77 | 0.8b | sampled/deployed minus continued_sft/temperature | brier | +0.0285 | +0.0352 | +0.0186 | +0.0275 | 0.0083 | [+0.0226, +0.0323] |
| main / BANKING77 | 4b | exact/deployed minus continued_sft/deployed | correct | +0.6169 pp | -0.1948 pp | +2.6948 pp | +1.0390 pp | 1.4903 pp | [+0.5628, +1.4940] pp |
| main / BANKING77 | 4b | sampled/deployed minus continued_sft/deployed | correct | -0.5519 pp | -0.5844 pp | -1.9481 pp | -1.0281 pp | 0.7968 pp | [-1.5801, -0.5088] pp |
| main / BANKING77 | 4b | sampled/deployed minus exact/deployed | correct | -1.1688 pp | -0.3896 pp | -4.6429 pp | -2.0671 pp | 2.2644 pp | [-2.5010, -1.6545] pp |
| main / BANKING77 | 4b | exact/deployed minus continued_sft/temperature | brier | +0.0093 | +0.0154 | -0.0013 | +0.0078 | 0.0085 | [+0.0047, +0.0109] |
| main / BANKING77 | 4b | sampled/deployed minus continued_sft/temperature | brier | +0.0197 | +0.0164 | +0.0439 | +0.0267 | 0.0150 | [+0.0231, +0.0302] |
| main / BANKING77 | 9b | exact/deployed minus continued_sft/deployed | correct | -0.2922 pp | +0.2273 pp | +0.6494 pp | +0.1948 pp | 0.4716 pp | [-0.3575, +0.7143] pp |
| main / BANKING77 | 9b | sampled/deployed minus continued_sft/deployed | correct | -0.5844 pp | -0.3571 pp | -0.8766 pp | -0.6061 pp | 0.2604 pp | [-1.1692, -0.0650] pp |
| main / BANKING77 | 9b | sampled/deployed minus exact/deployed | correct | -0.2922 pp | -0.5844 pp | -1.5260 pp | -0.8009 pp | 0.6447 pp | [-1.1909, -0.4006] pp |
| main / BANKING77 | 9b | exact/deployed minus continued_sft/temperature | brier | +0.0125 | +0.0374 | +0.0100 | +0.0200 | 0.0152 | [+0.0152, +0.0248] |
| main / BANKING77 | 9b | sampled/deployed minus continued_sft/temperature | brier | +0.0102 | +0.0482 | +0.0286 | +0.0290 | 0.0190 | [+0.0246, +0.0338] |
| precision / test | 4b | nf4 minus bf16 | correct | -1.2013 pp | +0.7792 pp | +2.8571 pp | +0.8117 pp | 2.0294 pp | [+0.2810, +1.3528] pp |
| precision / test | 4b | nf4 minus bf16 | brier | +0.0162 | -0.0051 | -0.0107 | +0.0002 | 0.0142 | [-0.0049, +0.0053] |
| transfer / distant | 0.8b | exact minus continued_sft | correct | -2.3438 pp | -1.1719 pp | +0.0000 pp | -1.1719 pp | 1.1719 pp | [-3.7760, +1.4323] pp |
| transfer / near | 0.8b | exact minus continued_sft | correct | -1.5625 pp | +1.5625 pp | +0.3906 pp | +0.1302 pp | 1.5787 pp | [-1.8229, +2.3438] pp |
| transfer / oos-none | 0.8b | exact minus continued_sft | correct | +0.0000 pp | +9.3750 pp | +0.0000 pp | +3.1250 pp | 5.4127 pp | [+1.8229, +4.4303] pp |
| transfer / unsupported-near-none | 0.8b | exact minus continued_sft | correct | +0.0000 pp | +10.6667 pp | +1.3333 pp | +4.0000 pp | 5.8119 pp | [+2.2222, +6.2222] pp |
| transfer / distant | 4b | exact minus continued_sft | correct | +0.3906 pp | -1.5625 pp | -2.7344 pp | -1.3021 pp | 1.5787 pp | [-3.7760, +1.1719] pp |
| transfer / near | 4b | exact minus continued_sft | correct | -3.5156 pp | +2.3438 pp | -1.1719 pp | -0.7812 pp | 2.9492 pp | [-2.4740, +0.7812] pp |
| transfer / oos-none | 4b | exact minus continued_sft | correct | -7.8125 pp | -51.1719 pp | +8.2031 pp | -16.9271 pp | 30.7190 pp | [-19.5312, -14.1927] pp |
| transfer / unsupported-near-none | 4b | exact minus continued_sft | correct | -4.0000 pp | -17.3333 pp | +6.0000 pp | -5.1111 pp | 11.7063 pp | [-8.8889, -1.5556] pp |
| transfer / distant | 9b | exact minus continued_sft | correct | -0.7812 pp | +0.7812 pp | -4.6875 pp | -1.5625 pp | 2.8168 pp | [-3.9062, +0.3906] pp |
| transfer / near | 9b | exact minus continued_sft | correct | -1.5625 pp | -3.1250 pp | +0.0000 pp | -1.5625 pp | 1.5625 pp | [-3.1250, -0.0000] pp |
| transfer / oos-none | 9b | exact minus continued_sft | correct | +9.7656 pp | -22.2656 pp | +6.2500 pp | -2.0833 pp | 17.5665 pp | [-4.2969, +0.2604] pp |
| transfer / unsupported-near-none | 9b | exact minus continued_sft | correct | +4.0000 pp | -26.0000 pp | +26.0000 pp | +1.3333 pp | 26.1024 pp | [-1.5556, +4.2222] pp |

Exact RL exceeds continued supervision at 4B in two of three seeds and in the mean. Its conditional test-group interval excludes zero, while the observed seed deltas include a negative value. At 0.8B continued supervision leads exact RL in all three seeds. These statements describe the observed runs, not future-run guarantees. Original Brier contrasts use different confidence sources; the matched post-hoc follow-up is reported separately.

Reproduce: `python scripts/review_seed_statistics.py`.
