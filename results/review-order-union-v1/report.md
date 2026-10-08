# Requests affected by any recorded candidate permutation

Exploratory descriptive union over the three recorded per-request permutations, relative to the original order. Each of 3080 requests is counted once per released seed-11 model. No new inference, training, calibration fit, uncertainty interval, or claim about arbitrary orders.

The original report measures each permutation separately. This table takes their union: a request that changed under two orders still counts once. The fraction depends on these three specific permutations and is not the probability of a change under every possible order.

| Size | Release | Changed under orders 101 / 202 / 303 | Changed under any of three | Fraction of 3,080 requests |
|---|---|---|---:|---:|
| 0.8b | continued_sft | 243 / 248 / 253 | 413 | 13.41% |
| 0.8b | exact | 267 / 283 / 299 | 478 | 15.52% |
| 4b | continued_sft | 123 / 130 / 123 | 217 | 7.05% |
| 4b | exact | 112 / 113 / 116 | 199 | 6.46% |
| 9b | continued_sft | 115 / 112 / 125 | 204 | 6.62% |
| 9b | exact | 152 / 153 / 146 | 240 | 7.79% |

Regenerate with `python scripts/summarize_order_union.py`. The summary hashes the original protocol, text-free calibration inputs, recorded predictions and per-permutation metrics. Per-permutation disagreements are checked against their original metrics before taking the union. No original result files are rewritten.
