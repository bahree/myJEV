# Decision-1 and local myJEV: one frozen diagnostic

Each completed model uses the same 1,000 calibration requests, 64 test requests under six views, and seven authored demos. The sample is diagnostic and does not reproduce Microsoft's benchmark. Native selection probabilities and an equally fitted post-hoc temperature are separate views. Vendor confidence is retained in the records without assuming it estimates selected-answer correctness.

## Original request quality

| Model | Confidence | Accuracy | Correctness Brier | ECE | Test coverage at 80% calibration threshold | Accepted-case error |
|---|---|---:|---:|---:|---:|---:|
| decision-1 | native-selection | 92.19% | 0.0586 | 0.0910 | 85.94% | 3.64% |
| decision-1 | temperature | 92.19% | 0.0569 | 0.0798 | 85.94% | 3.64% |
| myjev-4b | native-selection | 93.75% | 0.0348 | 0.1029 | 76.56% | 0.00% |
| myjev-4b | temperature | 93.75% | 0.0348 | 0.1029 | 76.56% | 0.00% |
| myjev-4b-rl | native-selection | 95.31% | 0.0420 | 0.0442 | 84.38% | 3.70% |
| myjev-4b-rl | temperature | 95.31% | 0.0364 | 0.0561 | 85.94% | 3.64% |

Accuracy, macro-F1, Brier metrics, correctness AUROC, 15-bin ECE, group intervals and accepted counts are in `summary.json`. Thresholds use calibration labels only. Their observed error targets provide no deployment guarantee. Temperature is fitted to log probabilities after clipping zeros at 1e-12, and does not change the returned choice. No model or setting is selected using these test outcomes.

## Changes under each perturbation

Each row uses the original model settings and threshold. Repeated views share test requests and are never pooled as independent samples. Renamed IDs are mapped back before comparison. The instruction paraphrase is authored and has no independent equivalence audit.

| Model | View | Changed answers / 64 | Accuracy | Accepted-case error at original 80% threshold |
|---|---|---:|---:|---:|
| decision-1 | original | 0 | 92.19% | 3.64% |
| decision-1 | reverse | 1 | 90.62% | 3.64% |
| decision-1 | shuffle | 0 | 92.19% | 3.64% |
| decision-1 | renamed-ids | 5 | 90.62% | 5.26% |
| decision-1 | whitespace | 1 | 93.75% | 3.51% |
| decision-1 | instruction-paraphrase | 1 | 90.62% | 5.00% |
| myjev-4b | original | 0 | 93.75% | 0.00% |
| myjev-4b | reverse | 3 | 93.75% | 0.00% |
| myjev-4b | shuffle | 3 | 96.88% | 0.00% |
| myjev-4b | renamed-ids | 0 | 93.75% | 0.00% |
| myjev-4b | whitespace | 2 | 96.88% | 0.00% |
| myjev-4b | instruction-paraphrase | 1 | 95.31% | 0.00% |
| myjev-4b-rl | original | 0 | 95.31% | 3.70% |
| myjev-4b-rl | reverse | 1 | 93.75% | 5.77% |
| myjev-4b-rl | shuffle | 1 | 96.88% | 1.89% |
| myjev-4b-rl | renamed-ids | 0 | 95.31% | 3.70% |
| myjev-4b-rl | whitespace | 1 | 93.75% | 4.00% |
| myjev-4b-rl | instruction-paraphrase | 0 | 95.31% | 3.70% |

## Sequential latency and billed usage

Latency uses 64 original test requests after the five warmups and calibration calls. Local scoring includes tokenization and conversion, with GPU synchronization; hosted scoring also includes network and provider queue time. This is a system-level observation across different hardware and request rendering. No matched-hardware speed claim is made.

| Model | p50 ms | p95 ms | Reported cost USD | Calls with cost / all calls |
|---|---:|---:|---:|---:|
| decision-1 | 380.34 | 527.43 | 0.047764 | 1396 / 1396 |
| myjev-4b | 214.66 | 216.76 | not returned | 0 / 1396 |
| myjev-4b-rl | 210.08 | 211.50 | not returned | 0 / 1396 |

Local GPU cost is not estimated here. Missing provider usage is reported as unknown, never zero. A reported total covers only calls returning cost metadata and may exclude failed attempts. Model/service identifiers and observation timestamps are retained; an unchanged API name does not establish unchanged hosted weights.

## All seven authored demos

| Model | Request | Expected | Selected | Selected-option probability |
|---|---|---|---|---:|
| decision-1 | billing | billing | billing | 0.9971 |
| decision-1 | technical | technical | technical | 0.9128 |
| decision-1 | none-of-these | other | other | 0.9938 |
| decision-1 | refund-day-13 | approve | approve | 0.9991 |
| decision-1 | refund-day-14 | deny | deny | 0.9993 |
| decision-1 | quoted-instruction | billing | billing | 0.9975 |
| decision-1 | synthetic-blog-format | tutorial | tutorial | 0.9992 |
| myjev-4b | billing | billing | billing | 0.9834 |
| myjev-4b | technical | technical | technical | 0.9669 |
| myjev-4b | none-of-these | other | other | 0.9828 |
| myjev-4b | refund-day-13 | approve | approve | 0.9672 |
| myjev-4b | refund-day-14 | deny | deny | 0.9835 |
| myjev-4b | quoted-instruction | billing | billing | 0.9383 |
| myjev-4b | synthetic-blog-format | tutorial | tutorial | 0.9975 |
| myjev-4b-rl | billing | billing | billing | 1.0000 |
| myjev-4b-rl | technical | technical | technical | 1.0000 |
| myjev-4b-rl | none-of-these | other | other | 1.0000 |
| myjev-4b-rl | refund-day-13 | approve | approve | 1.0000 |
| myjev-4b-rl | refund-day-14 | deny | deny | 1.0000 |
| myjev-4b-rl | quoted-instruction | billing | billing | 1.0000 |
| myjev-4b-rl | synthetic-blog-format | tutorial | tutorial | 1.0000 |

These examples are unselected diagnostics outside BANKING77. Their confidence has no demonstrated calibration on these new tasks.

## Replay

Run `python scripts/summarize_decision_comparison.py` from the repository root. No model download, credentials or GPU is required. The [comparison guide](../../docs/decision-1.md) describes the protocol, inference commands, cost assumptions and external sources.
