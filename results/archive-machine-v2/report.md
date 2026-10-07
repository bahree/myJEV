# Exploratory machine-label archive adaptation

Archive accuracy means agreement with an unreviewed local machine teacher; BANKING accuracy uses its existing dataset labels.

| Condition | N | Reference agreement / accuracy | Correctness Brier |
|---|---:|---:|---:|
| unadapted | 90 | 42.22% | 0.5640 |
| adapted | 90 | 61.11% | 0.2587 |
| banking_before | 3080 | 89.94% | 0.0939 |
| banking_after | 3080 | 90.10% | 0.0843 |

Archive confidence and thresholds are evaluated against machine labels, with group-aware intervals retained in JSON. Human label reliability, related-post grouping and production error are not established. Teacher/student family overlap can inflate agreement. The fixed one-seed adaptation budget does not establish convergence.

Inspect `data-manifest.json` for excluded annotations and split counts, and `summary.json` for per-rubric metrics, exact source hashes and forgetting. All raw annotations stay private.
