# Archive majority-per-rubric control

Post-hoc exploratory control, not preregistered. Select each rubric majority using training labels only; lexical label-ID order resolves training ties. Confidence is selected-class correctness frequency on calibration labels only. Test labels are used only for the final metrics, not selection, confidence or thresholds. Agreement is against unreviewed machine annotations, not human ground truth. Multiple rubric decisions from one post are correlated; no IID or population intervals are claimed. This control tests whether label imbalance is a plausible explanation of adaptation performance, not a causal decomposition of any gain.

| Rubric | Train N | Calibration N | Selected label | Calibration confidence | Test N | Reference agreement | Correctness Brier |
|---|---:|---:|---|---:|---:|---:|---:|
| claim_support | 83 | 13 | 0 | 0.3846 | 27 | 18.52% | 0.1907 |
| format | 96 | 16 | 0 | 0.1875 | 32 | 25.00% | 0.1914 |
| instructional_completeness | 92 | 15 | 0 | 0.4667 | 31 | 51.61% | 0.2522 |

Overall: 90 test decisions, 32.22% reference agreement, correctness Brier 0.2121.

Reproduce where private prepared data are available: `python scripts/archive_majority_control.py`. Public output contains rubric-level aggregate counts and file hashes only; no archive text, request IDs or individual annotations.
