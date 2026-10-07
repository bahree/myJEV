# Blog archive: post classification and adaptation study

## Why this dataset

Amit Bahree's [blog archive](https://blog.desigeek.com/) provides a practical example of classifying documents under new rubrics, beyond banking intent routing. The intended study asks whether a decision model can identify a post's format and assess properties of the supplied text, then whether domain adaptation improves those decisions without degrading public-task performance.

## Tasks

| Rubric | Candidate labels | Interpretation |
|---|---|---|
| Post format | Tutorial, reference, opinion, announcement, narrative, mixed/unclear | What kind of post is this? |
| Instructional completeness | No procedure, actionable with missing prerequisites, self-contained procedure, insufficient supplied text | Does the supplied text contain a usable procedure? |
| Claim support | No checkable central claim, explicit support, partial support, unsupported within text | Does the supplied text support its central checkable claim? |

Claim support does not establish external factual truth. None of these labels measures vague overall writing quality, author prestige, age or length.

## Provenance, preparation and partitions

The owner supplied the [RSS feed](https://blog.desigeek.com/index.xml). The October 4, 2026 snapshot contained 1,332 entries. Sampling across chronological quintiles produced a 200-post pilot in 199 provisional related-post groups.

| Partition | Posts | Status |
|---|---:|---|
| Rubric development | 40 | First machine-annotation pilot complete; human review pending |
| Training | 96 | Frozen provisional groups; adaptation completed |
| Validation | 16 | Provisional |
| Calibration | 16 | Provisional |
| Test | 32 | Frozen provisional groups; 90 usable rubric decisions evaluated |

Preparation extracted model-visible text. Six posts use explicitly materialized 2,048-token prefix excerpts; the others use the extracted full text. Labels refer to that supplied text, not unseen images, linked pages or omitted content. Related posts, excerpts and variants must remain grouped; the provisional groups were frozen for the exploratory experiment, but semantic grouping and rubric reliability still need human review.

## Completed exploratory machine-reference study

The earlier development annotation pilot produced 120 session judgments, 95 without uncertainty flags and 25 uncertain. A separate local Qwen3.5-9B NF4 judge initially passed only 50/120 strict output checks, primarily because quote normalization failed exact-substring validation. Returning IDs for lossless source spans repaired that interface: 118/120 development outputs passed, above the unchanged 80% engineering gate. Structural validity does not establish correct labels. Agreement with the earlier judge was only 50/92 comparable labelled decisions (54.35%), including 33.33% on claim support.

The frozen source-span protocol then processed the remaining 480 requests. Excluding 29 invalid, uncertain or inapplicable annotations left 271 training, 46 validation, 44 calibration and 90 test decisions. These are rubric decisions, not independent posts. Existing provisional groups were held fixed across adaptation; human semantic grouping review remains outstanding.

One 4B continued-supervision checkpoint completed a 100-update fit pilot and a separate 400-update adaptation stage. Its native scalar-confidence evaluation changed as follows:

| Condition | Test N | Machine-reference agreement | Correctness Brier |
|---|---:|---:|---:|
| Unadapted | 90 | 42.22% | 0.5640 |
| Adapted | 90 | 61.11% | 0.2587 |
| Majority per rubric | 90 | 32.22% | 0.2121 |

The majority control selects labels from training counts and sets confidence from calibration-only correctness frequency. It is a post-hoc exploratory control. Its lower Brier despite weaker selection demonstrates why Brier alone cannot establish a useful classifier or an adaptation benefit in confidence.

Per-rubric agreement changed from 43.75% to 56.25% for format (32 decisions), 29.03% to 67.74% for instructional completeness (31), and 55.56% to 59.26% for claim support (27). Numeric label IDs have different meanings across rubrics, so overall macro-F1 is not interpreted as a combined classification metric. Reference agreement and correctness Brier remain valid aggregate outcome summaries.

## Forgetting and release implications

BANKING77 official-test accuracy changed from 89.94% to 90.10% (3,080 examples). On frozen CLINC subsets, nearby/distant intent accuracy changed from 89.45%/81.25% to 90.62%/83.20%, but explicit none-option accuracy fell from 82.42% to 71.48% for OOS and from 47.33% to 34.67% for nearby unsupported intents. These are descriptive one-seed findings, not a general claim that adaptation preserves capability.

Deferral-only cohorts omit a correct candidate and therefore have zero selection accuracy by construction. Their saved acceptance metrics use original thresholds and separately BANKING-recalibrated thresholds; no CLINC tuning occurs. Published Qwen releases remain BANKING-trained, not archive-adapted.

Evidence: [adaptation report](../../results/archive-machine-v2/report.md), [majority control](../../results/archive-machine-v2/majority-control.md), and [CLINC forgetting](../../results/archive-machine-v2/clinc-forgetting-report.md). Source hashes, exclusion counts and group-aware metric intervals accompany the reports. The experiment does not claim convergence or human-audited correctness.

## Remaining limitations and rights

Teacher and student share the Qwen family, so increased agreement can reflect shared errors. A blinded human audit and semantic grouping review remain separate work. This small subjective sample from one author, with historical imbalance and possible pretraining exposure, cannot establish generalist capability or a low production error rate.

The owner supplied the archive for this study; no broader redistribution license is inferred. Post text, individual labels and private annotations remain outside the public repository. This card and the public reports contain design, aggregate results and provenance only.
