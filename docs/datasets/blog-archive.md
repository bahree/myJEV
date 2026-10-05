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
| Training | 96 | Provisional, unused for adaptation |
| Validation | 16 | Provisional |
| Calibration | 16 | Provisional |
| Test | 32 | Provisional; evaluation not run |

Preparation extracted model-visible text. Six posts use explicitly materialized 2,048-token prefix excerpts; the others use the extracted full text. Labels refer to that supplied text, not unseen images, linked pages or omitted content. Related posts, excerpts and variants must remain grouped; the grouping and rubric need review before final evaluation groups are frozen.

## What has actually been completed

The 40 development posts received 120 LLM-generated judgments across the three rubrics. Of these, 95 have a proposed label without an uncertainty flag and 25 are flagged uncertain. These are unreviewed machine annotations, not human ground truth or measured myJEV accuracy. A blinded audit subset has been prepared; a human audit and independent judge cross-check remain pending.

No myJEV archive fine-tuning, archive transfer benchmark or forgetting comparison has been completed. The current 0.8B/4B/9B training batch uses BANKING77, not these blog posts. The blog archive is a prepared example and annotation pilot for the later study.

## Next evaluation and limitations

Review the rubrics and grouping, audit labels, expand annotation and freeze evaluation groups before adaptation. Compare unadapted and adapted artifacts on the same held-out posts and re-evaluate BANKING77/CLINC150 for forgetting. Report machine-label agreement separately from human-audited correctness, with uncertainty grouped by related posts.

The sample is small, subjective and from one author, with historical imbalance and possible pretraining exposure. It cannot establish a low production error rate or broad generalist capability. The author supplied the archive for this study; no broader redistribution license is inferred. Post text, per-post labels and private annotations are not included in this repository. This card publishes the study design and aggregate status only.
