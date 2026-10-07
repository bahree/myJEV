# Open-Jev, LLM-as-Jev and System One review

Reviewed 6 October 2026. Recommendation: strengthen the untouched-backbone readout comparison and introduce a small, separately frozen verification diagnostic. Do not replace the completed studies, import every new dataset, or launch another size matrix.

## What the new paper changes

Li and Wagle's paper uses bracketed numeric suffix probabilities, cached prompt prefill and parallel suffix passes. Fine-tuning uses tree-factorized listwise loss with KL anchors. This is neither our joint answer/confidence RL nor necessarily one backbone forward. Tree-local training and globally normalized inference coincide only under a legal-continuation-mass condition.

Its 4B untouched readout reaches 81.4% on 231 public JevBench items; LoRA reaches 84.0%. These are public diagnostics, not sealed leaderboard results. Mixtures were revised after public error analysis. It uses one training seed; inference precision differs from comparison systems. Its Brier measures candidate distributions, unlike our separate selected-correctness Brier.

Training includes CLINC150 and datasets with noncommercial, research-only or unstated terms. Adopting that mixture would invalidate our untouched-CLINC task-transfer description. The authors filter lexical overlap but acknowledge incidental paragraph overlap elsewhere. Their calibration findings cannot establish universal native calibration. The current v2 title is *LLM-as-Jev*; v1 used *LLM2Jev*. No author-linked implementation repository was identified in the paper HTML. Section 2 describes Yinsongxu’s similarly named LLM2Jev as a separate community system with one yes/no prompt per option, contrasting it with the paper’s joint numeric-suffix method. That citation is not an author code release. [Full paper, including appendices](https://arxiv.org/html/2610.02076v2), [HF listing](https://huggingface.co/papers/2610.02076).

Our inference: readout quality is a missing alternative explanation for weak untouched-backbone performance. An improved zero-shot interface could narrow the apparent value of adaptation. That possibility does not erase the measured matched comparison under the original interface.

## The requested Open-Jev dataset is a different project

The original dataset is pinned at `c67699e13d0ae25e35b77165a4b6b079bedc8aba`. Its default redistributable configuration contains 79,116 training records and 113,568 total records. It is contained within the larger browser/drone configuration, so concatenating configurations duplicates examples. Each typed question is a row; related questions and variants share groups. The target is a reference distribution, not observed model confidence. `metadata_json` and `record_json` can contain privileged controls and labels; they must never be serialized wholesale into model input. [Dataset card](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/README.md).

The export omits 2,253 Wikispeedia records from the original mixture used by the released 2B/9B checkpoints. Reproducing the published model training exactly therefore requires separately obtained source material. The original export manifest enumerates seven configurations, while the current card lists twelve. Later additions have separate provenance; the original manifest alone is not a complete integrity inventory for the expanded release. [Export manifest](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/export-manifest.json), [reconstruction instructions](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/REPRODUCTION.md).

Original generated records have CC0 provenance, source code is MIT, and weights retain upstream terms. Some customer-control question descriptions have unverified upstream licensing. An overall dataset tag must not be treated as a license for every embedded source. [Third-party notices](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/THIRD_PARTY_NOTICES.md).

Implementation inspection found an independently scored candidate architecture: Qwen hidden states feed a scalar head initialized from the Yes-minus-No output embedding; candidate prompts form a batch. The training loss is reference-distribution cross-entropy plus Brier, followed by fitted temperature. There is no separate selected-correctness confidence policy in this implementation. A batched call over candidate pairs is not equivalent in cost or interactions to our single shared candidate-slate prompt. [Frozen model implementation](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/reproduce/source-code/jev/model.py), [training implementation](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/reproduce/source-code/jev/train.py).

The customer generator derives targets from explicit rules or defined latent worlds. Those are synthetic controls, not teacher opinions or human production outcomes. They can test recovery of known uncertainty but cannot establish that ambiguity in real customer requests has the same distribution. [Generator](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/reproduce/source-code/jev/case_customer.py).

## Newer material worth knowing about

The live public repository at `bd4118882f733574a3250a4b65fe4d884130c08b` has a 27B v1.1 release and broader controls. It also retains natural-support failures, calibration regressions and backend-equivalence failures. Its data, prior training and scale change together, so this is not a clean capacity comparison. The dataset's `Open-Jev-Dev` repository link returned 404 during this review; the public `Zefan-Cai/Open-Jev` repository and bundled export snapshot remain accessible. [Current public repository](https://github.com/Zefan-Cai/Open-Jev/tree/bd4118882f733574a3250a4b65fe4d884130c08b).

The newer v1.1 dataset adds WANLI-derived NLI examples and authored policy controls. WANLI derivatives retain CC BY 4.0; generated content retains separate terms. Connected seed/premise components define splits. Its controlled OOD is not a new-language or real-world-domain guarantee. It reports a lexical benchmark screen, which cannot rule out paraphrases or pretraining exposure. This release is also a filtered projection. [v1.1 dataset card](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1/blob/10ad6888333fa97f8c948192797bad3de3040802/README.md).

## What “System One” adds to the search

TypeSafe uses the term for typed probabilistic decision models and describes a proprietary architecture and RLCD. Its release claims are service measurements, not a published architecture specification we can reproduce. Schema validity does not imply semantic correctness. [TypeSafe introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

A concrete alternative is `shreyanbr/system-one-distilled`, a 70.8M DeBERTa cross-encoder distilled from Haiku answers with explicit calibration files. Its own card flags BANKING77 exposure in the base, a noncommercial ticket source and single-run limitations. It cannot serve as an uncontaminated BANKING comparison. Its zero-shot sibling supplies a useful architectural contrast, but published scores remain external author reports. [Distilled card](https://huggingface.co/shreyanbr/system-one-distilled), [zero-shot card](https://huggingface.co/shreyanbr/system-one-zeroshot).

## Bounded next work

1. Freeze a new readout-only protocol on the untouched 4B backbone. Compare our existing aliases with the paper's numeric suffix method. Verify token boundaries, cached/uncached parity and candidate-order behavior before evaluation. Use existing validation for any choice; label previously viewed test results exploratory. Report prefill, suffix work, latency and memory separately.
2. Select at most 200 grouped verification/none-of-the-options diagnostics from clearly licensed original controls. Pin rows before model inference. Retain soft targets and score distributions with proper scoring metrics; report hard accuracy only where a unique hard reference exists. Do not recast reference probability as the model's confidence.
3. Before any external-data training, audit exact text, normalized text, long n-grams and near-duplicates against every BANKING and CLINC partition. Audit generated variants by parent group. No exhaustive row-level overlap audit was performed here: source inventories alone do not prove disjointness. Keep a new adaptation track separate from the completed frozen studies.
4. Keep independent candidate encoders as an efficiency alternative. Compare full-request work at equal candidate counts, not just one batched API invocation. Reuse the bounded encoder control already being implemented rather than immediately adding another distilled model.

These are recommendations, not newly completed experiments. Retrieval hashes and revisions are saved in `results/openjev-review-20261006/sources.json`. No external checkpoint was executed and no training was started for this review.

## Implemented follow-up: a small readout probe

A subsequent frozen seven-case probe ran on the untouched BF16 4B backbone. Both the existing aliases and numeric suffix scoring answered all seven previously disclosed authored demos correctly. The new method includes closing-bracket probabilities, checks prefix/suffix token boundaries and sums full-vocabulary log probabilities without length normalization. Synthetic tests verify the causal shift and scoring arithmetic.

This correctness-first implementation repeats prompts in candidate batches rather than sharing a cached prefill. Prompt format and readout both change; no calibration, speed or architecture-superiority conclusion follows. The released architecture remains unchanged. See `results/numeric-readout-v1/report.md` and `scripts/numeric_readout_probe.py`. The broader untouched-readout comparison, cache parity and external-data overlap audit above remain future work.
