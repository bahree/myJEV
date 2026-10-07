# Unrelated human-labelled sentiment transfer

All 872 sentences in the official SST-2 validation split were held out solely for this myJEV transfer evaluation. This is the public validation split, not the official test set (whose labels are hidden). The three already-released supervised seed 11 Qwen artifacts used the same frozen two-candidate request and released temperature settings. No SST-2 training, prompt tuning, calibration, threshold selection or checkpoint selection was performed.

| Public release | Accuracy | Macro-F1 | Confidence Brier | ECE | Coverage at original BANKING 80% threshold | Accepted error |
|---|---:|---:|---:|---:|---:|---:|
| bahree/myJEV-0.8B | 84.86% | 0.8483 | 0.1183 | 0.0595 | 80.73% | 9.80% |
| bahree/myJEV-4B | 94.27% | 0.9427 | 0.0496 | 0.0341 | 93.12% | 4.06% |
| bahree/myJEV-9B | 95.30% | 0.9530 | 0.0404 | 0.0186 | 95.07% | 3.38% |

This is one familiar binary movie-review benchmark, with neutral sentences removed. It is unrelated to banking intent routing but may have appeared in backbone pretraining. It cannot establish unseen-data novelty, universal generalization, broad reasoning or an RL benefit. A BANKING-calibrated threshold is deliberately evaluated under shift and does not guarantee matching coverage or error. Per-model JSON preserves group-bootstrap intervals and all frozen-threshold operating points.

## Provenance, labels and license

The [Stanford source](https://nlp.stanford.edu/sentiment/) and [dataset card](https://huggingface.co/datasets/stanfordnlp/sst2) describe movie-review phrases annotated by three human judges. Binary SST discards neutral examples. The Hub card lists the dataset license as unknown. Consequently, this public evidence includes source row IDs, text hashes, reference labels and predictions, but no review text. Reproduction downloads the pinned upstream dataset; it does not treat third-party text as MIT-licensed project material.

The exact dataset revision, fixed prompt, candidates, three model revisions and original BANKING calibration paths are in `frozen-plan.json`. `selection.jsonl` identifies every evaluated row. `data-preparation.json` pins the locally prepared request file without redistributing it. The dataset library downloaded its packaged splits, but the evaluator only reads the validation split; it does not train on any SST-2 partition.

Run `HF_HOME="$PWD/.cache/huggingface" .venv/bin/python scripts/evaluate_sst2_transfer.py --prepare`, then `CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/.cache/huggingface" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/evaluate_sst2_transfer.py`. Existing completed model results are reused. Generate this table with `python scripts/summarize_bounded_controls.py`.
