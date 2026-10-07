"""Regenerate the two bounded follow-up reports from saved metrics."""
import json
from pathlib import Path

root=Path('results/encoder-control-v1')
if (root/'summary.json').exists():
    j=json.loads((root/'summary.json').read_text());m=j['metrics'];bench=j['model_latency']
    fresh=root/'fresh-process-benchmark.json'
    if fresh.exists():
        isolated=json.loads(fresh.read_text());bench={**bench,**isolated,'tokens':isolated['input_tokens']}
    text=f'''# Adapted fixed-taxonomy encoder control

A single ModernBERT-base checkpoint was fully fine-tuned for the fixed 77 BANKING intents. This is a practical operational control, not a generalist candidate-conditioned replacement for myJEV, and not a matched architecture or objective comparison.

- Pinned model: `answerdotai/ModernBERT-base` at `{j['revision']}`; {j['parameters']:,} classifier parameters.
- Three epochs, {j['updates']} updates, {j['examples']:,} example exposures; batch32, AdamW learning rate2e-5, weight decay0.01, linear decay with10%warmup, seed11. Best validation-NLL checkpoint: epoch{j['selected_epoch']}.
- Existing grouped BANKING splits retained. Temperature fitted only on the reserved calibration split: {j['temperature']:.4f}. Thresholds also fitted only there. Full official test evaluated after model selection.
- FP32 parameters/optimizer states, BF16 autocast, SDPA attention on one A30. Train/validation/checkpoint elapsed: {j['training_seconds_including_validation_and_checkpoints']:.2f} seconds. Peak allocated training memory: {j['training_peak_allocated_bytes']/2**30:.2f}GiB.

| Confidence | Test accuracy | Macro-F1 | Correctness Brier | ECE |
|---|---:|---:|---:|---:|
'''
    for name in ['raw','temperature']:
        r=m[name];text+=f"| {name} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} | {r['correctness_brier']:.4f} | {r['ece']:.4f} |\n"
    text+=f'''
Model-only latency on one frozen short BANKING request: p50 **{bench['p50_seconds']*1000:.2f}ms**, p95 **{bench['p95_seconds']*1000:.2f}ms**, {bench['tokens']} input tokens, 10warmups and100measurements. Peak inference allocated memory: {bench['peak_allocated_bytes']/2**30:.2f}GiB. The saved full encoder/classifier artifact occupies {j['artifact_bytes']/2**20:.1f}MiB. These are not HTTP measurements or a worst-case512-token benchmark. The final report uses the separate fresh-process benchmark when present; the original training-process memory measurement is preserved but can include retained model/gradient references. The fixed taxonomy is encoded in the classifier head, so its input contains only the utterance. Qwen reads instructions and supplied candidate descriptions as well; equal numbers of output classes do not imply equal input work.

The Qwen main continued-supervision arms saw8,000 examples, while this control saw23,997. This one-seed run also has a different optimizer schedule, pretraining history and fixed-label output contract. Compare practical task quality and resource cost, not causal superiority. It cannot classify a newly supplied taxonomy without replacing/fitting its head. The frozen100-update pilot is retained as part of the training run, not additional exposure. No learning rate, seed or epoch was selected from test performance.

## Reproduce and provenance

Run `CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/.cache/huggingface" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/run_encoder_control.py` in a fresh output directory matching the script constants, after preparing the frozenBANKINGsplits. Original results intentionally refuse overwrite. `scripts/summarize_bounded_controls.py` regenerates this report from saved metrics.

The original model is Apache-2.0: [model card](https://huggingface.co/answerdotai/ModernBERT-base), [paper](https://arxiv.org/abs/2412.13663). BANKING provenance and licenses remain those of the existing dataset card. Local full-model artifacts are excluded fromGit; no new model release is claimed. Protocol/source hashes, startup compatibility failure, per-step losses, validation checkpoints, calibration/test predictions and timings are retained. One rejected constructor option was removed before any training; the execution amendment records this without changing the scientific budget.
'''
    # Keep prose readable; only compact code identifiers remain compact.
    for a,b in [('batch32','batch 32'),('rate2e-5','rate 2e-5'),('decay0.01','decay 0.01'),('with10%warmup','with 10% warmup'),('seed11','seed 11'),('epoch'+str(j['selected_epoch']),'epoch '+str(j['selected_epoch'])),('GiB',' GiB'),('MiB',' MiB'),('10warmups and100measurements','10 warmups and 100 measurements'),('saw8,000','saw 8,000'),('saw23,997','saw 23,997'),('worst-case512-token','worst-case 512-token'),('frozen100-update','frozen 100-update'),('frozenBANKINGsplits','frozen BANKING splits'),('fromGit','from Git')]:text=text.replace(a,b)
    (root/'report.md').write_text(text)
root=Path('results/sst2-transfer-v1')
if (root/'summary.json').exists():
    j=json.loads((root/'summary.json').read_text())
    text='''# Unrelated human-labelled sentiment transfer

All872 sentences in the official SST-2 validation split were held out solely for this myJEV transfer evaluation. This is the public validation split, not the official test set (whose labels are hidden). The three already-released supervised seed11 Qwen artifacts used the same frozen two-candidate request and released temperature settings. No SST-2 training, prompt tuning, calibration, threshold selection or checkpoint selection was performed.

| Public release | Accuracy | Macro-F1 | Confidence Brier | ECE | Coverage at original BANKING80% threshold | Accepted error |
|---|---:|---:|---:|---:|---:|---:|
'''
    for r in j['models']:
        point=r['operating_points']['coverage_0.8'];error=f"{point['error']:.2%}" if point['error'] is not None else 'undefined'
        text+=f"| {r['model']} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} | {r['correctness_brier']:.4f} | {r['ece']:.4f} | {point['coverage']:.2%} | {error} |\n"
    text+='''
This is one familiar binary movie-review benchmark, with neutral sentences removed. It is unrelated to banking intent routing but may have appeared in backbone pretraining. It cannot establish unseen-data novelty, universal generalization, broad reasoning or an RL benefit. A BANKING-calibrated threshold is deliberately evaluated under shift and does not guarantee matching coverage or error. Per-model JSON preserves group-bootstrap intervals and all frozen-threshold operating points.

## Provenance, labels and license

The [Stanford source](https://nlp.stanford.edu/sentiment/) and [dataset card](https://huggingface.co/datasets/stanfordnlp/sst2) describe movie-review phrases annotated by three human judges. Binary SST discards neutral examples. The Hub card lists the dataset license as unknown. Consequently, this public evidence includes source row IDs, text hashes, reference labels and predictions, but no review text. Reproduction downloads the pinned upstream dataset; it does not treat third-party text as MIT-licensed project material.

The exact dataset revision, fixed prompt, candidates, three model revisions and original BANKING calibration paths are in `frozen-plan.json`. `selection.jsonl` identifies every evaluated row. `data-preparation.json` pins the locally prepared request file without redistributing it. The dataset library downloaded its packaged splits, but the evaluator only reads the validation split; it does not train on any SST-2 partition.

Run `HF_HOME="$PWD/.cache/huggingface" .venv/bin/python scripts/evaluate_sst2_transfer.py --prepare`, then `CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/.cache/huggingface" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/evaluate_sst2_transfer.py`. Existing completed model results are reused. Generate this table with `python scripts/summarize_bounded_controls.py`.
'''
    text=text.replace('All872','All 872').replace('seed11','seed 11').replace('BANKING80%','BANKING 80%')
    (root/'report.md').write_text(text)
