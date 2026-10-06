"""Apply original BANKING calibration temperatures/thresholds to saved diagnostics.

No additional fitting, model calls, candidate changes, or transfer tuning.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from myjev.data import read_jsonl
from myjev.metrics import summarize

ROOT=Path('results/generalization-v1')
plan={'purpose':'Supplement native confidence with the original-plan temperature control for continued SFT; analysis only, no transfer tuning.',
      'sources':{},'selection_unchanged':True,'calibration':'Original BANKING77 calibration only; preserve temperature and operating thresholds.'}
for size in ('0.8b','4b','9b'):
    for seed in (11,22,33):
        path=Path(f'results/longer-v1/{size}/main/seed-{seed}/continued_sft/posthoc/temperature-metrics.json')
        plan['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
frozen=ROOT/'temperature-control-plan.json'
if frozen.exists() and json.loads(frozen.read_text())!=plan:raise ValueError('Temperature sources changed')
frozen.write_text(json.dumps(plan,indent=2)+'\n')
count=0
for size in ('0.8b','4b','9b'):
    for seed in (11,22,33):
        cal=Path(f'results/longer-v1/{size}/main/seed-{seed}/continued_sft/posthoc/temperature-metrics.json')
        source=json.loads(cal.read_text());t=source['temperature']
        if not np.isfinite(t) or t<=0:raise ValueError('Invalid frozen temperature')
        thresholds={k:v['threshold'] for k,v in source['operating_points'].items()}
        parent=ROOT/size/f'seed-{seed}'/'continued_sft'
        for kind in ('transfer','robustness'):
            if not (parent/f'{kind}-complete.json').exists():continue
            for path in sorted((parent/kind).glob('*-predictions.jsonl')):
                name=path.name.removesuffix('-predictions.jsonl')
                dest=parent/'temperature-controls'/kind/f'{name}-metrics.json'
                if dest.exists():continue
                rows=read_jsonl(path)
                if not rows:continue
                for row in rows:
                    z=np.asarray(row['selection_logits'],dtype=float)/t
                    p=np.exp(z-z.max());p/=p.sum()
                    scores=dict(zip(row['selection_scores'],p.tolist(),strict=True))
                    # Positive temperature preserves the original logit argmax.
                    if row['selected_id']!=list(scores)[int(z.argmax())]:raise ValueError('Selection differs from original logits')
                    row.update(selection_scores=scores,confidence=max(scores.values()),confidence_mode='temperature')
                metrics=summarize(rows,thresholds)
                metrics.update(source_predictions=str(path),source_predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               temperature=t,threshold_source=str(cal),threshold_source_sha256=hashlib.sha256(cal.read_bytes()).hexdigest(),
                               confidence_mode='temperature',selection_unchanged=True,
                               scope='Saved diagnostic predictions; original BANKING temperature and thresholds unchanged. No transfer fitting.')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(metrics,indent=2)+'\n');count+=1
print(json.dumps({'new_temperature_cohorts':count}))
