"""Summarize the frozen diagnostic extension, explicitly retaining partial status."""
import hashlib
import json
from pathlib import Path

ROOT=Path('results/generalization-v1')
plan=json.loads((ROOT/'frozen-plan.json').read_text())
rows=[];sources={};completed=0
for size in plan['sizes']:
    for seed in plan['seeds']:
        for method in plan['methods']:
            root=ROOT/size/f'seed-{seed}'/method
            for kind in ('transfer','robustness'):
                if not (root/f'{kind}-complete.json').exists():continue
                completed+=1
                paths=sorted((root/kind).glob('*-metrics.json')) if kind=='transfer' else [root/kind/'report.json']
                for path in paths:
                    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    report=json.loads(path.read_text())
                    cohorts={path.name.removesuffix('-metrics.json'):report} if kind=='transfer' else report['cohorts']
                    for name,m in cohorts.items():
                        rows.append(dict(size=size,seed=seed,method=method,kind=kind,cohort=name,
                            n=m.get('n',0),requested=m.get('requested',m.get('n',0)),rejected=m.get('rejected',0),
                            accuracy=m.get('accuracy'),correctness_brier=m.get('correctness_brier'),
                            agreement=m.get('selection_agreement_with_original'),
                            operating_points=m.get('operating_points',{})))
# Supplemental temperature control reuses saved predictions and BANKING thresholds.
for path in sorted(ROOT.glob('*/seed-*/continued_sft/temperature-controls/*/*-metrics.json')):
    rel=path.relative_to(ROOT).parts;size,seed=rel[0],int(rel[1].removeprefix('seed-'))
    m=json.loads(path.read_text());sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(dict(size=size,seed=seed,method='continued_sft_temperature',kind=rel[4],cohort=path.name.removesuffix('-metrics.json'),
                     n=m['n'],requested=m['n'],rejected=0,accuracy=m['accuracy'],correctness_brier=m['correctness_brier'],agreement=None,operating_points=m['operating_points']))
summary=dict(completed_jobs=completed,planned_jobs=36,complete=completed==36,rows=rows,sources=sources,
             scope='Frozen subsample diagnostics with native scalar/policy confidence and unchanged BANKING77 thresholds. Near/distant use CLINC descriptions. None-option classification and confidence-based deferral are separate tasks. No transfer tuning.',
             limitations=plan['limitations'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Generalization extension','',f'Completed jobs: **{completed}/36**. '+('Final frozen diagnostic matrix.' if completed==36 else 'Partial progress only; no final cross-size ranking.'),'',summary['scope'],'',
       '| Size | Seed | Method | Task | Cohort | N | Accuracy | Coverage at BANKING 80% threshold | Accepted error | Rejected |',
       '|---|---:|---|---|---|---:|---:|---:|---:|---:|']
def pct(v):return f'{100*v:.2f}%' if v is not None else 'n/a'
for r in rows:
    op=r['operating_points'].get('coverage_0.8',{})
    lines.append(f"| {r['size']} | {r['seed']} | {r['method']} | {r['kind']} | {r['cohort']} | {r['n']} | {pct(r['accuracy'])} | {pct(op.get('coverage'))} | {pct(op.get('error'))} | {r['rejected']} |")
lines+=['','For deferral-only cohorts the true answer is absent, so selection accuracy is zero by construction. Lower acceptance is desirable there; this must not be mixed with explicit none-option classification accuracy. Fixed thresholds can yield very different coverage after distribution shift.','',
        'Primary-matrix limits: '+ '; '.join(plan['limitations'])+'. Supplemental temperature rows apply the original BANKING calibration values to saved predictions; see temperature-control-plan.json. They do not change the frozen primary matrix or fit on transfer outcomes.', '',
        'Regenerate from recorded metrics: `python scripts/summarize_generalization.py`. Per-run predictions and failure logs are retained privately; aggregate metrics and source hashes are public evidence.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'completed_jobs':completed,'planned_jobs':36,'metric_rows':len(rows)}))
