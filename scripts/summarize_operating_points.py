"""Report calibration-selected operating points without retuning thresholds."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('results/longer-v1')
OUT=ROOT/'operational'
OUT.mkdir(exist_ok=True)
rows=[];sources={}
for size in ('0.8b','4b','9b'):
    for seed in (11,22,33):
        for method,confidence,suffix in [('continued_sft','temperature','posthoc/temperature-metrics.json'),('exact','policy','evaluation/metrics.json'),('sampled','policy','evaluation/metrics.json')]:
            path=ROOT/size/'main'/f'seed-{seed}'/method/suffix
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            metrics=json.loads(path.read_text())
            for target,point in metrics['operating_points'].items():
                rows.append(dict(size=size,seed=seed,method=method,confidence=confidence,target=target,test_n=metrics['n'],**point))
report={'scope':'Full official BANKING77 test, thresholds fixed using each run calibration data. Individual seeds shown; no pooling of repeated test examples as independent observations.',
        'uncertainty':'error_upper_95 is the one-sided 95% Clopper-Pearson bound conditional on a fixed threshold, under independent Bernoulli sampling. error_cluster_ci95 is the existing group bootstrap interval. Neither accounts for repeated comparison selection or a new deployment distribution.',
        'warning':'Zero observed errors or a degenerate bootstrap interval do not establish zero risk. A calibration empirical-error target is not a test guarantee.',
        'sources':sources,'rows':rows}
(OUT/'operating-points.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Operational confidence after longer training','',report['scope'],'',report['uncertainty'],'',report['warning'],'',
       'Each cell below spans the three observed seeds. Coverage is the fraction accepted, error is among accepted examples, and the upper bound is a per-seed bound, not an interval for a pooled three-seed estimator. Threshold targets were set on calibration only.','',
       '| Size | Method/confidence | Calibration target | Test coverage range | Accepted-error range | One-sided upper bound range |',
       '|---|---|---|---:|---:|---:|']
for size in ('0.8b','4b','9b'):
    for method in ('continued_sft','exact','sampled'):
        for target in ('coverage_0.8','empirical_error_0.01','empirical_error_0.05'):
            selected=[r for r in rows if r['size']==size and r['method']==method and r['target']==target]
            def span(key):
                vals=[r[key]*100 for r in selected if r[key] is not None]
                return f'{min(vals):.2f}-{max(vals):.2f}%' if vals else 'No accepted cases'
            lines.append(f"| {size} | {method}/{selected[0]['confidence']} | {target} | {span('coverage')} | {span('error')} | {span('error_upper_95')} |")
lines+=['','Inspect `operating-points.json` for exact accepted counts, thresholds, seed identities and group intervals. No default release artifact is selected by this report.','',
        '![Observed error and upper bound at the calibration 80% coverage target](coverage80.png)','',
        'Regenerate: `python scripts/summarize_operating_points.py`. The raw metrics and their hashes identify the evidence; no new model calls or threshold fitting are performed.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,size in zip(axes,('0.8b','4b','9b')):
    for method,color in [('continued_sft','#2864a5'),('exact','#d17a22'),('sampled','#53813b')]:
        rs=[r for r in rows if r['size']==size and r['method']==method and r['target']=='coverage_0.8']
        x=np.array([r['coverage']*100 for r in rs]);y=np.array([r['error']*100 for r in rs]);upper=np.array([r['error_upper_95']*100 for r in rs])
        ax.errorbar(x,y,yerr=[np.zeros(3),upper-y],fmt='o',color=color,label=method,capsize=3)
    ax.set_title(size);ax.set_xlabel('Achieved test coverage (%)');ax.grid(alpha=.2)
axes[0].set_ylabel('Accepted-case error (%)');axes[-1].legend(fontsize=8)
fig.suptitle('Calibration target: 80% coverage; dots are individual seeds')
fig.text(.5,.01,'Bars: one-sided 95% binomial error upper bounds, conditional on fixed thresholds. Not uncertainty across seeds.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'coverage80.png',dpi=170);plt.close(fig)
print(json.dumps({'operating_points':len(rows),'sources':len(sources),'output':str(OUT)}))
