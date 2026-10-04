"""Regenerate three-seed accuracy and confidence figures from saved evaluations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS=['sft','continued_sft','exact','sampled']
LABELS=['Supervised (100)','Continued SFT (200)','Exact RL (100+100)','Sampled RL (100+100)']
rows=[]
for size in ('0.8b','4b','9b'):
    for method in METHODS:
        for seed in (11,22,33):
            root=Path(f'results/pilot-{size}-{method}-evaluation') if seed==11 else Path(f'results/three-seed-{size}/seed-{seed}/{method}')
            path=root/'policy-metrics.json'
            if path.exists():
                m=json.loads(path.read_text())
                rows.append({'size':size,'method':method,'seed':seed,'accuracy':m['accuracy'],'policy_brier':m['correctness_brier']})
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
colors={'0.8b':'#1f77b4','4b':'#d97706','9b':'#238636'}
for index,size in enumerate(colors):
    for ax,metric in zip(axes,['accuracy','policy_brier']):
        xs=[]; means=[]; errors=[]
        for j,method in enumerate(METHODS):
            vals=[r[metric] for r in rows if r['size']==size and r['method']==method]
            if len(vals)!=3:
                continue
            xs.append(j+(index-1)*.17);means.append(np.mean(vals));errors.append(np.std(vals,ddof=1))
        if xs:
            ax.errorbar(xs,means,yerr=errors,fmt='o',capsize=4,label=size.upper(),color=colors[size])
for ax in axes:
    ax.set_xticks(range(4),LABELS,rotation=15,ha='right',fontsize=9)
    ax.grid(axis='y',alpha=.2)
    ax.legend()
axes[0].set_ylabel('Accuracy (higher is better)');axes[0].set_ylim(0,1)
axes[1].set_ylabel('Policy-confidence Brier (lower is better)');axes[1].set_ylim(0,1)
fig.suptitle('Short BANKING77 study: mean ± sample standard deviation across three seeds')
fig.text(.5,.015,'256 fixed test examples. Error bars are seed variation, not confidence intervals. 9B uses NF4; smaller models use BF16.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.07,1,.93])
out=Path('results/figures');out.mkdir(exist_ok=True)
fig.savefig(out/'three-seed-study.png',dpi=180)
(out/'three-seed-study-source.json').write_text(json.dumps({'scope':'Only complete three-seed method/size groups are plotted. Untrained/convergence claims are unsupported.','rows':rows},indent=2))

# Read the same saved controls used by the report; never refit on test data.
control_rows=[]
for size in colors:
    for seed in (11,22,33):
        root=Path(f'results/pilot-{size}-continued_sft-evaluation') if seed==11 else Path(f'results/three-seed-{size}/seed-{seed}/continued_sft')
        posthoc=Path(f'results/pilot-{size}-continued_sft-posthoc') if seed==11 else root.with_name(root.name+'-posthoc')
        for mode in ('scalar','policy','temperature','constant'):
            source=(posthoc if mode in ('temperature','constant') else root)/f'{mode}-metrics.json'
            control_rows.append({'size':size,'seed':seed,'control':mode,'brier':json.loads(source.read_text())['correctness_brier']})
fig,ax=plt.subplots(figsize=(9,4.5))
for i,size in enumerate(colors):
    groups=[[r['brier'] for r in control_rows if r['size']==size and r['control']==mode] for mode in ('scalar','policy','temperature','constant')]
    ax.errorbar(np.arange(4)+(i-1)*.15,[np.mean(v) for v in groups],yerr=[np.std(v,ddof=1) for v in groups],fmt='o',capsize=4,label=size.upper(),color=colors[size])
ax.set_xticks(range(4),['Scalar head','Confidence policy','Temperature scaling','Constant base rate'])
ax.set_ylabel('Correctness Brier (lower is better)');ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2);ax.legend()
ax.set_title('Continued supervised training: confidence controls')
fig.text(.5,.015,'Three seeds; mean ± seed SD, not confidence intervals. Fixed 256-example test subset.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'confidence-controls.png',dpi=180)
(out/'confidence-controls-source.json').write_text(json.dumps(control_rows,indent=2))
