"""Paired group-bootstrap comparisons, conditional on the three observed seeds."""
import json
from pathlib import Path
from fetch_evidence import require_evidence

require_evidence("pilot")
import numpy as np


def read(path):
    return {r['id']:r for r in (json.loads(line) for line in path.read_text().splitlines())}


def predictions(size,seed,method):
    root=Path(f'results/pilot-{size}-{method}-evaluation') if seed==11 else Path(f'results/three-seed-{size}/seed-{seed}/{method}')
    return root/'predictions.jsonl' if (root/'metrics.json').exists() else None

rows=[]
for size in ('0.8b','4b','9b'):
    for method in ('exact','sampled'):
        sources=[(predictions(size,seed,'continued_sft'),predictions(size,seed,method)) for seed in (11,22,33)]
        if any(a is None or b is None for a,b in sources):
            continue
        per_seed=[]; group_keys=None; ids=None
        for a,b in sources:
            baseline,candidate=read(a),read(b)
            if set(baseline)!=set(candidate):
                raise ValueError('paired test IDs differ')
            current=sorted(baseline)
            if ids is not None and current!=ids:
                raise ValueError('seed test IDs differ')
            ids=current; group_keys=[baseline[k]['group'] for k in ids]
            delta=[]
            for k in ids:
                x,y=baseline[k],candidate[k]
                if x['label']!=y['label'] or x['group']!=y['group']:
                    raise ValueError('paired labels/groups differ')
                cx=float(x['selected_id']==x['label']);cy=float(y['selected_id']==y['label'])
                bx=(x['confidence_controls']['policy']-cx)**2
                by=(y['confidence_controls']['policy']-cy)**2
                delta.append([cy-cx,by-bx])
            per_seed.append(delta)
        array=np.array(per_seed)
        groups=sorted(set(group_keys)); indices={g:np.array([i for i,x in enumerate(group_keys) if x==g]) for g in groups}
        rng=np.random.default_rng(42); estimates=[]
        for _ in range(2000):
            selected=rng.choice(groups,size=len(groups),replace=True)
            sample=np.concatenate([indices[g] for g in selected])
            estimates.append(array[:,sample,:].mean(axis=(0,1)))
        rows.append({'size':size,'method':method,'reference':'continued_sft','seeds':[11,22,33],
            'test_n':len(ids),'groups':len(groups),'accuracy_delta':float(array[:,:,0].mean()),
            'policy_brier_delta':float(array[:,:,1].mean()),
            'accuracy_delta_group_ci95':np.quantile(np.array(estimates)[:,0],[.025,.975]).tolist(),
            'policy_brier_delta_group_ci95':np.quantile(np.array(estimates)[:,1],[.025,.975]).tolist(),
            'per_seed_deltas':array.mean(axis=1).tolist()})
Path('results/paired-comparisons.json').write_text(json.dumps({'scope':'Paired group-bootstrap across fixed test groups, averaging three observed seeds. Intervals are conditional on those seeds, not uncertainty over all random initializations. Accuracy positive is better; Brier negative is better. Exploratory short exposure; multiple comparisons unadjusted.','comparisons':rows},indent=2))
print(json.dumps(rows,indent=2))
