import json
from pathlib import Path
import numpy as np
rows=[]
for size in ('0.8b','4b','9b'):
    for method in ('sft','continued_sft','exact','sampled'):
        metrics=[]
        for seed in (11,22,33):
            directory=Path(f'results/pilot-{size}-{method}-evaluation') if seed==11 else Path(f'results/three-seed-{size}/seed-{seed}/{method}')
            path=directory/'policy-metrics.json'
            if not path.exists():
                continue
            m=json.loads(path.read_text())
            metrics.append(m)
            rows.append({'size':size,'method':method,'seed':seed,'n':m['n'],'confidence_mode':'policy',
                         'accuracy':m['accuracy'],'correctness_brier':m['correctness_brier'],
                         'coverage80':m['operating_points']['coverage_0.8'],
                         'artifact_revision':m['artifact_revision']})
summary=[]
for size in ('0.8b','4b','9b'):
    for method in ('sft','continued_sft','exact','sampled'):
        rs=[r for r in rows if r['size']==size and r['method']==method]
        if not rs:
            continue
        summary.append({'size':size,'method':method,'seeds_observed':[r['seed'] for r in rs],
                        'accuracy_mean':float(np.mean([r['accuracy'] for r in rs])),
                        'accuracy_seed_std':float(np.std([r['accuracy'] for r in rs],ddof=1)) if len(rs)>1 else None,
                        'brier_mean':float(np.mean([r['correctness_brier'] for r in rs]))})
Path('results/study-summary.json').write_text(json.dumps({'scope':'100 initial updates; 100 matched continuation updates; fixed random 256-example evaluation subsets; policy confidence for all compared methods',
    'limitations':['short exposure','256 calibration examples','256 test examples','9B uses NF4; BF16/NF4 capacity differences need a same-size control'],
    'runs':rows,'summary':summary},indent=2))
for r in summary:
    print(r)
