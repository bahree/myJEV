"""Recompute the head-study metrics from saved, text-free logits without a GPU."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from myjev.calibration import fit_temperature
from myjev.metrics import summarize, thresholds_from_calibration
from myjev.head_comparison import permute_request,digest
from run_head_comparison import probability_rows
from fetch_evidence import require_evidence


def read(path):return json.loads(path.read_text())
def compressed(path):return json.loads(gzip.decompress(path.read_bytes()))


def compare(expected,actual,where):
    if isinstance(expected,dict):
        if expected.keys()!=actual.keys():raise ValueError(f'{where}: different keys')
        for k in expected:compare(expected[k],actual[k],f'{where}.{k}')
    elif isinstance(expected,list):
        if len(expected)!=len(actual):raise ValueError(f'{where}: different length')
        for i,(x,y) in enumerate(zip(expected,actual)):compare(x,y,f'{where}[{i}]')
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not np.isclose(expected,actual,atol=1e-12,rtol=1e-12):
            raise ValueError(f'{where}: numerical difference {expected} / {actual}')
    elif expected!=actual:raise ValueError(f'{where}: values differ')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=['readout','candidate'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();files={};checks=[]
    manifest=Path('results/evidence-manifest.json')
    if manifest.exists() and args.study in read(manifest)['bundles']:require_evidence(args.study)
    plan=read(Path('configs/head-comparison-v1.json' if args.study=='readout' else 'configs/candidate-head-v1.json'))
    root=Path('results/unsloth-head-v1' if args.study=='readout' else 'results/candidate-head-v1')
    locations=[root/f'main/seed-{seed}/{arm}' for seed in plan['seeds'] for arm in plan['arms']] if args.study=='readout' else [root/'main']
    def remember(path):files[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for base in locations:
        path=base/'inputs.json.gz';remember(path);inputs=compressed(path)
        rows=inputs['calibration']
        t=fit_temperature([r['logits'] for r in rows],[r['keys'].index(r['label']) for r in rows])
        compare(read(base/'calibration.json')['temperature'],t,str(base/'calibration.json'))
        for mode,temp in [('raw',1.),('temperature',t)]:
            cal=probability_rows(rows,temp);test=probability_rows(inputs['test'],temp)
            thresholds=thresholds_from_calibration([r['label']==r['selected_id'] for r in cal],[r['confidence'] for r in cal])
            metrics=summarize(test,thresholds)
            saved=read(base/f'{mode}-metrics.json');remember(base/f'{mode}-metrics.json')
            compare({k:saved[k] for k in metrics},metrics,str(base/mode))
            checks.append({'path':str(base/mode),'n':len(test),'metrics_recomputed':True})
        if args.study=='readout' and base.parent.name==f"seed-{plan['order_seed']}":
            original=probability_rows(inputs['test'],t)
            for seed in plan['order_permutations']:
                path=base/f'order-{seed}-inputs.json.gz';remember(path);rows=compressed(path)
                if [(r['id'],r['group'],r['label']) for r in rows]!=[(r['id'],r['group'],r['label']) for r in inputs['test']]:
                    raise ValueError('Permutation changed test row identity')
                for source,actual in zip(inputs['test'],rows):
                    request={'candidates':[{'id':key} for key in source['keys']]}
                    wanted=[c['id'] for c in permute_request(request,source['id'],seed)['candidates']]
                    if wanted!=actual['keys']:raise ValueError('Per-request permutation does not reproduce')
                converted=probability_rows(rows,t);m=summarize(converted,thresholds)
                m.update(permutation_seed=seed,temperature=t,
                    permutation_sha256=digest([{'id':r['id'],'keys':r['keys']} for r in rows]),
                    changed_answer_fraction=float(np.mean([a['selected_id']!=b['selected_id'] for a,b in zip(converted,original)])))
                saved=base/f'order-{seed}-metrics.json';remember(saved);compare(read(saved),m,str(saved))
                checks.append({'path':str(saved),'n':len(rows),'frozen_thresholds_and_permutations_verified':True})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'passed':True,'study':args.study,'tolerance':{'absolute':1e-12,'relative':1e-12},
        'scope':'Fits use calibration logits only. Test metrics and fixed-calibration order checks are recomputed without a model or GPU. No new setting or winner is selected.','checks':checks,'source_sha256':files},indent=2)+'\n')
    print(json.dumps({'passed':True,'study':args.study,'checks':len(checks)}))


if __name__=='__main__':main()
