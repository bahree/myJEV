"""Frozen post-training evaluation, resumable at completed job boundaries.

Run one size per idle GPU. No training or threshold selection occurs here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/generalization-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    inputs = sorted((ROOT/'data/clinc150').glob('*.jsonl')) + [ROOT/'data/banking77/test.jsonl']
    artifacts = [ROOT/f'artifacts/longer-v1/{size}/main/seed-{seed}/{method}/artifact/manifest.json'
                 for size in ('0.8b','4b','9b') for seed in (11,22,33)
                 for method in ('continued_sft','exact')]
    evaluations = [ROOT/f'results/longer-v1/{size}/main/seed-{seed}/{method}/evaluation/metrics.json'
                   for size in ('0.8b','4b','9b') for seed in (11,22,33)
                   for method in ('continued_sft','exact')]
    plan = dict(sizes=['0.8b','4b','9b'], seeds=[11,22,33], methods=['continued_sft','exact'],
                transfer_limit_per_cohort=256, robustness_limit=64, selection_seed=42,
                confidence='native deployed scalar for continued_sft and policy for exact',
                threshold_source='unchanged corresponding BANKING77 calibration operating points',
                purpose='Post-training diagnostic extension, not tuning or release selection on transfer data.',
                limitations=['Fixed subsamples, not full CLINC evaluation', 'Paraphrases cover eight labels and are not human validated',
                             'Sampled RL not included in this bounded transfer extension', 'No temperature control in this first extension'],
                hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs+artifacts+evaluations},
                scripts={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'scripts/evaluate_transfer.py',ROOT/'scripts/evaluate_robustness.py']})
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'frozen-plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Inputs changed after freeze; use a new study version')
    else:
        path.write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--size',choices=['0.8b','4b','9b'])
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    plan=freeze()
    if a.freeze:
        print('Frozen generalization-v1, 18 checkpoints, 36 evaluation jobs')
        return
    if not a.size:
        p.error('--size is required to execute')
    status_path=OUT/f'{a.size}-status.json'
    status={'size':a.size,'jobs':[],'state':'running','started':time.time()}
    for seed in plan['seeds']:
        for method in plan['methods']:
            for kind in ('transfer','robustness'):
                directory=OUT/a.size/f'seed-{seed}'/method/kind
                marker=directory.parent/f'{kind}-complete.json'
                if marker.exists():
                    status['jobs'].append(json.loads(marker.read_text()))
                    continue
                if directory.exists():
                    # Retain interrupted partial evidence; never overwrite it.
                    directory.rename(directory.with_name(kind+f'-incomplete-{time.time_ns()}'))
                directory.parent.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,f'scripts/evaluate_{kind}.py','--artifact',f'artifacts/longer-v1/{a.size}/main/seed-{seed}/{method}/artifact',
                     '--banking-evaluation',f'results/longer-v1/{a.size}/main/seed-{seed}/{method}/evaluation',
                     '--output',str(directory.relative_to(ROOT)), '--limit',str(256 if kind=='transfer' else 64),'--device',a.device]
                status.update(current=dict(seed=seed,method=method,kind=kind),updated=time.time())
                status_path.write_text(json.dumps(status,indent=2)+'\n')
                start=time.time()
                with (directory.parent/f'{kind}-{time.time_ns()}.log').open('w') as log:
                    result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                        'HF_HOME':os.environ.get('HF_HOME',str(ROOT/'.cache/huggingface')),'HF_HUB_OFFLINE':'1'})
                job=dict(seed=seed,method=method,kind=kind,seconds=time.time()-start,returncode=result.returncode)
                if result.returncode:
                    status.update(state='failed',failure=job,updated=time.time())
                    status_path.write_text(json.dumps(status,indent=2)+'\n')
                    raise SystemExit(result.returncode)
                marker.write_text(json.dumps(job,indent=2)+'\n')
                status['jobs'].append(job)
    status.update(state='completed',updated=time.time(),current=None)
    status_path.write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':
    main()
