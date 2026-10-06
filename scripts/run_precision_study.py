"""Replicate the 4B precision control at the frozen longer SFT exposure.

Reuses the three completed BF16 SFT controls, adds NF4 SFT with identical
seeds/data order/LR/exposure, and waits for the assigned evaluation GPU.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/precision-v1'
ART=ROOT/'artifacts/precision-v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/longer-v1/4b/main/seed-11/sft/config.json'
    cfg=json.loads(source.read_text());cfg.update(precision='nf4',nonquantized_dtype='fp32',purpose='replicated-precision-v1')
    inputs=[ROOT/f'data/banking77/{part}.jsonl' for part in ('train','validation','calibration','test')]
    inputs += [ROOT/f'results/longer-v1/4b/main/seed-{seed}/sft/{name}' for seed in (11,22,33) for name in ('config.json','artifact-manifest.json','evaluation/metrics.json')]
    inputs += [ROOT/'src/myjev'/name for name in ('train.py','model.py','prompt.py','objectives.py')]
    plan=dict(seeds=[11,22,33],pilot_updates=100,main_updates=4000,method='sft',config=cfg,
              reference='Completed longer-v1 4B BF16 SFT, same seeds, training data order, 4000 examples and learning rate. No new hyperparameter selection.',
              scope='Training-configuration precision control: NF4 plus FP32 nonquantized modules versus BF16, not solely inference rounding. SFT only; does not isolate RL or 9B capacity.',
              hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    dest=OUT/'frozen-plan.json'
    if dest.exists() and json.loads(dest.read_text())!=plan:raise ValueError('Frozen inputs changed')
    dest.write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--wait-for',type=Path);p.add_argument('--gpu',default='0');a=p.parse_args()
    plan=freeze()
    if a.freeze:print('Frozen: 100-update fit pilot plus three 4000-update NF4 SFT runs');return
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'state':'waiting','completed':[],'started':time.time(),'gpu':a.gpu}
    def status(**kw):
        state.update(kw,updated=time.time());tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'status.json')
    status()
    if a.wait_for:
        while True:
            if a.wait_for.exists():
                wait=json.loads(a.wait_for.read_text())
                if wait['state']=='completed':break
                if wait['state']=='failed':status(state='blocked',reason='Prerequisite evaluation failed');return
            time.sleep(15)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu,'HF_HOME':str(ROOT/'.cache/huggingface'),'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MYJEV_TRACKING_MODE':'offline'}
    # Existing owner-configured W&B settings are optional; never print credentials.
    if (ROOT/'.env.wandb').exists():
        from sync_wandb_study import settings
        settings();env.update({k:os.environ[k] for k in ('WANDB_API_KEY','MYJEV_WANDB_ENTITY','MYJEV_WANDB_PROJECT')});env['MYJEV_TRACKING_MODE']='online'
    def run(cmd,log):
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than 8 GiB free disk')
        with log.open('a') as f:
            subprocess.run([sys.executable,*cmd],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        for label,seed,updates in [('pilot',11,100)]+[(f'seed-{s}',s,4000) for s in plan['seeds']]:
            output=ART/label;record=OUT/label;record.mkdir(parents=True,exist_ok=True)
            status(state='running',current={'label':label,'phase':'training','updates':updates})
            manifest=output/'artifact/manifest.json'
            finished=manifest.exists() and json.loads(manifest.read_text()).get('training',{}).get('steps')==updates
            if not finished:
                cmd=['-m','myjev.train','--config',str(OUT/'config.json'),'--data','data/banking77/train.jsonl','--output',str(output),'--seed',str(seed),'--updates',str(updates),'--method','sft']
                resume=output/'resume.pt'
                if resume.exists():
                    import torch
                    step=torch.load(resume,map_location='cpu',weights_only=False)['step']
                    log=output/'training.jsonl'
                    if log.exists():
                        original=log.read_text();(record/f'pre-resume-{time.time_ns()}.jsonl').write_text(original)
                        log.write_text(''.join(line+'\n' for line in original.splitlines() if json.loads(line)['step']<=step))
                    cmd+=['--resume',str(resume)]
                run(cmd,record/'train.log')
            shutil.copy2(output/'training.jsonl',record/'training.jsonl');shutil.copy2(manifest,record/'artifact-manifest.json')
            if label!='pilot':
                status(current={'label':label,'phase':'evaluation'})
                if not (record/'evaluation/metrics.json').exists():
                    run(['-m','myjev.evaluate','--artifact',str(output/'artifact'),'--data','data/banking77/test.jsonl','--calibration','data/banking77/calibration.jsonl','--output',str(record/'evaluation')],record/'evaluation.log')
                if not (record/'posthoc/temperature-artifact/manifest.json').exists():
                    run(['scripts/posthoc_controls.py','--artifact',str(output/'artifact'),'--evaluation',str(record/'evaluation'),'--output',str(record/'posthoc')],record/'posthoc.log')
            state['completed'].append(label);status()
        status(state='completed',current=None)
    except Exception as e:
        status(state='failed',error=str(e));raise

if __name__=='__main__':main()
