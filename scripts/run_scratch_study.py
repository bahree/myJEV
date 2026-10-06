"""Frozen, bounded synthetic scratch comparison with complete local evidence."""
import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import torch
import torch.nn.functional as F
from run_scratch import train, evaluate, request_from_row
from myjev.data import read_jsonl, validate_isolation
from myjev.scratch.model import ScratchDecisionModel, collate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/scratch-study-v1.json');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    plan=json.loads(Path(a.config).read_text());root=Path('results/scratch-study-v1');artifacts=Path('artifacts/scratch-study-v1')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'frozen.json').exists():raise ValueError('study already started; inspect records rather than overwrite')
    data=Path(plan['data']);parts={k:read_jsonl(data/f'{k}.jsonl') for k in ('train','validation','calibration','test')}
    validate_isolation(parts)
    frozen=dict(plan=plan,source_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                data_sha256={k:hashlib.sha256((data/f'{k}.jsonl').read_bytes()).hexdigest() for k in parts})
    (root/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    def status(state,stage):
        (root/'status.json').write_text(json.dumps(dict(state=state,stage=stage),indent=2)+'\n');print(state,stage,flush=True)
    def run(name,seed,lr,method,updates,initial=None,skip=0):
        status('running',name);out=artifacts/name
        args=Namespace(command='train',data=str(data/'train.jsonl'),output=str(out),method=method,initial=str(initial) if initial else None,
                       updates=updates,batch_size=plan['batch_size'],lr=lr,seed=seed,width=plan['width'],layers=plan['layers'],context_bytes=256,candidate_bytes=64,
                       max_candidates=32,skip_examples=skip,beta=plan['beta'],device=a.device,threads=2)
        train(args);ev=root/name;ev.mkdir(parents=True,exist_ok=True)
        for f in ('run.json','training.jsonl','training-summary.json'):shutil.copy2(out/f,ev/f)
        shutil.copy2(out/'artifact/manifest.json',ev/'artifact-manifest.json')
        return out/'artifact'
    try:
        trials=[]
        for i,lr in enumerate(plan['learning_rates']):
            artifact=run(f'tuning/lr-{i}',plan['tuning_seed'],lr,'sft',plan['tuning_updates'])
            m=ScratchDecisionModel.load(artifact,a.device);correct=0;nll=0
            with torch.inference_mode():
                for row in parts['validation']:
                    req=request_from_row(row);logits=m.network(**collate([req],m.config,a.device))['answer'][0]
                    target=next(i for i,c in enumerate(req.candidates) if c.id==row['label'])
                    correct+=int(logits.argmax())==target;nll+=float(-F.log_softmax(logits,0)[target])
            record=dict(lr=lr,accuracy=correct/len(parts['validation']),nll=nll/len(parts['validation']))
            (root/f'tuning/lr-{i}/validation.json').write_text(json.dumps(record,indent=2)+'\n');trials.append(record)
        selected=sorted(trials,key=lambda r:(-r['accuracy'],r['nll'],r['lr']))[0]
        (root/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
        for seed in plan['seeds']:
            base=None
            for method in plan['methods']:
                name=f'main/seed-{seed}/{method}'
                artifact=run(name,seed,selected['lr'],'sft' if method=='continued_sft' else method,
                             plan['main_updates'],base if method!='sft' else None,
                             plan['main_updates']*plan['batch_size'] if method!='sft' else 0)
                if method=='sft':base=artifact
                modes=[False,True] if method in ('sft','continued_sft') else [False]
                for temperature in modes:
                    status('running',name+('/temperature' if temperature else '/evaluation'))
                    evaluate(Namespace(data=str(data/'test.jsonl'),calibration=str(data/'calibration.jsonl'),artifact=str(artifact),
                                       output=str(root/name/('temperature' if temperature else 'evaluation')),
                                       temperature=temperature,device=a.device,threads=2))
        status('completed','all synthetic training and evaluation complete')
    except BaseException:
        status('failed','inspect study log; no automatic partial-run resume');raise


if __name__=='__main__':main()
