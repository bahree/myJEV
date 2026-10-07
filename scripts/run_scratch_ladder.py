"""Prospective bounded diagnostic ladder; does not replace original scratch study."""
import argparse
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from run_scratch import train, predict
from myjev.data import write_jsonl
from myjev.scratch.model import ScratchDecisionModel

ROOT=Path('results/scratch-ladder-v1');DATA=Path('data/scratch-ladder-v1');ART=Path('artifacts/scratch-ladder-v1')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fixtures(stage,split,n):
    colors=['red','blue','green','yellow'];rows=[]
    for i in range(n):
        color=colors[i%4]
        context=f'signal: {color}' if stage=='tiny-overfit' else f'signal: {color}; record {split}-{i:03d}'
        rows.append({'id':f'{stage}-{split}-{i}','group':f'{stage}-{split}-{i}', 'context':context,
                     'instructions':'Choose the named signal color.', 'candidates':[{'id':c,'description':c} for c in colors], 'label':color})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'frozen-plan.json').exists():raise ValueError('Ladder already frozen; retain it and use a new protocol for reruns')
    datasets={}
    for stage in ('tiny-overfit','deterministic-rule'):
        folder=DATA/stage;folder.mkdir(parents=True,exist_ok=True)
        tr=fixtures(stage,'train',8 if stage=='tiny-overfit' else 128)
        te=tr if stage=='tiny-overfit' else fixtures(stage,'test',64)
        write_jsonl(folder/'train.jsonl',tr);write_jsonl(folder/'evaluation.jsonl',te);datasets[stage]=(tr,te)
    plan={'frozen_utc':datetime.now(timezone.utc).isoformat(),'seeds':[11,22,33],'updates_per_run':300,'batch_size':8,'learning_rate':.001,
          'architecture':'Original scratch width64/layers2, maxcontext256/maxcandidate64','method':'sft: original selection + scalar BCE + policy Brier',
          'total_updates_cap':1800,'device':a.device,'thresholds':{'tiny-overfit':1.0,'deterministic-rule':.95},
          'scope':'Diagnostic, not an optimization comparison. Tiny stage evaluates its training examples intentionally. Deterministic stage uses new record IDs but same template and colors, so it tests elementary rule recovery, not unfamiliar-language transfer. No uncertainty/noise, missing options, or test-selected hyperparameters. All three seeds reported; no additional tuning on failure.',
          'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path('scripts/run_scratch.py'),Path('src/myjev/scratch/model.py')]},
          'data_sha256':{str(p):digest(p) for p in DATA.glob('*/*.jsonl')}}
    (ROOT/'frozen-plan.json').write_text(json.dumps(plan,indent=2)+'\n');results=[]
    for stage,(_,test) in datasets.items():
        for seed in plan['seeds']:
            name=f'{stage}/seed-{seed}';out=ART/name
            train(Namespace(command='train',data=str(DATA/stage/'train.jsonl'),output=str(out),method='sft',initial=None,updates=300,batch_size=8,lr=.001,seed=seed,width=64,layers=2,context_bytes=256,candidate_bytes=64,max_candidates=32,skip_examples=0,beta=.05,device=a.device,threads=2))
            model=ScratchDecisionModel.load(out/'artifact',a.device);preds=predict(model,test);target=ROOT/name;target.mkdir(parents=True)
            write_jsonl(target/'predictions.jsonl',preds)
            for f in ('run.json','training.jsonl','training-summary.json'):shutil.copy2(out/f,target/f)
            accuracy=sum(r['selected_id']==r['label'] for r in preds)/len(preds)
            results.append({'stage':stage,'seed':seed,'n':len(preds),'accuracy':accuracy,'criterion':plan['thresholds'][stage],'passed':accuracy>=plan['thresholds'][stage]})
            (ROOT/'progress.json').write_text(json.dumps({'completed':len(results),'planned':6,'last':results[-1]},indent=2)+'\n')
    summary={'scope':plan['scope'],'frozen_plan_sha256':digest(ROOT/'frozen-plan.json'),'results':results}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Scratch diagnostic ladder','',plan['scope'],'','| Stage | Seed | N | Accuracy | Predeclared criterion | Pass |','|---|---:|---:|---:|---:|---|']
    for r in results:lines.append(f"| {r['stage']} | {r['seed']} | {r['n']} | {100*r['accuracy']:.2f}% | {100*r['criterion']:.0f}% | {r['passed']} |")
    lines+=['','Budget: six independent random initializations, 300 updates each, batch eight, fixed learning rate 0.001. No pretrained weights. Passing a rung demonstrates learnability in that narrow setting; failing under this budget does not establish impossibility. The original noisy/template-transfer and BANKING studies are unchanged.','', 'Reproduce in a fresh checkout without this local output: `python scripts/run_scratch_ladder.py --device cuda:0`. The source script refuses to overwrite a frozen run. Training logs and all fixture predictions are retained.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
