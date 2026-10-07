"""Confound-corrected deterministic rung; keep v1 results and settings intact."""
from argparse import ArgumentParser,Namespace
from datetime import datetime,timezone
import hashlib,json,random,string,shutil
from pathlib import Path
from run_scratch import train,predict
from myjev.data import write_jsonl
from myjev.scratch.model import ScratchDecisionModel
ROOT=Path('results/scratch-ladder-v2');DATA=Path('data/scratch-ladder-v2');ART=Path('artifacts/scratch-ladder-v2')
COLORS=('red','blue','green','yellow')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixtures():
    rng=random.Random(82517);nonces=set();parts={}
    for split,n in [('train',32),('test',16)]:
        rows=[]
        for i in range(n):
            nonce=''.join(rng.choices(string.ascii_lowercase,k=8))
            while nonce in nonces:nonce=''.join(rng.choices(string.ascii_lowercase,k=8))
            nonces.add(nonce)
            for color in COLORS:
                rows.append({'id':f'{split}-{nonce}-{color}','group':nonce,'context':f'signal: {color}; record {nonce}',
                             'instructions':'Choose the named signal color.','candidates':[{'id':c,'description':c} for c in COLORS],'label':color})
        parts[split]=rows
    # Every nuisance record occurs with all four classes. A nonce-only classifier
    # cannot exceed 25% on either split; split IDs never enter the network input.
    for rows in parts.values():
        for group in {r['group'] for r in rows}:
            subset=[r for r in rows if r['group']==group]
            assert {r['label'] for r in subset}==set(COLORS) and len(subset)==4
            assert len({r['context'].replace('signal: '+r['label'],'signal: <cue>') for r in subset})==1
    assert not {r['group'] for r in parts['train']}&{r['group'] for r in parts['test']}
    return parts

def main():
    p=ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args()
    ROOT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    if (ROOT/'frozen-plan.json').exists():raise ValueError('Do not overwrite completed/frozen diagnostic')
    parts=fixtures()
    for split,rows in parts.items():write_jsonl(DATA/f'{split}.jsonl',rows)
    plan={'frozen_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Correct v1 nuisance-index/label correlation without changing architecture, learning rate, objective or per-run budget.',
          'seeds':[11,22,33],'updates':300,'batch_size':8,'learning_rate':.001,'total_updates_cap':900,'criterion':.95,'device':a.device,
          'scope':'Post-v1 structural correction, not an independent untouched scientific test. Every random nuisance nonce appears with each of four colors; train/test nonces disjoint. Only color cue distinguishes labels within nonce. Same-template deterministic recovery; no general-language/noisy-task claim. All results retained, no retuning.',
          'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path('scripts/run_scratch.py'),Path('src/myjev/scratch/model.py')]},
          'data_sha256':{str(p):digest(p) for p in DATA.glob('*.jsonl')}}
    (ROOT/'frozen-plan.json').write_text(json.dumps(plan,indent=2)+'\n');results=[]
    for seed in plan['seeds']:
        out=ART/f'seed-{seed}'
        train(Namespace(command='train',data=str(DATA/'train.jsonl'),output=str(out),method='sft',initial=None,updates=300,batch_size=8,lr=.001,seed=seed,width=64,layers=2,context_bytes=256,candidate_bytes=64,max_candidates=32,skip_examples=0,beta=.05,device=a.device,threads=2))
        model=ScratchDecisionModel.load(out/'artifact',a.device);preds=predict(model,parts['test']);target=ROOT/f'seed-{seed}';target.mkdir()
        write_jsonl(target/'predictions.jsonl',preds)
        for f in ('run.json','training.jsonl','training-summary.json'):shutil.copy2(out/f,target/f)
        accuracy=sum(r['selected_id']==r['label'] for r in preds)/len(preds)
        results.append({'seed':seed,'n':len(preds),'nonce_groups':16,'accuracy':accuracy,'criterion':.95,'passed':accuracy>=.95})
        (ROOT/'progress.json').write_text(json.dumps({'completed':len(results),'planned':3,'last':results[-1]},indent=2)+'\n')
    (ROOT/'summary.json').write_text(json.dumps({'scope':plan['scope'],'frozen_plan_sha256':digest(ROOT/'frozen-plan.json'),'results':results},indent=2)+'\n')
    lines=['# Scratch deterministic-rule rung, corrected nuisance design','',plan['scope'],'','V1 assigned colors cyclically by numeric case index, so its success could reflect a shortcut. That confounded run is preserved. V2 pairs each random nonce with all four colors, making nonce-only accuracy exactly 25%; evaluation nonces are unseen. This correction was frozen before V2 inference and did not retune any training setting.','','| Seed | Test decisions / nonce groups | Accuracy | Predeclared criterion | Pass |','|---|---:|---:|---:|---|']
    for r in results:lines.append(f"| {r['seed']} | 64 / 16 | {100*r['accuracy']:.2f}% | 95% | {r['passed']} |")
    lines+=['','Additional bounded budget: 900 updates total (three runs of 300). Including v1, the entire diagnostic extension uses 2,700 updates. The original 13,000-update scratch study and its failures are unchanged. Passing this rung is a basic rule-recovery sanity check, not proof of transfer to noisy or natural-language tasks.','', 'Reproduce in a fresh output checkout with `python scripts/run_scratch_ladder_v2.py --device cuda:0`. All generated fixtures, hashes, loss logs and predictions are retained.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
