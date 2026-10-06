"""Train/evaluate the scratch scorer. No pretrained weights or network access."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from myjev.data import read_jsonl, request_from_row as raw_request, write_jsonl
from myjev.schema import ScoreRequest
from myjev.objectives import exact_loss, sampled_loss, joint_log_probs, supervised_loss
from myjev.scratch.data import build
from myjev.scratch.model import ScratchConfig, ScratchNetwork, ScratchDecisionModel, collate
from myjev.metrics import summarize, thresholds_from_calibration


def request_from_row(row, rng=None):
    return ScoreRequest.model_validate(raw_request(row, rng)[0])


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(a):
    output=Path(a.output)
    if output.exists() and any(output.iterdir()):raise ValueError('output must be a new directory')
    output.mkdir(parents=True,exist_ok=True)
    if a.updates<1 or a.batch_size<1 or a.lr<=0:raise ValueError('positive training settings required')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);rng=random.Random(a.seed)
    data=read_jsonl(a.data)
    if a.initial:
        net=ScratchDecisionModel.load(a.initial,a.device).network.train()
    else:
        net=ScratchNetwork(ScratchConfig(width=a.width,layers=a.layers,max_context_bytes=a.context_bytes,
                          max_candidate_bytes=a.candidate_bytes,max_candidates=a.max_candidates)).to(a.device).train()
    if a.method in ('exact','sampled'):
        if not a.initial:raise ValueError('RL requires a supervised initial artifact')
        reference=ScratchDecisionModel.load(a.initial,a.device).network.eval()
        reference.requires_grad_(False)
    else:reference=None
    # Validate every example before optimizing; reject oversized inputs rather than truncate.
    for row in data:collate([request_from_row(row)],net.config)
    order=list(range(len(data)));rng.shuffle(order);cursor=0
    # Continuations consume subsequent examples in the same seed-defined stream.
    def draw():
        nonlocal cursor
        if cursor==len(order):rng.shuffle(order);cursor=0
        row=data[order[cursor]];cursor+=1
        return row
    for _ in range(a.skip_examples):draw()
    params=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr)
    config=dict(**vars(a),architecture=asdict(net.config),parameters=sum(p.numel() for p in net.parameters()),
                training_sha256=digest(a.data),initial_sha256=digest(Path(a.initial)/'manifest.json') if a.initial else None,
                random_initialization=not bool(a.initial),confidence_encoder_gradients=True)
    (output/'run.json').write_text(json.dumps(config,indent=2)+'\n')
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
    start=time.perf_counter()
    with (output/'training.jsonl').open('w') as log:
        for step in range(1,a.updates+1):
            rows=[draw() for _ in range(a.batch_size)]
            # Independent candidate-order RNG preserves data stream across objectives.
            reqs=[request_from_row(r,random.Random(a.seed+step*1000+i)) for i,r in enumerate(rows)]
            targets=torch.tensor([next(i for i,c in enumerate(req.candidates) if c.id==row['label']) for req,row in zip(reqs,rows)],device=a.device)
            batch=collate(reqs,net.config,a.device);out=net(**batch)
            with torch.no_grad():ref=reference(**batch) if reference else None
            losses=[]
            for i,req in enumerate(reqs):
                k=len(req.candidates);answers=out['answer'][i:i+1,:k];policy=out['policy'][i:i+1,:k]
                if a.method=='selection':loss=F.cross_entropy(answers,targets[i:i+1])
                elif a.method=='sft':loss=supervised_loss(answers,policy,out['scalar'][i:i+1,:k],targets[i:i+1])
                else:
                    correct=F.one_hot(targets[i:i+1],k).to(answers)
                    logref=joint_log_probs(ref['answer'][i:i+1,:k],ref['policy'][i:i+1,:k])
                    loss=(exact_loss if a.method=='exact' else sampled_loss)(answers,policy,correct,logref,a.beta)
                losses.append(loss)
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            record=dict(step=step,loss=float(loss.detach()),examples=a.skip_examples+step*a.batch_size,seconds=time.perf_counter()-start)
            log.write(json.dumps(record)+'\n');log.flush()
            if step%100==0:print(json.dumps(record),flush=True)
    if a.device.startswith('cuda'):torch.cuda.synchronize(a.device)
    mode={'selection':'untrained','sft':'scalar','exact':'policy','sampled':'policy'}[a.method]
    model=ScratchDecisionModel(net,confidence_mode=mode);model.save(output/'artifact')
    summary=dict(parameters=config['parameters'],updates=a.updates,seconds=time.perf_counter()-start,
                 peak_allocated_vram=torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else None,
                 artifact_revision=model.revision)
    (output/'training-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


@torch.inference_mode()
def predict(model,rows):
    output=[]
    for row in rows:
        if next(model.network.parameters()).is_cuda:torch.cuda.synchronize()
        start=time.perf_counter();r=model.score(request_from_row(row))
        if next(model.network.parameters()).is_cuda:torch.cuda.synchronize()
        r.update(id=row['id'],group=row['group'],label=row['label'],seconds=time.perf_counter()-start)
        if 'known_probabilities' in row:
            q=row['known_probabilities'];r['known_selected_probability']=q[r['selected_id']]
            r['known_distribution_brier']=sum((r['selection_scores'][k]-v)**2 for k,v in q.items())
        output.append(r)
    return output


def evaluate(a):
    torch.set_num_threads(a.threads)
    output=Path(a.output)
    if output.exists() and any(output.iterdir()):raise ValueError('output must be new')
    rows,cal=read_jsonl(a.data),read_jsonl(a.calibration)
    if {r['group'] for r in rows}&{r['group'] for r in cal}:raise ValueError('evaluation/calibration leakage')
    model=ScratchDecisionModel.load(a.artifact,a.device)
    if a.temperature:
        logits=[];targets=[]
        with torch.inference_mode():
            for r in cal:
                req=request_from_row(r);logits.append(model.network(**collate([req],model.config,a.device))['answer'][0].cpu().double())
                targets.append(next(i for i,c in enumerate(req.candidates) if c.id==r['label']))
        # Fixed positive grid; only calibration data selects temperature.
        candidates=np.geomspace(.1,10,101)
        losses=[float(torch.stack([-F.log_softmax(x/t,0)[y] for x,y in zip(logits,targets)]).mean()) for t in candidates]
        model.temperature=float(candidates[int(np.argmin(losses))]);model.confidence_mode='temperature'
        model.calibration_revision=digest(a.calibration)
    if model.confidence_mode=='untrained':raise ValueError('selection-only artifact has no trained confidence; use --temperature')
    calibrated=predict(model,cal)
    thresholds=thresholds_from_calibration([r['selected_id']==r['label'] for r in calibrated],[r['confidence'] for r in calibrated])
    predicted=predict(model,rows)
    metrics=summarize(predicted,thresholds)
    metrics.update(artifact_revision=model.revision,temperature=model.temperature,confidence_mode=model.confidence_mode,
                   calibration_sha256=digest(a.calibration),evaluation_sha256=digest(a.data),
                   latency_p50_p95_seconds=np.quantile([r['seconds'] for r in predicted],[.5,.95]).tolist())
    if all('known_selected_probability' in r for r in predicted):
        metrics['known_probability_confidence_mse']=float(np.mean([(r['confidence']-r['known_selected_probability'])**2 for r in predicted]))
        metrics['known_distribution_brier']=float(np.mean([r['known_distribution_brier'] for r in predicted]))
    output.mkdir(parents=True)
    write_jsonl(output/'predictions.jsonl',predicted);write_jsonl(output/'calibration-predictions.jsonl',calibrated)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    if a.temperature:model.save(output/'calibrated-artifact')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('reliability','operating_points')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('data');d.add_argument('--output',required=True);d.add_argument('--version',type=int,choices=[1,2],default=1)
    t=sub.add_parser('train');t.add_argument('--data',required=True);t.add_argument('--output',required=True)
    t.add_argument('--method',choices=['selection','sft','exact','sampled'],default='selection');t.add_argument('--initial')
    t.add_argument('--updates',type=int,default=100);t.add_argument('--batch-size',type=int,default=8);t.add_argument('--lr',type=float,default=.001)
    t.add_argument('--seed',type=int,default=11);t.add_argument('--width',type=int,default=64);t.add_argument('--layers',type=int,default=2)
    t.add_argument('--context-bytes',type=int,default=256);t.add_argument('--candidate-bytes',type=int,default=64);t.add_argument('--max-candidates',type=int,default=32)
    t.add_argument('--skip-examples',type=int,default=0);t.add_argument('--beta',type=float,default=.05)
    e=sub.add_parser('evaluate');e.add_argument('--data',required=True);e.add_argument('--calibration',required=True);e.add_argument('--artifact',required=True);e.add_argument('--output',required=True);e.add_argument('--temperature',action='store_true')
    for parser in (t,e):parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.command=='data':print(json.dumps(build(a.output,version=a.version),indent=2))
    elif a.command=='train':train(a)
    else:evaluate(a)
