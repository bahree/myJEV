"""Train, evaluate and score the original small attention-head teaching prototype."""
import argparse
import gzip
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
import numpy as np
from myjev.candidate_head import CandidateHeadModel
from myjev.head_comparison import frozen_splits,exposure,exposure_digest,digest
from myjev.data import request_from_row
from myjev.calibration import fit_temperature
from myjev.metrics import summarize,thresholds_from_calibration
from myjev.tracking import start_run
from run_head_comparison import write,load_tracking_env,probability_rows


def tiny_fit(args,plan,model,parts):
    examples=exposure(parts['train'],plan['pilot_seed'],plan['tiny_overfit']['rows'])
    encoded=[model.encode(r) for _,r,_ in examples]
    hidden,masks=model.features(encoded)
    targets=torch.tensor([e['keys'].index(r['label']) for (r,_,_),e in zip(examples,encoded)],device=hidden.device)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    model.head.train();trace=[];updates=0
    for step in range(plan['tiny_overfit']['max_updates']):
        optimizer.zero_grad(set_to_none=True);logits=model.head(hidden,*masks)
        loss=torch.nn.functional.cross_entropy(logits,targets)
        accuracy=float((logits.argmax(-1)==targets).float().mean())
        trace.append({'step':step,'loss':float(loss.detach()),'training_accuracy':accuracy})
        if accuracy>=plan['tiny_overfit']['stop_accuracy'] and float(loss)<plan['tiny_overfit']['stop_loss']:break
        loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),plan['gradient_clip'],error_if_nonfinite=True);optimizer.step();updates+=1
    model.head.eval()
    with torch.inference_mode():
        logits=model.head(hidden,*masks);final_loss=float(torch.nn.functional.cross_entropy(logits,targets))
        final_accuracy=float((logits.argmax(-1)==targets).float().mean())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'tiny-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
    write(args.output/'tiny-fit.json',{'rows':len(examples),'unique_labels':len({r['label'] for r,_,_ in examples}),
        'training_accuracy':final_accuracy,'training_loss':final_loss,'updates':updates,
        'passed':final_accuracy==1. and final_loss<plan['tiny_overfit']['stop_loss'],
        'exposure_sha256':exposure_digest(examples),'feature_backbone_calls':model.calls,
        'scope':'Memorization check on 16 training rows with fixed candidate permutations. Features cached from a frozen backbone. No held-out quality claim; this head is discarded.'})


def train(args,plan,model,parts):
    updates=plan['pilot_updates'] if args.action=='pilot' else plan['main_updates']
    seed=plan['pilot_seed'] if args.action=='pilot' else plan['seed']
    batch,accum=plan['microbatch'],plan['accumulation'];count=updates*batch*accum
    examples=exposure(parts['train'],seed,count)
    encoded=[model.encode(r) for _,r,_ in examples]
    targets=[e['keys'].index(r['label']) for (r,_,_),e in zip(examples,encoded)]
    spec={'plan':plan,'seed':seed,'updates':updates,'examples':count,'exposure_sha256':exposure_digest(examples)}
    args.output.mkdir(parents=True,exist_ok=True);args.artifact.mkdir(parents=True,exist_ok=True)
    path=args.output/'run.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Run specification changed')
    write(path,spec)
    write(args.output/'initialization.json',{'head_parameters':sum(p.numel() for p in model.head.parameters()),
        'backbone_parameters':sum(p.numel() for p in model.backbone.parameters()),
        'backbone_trainable_parameters':sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
        'min_tokens':min(len(e['ids']) for e in encoded),'max_tokens':max(len(e['ids']) for e in encoded),
        'mean_tokens':float(np.mean([len(e['ids']) for e in encoded])),
        'code_sha256':{p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),Path('src/myjev/candidate_head.py')]}})
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    resume=args.artifact/'resume.pt';start=0
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['spec_sha256']!=digest(spec):raise ValueError('Resume specification mismatch')
        model.head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer']);start=state['step']
    log=args.output/'training.jsonl'
    if log.exists():
        rows=log.read_text().splitlines()
        if any(json.loads(r)['step']>start for r in rows):
            log.rename(args.output/f'training-interrupted-{time.time_ns()}.jsonl')
            log.write_text(''.join(r+'\n' for r in rows if json.loads(r)['step']<=start))
    load_tracking_env();tracking=start_run(args.output,spec)
    if tracking:tracking.name=f'candidate-head-frozen-s{seed}-{updates}updates'
    model.head.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
    for step in range(start,updates):
        optimizer.zero_grad(set_to_none=True);value=0.
        for micro in range(accum):
            offset=(step*accum+micro)*batch
            logits=model.logits(encoded[offset:offset+batch])
            target=torch.tensor(targets[offset:offset+batch],device=logits.device)
            loss=torch.nn.functional.cross_entropy(logits,target)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            (loss/accum).backward();value+=float(loss.detach())/accum
        grad=torch.nn.utils.clip_grad_norm_(model.head.parameters(),plan['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        row={'step':step+1,'examples':(step+1)*batch*accum,'loss':value,'gradient_norm':float(grad),
             'session_seconds':time.perf_counter()-began,'epoch_fraction':(step+1)*batch*accum/len(parts['train']),
             'total_updates':updates,'progress_percent':100*(step+1)/updates,
             'peak_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        with log.open('a') as stream:stream.write(json.dumps(row)+'\n')
        if tracking:tracking.log(row,step=step+1)
        if (step+1)%25==0:print(json.dumps(row),flush=True)
        if (step+1)%100==0 or step+1==updates:
            torch.save({'spec_sha256':digest(spec),'step':step+1,'head':model.head.state_dict(),
                        'optimizer':optimizer.state_dict()},args.artifact/'resume.tmp.pt')
            (args.artifact/'resume.tmp.pt').replace(resume)
    if tracking:tracking.finish()
    if any(p.grad is not None or p.requires_grad for p in model.backbone.parameters()):raise ValueError('Backbone was not frozen')
    model.manifest['training_spec_sha256']=digest(spec)
    model.save(args.artifact/'model')
    fixture=json.loads(Path('examples/request.json').read_text())
    write(args.output/'reload-fixture.json',{'request':fixture,'prediction':model.predict([fixture],batch=1)[0]})
    write(args.output/'complete.json',{'updates':updates,'examples':count,'resumed_from_step':start,
        'session_seconds':time.perf_counter()-began,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
        'backbone_frozen':True,'head_parameters':sum(p.numel() for p in model.head.parameters())})


def verify(args,model):
    fixture=json.loads((args.output/'reload-fixture.json').read_text());actual=model.predict([fixture['request']],batch=1)[0]
    if actual!=fixture['prediction']:raise ValueError('Saved/reloaded logits changed')
    write(args.output/'reload.json',{'equal':True,'single_backbone_pass':True,'prediction':actual})


def evaluate(args,plan,model,parts):
    verify(args,model);raw={}
    for split in ('validation','calibration','test'):
        predicted=model.predict([request_from_row(r)[0] for r in parts[split]])
        raw[split]=[{**{k:r[k] for k in ('id','group','label')},**v} for r,v in zip(parts[split],predicted)]
    cal=raw['calibration'];t=fit_temperature([r['logits'] for r in cal],[r['keys'].index(r['label']) for r in cal])
    validation=probability_rows(raw['validation'],1.)
    write(args.output/'validation.json',{'accuracy':float(np.mean([r['label']==r['selected_id'] for r in validation])),
        'n':len(validation),'scope':'Descriptive only; no setting or checkpoint selected.'})
    calibration={'temperature':t,'data_sha256':plan['data_sha256']['calibration'],'fit':'selection NLL; calibration only'}
    write(args.output/'calibration.json',calibration)
    for mode,temp in [('raw',1.),('temperature',t)]:
        calibration_rows=probability_rows(cal,temp);test=probability_rows(raw['test'],temp)
        thresholds=thresholds_from_calibration([r['label']==r['selected_id'] for r in calibration_rows],
                                               [r['confidence'] for r in calibration_rows])
        metrics=summarize(test,thresholds);metrics.update(temperature=temp,seed=plan['seed'],scope=plan['limitations'])
        write(args.output/f'{mode}-metrics.json',metrics)
    (args.output/'inputs.json.gz').write_bytes(gzip.compress(json.dumps(raw,separators=(',',':')).encode(),mtime=0))
    model.manifest.update(temperature=t,calibration_sha256=digest(calibration))
    model.save(args.artifact/'calibrated')
    requests=[json.loads(line) for line in Path('examples/demo-requests.jsonl').read_text().splitlines()]
    expected=json.loads(Path('examples/demo-expectations.json').read_text())['cases']
    answers=[model.score(r) for r in requests]
    write(args.output/'demos.json',{'scope':'Seven pre-existing authored diagnostics, without example selection. BANKING calibration does not establish calibration on these tasks.',
        'cases':[{'request':r,**e,'response':a,'correct':a['selected_id']==e['expected_id']} for r,e,a in zip(requests,expected,answers)]})
    # Release the original GPU allocation before measuring a single loaded model.
    model.hook.remove()
    model.head.cpu();model.backbone.cpu()
    del model.decoder,model.backbone
    gc.collect();torch.cuda.empty_cache()
    reloaded=CandidateHeadModel.load(args.artifact/'calibrated')
    actual=[reloaded.score(r) for r in requests]
    if actual!=answers:raise ValueError('Calibrated save/reload demo responses changed')
    write(args.output/'calibrated-reload.json',{'all_seven_equal':True,'single_pass_checked':True})
    elapsed=[]
    for _ in range(10):reloaded.score(requests[0])
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for _ in range(100):
        torch.cuda.synchronize();began=time.perf_counter();reloaded.score(requests[0]);torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-began)
    write(args.output/'latency.json',{'seconds':elapsed,'p50_ms':float(np.quantile(elapsed,.5)*1000),
        'p95_ms':float(np.quantile(elapsed,.95)*1000),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'warmup_requests':10,'measured_requests':100,'loaded_gpu_models':1,
        'scope':'Warm three-candidate scoring including tokenization, no HTTP. One loaded model on the measured GPU. PyTorch memory excludes CUDA context memory.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['diagnostic','pilot','main','verify','evaluate','score'])
    p.add_argument('--plan',type=Path,default=Path('configs/candidate-head-v1.json'))
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--input',type=Path,default=Path('examples/request.json'))
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if args.action=='score':
        model=CandidateHeadModel.load(args.artifact)
        print(json.dumps(model.score(json.loads(args.input.read_text())),indent=2));return
    if args.output is None:p.error('--output is required for experiments')
    parts=frozen_splits(plan)
    if args.action in ('evaluate','verify'):
        model=CandidateHeadModel.load(args.artifact/'model')
        spec=json.loads((args.output/'run.json').read_text())
        if spec['plan']!=plan or model.manifest.get('training_spec_sha256')!=digest(spec):
            raise ValueError('Artifact and frozen run specification differ')
    else:
        torch.manual_seed(plan['seed'] if args.action=='main' else plan['pilot_seed'])
        model=CandidateHeadModel.fresh(plan)
    if args.action in ('pilot','main'):train(args,plan,model,parts)
    elif args.action=='diagnostic':tiny_fit(args,plan,model,parts)
    elif args.action=='verify':verify(args,model)
    else:evaluate(args,plan,model,parts)


if __name__=='__main__':main()
