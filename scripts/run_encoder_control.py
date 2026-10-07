"""Bounded fixed-taxonomy ModernBERT control; not the myJEV candidate interface."""
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from myjev.data import read_jsonl, write_jsonl
from myjev.metrics import summarize, thresholds_from_calibration

OUT=Path('results/encoder-control-v1')
ARTIFACT=Path('artifacts/encoder-control-v1/best')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_temperature(logits, labels):
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(labels)
    def nll(t):
        z=x/t
        return float((logsumexp(z,axis=1)-z[np.arange(len(y)),y]).mean())
    fitted=minimize_scalar(nll,bounds=(.05,10),method='bounded')
    if not fitted.success:raise ValueError('temperature fit failed')
    return float(fitted.x)


def make_predictions(rows, logits, classes, temperature):
    probabilities=torch.tensor(logits,dtype=torch.float64).div(temperature).softmax(-1).tolist()
    return [{'id':row['id'],'group':row['group'],'label':row['label'],
             'selected_id':classes[int(np.argmax(probs))],
             'selection_scores':dict(zip(classes,probs,strict=True)),
             'confidence':float(max(probs)),'confidence_mode':'temperature_scaled_selection' if temperature!=1 else 'selection'}
            for row,probs in zip(rows,probabilities,strict=True)]


def main():
    plan=json.loads((OUT/'frozen-plan.json').read_text())
    execution=json.loads((OUT/'execution-freeze.json').read_text())
    if file_sha(__file__)!=execution['script_sha256']:raise ValueError('script changed after freeze')
    for details in plan['splits'].values():
        if file_sha(details['path'])!=details['sha256']:raise ValueError('split changed')
    if (OUT/'summary.json').exists():raise ValueError('completed run exists')
    random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])
    rows={k:read_jsonl(v['path']) for k,v in plan['splits'].items()}
    groups={k:{r['group'] for r in v} for k,v in rows.items()}
    for a in groups:
        for b in groups:
            if a!=b and groups[a]&groups[b]:raise ValueError('split group leakage')
    classes=[c['id'] for c in rows['train'][0]['candidates']];label_ids={k:i for i,k in enumerate(classes)}
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],revision=plan['model_revision'])
    def encode_batch(batch):
        encoded=tokenizer([r['context'] for r in batch],padding=True,truncation=False,return_tensors='pt')
        if encoded['input_ids'].shape[1]>plan['max_tokens']:raise ValueError('input exceeds declared bound')
        return {k:v.to('cuda:0') for k,v in encoded.items()}
    start=time.perf_counter()
    model=AutoModelForSequenceClassification.from_pretrained(plan['model'],revision=plan['model_revision'],num_labels=len(classes),id2label=dict(enumerate(classes)),label2id=label_ids,attn_implementation='sdpa').to('cuda:0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    total_steps=math.ceil(len(rows['train'])/plan['batch_size'])*plan['epochs']
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*plan['warmup_fraction']),total_steps)
    def infer(part):
        model.eval(); output=[]
        with torch.inference_mode():
            for begin in range(0,len(part),plan['batch_size']):
                inputs=encode_batch(part[begin:begin+plan['batch_size']])
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs).logits
                output.append(logits.float().cpu())
        return torch.cat(output)
    log=(OUT/'training.jsonl').open('w');best=math.inf;best_epoch=None;step=0;seen=0;history=[]
    torch.cuda.reset_peak_memory_stats();train_start=time.perf_counter()
    for epoch in range(1,plan['epochs']+1):
        order=list(range(len(rows['train'])));random.Random(plan['seed']+epoch).shuffle(order);model.train()
        for begin in range(0,len(order),plan['batch_size']):
            batch=[rows['train'][i] for i in order[begin:begin+plan['batch_size']]]
            inputs=encode_batch(batch);labels=torch.tensor([label_ids[r['label']] for r in batch],device='cuda:0')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**inputs,labels=labels).loss
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step()
            step+=1;seen+=len(batch)
            record={'step':step,'epoch':epoch,'examples':seen,'loss':float(loss.detach()),'learning_rate':scheduler.get_last_lr()[0],'seconds':time.perf_counter()-train_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            log.write(json.dumps(record)+'\n');log.flush()
            if step%25==0:print(json.dumps(record),flush=True)
            if step==100:(OUT/'100-update-pilot.json').write_text(json.dumps(record,indent=2)+'\n')
        validation_logits=infer(rows['validation']);labels=torch.tensor([label_ids[r['label']] for r in rows['validation']])
        nll=float(F.cross_entropy(validation_logits,labels));accuracy=float((validation_logits.argmax(-1)==labels).float().mean())
        history.append({'epoch':epoch,'step':step,'validation_nll':nll,'validation_accuracy':accuracy});print(json.dumps(history[-1]),flush=True)
        if nll<best:
            best=nll;best_epoch=epoch;ARTIFACT.mkdir(parents=True,exist_ok=True);model.save_pretrained(ARTIFACT);tokenizer.save_pretrained(ARTIFACT)
    log.close();training_seconds=time.perf_counter()-train_start;peak=torch.cuda.max_memory_allocated()
    del optimizer,scheduler,model
    import gc
    gc.collect();torch.cuda.empty_cache()
    model=AutoModelForSequenceClassification.from_pretrained(ARTIFACT,attn_implementation='sdpa').to('cuda:0')
    calibration_logits=infer(rows['calibration']);cal_labels=[label_ids[r['label']] for r in rows['calibration']]
    temperature=fit_temperature(calibration_logits.numpy(),cal_labels)
    test_logits=infer(rows['test'])
    reports={}
    for name,temp in [('raw',1.0),('temperature',temperature)]:
        calibration=make_predictions(rows['calibration'],calibration_logits.tolist(),classes,temp)
        test=make_predictions(rows['test'],test_logits.tolist(),classes,temp)
        thresholds=thresholds_from_calibration([r['selected_id']==r['label'] for r in calibration],[r['confidence'] for r in calibration])
        write_jsonl(OUT/f'{name}-calibration-predictions.jsonl',calibration);write_jsonl(OUT/f'{name}-test-predictions.jsonl',test)
        reports[name]=summarize(test,thresholds)
    # One frozen existing request, model-only timing, tokenization excluded.
    inputs=encode_batch([rows['test'][0]]);model.eval();times=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(10):model(**inputs)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for _ in range(100):
            t=time.perf_counter();model(**inputs);torch.cuda.synchronize();times.append(time.perf_counter()-t)
    result={'model':plan['model'],'revision':plan['model_revision'],'classes':classes,'parameters':sum(p.numel() for p in model.parameters()),'updates':step,'examples':seen,'selected_epoch':best_epoch,'validation_history':history,'temperature':temperature,'training_seconds_including_validation_and_checkpoints':training_seconds,'total_seconds':time.perf_counter()-start,'training_peak_allocated_bytes':peak,'artifact_bytes':sum(p.stat().st_size for p in ARTIFACT.rglob('*') if p.is_file()),'model_latency':{'test_row_id':rows['test'][0]['id'],'tokens':inputs['input_ids'].shape[1],'warmup':10,'repeats':100,'seconds':times,'p50_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'scope':'One short BANKING request; model-only, excludes tokenization/HTTP; fixed77 outputs, not request-suppliedcandidate scoring.'},'metrics':reports,'limits':plan['limits']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (ARTIFACT/'control-manifest.json').write_text(json.dumps({'format':'fixed-taxonomy-encoder-control','source_revision':plan['model_revision'],'classes':classes,'temperature':temperature,'max_tokens':plan['max_tokens'],'precision':plan['precision'],'myjev_loader_compatible':False,'selection':history[best_epoch-1]},indent=2)+'\n')
    print(json.dumps({'complete':True,'selected_epoch':best_epoch,'accuracy':reports['temperature']['accuracy'],'brier':reports['temperature']['correctness_brier']}),flush=True)

if __name__=='__main__':main()
