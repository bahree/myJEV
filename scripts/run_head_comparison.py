"""Optional Unsloth alias/Clef study runner; use the separate dependency lock."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def load_tracking_env():
    path = Path('.env.wandb')
    allowed = {'WANDB_API_KEY', 'MYJEV_WANDB_PROJECT', 'MYJEV_WANDB_ENTITY', 'MYJEV_TRACKING_MODE'}
    if path.exists():
        for line in path.read_text().splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip() in allowed:
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))


class Readout:
    def __init__(self, plan, arm, seed):
        # Unsloth must load before Transformers/PEFT. This module's CLI imports neither earlier.
        for key, value in plan.get('runtime_environment', {}).items():
            os.environ[key] = value
        from unsloth import FastDecisionModel
        import torch
        from myjev.prompt import discover_aliases
        self.torch, self.plan, self.arm = torch, plan, arm
        torch.manual_seed(seed)
        self.model, processor = FastDecisionModel.from_pretrained(
            plan['backbone'], revision=plan['backbone_revision'], decision_head='clef',
            dtype=torch.bfloat16, load_in_4bit=False, max_seq_length=plan['max_tokens'],
            random_state=seed)
        self.tokenizer = getattr(processor, 'tokenizer', processor)
        self.model = FastDecisionModel.get_peft_model(
            self.model, r=plan['lora_rank'], lora_alpha=plan['lora_alpha'],
            target_modules=plan['lora_targets'], lora_dropout=0.0,
            use_gradient_checkpointing='unsloth', random_state=seed)
        if arm == 'alias':
            # The alias control has no new decision/confidence head.
            self.model.head = torch.nn.Identity()
        self.base = self.model._backbone()
        self.decoder = getattr(self.base, 'model', None) or self.base.base_model
        self.decoder = getattr(self.decoder, 'language_model', self.decoder)
        self.aliases, self.alias_ids = discover_aliases(self.tokenizer, plan['max_candidates'])
        self.calls = 0
        self.hook = self.decoder.register_forward_pre_hook(self._called)
        self.model.train()

    def _called(self, module, args):
        self.calls += 1

    def encode(self, request):
        from myjev.prompt import encode
        from myjev.schema import ScoreRequest
        from myjev.head_comparison import clef_request
        checked = ScoreRequest.model_validate(request)
        if self.arm == 'alias':
            ids = encode(self.tokenizer, checked, self.aliases, self.plan['max_tokens'])['input_ids'][0].tolist()
            return {'ids': ids, 'keys': [c['id'] for c in request['candidates']], 'record': None}
        from unsloth.models.clef import encode_record
        record, mapping = clef_request(request)
        # Encode without a practical truncation limit, then enforce the study limit.
        encoded = encode_record(self.tokenizer, record, max_length=10**8)
        if len(encoded.input_ids) > self.plan['max_tokens']:
            raise ValueError('Clef input exceeds the frozen token limit; no truncation allowed')
        keys = [mapping[k] for k in encoded.questions[0].option_ids]
        assert keys == [c['id'] for c in request['candidates']]
        return {'ids': list(encoded.input_ids), 'keys': keys, 'record': encoded}

    def forward(self, encoded):
        torch = self.torch
        lengths = [len(e['ids']) for e in encoded]
        ids = torch.full((len(encoded), max(lengths)), self.tokenizer.pad_token_id,
                         dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        for i, row in enumerate(encoded):
            ids[i, :lengths[i]] = torch.tensor(row['ids'], device='cuda'); mask[i, :lengths[i]] = 1
        if self.arm == 'clef':
            logits, _ = self.model(input_ids=ids, attention_mask=mask, records=[r['record'] for r in encoded])
        else:
            hidden = self.decoder(input_ids=ids, attention_mask=mask, use_cache=False,
                                  return_dict=True).last_hidden_state
            final = hidden[torch.arange(len(encoded), device='cuda'), torch.tensor(lengths, device='cuda')-1].float()
            count = max(len(r['keys']) for r in encoded)
            weights = self.base.get_output_embeddings().weight[self.alias_ids[:count]].float()
            logits = final @ weights.T
        for i, row in enumerate(encoded):
            if len(row['keys']) < logits.shape[1]:
                logits[i, len(row['keys']):] = -1e4
        return logits.float()

    def predict(self, requests, batch=None):
        torch = self.torch; self.model.eval()
        encoded = [self.encode(request) for request in requests]
        outputs = []; batch = batch or self.plan['evaluation_microbatch']
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, len(encoded), batch):
                items = encoded[start:start+batch]; before = self.calls
                logits = self.forward(items)
                if self.calls - before != 1:
                    raise ValueError('Inference must perform exactly one backbone call per batch')
                outputs.extend({'keys': row['keys'], 'logits': z[:len(row['keys'])].tolist(),
                                'tokens': len(row['ids'])} for row, z in zip(items, logits.cpu()))
        return outputs

    def parameters(self):
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def state(self):
        return {n: p.detach().cpu() for n, p in self.parameters().items()}

    def load(self, weights):
        current = self.parameters()
        if current.keys() != weights.keys(): raise ValueError('Trainable parameter schema changed')
        with self.torch.no_grad():
            for n, p in current.items(): p.copy_(weights[n])


def train(args, plan, model, splits):
    torch = model.torch
    from myjev.head_comparison import exposure, exposure_digest, digest
    from myjev.tracking import start_run
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact; artifact.mkdir(parents=True, exist_ok=True)
    batch = plan['microbatch']; accum = plan['accumulation']; count = args.updates * batch * accum
    examples = exposure(splits['train'], args.seed, count)
    spec = {'plan': plan, 'arm': args.arm, 'seed': args.seed, 'updates': args.updates,
            'learning_rate': args.learning_rate, 'example_exposure': count,
            'exposure_sha256': exposure_digest(examples)}
    config_path = output/'run.json'
    if config_path.exists() and json.loads(config_path.read_text()) != spec:
        raise ValueError('Run specification changed; use a new output path')
    write(config_path, spec)
    parameters = model.parameters()
    dtypes = {}
    for parameter in model.model.parameters():
        key = str(parameter.dtype); dtypes[key] = dtypes.get(key, 0) + parameter.numel()
    adapter_fingerprint = hashlib.sha256()
    for name, value in parameters.items():
        if 'lora_' in name: adapter_fingerprint.update(value.detach().cpu().float().numpy().tobytes())
    write(output/'initialization.json', {'trainable_parameters': sum(p.numel() for p in parameters.values()),
        'total_parameters': sum(p.numel() for p in model.model.parameters()), 'dtypes': dtypes,
        'initial_adapter_sha256': adapter_fingerprint.hexdigest(),
        'head_config': getattr(model.model.head, 'config', None),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()})
    started = time.perf_counter()
    encoded = [model.encode(request) for _, request, _ in examples]
    targets = [e['keys'].index(row['label']) for (row, _, _), e in zip(examples, encoded)]
    write(output/'encoding.json', {'examples': len(encoded), 'min_tokens': min(len(e['ids']) for e in encoded),
        'max_tokens': max(len(e['ids']) for e in encoded), 'mean_tokens': sum(len(e['ids']) for e in encoded)/len(encoded),
        'seconds': time.perf_counter()-started, 'truncated': 0, 'skipped': 0})
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate, weight_decay=plan['weight_decay'])
    resume = artifact/'resume.pt'; start = 0
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['spec_sha256'] != digest(spec): raise ValueError('Resume spec mismatch')
        model.load(saved['parameters']); optimizer.load_state_dict(saved['optimizer']); start = saved['step']
        torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    log_path = output/'training.jsonl'
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        if any(json.loads(line)['step'] > start for line in lines):
            log_path.rename(output/f'training-interrupted-{time.time_ns()}.jsonl')
            log_path.write_text(''.join(line+'\n' for line in lines if json.loads(line)['step'] <= start))
    load_tracking_env()
    tracking = start_run(output, {**spec, 'readout': args.arm})
    if tracking:
        tracking.name = f"head-v1-{args.arm}-s{args.seed}-{args.updates}updates-lr{args.learning_rate:g}"
        tracking.tags = tuple(tracking.tags) + ('head-comparison-v1',)
    model.model.train(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(start, args.updates):
        optimizer.zero_grad(set_to_none=True); loss_value = 0
        for micro in range(accum):
            offset = (step*accum + micro)*batch; items = encoded[offset:offset+batch]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model.forward(items)
                target = torch.tensor(targets[offset:offset+batch], device='cuda')
                loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            (loss/accum).backward(); loss_value += float(loss.detach())/accum
        grad = torch.nn.utils.clip_grad_norm_(parameters.values(), plan['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        row = {'step': step+1, 'examples': (step+1)*batch*accum, 'loss': loss_value,
               'gradient_norm': float(grad), 'total_updates': args.updates,
               'epoch_fraction': (step+1)*batch*accum/len(splits['train']),
               'planned_epochs': count/len(splits['train']),
               'progress_percent': 100*(step+1)/args.updates, 'session_seconds': time.perf_counter()-started,
               'peak_vram_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'learning_rate': args.learning_rate}
        with log_path.open('a') as stream: stream.write(json.dumps(row)+'\n')
        if tracking: tracking.log(row, step=step+1)
        if (step+1)%10 == 0: print(json.dumps(row), flush=True)
        if (step+1)%plan['checkpoint_every'] == 0 or step+1 == args.updates:
            state = {'spec_sha256': digest(spec), 'step': step+1, 'parameters': model.state(),
                     'optimizer': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all()}
            torch.save(state, artifact/'resume.tmp.pt'); (artifact/'resume.tmp.pt').replace(resume)
    if tracking: tracking.finish()
    torch.save(model.state(), artifact/'readout.pt')
    write(artifact/'manifest.json', {'artifact_type': 'experimental-head-comparison', 'spec': spec,
        'weights_sha256': hashlib.sha256((artifact/'readout.pt').read_bytes()).hexdigest(),
        'loader': 'scripts/run_head_comparison.py; not a standard myJEV release artifact'})
    fixture = json.loads(Path('examples/request.json').read_text())
    write(artifact/'reload-fixture.json', {'request': fixture, 'prediction': model.predict([fixture], batch=1)[0]})
    write(output/'complete.json', {'updates': args.updates, 'examples': count,
        'session_seconds': time.perf_counter()-started, 'resumed_from_step': start,
        'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def load_artifact(args, plan, model):
    manifest = json.loads((args.artifact/'manifest.json').read_text())
    spec = manifest['spec']
    if spec['plan'] != plan or spec['arm'] != args.arm or spec['seed'] != args.seed:
        raise ValueError('Artifact does not match the requested frozen configuration')
    if hashlib.sha256((args.artifact/'readout.pt').read_bytes()).hexdigest() != manifest['weights_sha256']:
        raise ValueError('Artifact weights checksum mismatch')
    model.load(model.torch.load(args.artifact/'readout.pt', map_location='cpu', weights_only=True))


def evaluate(args, plan, model, splits):
    import numpy as np
    from myjev.data import request_from_row
    from myjev.calibration import fit_temperature
    from myjev.metrics import summarize, thresholds_from_calibration
    load_artifact(args, plan, model)
    parts = ['validation'] if args.action == 'validate' else ['calibration', 'test']
    raw = {}
    for split in parts:
        rows = splits[split]; values = model.predict([request_from_row(r)[0] for r in rows])
        raw[split] = [{**{k:r[k] for k in ('id','group','label')}, **v} for r,v in zip(rows,values)]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'validate':
        rows = raw['validation']; correct = [r['keys'][int(np.argmax(r['logits']))] == r['label'] for r in rows]
        from scipy.special import logsumexp
        nll = [logsumexp(r['logits']) - r['logits'][r['keys'].index(r['label'])] for r in rows]
        write(args.output/'validation.json', {'accuracy': float(np.mean(correct)), 'selection_nll': float(np.mean(nll)),
            'learning_rate': args.learning_rate, 'n': len(rows)})
        return
    temperature = fit_temperature([r['logits'] for r in raw['calibration']],
                                  [r['keys'].index(r['label']) for r in raw['calibration']])
    write(args.output/'calibration.json', {'temperature': temperature, 'source_sha256': plan['data_sha256']['calibration'],
        'rule': 'Positive selection temperature fitted on calibration only; no test selection'})
    for mode, t in [('raw',1.0),('temperature',temperature)]:
        converted = {}
        for split, rows in raw.items():
            converted[split] = []
            for row in rows:
                z = np.array(row['logits'])/t; probabilities = np.exp(z-z.max()); probabilities /= probabilities.sum()
                converted[split].append({**{k:row[k] for k in ('id','group','label')},
                    'selected_id': row['keys'][int(probabilities.argmax())], 'confidence': float(probabilities.max()),
                    'selection_scores': dict(zip(row['keys'], probabilities.tolist()))})
        cal = converted['calibration']
        thresholds = thresholds_from_calibration([r['selected_id']==r['label'] for r in cal],[r['confidence'] for r in cal])
        metrics = summarize(converted['test'], thresholds); metrics.update(temperature=t,arm=args.arm,seed=args.seed)
        write(args.output/(mode+'-metrics.json'), metrics)
    data = json.dumps(raw,separators=(',',':')).encode()
    (args.output/'inputs.json.gz').write_bytes(gzip.compress(data,mtime=0))
    print(json.dumps({'evaluation_complete':str(args.output),'temperature':temperature}),flush=True)



def probability_rows(rows, temperature):
    import numpy as np
    result = []
    for row in rows:
        z = np.array(row['logits'])/temperature
        p = np.exp(z-z.max()); p /= p.sum()
        result.append({**{k:row[k] for k in ('id','group','label')},
                       'selected_id':row['keys'][int(p.argmax())], 'confidence':float(p.max()),
                       'selection_scores':dict(zip(row['keys'],p.tolist()))})
    return result


def probe(args, plan, model, splits):
    import numpy as np
    from myjev.data import request_from_row
    from myjev.head_comparison import permute_request, digest
    from myjev.metrics import summarize
    load_artifact(args,plan,model)
    fixture=json.loads((args.artifact/'reload-fixture.json').read_text())
    actual=model.predict([fixture['request']],batch=1)[0]
    if actual != fixture['prediction']:
        write(args.output/'reload-failure.json', {'expected':fixture['prediction'],'actual':actual})
        raise ValueError('Saved/reloaded fixture is not output-identical')
    evidence={'arm':args.arm,'seed':args.seed,'reload_equal':True,'fixture':actual,
              'single_backbone_call_checked':True}
    if args.seed == plan['order_seed']:
        original=json.loads(gzip.decompress((args.output/'inputs.json.gz').read_bytes()))['test']
        temperature=json.loads((args.output/'calibration.json').read_text())['temperature']
        baseline=json.loads((args.output/'temperature-metrics.json').read_text())
        thresholds={k:v['threshold'] for k,v in baseline['operating_points'].items()}
        order=[]
        for seed in plan['order_permutations']:
            requests=[permute_request(request_from_row(r)[0],r['id'],seed) for r in splits['test']]
            predictions=model.predict(requests)
            rows=[{**{k:r[k] for k in ('id','group','label')},**v} for r,v in zip(splits['test'],predictions)]
            converted=probability_rows(rows,temperature)
            metrics=summarize(converted,thresholds)
            metrics.update(permutation_seed=seed,temperature=temperature,
                           permutation_sha256=digest([{'id':r['id'],'keys':r['keys']} for r in rows]),
                           changed_answer_fraction=float(np.mean([a['selected_id']!=b['selected_id']
                               for a,b in zip(converted,probability_rows(original,temperature))])))
            write(args.output/f'order-{seed}-metrics.json',metrics)
            (args.output/f'order-{seed}-inputs.json.gz').write_bytes(
                gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
            order.append({'permutation_seed':seed,'accuracy':metrics['accuracy'],
                          'changed_answer_fraction':metrics['changed_answer_fraction']})
        evidence['order']=order
        cases={'three_candidates':fixture['request'],
               'seventy_seven_candidates':request_from_row(splits['test'][0])[0]}
        timings={}; torch=model.torch
        for name,request in cases.items():
            for _ in range(10): model.predict([request],batch=1)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); elapsed=[]
            for _ in range(100):
                torch.cuda.synchronize(); start=time.perf_counter()
                value=model.predict([request],batch=1)[0]
                torch.cuda.synchronize(); elapsed.append(time.perf_counter()-start)
            timings[name]={'seconds':elapsed,'p50_ms':float(np.quantile(elapsed,.5)*1000),
                           'p95_ms':float(np.quantile(elapsed,.95)*1000), 'tokens':value['tokens'],
                           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                           'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                           'candidate_count':len(request['candidates'])}
        evidence['latency']=timings
        evidence['latency_scope']='Warm single-request model scoring including tokenization and Python conversion, no HTTP; exclusive target A30, other GPUs may run other seeds.'
        if args.arm=='clef':
            evidence['multi_question']=multi_question_demo(model,plan)
    write(args.output/'probe.json',evidence)


def multi_question_demo(model,plan):
    from unsloth.models.clef import encode_record
    record={
        'state':'I was charged twice for the same invoice. Please refund the duplicate charge. The account is working and this is not urgent.',
        'questions':{
            'route':{'type':'choice','instructions':'Choose the support route.',
                     'criteria':{'billing':'Charges, invoices and refunds','technical':'Errors and configuration','other':'Neither route applies'}},
            'refund_requested':{'type':'choice','instructions':'Does the customer explicitly ask for a refund?',
                                'criteria':{'yes':'A refund is requested','no':'No refund is requested'}},
            'urgency':{'type':'choice','instructions':'Use the stated urgency, without inferring an emergency.',
                       'criteria':{'routine':'Routine request, not urgent','urgent':'Explicitly urgent or a service outage'}}}}
    encoded=encode_record(model.tokenizer,record,max_length=10**8)
    if len(encoded.input_ids)>plan['max_tokens']: raise ValueError('Oversize multi-question demo')
    torch=model.torch; model.model.eval(); before=model.calls
    ids=torch.tensor([encoded.input_ids],device='cuda'); mask=torch.ones_like(ids)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits,_=model.model(input_ids=ids,attention_mask=mask,records=[encoded])
    if model.calls-before!=1: raise ValueError('Multi-question demo made multiple backbone calls')
    answers={}
    for q,z in zip(encoded.questions,logits):
        keys=list(q.option_ids); probs=z[:len(keys)].float().softmax(-1).tolist()
        answers[q.question_id]={'selected_id':keys[max(range(len(probs)),key=probs.__getitem__)],
                        'selection_scores':dict(zip(keys,probs))}
    return {'request':record,'answers':answers,'backbone_calls':model.calls-before,
            'expected':{'route':'billing','refund_requested':'yes','urgency':'routine'},
            'semantics':'Original qualitative demo, outside BANKING training. Raw scores; BANKING calibration does not transfer to these questions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train','validate','evaluate','probe'])
    parser.add_argument('--plan', type=Path, default=Path('configs/head-comparison-v1.json'))
    parser.add_argument('--arm', choices=['alias','clef'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    model = Readout(plan,args.arm,args.seed)
    from myjev.head_comparison import frozen_splits
    splits = frozen_splits(plan)
    if args.action == 'train': train(args,plan,model,splits)
    elif args.action == 'probe': probe(args,plan,model,splits)
    else: evaluate(args,plan,model,splits)


if __name__ == '__main__': main()
