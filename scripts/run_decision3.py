"""Evaluate pinned Decision 3.0 and myJEV releases on the frozen local comparison."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from myjev.decision3 import comparison_items, file_sha, normalize_d3, public_record, rows
from myjev.decision_comparison import digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Scorer:
    def __init__(self, spec, plan, device):
        import torch
        self.torch, self.spec, self.device = torch, spec, device
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        self.forward_count = 0
        self.events = []
        start = time.perf_counter()
        if spec['backend'] == 'd3':
            if importlib.metadata.version('transformers') != '5.17.0':
                raise ValueError('Run D3 with the pinned Transformers 5.17.0 environment')
            from huggingface_hub import snapshot_download
            folder = Path(snapshot_download(spec['repo'], revision=spec['revision'],
                                           local_files_only=True, ignore_patterns=['assets/*']))
            sys.path.insert(0, str(folder))
            from d3_runtime import D3
            self.model = D3.from_pretrained(str(folder), device=device, batch_size=1,
                                           verify='full', max_length=plan['max_input_tokens'],
                                           permutation_average=False)
            backbone = self.model.backbone
            self.metadata = {'decision_config': self.model.config,
                             'kernels': self.model.kernels,
                             'amd_fast_path_active': self.model.fast is not None,
                             'amd_fast_path_skipped': self.model.fast_skipped,
                             'source_sha256': {p.name: file_sha(p) for p in folder.glob('*.py')},
                             'manifest_sha256': file_sha(folder / 'MODEL_MANIFEST.json')}
            # The 255 code strings and token IDs remain in the pinned upstream config.
            self.metadata['decision_config'] = {k: v for k, v in self.metadata['decision_config'].items()
                                                if k not in ('codes', 'token_ids')}
            if self.model.fast is not None:
                raise ValueError('Protocol expects the NVIDIA reference path')
        else:
            if importlib.metadata.version('transformers') != '5.18.0':
                raise ValueError('Use the existing Transformers 5.18.0 release environment for myJEV')
            from myjev.inference import DecisionModel
            self.model = DecisionModel.load(spec['repo'], revision=spec['revision'], device=device)
            wrapped = self.model.network.backbone
            base = wrapped.get_base_model() if hasattr(wrapped, 'peft_config') else wrapped
            backbone = getattr(base, base.base_model_prefix)
            self.metadata = {'artifact_manifest': self.model.manifest}
        torch.cuda.synchronize(device)
        self.load_seconds = time.perf_counter() - start

        def before(module, inputs):
            self.forward_count += 1
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            self.events.append((begin, end))

        def after(module, inputs, outputs):
            self.events[-1][1].record()

        self.hooks = [backbone.register_forward_pre_hook(before), backbone.register_forward_hook(after)]

    def score(self, request):
        torch = self.torch
        self.events = []
        previous = self.forward_count
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        if self.spec['backend'] == 'd3':
            raw = self.model.system_one(state=request['context'], questions={
                'route': {'type': 'choice', 'instructions': request['instructions'],
                          'criteria': {c['id']: c['description'] for c in request['candidates']}}})
            response = normalize_d3(raw, request)
        else:
            raw = self.model.score(request)
            response = {**raw, 'native_confidence': raw['confidence'],
                        'native_confidence_mode': raw['confidence_mode'],
                        'confidence': raw['selection_scores'][raw['selected_id']],
                        'confidence_mode': 'selected-option-probability'}
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        forwards = self.forward_count - previous
        if forwards != 1 or len(self.events) != 1:
            raise ValueError(f'Expected one backbone forward, observed {forwards}')
        return {**response, 'seconds': seconds,
                'backbone_forward_ms': sum(a.elapsed_time(b) for a, b in self.events),
                'backbone_forward_count': forwards}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--phase', choices=('smoke', 'quality', 'benchmark'), default='quality')
    p.add_argument('--output', type=Path, default=Path('results/decision3-v1'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wandb', action='store_true')
    args = p.parse_args()
    plan_path = Path('configs/decision3-v1.json')
    plan = json.loads(plan_path.read_text())
    spec = plan['models'][args.model]
    all_items = comparison_items(plan)
    if digest(all_items) != plan['requests_sha256']:
        raise ValueError('Frozen requests changed')
    items = all_items
    if args.phase == 'smoke':
        items = [item for item in all_items if item['phase'] == 'calibration'][:20]
    elif args.phase == 'benchmark':
        lookup = {item['id']: item for item in all_items if item['phase'] == 'test'}
        bank = [lookup[key] for key in plan['benchmark']['banking_ids']]
        demos = [item for item in all_items if item['phase'] == 'demo']
        items = []
        for workload, subset, repeats in [('banking', bank, 3), ('short-demos', demos, 5)]:
            for repeat in range(repeats):
                for item in subset:
                    items.append({**item, 'phase': workload, 'key': f"{workload}/{repeat}/{item['id']}/original"})
    output = args.output / args.model / args.phase
    output.mkdir(parents=True, exist_ok=True)
    record_path, run_path = output / 'records.jsonl', output / 'run.json'
    identity = {'model': args.model, 'spec': spec, 'phase': args.phase,
                'protocol_sha256': file_sha(plan_path), 'requests_sha256': digest(items),
                'code_sha256': {s: file_sha(s) for s in ['scripts/run_decision3.py', 'src/myjev/decision3.py',
                                                       'src/myjev/decision_comparison.py']}}
    existing = rows(record_path) if record_path.exists() else []
    if run_path.exists():
        if not args.resume or json.loads(run_path.read_text())['identity'] != identity:
            raise ValueError('Choose a fresh output or resume an identical run')
        if (output / 'complete.json').exists():
            print(json.dumps({'already_complete': str(output)})); return
    if [r['key'] for r in existing] != [i['key'] for i in items[:len(existing)]]:
        raise ValueError('Saved output is not the expected request prefix')
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'numpy', 'huggingface_hub', 'safetensors', 'flash-linear-attention', 'causal-conv1d'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    run = {'identity': identity, 'started_utc': utc(), 'packages': packages, 'python': platform.python_version(),
           'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'scope': 'Local text scoring. Quality jobs can overlap on different GPUs; use the separate isolated benchmark for latency.',
           'expected_requests': len(items)}
    save(run_path, run)
    model = Scorer(spec, plan, args.device)
    run.update(loader=model.metadata, cached_load_seconds=model.load_seconds,
               hardware=model.torch.cuda.get_device_name(args.device), device=args.device,
               gpu_total_bytes=model.torch.cuda.get_device_properties(args.device).total_memory)
    save(run_path, run)
    warmup = items[0]['request']
    for _ in range(plan['warmup_requests']): model.score(warmup)
    tracking = None
    if args.wandb:
        from sync_wandb_study import settings
        settings('.env.wandb')
        import wandb
        tracking = wandb.init(entity=os.environ['MYJEV_WANDB_ENTITY'], project=os.environ['MYJEV_WANDB_PROJECT'],
                              id='decision3-v1-' + args.model, resume='allow', name='Decision 3.0 comparison: ' + args.model,
                              group='decision3-v1', job_type='evaluation', dir='.cache', config=identity,
                              save_code=False, settings=wandb.Settings(console='off', disable_git=True))
        save(output / 'wandb.json', {'url': tracking.url})
    began = time.perf_counter()
    completed = len(existing)
    workload = None
    with record_path.open('a') as stream:
        for item in items[len(existing):]:
            if args.phase == 'benchmark' and item['phase'] != workload:
                for _ in range(plan['warmup_requests']): model.score(item['request'])
                workload = item['phase']
            try:
                response = model.score(item['request'])
                record = public_record(item, response)
            except Exception as error:
                save(output / 'failure.json', {'utc': utc(), 'key': item['key'], 'exception': type(error).__name__,
                                               'message': str(error), 'completed': completed})
                raise
            stream.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
            stream.flush()
            completed += 1
            if completed % 50 == 0 or completed == len(items):
                elapsed = time.perf_counter() - began
                rate = (completed - len(existing)) / elapsed
                status = {'model': args.model, 'phase': args.phase, 'completed': completed, 'total': len(items),
                          'percent': 100 * completed / len(items), 'requests_per_second': rate,
                          'eta_seconds': (len(items) - completed) / rate, 'utc': utc()}
                save(output / 'progress.json', status)
                print(json.dumps(status), flush=True)
                if tracking: tracking.log({'progress/requests': completed, 'progress/percent': status['percent'],
                                           'progress/eta_minutes': status['eta_seconds'] / 60})
    complete = {'finished_utc': utc(), 'requests': completed, 'records_sha256': file_sha(record_path),
                'session_seconds': time.perf_counter() - began,
                'peak_allocated_bytes': model.torch.cuda.max_memory_allocated(args.device),
                'peak_reserved_bytes': model.torch.cuda.max_memory_reserved(args.device)}
    save(output / 'complete.json', complete)
    if tracking: tracking.finish()
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
