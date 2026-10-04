"""Resume-safe validation-selected study; one size per GPU, no cloud services."""
import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from myjev.validation import select_trial


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('size', choices=['0.8b', '4b', '9b'])
    p.add_argument('--plan', default='configs/study-v1.json')
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    cfg = json.loads(Path(f'configs/{a.size}.json').read_text())
    data = Path('data/banking77')
    root = Path(f'artifacts/longer-v1/{a.size}')
    evidence = Path(f'results/longer-v1/{a.size}')
    root.mkdir(parents=True, exist_ok=True); evidence.mkdir(parents=True, exist_ok=True)
    lock = (root/'.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data_manifest = json.loads((data/'manifest.json').read_text())
    for split in ('train','validation','calibration','test'):
        if hashlib.sha256((data/f'{split}.jsonl').read_bytes()).hexdigest() != data_manifest['sha256'][split]:
            raise ValueError(f'frozen partition changed: {split}')
    frozen = {**plan, 'size': a.size, 'base_config': cfg,
              'dataset_manifest_sha256': hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()}
    frozen_path = evidence/'frozen-plan.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError('frozen study changed; use a new study directory')
    atomic(frozen_path, frozen)
    atomic(evidence/'execution.json', {'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
           'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    status = evidence/'status.json'

    def record(**values):
        atomic(status, {'size': a.size, 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(), **values})

    def run(args, log, stage):
        if shutil.disk_usage('.').free < 8*1024**3:
            raise RuntimeError('Less than 8 GiB free disk; refusing next stage')
        record(state='running', stage=stage)
        with log.open('a') as f:
            f.write('\n'+json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'command':args})+'\n'); f.flush()
            subprocess.run([sys.executable, *args], check=True, stdout=f, stderr=subprocess.STDOUT)

    def train(name, method, seed, rate, updates, initial=None):
        dest = root/name; out = evidence/name
        dest.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
        run_cfg = {**cfg, 'learning_rate': rate, 'updates': updates,
                   'save_every': plan['save_every'], 'accumulation': plan['accumulation'],
                   'purpose': plan['study'], 'data_offset': updates if initial else 0}
        config = out/'config.json'
        if config.exists() and json.loads(config.read_text()) != run_cfg:
            raise ValueError('existing run configuration changed')
        atomic(config, run_cfg)
        manifest = dest/'artifact/manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['training']['steps'] == updates:
            return dest/'artifact'
        cmd = ['-m','myjev.train','--config',str(config),'--data',str(data/'train.jsonl'),
               '--output',str(dest),'--method',method,'--seed',str(seed)]
        if initial: cmd += ['--initial',str(initial)]
        resume = dest/'resume.pt'
        if resume.exists():
            import torch
            step = torch.load(resume, map_location='cpu', weights_only=False)['step']
            log = dest/'training.jsonl'
            if log.exists():
                lines = log.read_text().splitlines()
                if any(json.loads(line)['step'] > step for line in lines):
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    shutil.copy2(log, dest/f'training-interrupted-{stamp}.jsonl')
                    log.write_text(''.join(line+'\n' for line in lines if json.loads(line)['step'] <= step))
            cmd += ['--resume',str(resume)]
        elif (dest/'training.jsonl').exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            (dest/'training.jsonl').rename(dest/f'training-interrupted-{stamp}.jsonl')
        run(cmd, out/'train.log', name+'/training')
        shutil.copy2(manifest, out/'artifact-manifest.json')
        shutil.copy2(dest/'training.jsonl', out/'training.jsonl')
        return dest/'artifact'

    try:
        selected = {}
        tuning_initial = None
        for method in plan['methods']:
            trials = []
            for index, rate in enumerate(plan['learning_rates']):
                name = f'tuning/{method}/lr-{index}'
                artifact = train(name, method, plan['tuning_seed'], rate, plan['tuning_updates'],
                                 tuning_initial if method != 'sft' else None)
                out = evidence/name
                if not (out/'validation/metrics.json').exists():
                    run(['-m','myjev.validation','--artifact',str(artifact),'--data',str(data/'validation.jsonl'),
                         '--output',str(out/'validation')],out/'validation.log',name+'/validation')
                report = json.loads((out/'validation/metrics.json').read_text())
                trials.append({**report,'learning_rate':rate,'artifact':str(artifact),'trial':name})
            selected[method] = select_trial(trials)
            atomic(evidence/f'selection-{method}.json', {'rule':plan['validation_selection'], 'trials':trials,'selected':selected[method]})
            if method == 'sft': tuning_initial = Path(selected[method]['artifact'])
        atomic(evidence/'selected-hyperparameters.json', selected)
        # Test/calibration evaluation begins only after all tuning decisions are frozen.
        for seed in plan['seeds']:
            initial = None
            for method in plan['methods']:
                name = f'main/seed-{seed}/{method}'
                artifact = train(name, method, seed, selected[method]['learning_rate'], plan['main_updates'],
                                 initial if method != 'sft' else None)
                if method == 'sft': initial = artifact
                out = evidence/name
                if not (out/'evaluation/metrics.json').exists():
                    run(['-m','myjev.evaluate','--artifact',str(artifact),'--data',str(data/'test.jsonl'),
                         '--calibration',str(data/'calibration.jsonl'),'--output',str(out/'evaluation')],
                         out/'evaluation.log',name+'/full-evaluation')
                if method in ('sft','continued_sft') and not (out/'posthoc/temperature-metrics.json').exists():
                    run(['scripts/posthoc_controls.py','--evaluation',str(out/'evaluation'),
                         '--artifact',str(artifact),'--output',str(out/'posthoc')],out/'posthoc.log',name+'/posthoc')
        record(state='completed', stage='all main comparisons and posthoc controls finished')
    except BaseException as error:
        record(state='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
