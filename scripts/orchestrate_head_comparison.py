"""Run the frozen optional readout study; every subprocess retains its console log."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import queue
from pathlib import Path
import subprocess
import sys

from myjev.head_comparison import selected_trial

ROOT = Path('results/unsloth-head-v1')
ARTIFACTS = Path('artifacts/head-comparison-v1')
PLAN = Path('configs/head-comparison-v1.json')


def run(arm, seed, updates, lr, suffix, gpu, actions):
    output, artifact = ROOT/suffix, ARTIFACTS/suffix
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    for action in actions:
        receipt = {'train':'complete.json', 'validate':'validation.json',
                   'evaluate':'temperature-metrics.json', 'probe':'probe.json'}[action]
        if (output/receipt).exists():
            continue
        command = [sys.executable, 'scripts/run_head_comparison.py', action, '--arm', arm,
                   '--seed', str(seed), '--updates', str(updates), '--learning-rate', str(lr),
                   '--artifact', str(artifact), '--output', str(output)]
        path = output/(action+'-console.log')
        if path.exists():
            path.rename(output/(action+'-console-interrupted-'+str(path.stat().st_mtime_ns)+'.log'))
        print(json.dumps({'action':action,'arm':arm,'seed':seed,'learning_rate':lr,'gpu':gpu}),flush=True)
        with path.open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def tuning(arm,lr,devices,plan):
    gpu=devices.get()
    try:
        run(arm,plan['tuning_seed'],plan['tuning_updates'],lr,f'tuning/{arm}/lr-{lr:g}',gpu,['train','validate'])
    finally:
        devices.put(gpu)


def seal(plan):
    choices = {}; inputs = {}
    for arm in plan['arms']:
        paths = [ROOT/f'tuning/{arm}/lr-{lr:g}/validation.json' for lr in plan['learning_rates']]
        trials = [json.loads(p.read_text()) for p in paths]
        choices[arm] = selected_trial(trials)
        inputs.update({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    selection = {'plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                 'rule':plan['validation_selection'],'chosen':choices,'validation_sha256':inputs,
                 'test_used':False}
    dest = ROOT/'selection.json'
    if dest.exists() and json.loads(dest.read_text()) != selection:
        raise ValueError('Frozen learning-rate selection changed')
    dest.write_text(json.dumps(selection,indent=2)+'\n')
    return choices


def main_seed(seed,gpu,plan,choices):
    for arm in plan['arms']:
        run(arm,seed,plan['main_updates'],choices[arm]['learning_rate'],
            f'main/seed-{seed}/{arm}',gpu,['train','evaluate','probe'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['tune','main'])
    args=p.parse_args(); plan=json.loads(PLAN.read_text())
    if args.phase=='tune':
        for arm in plan['arms']:
            if not (ROOT/'pilot'/arm/'complete.json').exists():
                raise ValueError('Both 100-update pilots must complete first')
            probe=ROOT/'pilot'/arm/'probe.json'
            if not probe.exists() or not json.loads(probe.read_text())['reload_equal']:
                raise ValueError('Both pilot checkpoints must pass reload verification first')
        initial=[json.loads((ROOT/'pilot'/a/'initialization.json').read_text()) for a in plan['arms']]
        if initial[0]['initial_adapter_sha256'] != initial[1]['initial_adapter_sha256']:
            raise ValueError('Pilot adapter initializations differ')
        devices=queue.Queue()
        for gpu in range(3): devices.put(gpu)
        # Start the slower pilot readout first; GPU time and tuning budgets remain explicit.
        arms=sorted(plan['arms'],key=lambda a:json.loads((ROOT/'pilot'/a/'complete.json').read_text())['session_seconds'],reverse=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(tuning,a,lr,devices,plan) for a in arms for lr in plan['learning_rates']]
            for job in jobs: job.result()
        seal(plan)
    else:
        choices=seal(plan)
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(main_seed,s,i,plan,choices) for i,s in enumerate(plan['seeds'])]
            for job in jobs: job.result()


if __name__=='__main__': main()
