"""Run independent quality jobs on three GPUs, then isolated serving measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path.cwd()
    output = root / 'results/decision3-v1'
    logdir = output / 'execution'
    logdir.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(HF_HOME=str(root / '.cache/huggingface'), PYTHONPATH=str(root / 'src'),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                       HF_HUB_OFFLINE='1')
    mutex = threading.Lock()

    def event(**record):
        record['utc'] = datetime.now(timezone.utc).isoformat()
        with mutex:
            with (output / 'events.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def run(model, gpu, phase):
        executable = str(root / '.venv-decision3/bin/python') if model.startswith('d3-') else sys.executable
        command = [executable, '-u', 'scripts/run_decision3.py', '--model', model,
                   '--phase', phase, '--device', 'cuda:' + str(gpu)]
        if args.resume: command.append('--resume')
        if args.wandb and phase == 'quality': command.append('--wandb')
        event(action='start', model=model, gpu=gpu, phase=phase)
        with (logdir / f'{model}-{phase}.log').open('a' if args.resume else 'w') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        event(action='finish', model=model, gpu=gpu, phase=phase, exit_code=result.returncode)
        if result.returncode: raise RuntimeError(f'{model} {phase} failed; inspect its saved log')

    def lane(gpu, models):
        for model in models: run(model, gpu, 'quality')

    lanes = [(0, ['d3-lite', 'myjev-4b-rl']), (1, ['d3-mini', 'myjev-0.8b-rl']), (2, ['myjev-0.8b', 'myjev-4b'])]
    with ThreadPoolExecutor(max_workers=3) as workers:
        jobs = [workers.submit(lane, gpu, models) for gpu,models in lanes]
        for job in jobs: job.result()
    # Exit rather than silently call a shared-GPU measurement an isolated benchmark.
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if busy: raise RuntimeError('GPU process remains active before isolated benchmark: ' + busy)
    event(action='isolated_benchmark_start', gpu=0)
    for model in ['myjev-4b', 'd3-lite', 'myjev-0.8b-rl', 'd3-mini', 'myjev-0.8b', 'myjev-4b-rl']:
        run(model, 0, 'benchmark')
    with (logdir / 'summarize.log').open('w') as log:
        subprocess.run([sys.executable, 'scripts/summarize_decision3.py', '--require-benchmark'],
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    event(action='complete')


if __name__ == '__main__': main()
