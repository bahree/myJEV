"""Run the frozen candidate-head prototype after the matched study releases GPU 1."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('results/candidate-head-v1')
ARTIFACTS = Path('artifacts/candidate-head-v1')


def run(action, phase):
    output, artifact = ROOT / phase, ARTIFACTS / phase
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'diagnostic': 'tiny-fit.json', 'pilot': 'complete.json',
               'main': 'complete.json', 'verify': 'reload.json',
               'evaluate': 'latency.json'}[action]
    if not (output / receipt).exists():
        log = output / f'{action}-console.log'
        if log.exists():
            log.rename(output / f'{action}-interrupted-{time.time_ns()}.log')
        command = [sys.executable, 'scripts/run_candidate_head.py', action,
                   '--output', str(output), '--artifact', str(artifact)]
        print(json.dumps({'action': action, 'phase': phase, 'gpu': 1}), flush=True)
        with log.open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    data = json.loads((output / receipt).read_text())
    if action == 'diagnostic' and not data['passed']:
        raise ValueError('Tiny training-set fit did not pass; main training was not started')
    if action == 'verify' and not data['equal']:
        raise ValueError('Save/reload verification failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-for-matched-study', action='store_true')
    args = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      PYTHONUNBUFFERED='1')
    if args.wait_for_matched_study:
        receipt = Path('results/unsloth-head-v1/main/seed-22/clef/probe.json')
        deadline = time.monotonic() + 6 * 3600
        while True:
            processes = subprocess.check_output([
                'nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip()
            if receipt.exists() and not processes:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('GPU 1 was not released by the matched study within six hours')
            time.sleep(30)
    for action, phase in [('diagnostic','diagnostic'), ('pilot','pilot'), ('verify','pilot'),
                          ('main','main'), ('evaluate','main')]:
        run(action, phase)
    print('Candidate-head training and evaluation complete.', flush=True)


if __name__ == '__main__':
    main()
