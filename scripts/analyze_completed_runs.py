"""Fit post-hoc controls only for completed SFT and continued-SFT evaluations."""
import json
from pathlib import Path
import subprocess
import sys

for size in ('0.8b','4b','9b'):
    for seed in (11,22,33):
        for method in ('sft','continued_sft'):
            if seed==11:
                evaluation=Path(f'results/pilot-{size}-{method}-evaluation')
                artifact=Path(f'artifacts/pilot-{size}'+('' if method=='sft' else f'-{method}'))/'artifact'
                output=Path(f'results/pilot-{size}-{method}-posthoc')
            else:
                evaluation=Path(f'results/three-seed-{size}/seed-{seed}/{method}')
                artifact=Path(f'artifacts/three-seed-{size}/seed-{seed}/{method}/artifact')
                output=Path(f'results/three-seed-{size}/seed-{seed}/{method}-posthoc')
            if not (evaluation/'metrics.json').exists() or (output/'temperature-artifact/manifest.json').exists():
                continue
            print(size,seed,method,flush=True)
            subprocess.run([sys.executable,'scripts/posthoc_controls.py','--evaluation',str(evaluation),
                            '--artifact',str(artifact),'--output',str(output)],check=True)
