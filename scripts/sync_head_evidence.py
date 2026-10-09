"""Upload completed head-study results to W&B; local files remain unchanged."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

# Keep this list aligned with reader evidence, never console/setup or tracking files.
PATTERNS = (
    'configs/head-comparison-v1.json', 'configs/candidate-head-v1.json',
    'results/unsloth-head-v1/environment.json',
    'results/unsloth-head-v1/selection.json',
    'results/unsloth-head-v1/report.md',
    'results/unsloth-head-v1/summary.json',
    'results/unsloth-head-v1/*sources.json',
    'results/unsloth-head-v1/*.png',
    'results/unsloth-head-v1/pilot/*/*.json',
    'results/unsloth-head-v1/pilot/*/training.jsonl',
    'results/unsloth-head-v1/tuning/*/*/validation.json',
    'results/unsloth-head-v1/tuning/*/*/training.jsonl',
    'results/unsloth-head-v1/main/seed-*/*/*.json',
    'results/unsloth-head-v1/main/seed-*/*/*inputs.json.gz',
    'results/unsloth-head-v1/main/seed-*/*/training.jsonl',
    'results/unsloth-head-v1/telemetry/gpu.csv',
    'results/unsloth-head-v1/telemetry/metadata.json',
    'results/unsloth-head-v1/telemetry/gpu-activity.png',
    'results/candidate-head-v1/environment.json',
    'results/candidate-head-v1/report.md',
    'results/candidate-head-v1/summary.json',
    'results/candidate-head-v1/*.png',
    'results/candidate-head-v1/diagnostic/tiny-fit.json',
    'results/candidate-head-v1/diagnostic/tiny-trace.jsonl',
    'results/candidate-head-v1/pilot/*.json',
    'results/candidate-head-v1/pilot/training.jsonl',
    'results/candidate-head-v1/main/*.json',
    'results/candidate-head-v1/main/inputs.json.gz',
    'results/candidate-head-v1/main/training.jsonl',
)


JSON_NAMES = {
    'head-comparison-v1.json', 'candidate-head-v1.json', 'environment.json',
    'selection.json', 'summary.json', 'report-sources.json', 'figure-sources.json',
    'validation.json', 'calibration.json', 'complete.json', 'encoding.json',
    'initialization.json', 'probe.json', 'raw-metrics.json', 'temperature-metrics.json',
    'run.json', 'reload-fixture.json', 'reload.json', 'calibrated-reload.json',
    'demos.json', 'latency.json', 'tiny-fit.json', 'metadata.json',
}


def evidence_files(root):
    found = set()
    for pattern in PATTERNS:
        for path in root.glob(pattern):
            relative = path.relative_to(root)
            if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                continue
            if any(part in {'tracking', 'wandb', 'setup'} for part in relative.parts):
                continue
            if path.suffix == '.json' and path.name not in JSON_NAMES and not re.fullmatch(r'order-(101|202|303)-metrics\.json', path.name):
                continue
            found.add(relative)
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true', help='Without this flag, list sizes and hashes only')
    parser.add_argument('--env-file', default='.env.wandb')
    parser.add_argument('--output', type=Path, default=Path('results/head-replay-v1/wandb.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    for name in ('unsloth-head-v1', 'candidate-head-v1'):
        if not (root / 'results' / name / 'summary.json').exists():
            raise SystemExit(f'Finish and summarize {name} before mirroring its evidence.')
    files = evidence_files(root)
    manifest = {str(p): {'bytes': (root / p).stat().st_size,
                         'sha256': hashlib.sha256((root / p).read_bytes()).hexdigest()}
                for p in files}
    receipt = {'files': manifest, 'total_bytes': sum(v['bytes'] for v in manifest.values())}
    if not args.upload:
        print(json.dumps(receipt, indent=2))
        return
    from sync_wandb_study import settings
    settings(args.env_file)
    import wandb
    run = wandb.init(entity=os.environ['MYJEV_WANDB_ENTITY'], project=os.environ['MYJEV_WANDB_PROJECT'],
                     name='Head studies: measured results', job_type='evaluation-evidence',
                     dir=str(root / '.cache'), save_code=False,
                     settings=wandb.Settings(console='off', disable_git=True, x_disable_stats=True))
    try:
        artifact = wandb.Artifact('myjev-head-evidence', type='evaluation-evidence',
                                  metadata={'scope': 'Matched readout study and separate one-seed frozen-backbone prototype'})
        for path in files:
            artifact.add_file(str(root / path), name=str(path))
        uploaded = run.log_artifact(artifact).wait()
        matched = json.loads(Path('results/unsloth-head-v1/summary.json').read_text())
        table = wandb.Table(columns=['seed', 'readout', 'accuracy', 'correctness_brier', 'ece'],
                            data=[[r['seed'], r['arm'], r['calibrated']['accuracy'],
                                   r['calibrated']['correctness_brier'], r['calibrated']['ece']]
                                  for r in matched['runs']])
        prototype = json.loads(Path('results/candidate-head-v1/summary.json').read_text())
        run.log({'matched/readout_results': table,
                 'prototype/accuracy': prototype['calibrated']['accuracy'],
                 'prototype/correctness_brier': prototype['calibrated']['correctness_brier'],
                 'matched/accuracy_and_reliability': wandb.Image('results/unsloth-head-v1/accuracy-reliability.png'),
                 'prototype/reliability': wandb.Image('results/candidate-head-v1/reliability.png')})
        receipt.update(run_url=run.url, artifact=uploaded.qualified_name, uploaded=True)
    finally:
        run.finish()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'uploaded': True, 'files': len(files), 'run_url': receipt['run_url']}))


if __name__ == '__main__':
    main()
