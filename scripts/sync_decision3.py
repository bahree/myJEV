"""Upload completed Decision 3.0 comparison evidence to W&B; keep local files."""
import argparse
import json
import os
from pathlib import Path

from myjev.decision3 import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--env-file', default='.env.wandb')
    parser.add_argument('--root', type=Path, default=Path('results/decision3-v1'))
    args = parser.parse_args()
    root = args.root
    plan = json.loads(Path('configs/decision3-v1.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if set(summary['models']) != set(plan['models']):
        raise ValueError('Complete every declared model before uploading')
    if summary['protocol_sha256'] != file_sha('configs/decision3-v1.json'):
        raise ValueError('Summary protocol differs from the current pinned comparison')
    paths = [Path('configs/decision3-v1.json'), Path('results/decision3-v1/source-review.json')]
    paths += [root / name for name in ('summary.json', 'report.md',
                                      'quality.png', 'latency.png', 'figure-sources.json')]
    quality, timing = [], []
    for name, result in summary['models'].items():
        if result['benchmark'] is None:
            raise ValueError('Complete isolated benchmark: ' + name)
        for phase in ('quality', 'benchmark'):
            folder = root / name / phase
            complete = json.loads((folder / 'complete.json').read_text())
            if file_sha(folder / 'records.jsonl') != complete['records_sha256']:
                raise ValueError('Changed predictions: ' + name + '/' + phase)
            if phase == 'quality' and complete['records_sha256'] != result['source_records_sha256']:
                raise ValueError('Summary describes different predictions: ' + name)
            paths += [folder / file for file in ('run.json', 'complete.json', 'records.jsonl')]
        for mode, views in result['modes'].items():
            m = views['phases']['test']
            point = m['operating_points']['coverage_0.8']
            quality.append([name, mode, m['n'], m['accuracy'], m['correctness_brier'],
                            m['ece'], point['coverage'], point['error']])
        for workload, values in result['benchmark']['workloads'].items():
            timing.append([name, workload, *values['local_scoring_p50_p95_ms'],
                           result['benchmark']['peak_allocated_gib']])
    receipt = {'files': {str(p): {'bytes': p.stat().st_size, 'sha256': file_sha(p)} for p in paths},
               'uploaded': False}
    if args.upload:
        from sync_wandb_study import settings
        settings(args.env_file)
        import wandb
        run = wandb.init(entity=os.environ['MYJEV_WANDB_ENTITY'], project=os.environ['MYJEV_WANDB_PROJECT'],
                         name='Decision 3.0 and myJEV: completed comparison', group='decision3-v1',
                         job_type='evaluation-evidence', dir='.cache', save_code=False,
                         settings=wandb.Settings(console='off', disable_git=True, x_disable_stats=True))
        try:
            artifact = wandb.Artifact('myjev-decision3-comparison', type='evaluation-evidence',
                                     metadata={'scope': summary['scope'], 'protocol_sha256': summary['protocol_sha256']})
            for path in paths:
                artifact.add_file(str(path), name=str(path))
            uploaded = run.log_artifact(artifact).wait()
            run.log({'quality': wandb.Table(columns=['model', 'confidence_view', 'test_rows', 'accuracy',
                                                     'correctness_brier', 'ece', 'coverage', 'accepted_error'], data=quality),
                     'local_timing': wandb.Table(columns=['model', 'workload', 'p50_ms', 'p95_ms', 'peak_allocated_gib'], data=timing),
                     'quality_plot': wandb.Image(str(root / 'quality.png')),
                     'latency_plot': wandb.Image(str(root / 'latency.png'))})
            receipt.update(uploaded=True, run_url=run.url, artifact=uploaded.qualified_name)
        finally:
            run.finish()
        (root / 'wandb.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt if not args.upload else {'uploaded': True, 'run_url': receipt['run_url']}, indent=2))


if __name__ == '__main__':
    main()
