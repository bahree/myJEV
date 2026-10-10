"""Mirror the completed decision comparison to W&B, preserving local evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--env-file', default='.env.wandb')
    args = parser.parse_args()
    root = Path('results/decision-comparison-v1')
    report = json.loads((root / 'summary.json').read_text())
    if set(report['models']) != {'myjev-4b', 'myjev-4b-rl', 'decision-1'}:
        raise ValueError('Complete all three comparisons before uploading')
    paths = [Path('configs/decision-comparison-v1.json')]
    paths += [root / name for name in ('summary.json', 'report.md', 'source-review.json', 'answer-changes.png', 'figure-sources.json')]
    for name in report['models']:
        paths += [root / name / file for file in ('run.json', 'complete.json', 'records.jsonl', 'attempts.jsonl')]
    files = {str(p): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    receipt = {'files': files, 'uploaded': False}
    if args.upload:
        from sync_wandb_study import settings
        settings(args.env_file)
        import wandb
        run = wandb.init(entity=os.environ['MYJEV_WANDB_ENTITY'], project=os.environ['MYJEV_WANDB_PROJECT'],
                         name='Decision-1 and local myJEV', job_type='evaluation-evidence', dir='.cache', save_code=False,
                         settings=wandb.Settings(console='off', disable_git=True, x_disable_stats=True))
        try:
            artifact = wandb.Artifact('myjev-decision-comparison', type='evaluation-evidence',
                                      metadata={'scope': 'One fixed 64-case diagnostic, separate from the training study'})
            for p in paths:
                artifact.add_file(str(p), name=str(p))
            uploaded = run.log_artifact(artifact).wait()
            table = []
            for name, model in report['models'].items():
                for mode, results in model['modes'].items():
                    m = results['views']['original']
                    table.append([name, mode, m['accuracy'], m['correctness_brier'], m['ece']])
            run.log({'quality': wandb.Table(columns=['model', 'confidence_view', 'accuracy', 'correctness_brier', 'ece'], data=table),
                     'answer_changes': wandb.Image(str(root / 'answer-changes.png')),
                     'hosted/cost_usd': report['models']['decision-1']['reported_cost_usd']})
            receipt.update(uploaded=True, run_url=run.url, artifact=uploaded.qualified_name)
        finally:
            run.finish()
        (root / 'wandb.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt if not args.upload else {'uploaded': True, 'run_url': receipt['run_url']}, indent=2))


if __name__ == '__main__':
    main()
