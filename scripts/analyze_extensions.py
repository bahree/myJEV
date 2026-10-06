"""Reproduce paired precision/transfer contrasts from compact, text-free evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/extension-analysis-v1')
SEEDS = (11, 22, 33)

def extract():
    records, sources = [], {}
    specs = [('precision', '4b', 'test', seed, method, path)
             for seed in SEEDS for method, path in (
                 ('bf16', Path(f'results/longer-v1/4b/main/seed-{seed}/sft/evaluation/predictions.jsonl')),
                 ('nf4', Path(f'results/precision-v1/seed-{seed}/evaluation/predictions.jsonl')))]
    specs += [('transfer', size, cohort, seed, method,
               Path(f'results/generalization-v1/{size}/seed-{seed}/{method}/transfer/{cohort}-predictions.jsonl'))
              for size in ('0.8b', '4b', '9b') for cohort in ('near', 'distant', 'oos-none', 'unsupported-near-none')
              for seed in SEEDS for method in ('continued_sft', 'exact')]
    for study, size, cohort, seed, method, path in specs:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            r = json.loads(line)
            correct = int(r['selected_id'] == r['label'])
            records.append(dict(study=study, size=size, cohort=cohort, seed=seed, method=method,
                                id=r['id'], group=r['group'], label=r['label'], correct=correct,
                                brier=(r['confidence']-correct)**2))
    ROOT.mkdir(exist_ok=True)
    (ROOT/'inputs.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in records).encode(), mtime=0))
    (ROOT/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')

def analyze():
    buckets = {}
    with gzip.open(ROOT/'inputs.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            bucket = buckets.setdefault(tuple(r[k] for k in ('study','size','cohort','seed','method')), {})
            assert r['id'] not in bucket
            bucket[r['id']] = r
    results = []
    for study, size, cohort in sorted({key[:3] for key in buckets}):
        ref, cand = ('bf16','nf4') if study == 'precision' else ('continued_sft','exact')
        for metric in ('correct','brier') if study == 'precision' else ('correct',):
            differences, identity = [], None
            for seed in SEEDS:
                a, b = (buckets[(study,size,cohort,seed,m)] for m in (ref,cand))
                assert set(a) == set(b)
                ids = sorted(a)
                current = [(i,a[i]['group'],a[i]['label']) for i in ids]
                assert identity is None or identity == current
                identity = current
                assert all((a[i]['group'],a[i]['label']) == (b[i]['group'],b[i]['label']) for i in ids)
                differences.append([b[i][metric]-a[i][metric] for i in ids])
            values = np.array(differences)
            groups, inverse = np.unique([x[1] for x in identity], return_inverse=True)
            sums = np.bincount(inverse, weights=values.mean(axis=0)); counts = np.bincount(inverse)
            rng = np.random.default_rng(42)
            draws = []
            for _ in range(2000):
                chosen = rng.integers(0,len(groups),len(groups))
                draws.append(sums[chosen].sum()/counts[chosen].sum())
            results.append(dict(study=study,size=size,cohort=cohort,metric=metric,reference=ref,candidate=cand,
                                n=len(ids),groups=len(groups),mean_delta=float(values.mean()),
                                per_seed_deltas=values.mean(axis=1).tolist(),ci95=np.quantile(draws,[.025,.975]).tolist()))
    scope = 'Paired group bootstrap, 2000 draws, RNG 42; average the three observed seeds before resampling groups. Conditional on these seeds, exploratory, no multiplicity adjustment. Positive correctness delta favors the candidate; positive Brier delta favors the reference. Precision compares training configurations including nonquantized dtype. Transfer is a fixed subsample and uses accuracy only.'
    (ROOT/'summary.json').write_text(json.dumps(dict(scope=scope,input_sha256=hashlib.sha256((ROOT/'inputs.jsonl.gz').read_bytes()).hexdigest(),comparisons=results),indent=2)+'\n')
    lines = ['# Completed extension contrasts','',scope,'','| Study | Size | Cohort | Contrast | Metric | Difference | 95% interval |','|---|---|---|---|---|---:|---|']
    for r in results:
        scale = 100 if r['metric']=='correct' else 1
        unit = ' pp' if scale==100 else ''
        lo,hi = r['ci95']
        lines.append(f"| {r['study']} | {r['size']} | {r['cohort']} | {r['candidate']} minus {r['reference']} | {r['metric']} | {scale*r['mean_delta']:+.4f}{unit} | [{scale*lo:+.4f}, {scale*hi:+.4f}]{unit} |")
    lines += ['', 'Reproduce: `python scripts/analyze_extensions.py`. Use `--extract` only where private original predictions are present. Compact inputs contain IDs, groups, labels and numeric outcomes, never request text. The intervals do not establish a precision effect for 9B or RL, and transfer uncertainty does not represent all unfamiliar tasks.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');args=parser.parse_args()
    if args.extract: extract()
    analyze()
