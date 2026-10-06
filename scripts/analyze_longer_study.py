"""Compact paired evidence, group bootstrap and figures for the completed study.

Run --extract on the private evidence checkout once. Default reads only the
compact public input file, which contains no request text or candidate scores.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/longer-v1')
SIZES = ('0.8b', '4b', '9b')
SEEDS = (11, 22, 33)


def extract():
    rows, sources = [], {}
    for size in SIZES:
        for seed in SEEDS:
            for method in ('continued_sft', 'exact', 'sampled'):
                root = ROOT / size / 'main' / f'seed-{seed}' / method
                for mode, suffix in [('deployed', 'evaluation/predictions.jsonl')] + (
                    [('temperature', 'posthoc/temperature-predictions.jsonl')]
                    if method == 'continued_sft' else []):
                    path = root / suffix
                    h = hashlib.sha256()
                    seen = set()
                    with path.open('rb') as f:
                        for line in f:
                            h.update(line)
                            r = json.loads(line)
                            assert r['id'] not in seen
                            seen.add(r['id'])
                            correct = int(r['selected_id'] == r['label'])
                            rows.append(dict(size=size, seed=seed, method=method, mode=mode,
                                             id=r['id'], group=r['group'], label=r['label'],
                                             correct=correct, brier=(r['confidence']-correct)**2))
                    assert len(seen) == 3080
                    sources[str(path)] = h.hexdigest()
    (ROOT / 'paired-inputs.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows).encode(), mtime=0))
    (ROOT / 'paired-input-sources.json').write_text(json.dumps(sources, indent=2)+'\n')


def analyze():
    index = {}
    with gzip.open(ROOT / 'paired-inputs.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            key = (r['size'], r['seed'], r['method'], r['mode'])
            bucket = index.setdefault(key, {})
            assert r['id'] not in bucket
            bucket[r['id']] = r
    comparisons = []
    for size in SIZES:
        for reference, candidate, metric in [
            (('continued_sft', 'deployed'), ('exact', 'deployed'), 'correct'),
            (('continued_sft', 'deployed'), ('sampled', 'deployed'), 'correct'),
            (('exact', 'deployed'), ('sampled', 'deployed'), 'correct'),
            (('continued_sft', 'temperature'), ('exact', 'deployed'), 'brier'),
            (('continued_sft', 'temperature'), ('sampled', 'deployed'), 'brier')]:
            arrays, identity = [], None
            for seed in SEEDS:
                a, b = index[(size, seed, *reference)], index[(size, seed, *candidate)]
                assert set(a) == set(b) and len(a) == 3080
                ids = sorted(a)
                current = [(k, a[k]['group'], a[k]['label']) for k in ids]
                assert identity is None or current == identity
                identity = current
                assert all((a[k]['group'], a[k]['label']) == (b[k]['group'], b[k]['label']) for k in ids)
                arrays.append([b[k][metric]-a[k][metric] for k in ids])
            values = np.array(arrays)
            groups, inverse = np.unique([x[1] for x in identity], return_inverse=True)
            sums = np.bincount(inverse, weights=values.mean(axis=0))
            counts = np.bincount(inverse)
            rng = np.random.default_rng(42)
            samples = []
            for _ in range(2000):
                selected = rng.integers(0, len(groups), len(groups))
                samples.append(sums[selected].sum()/counts[selected].sum())
            comparisons.append(dict(size=size, reference='/'.join(reference), candidate='/'.join(candidate),
                                    metric=metric, mean_delta=float(values.mean()),
                                    group_ci95=np.quantile(samples, [.025, .975]).tolist(),
                                    per_seed_deltas=values.mean(axis=1).tolist(), groups=len(groups), n=3080))
    report = dict(scope='Paired group bootstrap, 2000 draws, RNG 42; averages the three observed seeds. Conditional on those seeds, not a population interval over initializations. Exploratory contrasts, no multiple-comparison adjustment. Positive accuracy delta is better; positive Brier delta is worse. Confidence sources differ in Brier contrasts.',
                  input_sha256=hashlib.sha256((ROOT/'paired-inputs.jsonl.gz').read_bytes()).hexdigest(), comparisons=comparisons)
    (ROOT/'paired-analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Paired longer-study contrasts', '', report['scope'], '', '| Size | Candidate minus reference | Metric | Mean delta | 95% group interval |', '|---|---|---|---:|---|']
    for r in comparisons:
        scale = 100 if r['metric'] == 'correct' else 1
        unit = ' pp' if scale == 100 else ''
        lo, hi = r['group_ci95']
        lines.append(f"| {r['size']} | {r['candidate']} minus {r['reference']} | {r['metric']} | {r['mean_delta']*scale:.4f}{unit} | [{lo*scale:.4f}, {hi*scale:.4f}]{unit} |")
    lines += ['', 'Reproduce: `python3 scripts/analyze_longer_study.py`. Compact paired inputs retain example/group IDs, gold label IDs, correctness and squared confidence error. They omit request text. Source prediction hashes are recorded separately; original full predictions are retained in the private evidence archive.', '']
    (ROOT/'paired-analysis.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((ROOT/'summary.json').read_text())['aggregates']
    out = ROOT/'figures'; out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    for i, method in enumerate(('sft','continued_sft','exact','sampled')):
        rs = [next(r for r in rows if (r['size'],r['method'],r['mode']) == (z,method,'deployed')) for z in SIZES]
        ax.errorbar(np.arange(3)+(i-1.5)*.09, [100*r['accuracy_mean'] for r in rs],
                    yerr=[100*r['accuracy_seed_sd'] for r in rs], marker='o', capsize=4, label=method)
    ax.set(xticks=range(3),xticklabels=['0.8B BF16','4B BF16','9B NF4'],ylabel='Accuracy (%)',title='BANKING77: complete test split, three seeds')
    ax.legend(ncol=2); ax.grid(axis='y',alpha=.25)
    fig.text(.5, -.02, 'Error bars: seed SD, not confidence intervals. 4,000 initial + 4,000 continuation updates.',ha='center',fontsize=9)
    fig.savefig(out/'accuracy.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig, ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for method,mode,label in [('continued_sft','deployed','Continued SFT: scalar'),('continued_sft','temperature','Continued SFT: temperature'),('exact','deployed','Exact RL: policy'),('sampled','deployed','Sampled RL: policy')]:
        rs=[next(r for r in rows if (r['size'],r['method'],r['mode'])==(z,method,mode)) for z in SIZES]
        ax.plot(range(3),[r['brier_mean'] for r in rs],marker='o',label=label)
    ax.set(xticks=range(3),xticklabels=['0.8B BF16','4B BF16','9B NF4'],ylabel='Correctness Brier (lower is better)',title='Confidence source matters: three-seed means')
    ax.legend();ax.grid(axis='y',alpha=.25)
    fig.savefig(out/'confidence.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extract',action='store_true');a=p.parse_args()
    if a.extract: extract()
    analyze();plot()
