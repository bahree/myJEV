"""Regenerate synthetic scratch findings and figure from saved metrics."""
import json
from pathlib import Path
from statistics import mean,stdev
import hashlib

ROOT=Path('results/scratch-study-v1')


def main():
    assert json.loads((ROOT/'status.json').read_text())['state']=='completed'
    rows=[];sources={}
    for method in ('sft','continued_sft','exact','sampled'):
        for mode in ('evaluation','temperature','constant') if method in ('sft','continued_sft') else ('evaluation','constant'):
            ms=[]
            for seed in (11,22,33):
                path=ROOT/f'main/seed-{seed}/{method}/{mode}/metrics.json'
                raw=path.read_bytes();m=json.loads(raw);assert m['n']==512
                sources[str(path)]=hashlib.sha256(raw).hexdigest();ms.append(m)
            rows.append(dict(method=method,mode=mode,accuracy=mean(m['accuracy'] for m in ms),
                             accuracy_seed_sd=stdev(m['accuracy'] for m in ms),brier=mean(m['correctness_brier'] for m in ms),
                             known_confidence_mse=mean(m['known_probability_confidence_mse'] for m in ms),
                             known_distribution_brier=mean(m['known_distribution_brier'] for m in ms),
                             per_seed_accuracy=[m['accuracy'] for m in ms]))
    (ROOT/'summary.json').write_text(json.dumps(dict(rows=rows,source_sha256=sources),indent=2)+'\n')
    lines=['# Controlled scratch study', '',
           'Completed 13,000 updates: two 500-update SFT tuning trials, then three seeds with 1,000 initial SFT and three separate 1,000-update continuations. Batch size 8; FP32; 201,175 parameters. Same initial checkpoint and subsequent examples per seed. Shared SFT-selected learning rate across methods, unlike Qwen\'s method-specific tuning. This is a teaching comparison, not an optimized RL claim.', '',
           '| Method | Confidence | Accuracy mean | Seed SD | Correctness Brier | Confidence MSE against known probability |',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        mode=r['mode'] if r['mode'] in ('temperature','constant') else ('scalar' if r['method'] in ('sft','continued_sft') else 'policy')
        lines.append(f"| {r['method']} | {mode} | {r['accuracy']:.2%} | {r['accuracy_seed_sd']:.2%} | {r['brier']:.4f} | {r['known_confidence_mse']:.4f} |")
    lines+=['','The 512-row test has held-out templates and ambiguous pairings, with a quarter of rows having two equally likely outcomes. An ideal selector has expected 87.5% correctness under the generator, but realized sampled-label accuracy can differ. Known-probability MSE compares confidence with the actual conditional probability of the selected action, separating confidence error from latent-outcome noise.', '',
            'Seed SD is not a confidence interval. Synthetic success is not natural-language competence. These data expose whether direct scoring and confidence learning work under a declared artificial rule; there is no claim of Jev reproduction.', '',
            '![Synthetic accuracy across seeds](figures/synthetic-accuracy.png)', '',
            'Reproduce: `python scripts/summarize_scratch_study.py`. All source metrics, configurations and training logs are retained. The separate BANKING77 diagnostic has a different input budget and parameter count and must not be pooled into this table.', '']
    (ROOT/'summary.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r for r in rows if r['mode']=='evaluation']
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    ax.bar([r['method'] for r in x],[100*r['accuracy'] for r in x],yerr=[100*r['accuracy_seed_sd'] for r in x],capsize=5,color=['#5079a8','#5079a8','#b57536','#b57536'])
    ax.axhline(87.5,color='black',linestyle=':',label='Generator expected optimum')
    ax.set(ylabel='Sampled-label accuracy (%)',ylim=(0,100),title='201,175 parameters: controlled routing, three seeds')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.text(.5,-.02,'Error bars: seed SD. Toy-rule evidence, not natural-language performance.',ha='center',fontsize=9)
    (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/synthetic-accuracy.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
