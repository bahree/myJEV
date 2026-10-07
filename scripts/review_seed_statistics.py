"""Expose the seed axis alongside existing conditional group intervals."""
import hashlib
import json
from pathlib import Path
from statistics import stdev


def main():
    root=Path('results/review-seeds-v1');root.mkdir(parents=True,exist_ok=True)
    sources=[Path('results/longer-v1/paired-analysis.json'),Path('results/extension-analysis-v1/summary.json')]
    rows=[]
    for path in sources:
        for r in json.loads(path.read_text())['comparisons']:
            row=dict(r)
            row['study']=row.get('study','main')
            row['group_ci95']=row.get('group_ci95',row.get('ci95'))
            row['seed_delta_sd']=stdev(row['per_seed_deltas'])
            rows.append(row)
    result={'scope':'Existing paired estimates unchanged; expose seeds 11,22,33. Conditional group intervals hold seeds fixed. Seed SD is descriptive, not a confidence interval for future runs. No multiplicity adjustment.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'comparisons':rows}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Test-sample uncertainty and training-seed spread','','The same test examples were evaluated for each seed. The group-bootstrap intervals below hold the three trained runs fixed; they do not include uncertainty over future training seeds. Seed deltas and their sample SD show that separate axis. Three runs are insufficient for a dependable distributional model of training variability. No new training was performed.','','| Study / cohort | Size | Contrast | Metric | Seed 11 | Seed 22 | Seed 33 | Mean | Seed SD | Conditional group 95% interval |','|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        scale=100 if r['metric']=='correct' else 1;unit=' pp' if scale==100 else ''
        values=' | '.join(f'{x*scale:+.4f}{unit}' for x in r['per_seed_deltas'])
        lo,hi=r['group_ci95']
        lines.append(f"| {r['study']} / {r.get('cohort','BANKING77')} | {r['size']} | {r['candidate']} minus {r['reference']} | {r['metric']} | {values} | {r['mean_delta']*scale:+.4f}{unit} | {r['seed_delta_sd']*scale:.4f}{unit} | [{lo*scale:+.4f}, {hi*scale:+.4f}]{unit} |")
    lines+=['','Exact RL exceeds continued supervision at 4B in two of three seeds and in the mean. Its conditional test-group interval excludes zero, while the observed seed deltas include a negative value. At 0.8B continued supervision leads exact RL in all three seeds. These statements describe the observed runs, not future-run guarantees. Original Brier contrasts use different confidence sources; the matched post-hoc follow-up is reported separately.','', 'Reproduce: `python scripts/review_seed_statistics.py`.','']
    (root/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
