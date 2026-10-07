"""Post-hoc paired group contrasts; private archive predictions stay private."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('results/archive-machine-v2')
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def contrast(before,after,repeats=5000):
    a={r['id']:r for r in before};b={r['id']:r for r in after}
    assert len(a)==len(before) and len(b)==len(after) and a.keys()==b.keys()
    ids=sorted(a);assert all((a[i]['group'],a[i]['label'])==(b[i]['group'],b[i]['label']) for i in ids)
    groups,inverse=np.unique([a[i]['group'] for i in ids],return_inverse=True);counts=np.bincount(inverse)
    metrics={}
    for metric in ('agreement','correctness_brier'):
        def value(r):
            c=float(r['selected_id']==r['label']);return c if metric=='agreement' else (r['confidence']-c)**2
        delta=np.array([value(b[i])-value(a[i]) for i in ids]);sums=np.bincount(inverse,weights=delta)
        rng=np.random.default_rng(42);samples=[]
        for _ in range(repeats):
            ix=rng.integers(0,len(groups),len(groups));samples.append(sums[ix].sum()/counts[ix].sum())
        metrics[metric]={'after_minus_before':float(delta.mean()),'paired_group_ci95':np.quantile(samples,[.025,.975]).tolist()}
    return {'n':len(ids),'groups':len(groups),'metrics':metrics}

def main():
    pairs={'archive':(ROOT/'unadapted/predictions.jsonl',ROOT/'adapted/evaluation/predictions.jsonl'),
           'banking':(Path('results/longer-v1/4b/main/seed-11/continued_sft/evaluation/predictions.jsonl'),ROOT/'forgetting/predictions.jsonl')}
    for cohort in ('near','distant','oos-none','unsupported-near-none','oos-deferral','unsupported-near-deferral'):
        pairs['clinc/'+cohort]=(Path(f'results/generalization-v1/4b/seed-11/continued_sft/transfer/{cohort}-predictions.jsonl'),ROOT/f'clinc-forgetting/{cohort}-predictions.jsonl')
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');args=parser.parse_args()
    compact=ROOT/'paired-change-inputs.json.gz';provenance=ROOT/'paired-change-sources.json'
    if args.extract:
        encoded={}
        for name,(ap,bp) in pairs.items():
            before,after=read(ap),read(bp);a={r['id']:r for r in before};b={r['id']:r for r in after}
            assert len(a)==len(before) and len(b)==len(after) and a.keys()==b.keys()
            assert all((r['group'],r['label'])==(b[i]['group'],b[i]['label']) for i,r in a.items())
            group_rank={g:n for n,g in enumerate(sorted({r['group'] for r in a.values()}))}
            encoded[name]=[{'id':hashlib.sha256(('paired-example:'+i).encode()).hexdigest(),
                            'group':f"g{group_rank[a[i]['group']]:06d}-"+hashlib.sha256(('paired-group:'+a[i]['group']).encode()).hexdigest(),
                            'correct_before':int(a[i]['selected_id']==a[i]['label']),
                            'correct_after':int(b[i]['selected_id']==b[i]['label']),
                            'confidence_before':a[i]['confidence'],'confidence_after':b[i]['confidence']}
                           for i in sorted(a)]
        compact.write_bytes(gzip.compress(json.dumps(encoded,separators=(',',':')).encode(),mtime=0))
        provenance.write_text(json.dumps({'source_sha256':{str(p):digest(p) for pair in pairs.values() for p in pair},
                                         'privacy':'Only hashed example/group IDs, correctness events and confidence values; no label IDs, selected classes, text or annotation content.'},indent=2)+'\n')
    encoded=json.loads(gzip.decompress(compact.read_bytes()))
    results={}
    for name,rows in encoded.items():
        # Internal event labels normalize correctness only; no actual class labels
        # are present in the public compact input.
        views=[]
        for phase in ('before','after'):
            views.append([{'id':r['id'],'group':r['group'],'label':'correct',
                           'selected_id':'correct' if r['correct_'+phase] else 'incorrect',
                           'confidence':r['confidence_'+phase]} for r in rows])
        results[name]=contrast(*views)
    scope='Post-hoc paired group bootstrap, 5,000 draws, RNG42, after minus before. Resamples original post groups for archive and recorded example groups for public tasks. Conditional on this one trained seed and available sample; no multiplicity adjustment or uncertainty over teacher correctness. Archive agreement is with unreviewed machine labels. Brier uses each deployed scalar correctness head. Deferral accuracy is zero by construction and not a performance measure.'
    report={'scope':scope,'results':results,'source_sha256':json.loads(provenance.read_text())['source_sha256'],'compact_input_sha256':digest(compact),'script_sha256':digest(Path(__file__))}
    (ROOT/'paired-changes.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Paired archive and forgetting changes','',scope,'','| Cohort | N / groups | Agreement / accuracy change, pp | 95% paired group interval, pp | Brier change | 95% paired group interval |','|---|---:|---:|---|---:|---|']
    for name,r in results.items():
        c=r['metrics']['agreement'];b=r['metrics']['correctness_brier'];lo,hi=c['paired_group_ci95'];bl,bh=b['paired_group_ci95']
        lines.append(f"| {name} | {r['n']} / {r['groups']} | {100*c['after_minus_before']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {b['after_minus_before']:+.4f} | [{bl:+.4f}, {bh:+.4f}] |")
    lines+=['','Positive agreement change is better; positive Brier change is worse. These are changes in outcomes, not causal attribution: the study has one adaptation run and no seed population inference. Raw archive predictions/text are private. Regenerate intervals publicly with `python scripts/analyze_archive_changes.py` from compact hashed IDs/groups, correctness events and confidence values. `--extract` requires original private predictions and recreates that compact input; no model inference is needed.']
    (ROOT/'paired-changes.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
