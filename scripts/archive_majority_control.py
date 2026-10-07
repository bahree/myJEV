"""Exploratory rubric-majority control, fitted without held-out test labels.

Only aggregates and source hashes are emitted. Archive texts and row IDs remain
in the private data files. This post-hoc control is not a frozen main comparison.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def fit(train, calibration):
    rubrics=sorted({r['rubric'] for r in train})
    controls={}
    for rubric in rubrics:
        training=[r for r in train if r['rubric']==rubric]
        reserved=[r for r in calibration if r['rubric']==rubric]
        if not reserved:raise ValueError(f'No calibration examples for {rubric}')
        counts=Counter(r['label'] for r in training)
        selected=min(counts,key=lambda label:(-counts[label],label))
        confidence=sum(r['label']==selected for r in reserved)/len(reserved)
        controls[rubric]={'selected_label':selected,'confidence':confidence,
                         'training_n':len(training),'training_label_counts':dict(sorted(counts.items())),
                         'calibration_n':len(reserved),'calibration_selected_correct':sum(r['label']==selected for r in reserved)}
    if {r['rubric'] for r in calibration}-set(controls):raise ValueError('Unseen calibration rubric')
    return controls


def evaluate(test, controls):
    if not test:raise ValueError('Empty test set')
    if {r['rubric'] for r in test}-set(controls):raise ValueError('Unseen test rubric')
    def summarize(rows):
        correct=[int(r['label']==controls[r['rubric']]['selected_label']) for r in rows]
        return {'n':len(rows),'reference_agreement':sum(correct)/len(rows),
                'correctness_brier':sum((controls[r['rubric']]['confidence']-c)**2 for r,c in zip(rows,correct))/len(rows)}
    return {'overall':summarize(test),'per_rubric':{rubric:summarize([r for r in test if r['rubric']==rubric]) for rubric in sorted({r['rubric'] for r in test})}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/archive-machine-v2'))
    parser.add_argument('--output',type=Path,default=Path('results/archive-machine-v2'));args=parser.parse_args()
    paths={split:args.data/f'{split}.jsonl' for split in ('train','calibration','test')}
    parts={split:[json.loads(line) for line in path.read_text().splitlines() if line.strip()] for split,path in paths.items()}
    seen_groups={};seen_ids=set()
    for split,rows in parts.items():
        for row in rows:
            if row['split']!=split:raise ValueError('Split field mismatch')
            if row['id'] in seen_ids:raise ValueError('Duplicate record ID')
            seen_ids.add(row['id'])
            if row['group'] in seen_groups and seen_groups[row['group']]!=split:raise ValueError('Group overlap across partitions')
            seen_groups[row['group']]=split
            if row.get('human_reviewed') is not False or row.get('label_source')!='llm':raise ValueError('Unexpected label provenance')
    controls=fit(parts['train'],parts['calibration'])
    for split,rows in parts.items():
        for row in rows:
            if controls[row['rubric']]['selected_label'] not in {c['id'] for c in row['candidates']}:raise ValueError('Majority label absent from candidates')
    metrics=evaluate(parts['test'],controls)
    scope=('Post-hoc exploratory control, not preregistered. Select each rubric majority using training labels only; lexical label-ID order resolves training ties. '
           'Confidence is selected-class correctness frequency on calibration labels only. Test labels are used only for the final metrics, not selection, confidence or thresholds. '
           'Agreement is against unreviewed machine annotations, not human ground truth. Multiple rubric decisions from one post are correlated; no IID or population intervals are claimed. '
           'This control tests whether label imbalance is a plausible explanation of adaptation performance, not a causal decomposition of any gain.')
    report={'scope':scope,'human_reviewed':False,'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'controls':controls,**metrics}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'majority-control.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Archive majority-per-rubric control','',scope,'','| Rubric | Train N | Calibration N | Selected label | Calibration confidence | Test N | Reference agreement | Correctness Brier |','|---|---:|---:|---|---:|---:|---:|---:|']
    for rubric,m in metrics['per_rubric'].items():
        c=controls[rubric]
        lines.append(f"| {rubric} | {c['training_n']} | {c['calibration_n']} | {c['selected_label']} | {c['confidence']:.4f} | {m['n']} | {100*m['reference_agreement']:.2f}% | {m['correctness_brier']:.4f} |")
    m=metrics['overall'];lines+=['',f"Overall: {m['n']} test decisions, {100*m['reference_agreement']:.2f}% reference agreement, correctness Brier {m['correctness_brier']:.4f}.",'','Reproduce where private prepared data are available: `python scripts/archive_majority_control.py`. Public output contains rubric-level aggregate counts and file hashes only; no archive text, request IDs or individual annotations.']
    (args.output/'majority-control.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(metrics))

if __name__=='__main__':main()
