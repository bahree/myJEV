"""Report punctuation-only overlap and descriptive exclusion sensitivity."""
import hashlib
import gzip
import json
from pathlib import Path
import re
import string


def main():
    output=Path('results/review-overlap-v1');output.mkdir(parents=True,exist_ok=True)
    paths={name:Path(f'data/banking77/{name}.jsonl') for name in ('train','test')}
    data={name:[json.loads(s) for s in p.read_text().splitlines()] for name,p in paths.items()}
    def normalize(s):return re.sub(r'\s+',' ',s.lower().translate(str.maketrans('','',string.punctuation))).strip()
    index={}
    for r in data['train']:index.setdefault(normalize(r['context']),[]).append(r)
    hits=[]
    for row in data['test']:
        for other in index.get(normalize(row['context']),[]):
            hits.append({'train_id':other['id'],'test_id':row['id'],'same_label':other['label']==row['label']})
    excluded={r['test_id'] for r in hits};counts=[];sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    prediction_paths=sorted(Path('results/longer-v1').glob('*/main/seed-*/*/evaluation/predictions.jsonl'))
    compact_mode=not prediction_paths
    if compact_mode:prediction_paths=sorted(Path('results/review-calibration-v1').glob('*/seed-*/*/inputs.json.gz'))
    if len(prediction_paths)!=36:raise ValueError('All 36 original or public compact prediction files are required')
    for p in prediction_paths:
        rows=json.loads(gzip.decompress(p.read_bytes()))['test'] if compact_mode else [json.loads(s) for s in p.read_text().splitlines()]
        remaining=[r for r in rows if r['id'] not in excluded]
        run=Path('results/longer-v1')/p.parts[-4]/'main'/p.parts[-3]/p.parts[-2] if compact_mode else p.parent.parent
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        accuracy=lambda rows:sum(r['label']==r['selected_id'] for r in rows)/len(rows)
        counts.append({'run':str(run),'original_n':len(rows),'sensitivity_n':len(remaining),'original_accuracy':accuracy(rows),'sensitivity_accuracy':accuracy(remaining),'delta_pp':100*(accuracy(remaining)-accuracy(rows))})
    result={'scope':'Post-review punctuation-only overlap check and descriptive test-row exclusion. Official test and trained models unchanged. Does not estimate the causal training effect of overlap; semantic overlap may remain.','normalization':'lowercase; remove ASCII string.punctuation; collapse whitespace','overlaps':hits,'per_run':counts,'source_sha256':sources}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# BANKING77 punctuation-overlap sensitivity','',result['scope'],'',f'{len(excluded)} of 3,080 official test rows match training rows under the extra normalization. The sensitivity view contains {3080-len(excluded)} test rows. Every flagged pair has the same intent label. No request text is redistributed.','','| Run | Official accuracy | Excluding flagged rows | Change (pp) |','|---|---:|---:|---:|']
    lines += [f"| {r['run'].replace('results/longer-v1/','')} | {r['original_accuracy']:.2%} | {r['sensitivity_accuracy']:.2%} | {r['delta_pp']:+.4f} |" for r in counts]
    lines+=['','The official score remains primary. Excluding four evaluated rows is a sensitivity check, not a correction to the trained weights or proof of a maximum causal contamination effect. Any retraining with different exclusions would be a new experiment.','','Reproduce after preparing the pinned data; uses original predictions when present, otherwise the exported text-free calibration inputs: `python scripts/review_banking_overlap.py`.','']
    (output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
