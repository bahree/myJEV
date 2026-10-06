"""Export exploratory teacher-labeled splits; preserve abstentions and provenance."""
import argparse
import json
from pathlib import Path
from myjev.data import read_jsonl, write_jsonl, validate_isolation
from myjev.judging import sha, validate_judgment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--requests',default='annotation/requests.jsonl')
    p.add_argument('--labels',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    requests=read_jsonl(a.requests);records=read_jsonl(a.labels)
    labels={r['id']:r for r in records}
    if len(labels)!=len(records):raise ValueError('duplicate labels')
    parts={s:[] for s in ['train','validation','calibration','test']};exclusions=[]
    for row in requests:
        if row['split']=='development':continue
        label=labels.get(row['id'])
        if not label:raise ValueError('missing judge response: '+row['id'])
        if label.get('label_source')!='llm' or label.get('human_reviewed') is not False or label.get('reviewed') is not False:
            raise ValueError('expected explicitly machine-generated, unreviewed records')
        if label['visible_text_sha256']!=sha(row['context']):raise ValueError('text hash mismatch')
        if not label.get('valid'):
            exclusions.append({'id':row['id'],'split':row['split'],'reason':'invalid judge output'});continue
        value=validate_judgment(row,{k:label[k] for k in ['label','status','evidence','explanation']})
        if value['status']!='labelled' or value['label'] is None:
            exclusions.append({'id':row['id'],'split':row['split'],'reason':value['status']});continue
        parts[row['split']].append({**row,'label':value['label'],'label_source':'llm','human_reviewed':False,
            'judge_model':label['resolved_model'],'judge_prompt_sha256':label['prompt_sha256']})
    validate_isolation(parts)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for split,rows in parts.items():write_jsonl(out/f'{split}.jsonl',rows)
    write_jsonl(out/'excluded-requests.jsonl',exclusions)
    (out/'manifest.json').write_text(json.dumps({'purpose':'exploratory teacher-labeled archive adaptation',
        'label_source':'llm','human_reviewed':False,'metric_semantics':'agreement with machine reference labels',
        'requests_sha256':sha(Path(a.requests).read_text()),'labels_sha256':sha(Path(a.labels).read_text()),
        'counts':{k:len(v) for k,v in parts.items()},'excluded':len(exclusions),
        'limitations':['abstentions excluded explicitly; report coverage by rubric and split','human audit remains separate','development groups excluded']},indent=2))

if __name__=='__main__':main()
