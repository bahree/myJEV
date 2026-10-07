"""Unrelated human-labelled sentiment transfer; no fitting on SST-2."""
import hashlib
import json
import re
import time
from pathlib import Path
from datasets import load_dataset
from myjev.data import read_jsonl, write_jsonl
from myjev.inference import DecisionModel
from myjev.metrics import summarize

OUT=Path('results/sst2-transfer-v1')
DATA=Path('data/sst2-transfer-v1/requests.jsonl')


def sha_text(text):return hashlib.sha256(text.encode()).hexdigest()


def prepare(plan):
    source=load_dataset(plan['dataset'],revision=plan['dataset_revision'],split=plan['split'])
    rows=[];public=[]
    for index,item in enumerate(source):
        text=item['sentence'];label=['negative','positive'][int(item['label'])]
        row={'id':f'sst2-validation-{item["idx"]}','group':sha_text(re.sub(r'\s+',' ',text.lower()).strip()),'label':label,'label_source':'human','context':text,'instructions':plan['instructions'],'candidates':plan['candidates']}
        rows.append(row);public.append({'id':row['id'],'source_row':index,'label':label,'group':row['group'],'text_sha256':sha_text(text)})
    if len(rows)!=872:raise ValueError('unexpected validation size')
    DATA.parent.mkdir(parents=True,exist_ok=True);write_jsonl(DATA,rows);write_jsonl(OUT/'selection.jsonl',public)
    record={'rows':len(rows),'class_counts':{k:sum(r['label']==k for r in rows) for k in ['negative','positive']},'request_file_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((OUT/'selection.jsonl').read_bytes()).hexdigest(),'dataset_revision':plan['dataset_revision'],'text_redistributed':False}
    (OUT/'data-preparation.json').write_text(json.dumps(record,indent=2)+'\n')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    plan=json.loads((OUT/'frozen-plan.json').read_text());execution=json.loads((OUT/'execution-freeze.json').read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=execution['script_sha256']:raise ValueError('script changed')
    if a.prepare:prepare(plan);return
    preparation=json.loads((OUT/'data-preparation.json').read_text())
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=preparation['request_file_sha256']:raise ValueError('request data changed')
    rows=read_jsonl(DATA);reports=[]
    for spec in plan['models']:
        target=OUT/spec['size'];target.mkdir(exist_ok=True)
        if (target/'metrics.json').exists():
            reports.append(json.loads((target/'metrics.json').read_text()));continue
        model=DecisionModel.load(spec['repo_id'],revision=spec['revision'],device='cuda:0')
        bank=json.loads(Path(spec['threshold_source']).read_text());thresholds={k:v['threshold'] for k,v in bank['operating_points'].items()}
        if bank['calibration_sha256']!=model.manifest['calibration_revision'] or bank['temperature']!=model.manifest['temperature']:raise ValueError('threshold calibration mismatch')
        preds=[];t=time.perf_counter()
        with (target/'predictions.jsonl').open('w') as f:
            for row in rows:
                result=model.score({k:row[k] for k in ['context','instructions','candidates']})
                result.update({k:row[k] for k in ['id','group','label','label_source']})
                f.write(json.dumps(result)+'\n');f.flush();preds.append(result)
                if len(preds)%100==0:print(json.dumps({'size':spec['size'],'completed':len(preds),'total':len(rows),'elapsed':time.perf_counter()-t}),flush=True)
        metrics=summarize(preds,thresholds);metrics.update(model=spec['repo_id'],revision=spec['revision'],artifact_revision=model.manifest['artifact_revision'],banking_threshold_source=spec['threshold_source'],elapsed_seconds=time.perf_counter()-t,scope=plan['limits'])
        (target/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');reports.append(metrics)
        del model
        import gc,torch
        gc.collect();torch.cuda.empty_cache()
    (OUT/'summary.json').write_text(json.dumps({'dataset':plan['dataset'],'split':plan['split'],'n':len(rows),'calibration':plan['calibration'],'scope':plan['limits'],'models':reports},indent=2)+'\n')

if __name__=='__main__':main()
