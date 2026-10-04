"""Use BANKING77 calibration thresholds unchanged on CLINC150 cohorts."""
import argparse
import json
import random
from pathlib import Path
from myjev.data import read_jsonl, write_jsonl
from myjev.evaluate import predict
from myjev.inference import DecisionModel
from myjev.metrics import summarize
p=argparse.ArgumentParser()
p.add_argument('--artifact',required=True)
p.add_argument('--banking-evaluation',required=True)
p.add_argument('--output',required=True)
p.add_argument('--limit',type=int,default=256)
p.add_argument('--device',default='cuda:0')
a=p.parse_args()
bank=json.loads((Path(a.banking_evaluation)/'metrics.json').read_text())
model=DecisionModel.load(a.artifact,device=a.device)
if model.manifest['artifact_revision']!=bank['artifact_revision']:
    raise ValueError('thresholds must come from this exact artifact')
thresholds={k:v['threshold'] for k,v in bank['operating_points'].items()}
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
for name in ('near','distant','unsupported-near-none','unsupported-near-deferral','oos-none','oos-deferral'):
    rows=read_jsonl(f'data/clinc150/{name}.jsonl')
    rows=random.Random(42).sample(rows,min(a.limit,len(rows)))
    preds=predict(model,rows)
    write_jsonl(out/f'{name}-predictions.jsonl',preds)
    m=summarize(preds,thresholds)
    m.update(threshold_source=str(a.banking_evaluation),artifact_revision=model.manifest['artifact_revision'],subset_limit=a.limit)
    (out/f'{name}-metrics.json').write_text(json.dumps(m,indent=2))
