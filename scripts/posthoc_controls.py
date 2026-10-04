"""Fit temperature and constant controls from saved calibration predictions only."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from myjev.calibration import fit_temperature
from myjev.data import read_jsonl,write_jsonl
from myjev.metrics import summarize,thresholds_from_calibration
p=argparse.ArgumentParser()
p.add_argument('--evaluation',required=True)
p.add_argument('--artifact',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.evaluation)
source_manifest=json.loads((Path(a.artifact)/'manifest.json').read_text())
source_metrics=json.loads((root/'metrics.json').read_text())
if source_metrics['artifact_revision'] != source_manifest['artifact_revision']:
    raise ValueError('evaluation/artifact revision mismatch')
if (Path(a.output)/'temperature-artifact').exists():
    raise ValueError('calibrated artifact already exists; choose a new output')
cal=read_jsonl(root/'calibration-predictions.jsonl')
test=read_jsonl(root/'predictions.jsonl')
if {r['group'] for r in cal} & {r['group'] for r in test}:
    raise ValueError('calibration/test overlap')
logits=np.asarray([r['selection_logits'] for r in cal])
targets=[list(r['selection_scores']).index(r['label']) for r in cal]
t=fit_temperature(logits,targets)
base_rate=float(np.mean([r['selected_id']==r['label'] for r in cal]))
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
for mode in ('temperature','constant','half'):
    parts={}
    for name,rows in [('calibration',cal),('test',test)]:
        preds=[]
        for r in rows:
            z=np.asarray(r['selection_logits'])/t
            p=np.exp(z-z.max());p/=p.sum()
            scores=dict(zip(r['selection_scores'],p.tolist())) if mode=='temperature' else r['selection_scores']
            confidence=max(scores.values()) if mode=='temperature' else (base_rate if mode=='constant' else .5)
            preds.append({**r,'selection_scores':scores,'confidence':confidence,'confidence_mode':mode})
        parts[name]=preds
    thresholds=thresholds_from_calibration([r['selected_id']==r['label'] for r in parts['calibration']],[r['confidence'] for r in parts['calibration']])
    m=summarize(parts['test'],thresholds)
    m.update(temperature=t,base_rate=base_rate,calibration_n=len(cal),confidence_mode=mode,
             source_artifact_revision=source_manifest['artifact_revision'],
             calibration_sha256=hashlib.sha256((root/'calibration-predictions.jsonl').read_bytes()).hexdigest())
    (out/f'{mode}-metrics.json').write_text(json.dumps(m,indent=2))
    write_jsonl(out/f'{mode}-predictions.jsonl',parts['test'])
artifact=out/'temperature-artifact'
if artifact.exists():
    raise ValueError('calibrated artifact already exists; choose a new output')
shutil.copytree(a.artifact,artifact)
m=json.loads((artifact/'manifest.json').read_text())
m.pop('artifact_revision',None)
m.update(temperature=t,confidence_mode='selection',calibration_revision=hashlib.sha256((root/'calibration-predictions.jsonl').read_bytes()).hexdigest())
m['artifact_revision']=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
(artifact/'manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps({'temperature':t,'base_rate':base_rate,'artifact':str(artifact)}))
