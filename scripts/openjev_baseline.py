"""Optional external baseline. Never starts an endpoint or downloads weights."""
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
import httpx
from myjev.data import read_jsonl,write_jsonl,validate_isolation
from myjev.metrics import summarize,thresholds_from_calibration
p=argparse.ArgumentParser()
p.add_argument('--url',required=True)
p.add_argument('--revision',required=True)
p.add_argument('--data',default='data/banking77')
p.add_argument('--output',required=True)
p.add_argument('--allow-multipass',action='store_true')
p.add_argument('--limit',type=int)
a=p.parse_args()
if not re.fullmatch(r'[0-9a-f]{40}',a.revision):
    p.error('record an immutable model revision')
parts={s:read_jsonl(Path(a.data)/f'{s}.jsonl') for s in ('calibration','test')}
validate_isolation(parts)
if not a.allow_multipass and any(len(r['candidates'])>52 for rs in parts.values() for r in rs):
    p.error('OpenJev needs multiple passes above 52 choices; explicitly allow this and report it separately')
headers={'Authorization':f'Bearer {os.environ["OPENJEV_TOKEN"]}'} if os.getenv('OPENJEV_TOKEN') else {}
outputs={}
with httpx.Client(timeout=120,headers=headers) as client:
    for split,rows in parts.items():
        if a.limit:
            rows=random.Random(43 if split=='calibration' else 42).sample(rows,min(a.limit,len(rows)))
        outputs[split]=[]
        for row in rows:
            request={'model':'openjev','state':row['context'],'questions':{'decision':{
                'type':'choice','instructions':row['instructions'],
                'criteria':{c['id']:c['description'] for c in row['candidates']}}}}
            start=time.perf_counter()
            response=client.post(a.url.rstrip('/')+'/v1/systemone',json=request)
            response.raise_for_status()
            result=response.json()['answers']['decision']
            scores={k:float(v) for k,v in result['probabilities'].items()}
            if set(scores)!={c['id'] for c in row['candidates']} or result['choice'] not in scores:
                raise ValueError('upstream candidate mismatch')
            if any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()) or sum(scores.values())<=0:
                raise ValueError('invalid upstream probability distribution')
            total=sum(scores.values())
            scores={k:v/total for k,v in scores.items()}
            count=len(scores)
            outputs[split].append({'id':row['id'],'group':row['group'],'label':row['label'],
                'selected_id':result['choice'],'selection_scores':scores,'confidence':scores[result['choice']],
                'confidence_mode':'selected_probability_proxy','upstream_confidence':result['confidence'],
                'estimated_backbone_reads':1 if count<=52 else math.ceil(count/52)+1,
                'seconds':time.perf_counter()-start})
cal=outputs['calibration']
thresholds=thresholds_from_calibration([r['selected_id']==r['label'] for r in cal],[r['confidence'] for r in cal])
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
for name,rows in outputs.items():
    write_jsonl(out/f'{name}-predictions.jsonl',rows)
metrics=summarize(outputs['test'],thresholds)
metrics.update(model='openjev/openjev',declared_revision=a.revision,multipass_allowed=a.allow_multipass,
    probability_rounding='upstream four-decimal scores renormalized by this client',
    execution_scope='external endpoint; remote revision must be independently verified')
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
