"""Real inference at the artifact's configured caps; no accuracy guarantee implied."""
import argparse
import json
import time
from pathlib import Path
import torch
from myjev.inference import DecisionModel
from myjev.prompt import render
from myjev.schema import ScoreRequest
p=argparse.ArgumentParser()
p.add_argument('--artifact',required=True)
p.add_argument('--output',required=True)
p.add_argument('--device',default='cuda:0')
a=p.parse_args()
model=DecisionModel.load(a.artifact,device=a.device)
limit=model.manifest['max_tokens']
results=[]
for count in (2,model.manifest['max_candidates']):
    request={'context':'','instructions':'Choose one candidate.','candidates':[{'id':str(i),'description':f'Option {i}'} for i in range(count)]}
    def length(n):
        request['context']=' test'*n
        text=render(ScoreRequest.model_validate(request),model.manifest['aliases'])
        return len(model.tokenizer.encode(text,add_special_tokens=True))
    low,high=0,limit
    while low<high:
        mid=(low+high+1)//2
        if length(mid)<=limit:
            low=mid
        else:
            high=mid-1
    actual_tokens=length(low)
    assert actual_tokens==limit,(actual_tokens,limit)
    torch.cuda.reset_peak_memory_stats(a.device)
    start=time.perf_counter()
    response=model.score(request)
    elapsed=time.perf_counter()-start
    assert len(response['selection_scores'])==count
    assert abs(sum(response['selection_scores'].values())-1)<1e-5
    too_many=length(low+1)
    assert too_many>limit
    try:
        model.score(request)
        raise AssertionError('oversized input accepted')
    except ValueError:
        pass
    results.append({'candidates':count,'accepted_tokens':actual_tokens,'rejected_tokens':too_many,
                    'seconds':elapsed,'peak_vram_bytes':torch.cuda.max_memory_allocated(a.device)})
Path(a.output).write_text(json.dumps({'artifact_revision':model.manifest['artifact_revision'],
    'results':results,'scope':'synthetic cap validation only; latency measured during other study jobs'},indent=2))
