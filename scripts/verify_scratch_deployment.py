"""Verify scratch Python/CLI/HTTP equivalence and record local model timings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import httpx
from myjev import DecisionModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',required=True);p.add_argument('--url',default='http://127.0.0.1:18085')
    p.add_argument('--request',default='examples/scratch-request.json');p.add_argument('--output',required=True);p.add_argument('--gpu',default='cuda:2');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    request=json.loads(Path(a.request).read_text())
    cpu=DecisionModel.load(a.artifact,device='cpu');expected=cpu.score(request)
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    cli=json.loads(subprocess.check_output([sys.executable,'-m','myjev.cli','score','--artifact',a.artifact,'--device','cpu','--input',a.request],env=env))
    with httpx.Client(timeout=120) as client:
        client.get(a.url+'/readyz').raise_for_status()
        response=client.post(a.url+'/score',json=request);response.raise_for_status();http=response.json()
        invalid={**request,'context':'x'*300}
        assert client.post(a.url+'/score',json=invalid).status_code==422
    errors={}
    for name,r in [('cli',cli),('http',http)]:
        assert r['selected_id']==expected['selected_id']
        assert r['artifact_revision']==expected['artifact_revision']
        delta=max([abs(r['confidence']-expected['confidence'])]+[abs(r['selection_scores'][k]-v) for k,v in expected['selection_scores'].items()])
        assert delta<1e-5,(name,delta)
        errors[name]=delta
    results={}
    for device in ['cpu',a.gpu]:
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();m=DecisionModel.load(a.artifact,device=device)
        m.score(request)
        if device.startswith('cuda'):torch.cuda.synchronize(device)
        load=time.perf_counter()-start
        for _ in range(10):m.score(request)
        durations=[]
        for _ in range(100):
            if device.startswith('cuda'):torch.cuda.synchronize(device)
            start=time.perf_counter();m.score(request)
            if device.startswith('cuda'):torch.cuda.synchronize(device)
            durations.append(time.perf_counter()-start)
        results[device]=dict(warm_p50_p95_seconds=np.quantile(durations,[.5,.95]).tolist(),requests=100,
                             load_and_first_score_seconds=load,peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                             weights_bytes=(Path(a.artifact)/'model.safetensors').stat().st_size)
    report=dict(artifact_revision=expected['artifact_revision'],response=expected,max_absolute_difference=errors,
                oversized_http_status=422,model_timings=results,
                scope='Short four-candidate synthetic request; CPU two threads and one A30. HTTP container on same host. This is not a matched Qwen or Jev benchmark.')
    (out/'equivalence-and-model-timings.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
