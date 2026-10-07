"""Serve a pinned Hub artifact directly in a local GPU container and compare output."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import httpx


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True);p.add_argument('--repo',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--port',type=int,default=18089)
    p.add_argument('--output',type=Path,default=Path('results/release-container-v2/hub-http.json'))
    a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',a.revision):raise ValueError('Pinned artifact revision required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    name='myjev-hub-smoke-'+str(time.time_ns())
    cmd=['docker','run','-d','--name',name,'--gpus',f'device={a.gpu}','-p',f'127.0.0.1:{a.port}:8000',
         '-v',f'{a.cache.resolve()}:/hf:ro','-e','HF_HOME=/hf','-e','HF_HUB_OFFLINE=1',
         '-e',f'MYJEV_ARTIFACT={a.repo}','-e',f'MYJEV_REVISION={a.revision}',
         '-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',a.image]
    start=time.perf_counter();subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{a.port}',timeout=60) as client:
            while True:
                try:
                    if client.get('/readyz').status_code==200:break
                except httpx.HTTPError:pass
                if time.perf_counter()-start>240:raise TimeoutError('Readiness deadline')
                time.sleep(1)
            ready=time.perf_counter()-start
            request=json.loads(Path('examples/request.json').read_text())
            expected=json.loads(a.expected.read_text())['response']
            response=client.post('/score',json=request);response.raise_for_status();assert response.json()==expected
            alias=client.post('/generate',json=request);alias.raise_for_status();assert alias.json()==expected
            assert client.get('/health').status_code==200
            result={'image':a.image,'image_id':subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip(),
                    'repo_id':a.repo,'revision':a.revision,'direct_hub_load':True,'http_equal_to_host_python':True,
                    'managed_aliases_equal':True,'ready_seconds':ready,'response':response.json(),
                    'scope':'Pinned Hub reference loaded directly in source-refreshed container from read-only cached Hub storage; no cloud or registry deployment. Readiness observation is not an isolated latency benchmark.'}
            a.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        with a.output.with_suffix('.log').open('w') as log:subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['docker','rm','-f',name],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':main()
