"""Verify a pinned GPU patch image, input rejection, installed files, and CLI/HTTP equality."""
import argparse
import json
import hashlib
from datetime import datetime, timezone
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
    p.add_argument('--output',type=Path,default=Path('results/container-registry-v3/gpu-check.json'))
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
            statuses={}
            for route in ['/score','/generate']:
                for literal in ['NaN','Infinity','-Infinity']:
                    payload=json.dumps(request).replace(json.dumps(request['context']),literal,1)
                    rejected=client.post(route,content=payload,headers={'Content-Type':'application/json'})
                    assert rejected.status_code==422 and rejected.json()['detail']
                    assert all(set(e)=={'type','loc','msg'} for e in rejected.json()['detail'])
                    statuses[route+':'+literal]=rejected.status_code
            assert client.post('/score',content=b'x'*(1048576+1)).status_code==413
            assert client.post('/score',content=iter([b'x'*65536]*17)).status_code==413
            big={**request,'candidates':[{'id':str(i),'description':'x'*8192} for i in range(33)]}
            assert client.post('/score',json=big).status_code==422
            assert client.get('/readyz').status_code==200
            subprocess.run(['docker','cp','examples/request.json',name+':/tmp/request.json'],check=True,stdout=subprocess.DEVNULL)
            cli=subprocess.run(['docker','exec',name,'myjev','score','--artifact',a.repo,'--revision',a.revision,'--input','/tmp/request.json'],capture_output=True,text=True,check=True)
            a.output.with_suffix('.cli.log').write_text(cli.stderr+'\n'+cli.stdout)
            assert json.loads(cli.stdout)==expected
            hashcode="import pathlib,hashlib,myjev,json; p=pathlib.Path(myjev.__file__).parent; print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*.py')}))"
            installed=json.loads(subprocess.check_output(['docker','exec',name,'python','-c',hashcode],text=True))
            source={str(p.relative_to('src/myjev')):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/myjev').rglob('*.py')}
            assert installed==source
            info=json.loads(subprocess.check_output(['docker','image','inspect',a.image],text=True))[0]
            assert info['Config']['Healthcheck']['StartPeriod']==900_000_000_000
            uid=subprocess.check_output(['docker','exec',name,'id','-u'],text=True).strip()
            result={'verified_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'source_revision':info['Config']['Labels']['org.opencontainers.image.revision'],'user_uid':uid,'health_start_period_seconds':900,'nonfinite_statuses':statuses,'body_status':413,'chunked_status':413,'aggregate_status':422,'cli_equal':True,'installed_source_sha256':installed,'image':a.image,'image_id':subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip(),
                    'repo_id':a.repo,'revision':a.revision,'direct_hub_load':True,'http_equal_to_host_python':True,
                    'managed_aliases_equal':True,'ready_seconds':ready,'response':response.json(),
                    'scope':'Pinned Hub reference loaded directly in the specified container from read-only cached Hub storage; no managed cloud deployment. Readiness observation is not an isolated latency benchmark.'}
            a.output.with_suffix('.http.txt').write_text('GET /readyz\n200\n{"status":"ready"}\n\nPOST /score\n200\n'+json.dumps(response.json(),indent=2)+'\n\nInvalid non-finite inputs\n'+json.dumps(statuses,indent=2)+'\n')
            a.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:
        with a.output.with_suffix('.log').open('w') as log:subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['docker','rm','-f',name],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':main()
