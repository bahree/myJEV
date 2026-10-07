"""Validate an empty-model-cache startup and request limits, preserving existing caches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import httpx


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--repo',default='bahree/myJEV-4B')
    p.add_argument('--revision',default='38f7cca5a8530483309f576b0c3dd1756bc27c33')
    p.add_argument('--gpu',default='0')
    p.add_argument('--port',type=int,default=18092)
    p.add_argument('--cache-medium',choices=['volume','tmpfs'],default='volume')
    p.add_argument('--expected',type=Path,default=Path('results/release-validation-v1/myjev-4b-continued_sft-seed11/equivalence.json'))
    p.add_argument('--output',type=Path,default=Path('results/review-container-v1/first-use.json'))
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    name='myjev-first-use-'+str(time.time_ns());volume=None
    cmd=['docker','run','-d','--name',name,'--gpus',f'device={a.gpu}','-p',f'127.0.0.1:{a.port}:8000','-e','HF_HOME=/cache/huggingface','-e','HF_HUB_DISABLE_XET=1','-e',f'MYJEV_ARTIFACT={a.repo}','-e',f'MYJEV_REVISION={a.revision}','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2']
    if a.cache_medium=='tmpfs':cmd+=['--tmpfs','/cache/huggingface:rw,size=12g']
    else:
        volume=run('docker','volume','create',name)
        cmd+=['-v',f'{volume}:/cache/huggingface']
    cmd+=[a.image]
    start=time.perf_counter();result={'started_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'cache_initially_empty':True,'cache_medium':a.cache_medium,'existing_caches_mounted':False,'docker_layers_cached':True,'source_image':a.image}
    try:
        run(*cmd)
        with httpx.Client(base_url=f'http://127.0.0.1:{a.port}',timeout=120) as client:
            while True:
                try:
                    ready=client.get('/readyz')
                    if ready.status_code==200:break
                except httpx.HTTPError:pass
                state=run('docker','inspect','--format','{{.State.Status}}',name)
                if state=='exited':raise RuntimeError('container exited before readiness')
                if time.perf_counter()-start>1200:raise TimeoutError('first-use startup exceeded 1200 seconds')
                time.sleep(1)
            result['seconds_to_readiness']=time.perf_counter()-start
            result['cache_bytes_regular_files']=int(run('docker','exec',name,'python','-c',"import os; print(sum(os.path.getsize(os.path.join(p,f)) for p,ds,fs in os.walk('/cache/huggingface') for f in fs if not os.path.islink(os.path.join(p,f))))"))
            result['image_inspect']=json.loads(run('docker','image','inspect',a.image))[0]
            # Store only public provenance, not generic image config/environment.
            result['image_id']=result['image_inspect']['Id']
            result['source_revision']=result['image_inspect']['Config']['Labels']['org.opencontainers.image.revision']
            del result['image_inspect']
            request=json.loads(Path('examples/request.json').read_text())
            expected=json.loads(a.expected.read_text())['response']
            response=client.post('/score',json=request);response.raise_for_status();assert response.json()==expected
            alias=client.post('/generate',json=request);assert alias.status_code==200 and alias.json()==expected
            assert client.get('/health').status_code==200
            result.update(response=response.json(),http_equal_to_saved_python=True,managed_aliases_equal=True)
            oversized=client.post('/score',content=b'x'*(1024*1024+1),headers={'Content-Type':'application/json'})
            chunks=client.post('/score',content=iter([b'x'*65536]*17),headers={'Content-Type':'application/json'})
            aggregate={**request,'candidates':[{'id':str(i),'description':'x'*8192} for i in range(33)]}
            bigtext=client.post('/score',json=aggregate)
            duplicate={**request,'candidates':[request['candidates'][0],request['candidates'][0]]}
            dup=client.post('/score',json=duplicate)
            result['http_statuses']={'body':oversized.status_code,'chunked_body':chunks.status_code,'aggregate':bigtext.status_code,'duplicate_ids':dup.status_code}
            assert result['http_statuses']=={'body':413,'chunked_body':413,'aggregate':422,'duplicate_ids':422}
            assert client.get('/readyz').status_code==200
            transcript=f"GET /readyz\n{ready.status_code}\n{ready.text}\n\nPOST /score\n{json.dumps(request,indent=2)}\n\n{response.status_code}\n{json.dumps(response.json(),indent=2)}\n\nRejection status checks\n{json.dumps(result['http_statuses'],indent=2)}\n"
            a.output.with_suffix('.http.txt').write_text(transcript)
            # Exercise the installed CLI as a distinct entry point on the same artifact.
            run('docker','cp','examples/request.json',name+':/tmp/myjev-request.json')
            cli=subprocess.run(['docker','exec',name,'myjev','score','--artifact',a.repo,'--revision',a.revision,'--input','/tmp/myjev-request.json'],capture_output=True,text=True,check=True)
            a.output.with_suffix('.cli.log').write_text(cli.stderr+'\n'+cli.stdout)
            assert json.loads(cli.stdout)==expected
            result['cli_equal_to_http']=True
            result['passed']=True
            result['scope']='One empty model-cache download and warmup on an A30, followed by exact fixture equality and HTTP limit checks. Image layers were already present. Other GPUs were evaluating. Tmpfs, when selected, is RAM-backed and is not a cold-disk performance measurement. No paid managed deployment.'
    except Exception as e:
        result.update(passed=False,error=type(e).__name__+': '+str(e));raise
    finally:
        result['finished_utc']=datetime.now(timezone.utc).isoformat()
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        with a.output.with_suffix('.log').open('w') as log:subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['docker','rm','-f',name],check=False,stdout=subprocess.DEVNULL)
        if volume:subprocess.run(['docker','volume','rm',volume],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':main()
