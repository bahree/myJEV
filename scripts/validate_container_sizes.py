"""Check actual GPU containers against Python fixtures, retaining latency evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx


def command(*args):
    return subprocess.check_output(list(args),text=True).strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--port',type=int,default=18000)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    gpu_uuid=command('nvidia-smi','-i',a.gpu,'--query-gpu=uuid','--format=csv,noheader')
    processes=command('nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader')
    if gpu_uuid in processes:raise ValueError('selected GPU is not idle; do not benchmark over another job')
    image_id=command('docker','image','inspect',a.image,'--format','{{.Id}}')
    (out/'environment.json').write_text(json.dumps({'image':a.image,'image_id':image_id,'gpu_uuid':gpu_uuid,
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'One container on an otherwise idle GPU; other GPUs may run study jobs. Cached backbone, short three-candidate fixture; not maximum-context or isolated-host performance.'},indent=2))
    request=json.loads(Path('examples/request.json').read_text())
    for size in ('0.8b','4b','9b'):
        directory=out/size;directory.mkdir()
        name=f'myjev-sizecheck-{size}-{os.getpid()}'
        fixture=Path('results/artifact-equivalence.json') if size=='0.8b' else Path(f'results/artifact-equivalence-{size}.json')
        expected=json.loads(fixture.read_text())['response']
        started=time.perf_counter()
        command('docker','run','-d','--gpus',f'device={a.gpu}','--name',name,
                '-p',f'127.0.0.1:{a.port}:8000','-e','MYJEV_ARTIFACT=/artifact','-e','OMP_NUM_THREADS=4',
                '-v',f'{Path(f"artifacts/pilot-{size}/artifact").resolve()}:/artifact:ro',
                '-v',f'{Path(".cache/huggingface").resolve()}:/cache/huggingface',a.image)
        try:
            url=f'http://127.0.0.1:{a.port}'
            with httpx.Client(timeout=5) as client:
                while time.perf_counter()-started<180:
                    try:
                        if client.get(url+'/readyz').status_code==200:break
                    except httpx.HTTPError:pass
                    time.sleep(1)
                else:raise TimeoutError('container did not become ready')
                ready=time.perf_counter()-started
                response=client.post(url+'/score',json=request);response.raise_for_status()
                actual=response.json()
                if actual!=expected:raise AssertionError('container response differs from Python fixture')
            (directory/'equivalence.json').write_text(json.dumps({'equal':True,'response':actual,
                'cold_start_to_ready_seconds':ready,'cache_state':'backbone pre-cached; new container'},indent=2))
            for concurrency in (1,4):
                subprocess.run([sys.executable,'scripts/benchmark.py','--url',url,
                    '--requests','40','--concurrency',str(concurrency),'--output',str(directory/f'http-c{concurrency}.json')],check=True)
            (directory/'nvidia-smi.txt').write_text(command('nvidia-smi'))
        finally:
            log=subprocess.run(['docker','logs',name],capture_output=True,text=True)
            (directory/'container.log').write_text(log.stdout+log.stderr)
            subprocess.run(['docker','stop',name],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['docker','rm',name],check=True,stdout=subprocess.DEVNULL)
        print(json.dumps({'size':size,'container_equivalence':True,'benchmark_requests':80}),flush=True)

if __name__=='__main__':main()
