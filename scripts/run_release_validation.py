"""Queue isolated-host candidate verification and reference serving benchmarks.

No registry upload, Hub upload, public endpoint, or default selection occurs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/release-validation-v1'


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--image',default='myjev:scratch-checkpoint');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inputs=[ROOT/'results/release-candidates-v1/inventory.json']+[ROOT/'scripts'/name for name in ('benchmark_scoring_paths.py','verify_artifact.py','benchmark.py')]
    plan={'candidates':'Six seed-11 candidates, three sizes, continued SFT with temperature and exact RL',
          'workloads':[[3,0],[32,0],[32,128]],'model_repeats':20,'model_warmup':3,
          'http_requests_per_cell':100,'http_concurrency':[1,4], 'image_tag':a.image,
          'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    frozen=OUT/'frozen-plan.json'
    if frozen.exists() and json.loads(frozen.read_text())!=plan:raise ValueError('Release benchmark inputs changed after freeze')
    frozen.write_text(json.dumps(plan,indent=2)+'\n')
    if a.freeze:print('Frozen six-candidate verification and serving workloads');return
    state={'state':'waiting','completed':[],'started':time.time()}
    def update(**kw):
        state.update(kw,updated=time.time());tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'status.json')
    update()
    if a.wait:
        prerequisites=[ROOT/f'results/generalization-v1/{s}-status.json' for s in ('0.8b','4b','9b')]+[ROOT/'results/precision-v1/status.json',ROOT/'results/archive-machine-v1/status.json']
        while True:
            values=[]
            for path in prerequisites:
                try:values.append(json.loads(path.read_text()))
                except (FileNotFoundError,json.JSONDecodeError):values.append({})
            if all(v.get('state') in ('completed','failed','blocked') for v in values):
                update(preceding_gpu_jobs=[v.get('state') for v in values]);break
            time.sleep(15)
    for name,expected in plan['hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            update(state='failed',reason=f'Frozen input changed: {name}');return
    # Refuse claimed isolation if another GPU compute process is still present.
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:
        update(state='blocked',reason='GPU compute processes remain; cannot claim isolated-host benchmark');return
    inventory=json.loads((ROOT/'results/release-candidates-v1/inventory.json').read_text())
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','HF_HOME':str(ROOT/'.cache/huggingface'),'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    image_id=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
    (OUT/'environment.json').write_text(json.dumps({'image_tag':a.image,'image_id':image_id,'gpu_processes_before':active,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'scope':'Local prebuilt reference image; not registry published. Other GPU compute jobs required to finish first.'},indent=2)+'\n')
    def run(cmd,log):
        with log.open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        for candidate in inventory['candidates']:
            name=candidate['name'];dest=OUT/name;dest.mkdir(exist_ok=True);artifact=(ROOT/candidate['destination']).resolve()
            update(state='running',current={'candidate':name,'phase':'equivalence'})
            if not (dest/'equivalence.json').exists():run([sys.executable,'scripts/verify_artifact.py','--artifact',str(artifact),'--output',str(dest/'equivalence.json')],dest/'equivalence.log')
            update(current={'candidate':name,'phase':'scoring-generation'})
            if not (dest/'paths/report.json').exists():
                if (dest/'paths').exists():(dest/'paths').rename(dest/f'paths-interrupted-{time.time_ns()}')
                run([sys.executable,'scripts/benchmark_scoring_paths.py','--artifact',str(artifact),'--output',str(dest/'paths')],dest/'paths.log')
            update(current={'candidate':name,'phase':'docker-http'})
            container=f'myjev-release-{name}'
            if not (dest/'http-complete.json').exists():
                start=time.perf_counter()
                container_id=subprocess.check_output(['docker','run','-d','--name',container,'--gpus','device=0','-p','127.0.0.1:18086:8000','-v',f'{artifact}:/artifact:ro','-v',f'{ROOT}/.cache/huggingface:/hf:ro','-e','HF_HOME=/hf','-e','HF_HUB_OFFLINE=1','-e','MYJEV_ARTIFACT=/artifact','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',a.image],text=True).strip()
                try:
                    deadline=time.monotonic()+300
                    with httpx.Client(timeout=10) as client:
                        while True:
                            try:
                                response=client.get('http://127.0.0.1:18086/readyz')
                                if response.status_code==200:break
                            except httpx.HTTPError:pass
                            if time.monotonic()>deadline:raise TimeoutError('Container readiness exceeded 300 seconds')
                            time.sleep(1)
                    startup=time.perf_counter()-start
                    from benchmark_scoring_paths import request_for
                    for count,length in ((3,0),(32,0),(32,128)):
                        req=dest/f'request-{count}-{length}.json';req.write_text(json.dumps(request_for(count,length),indent=2)+'\n')
                        for concurrency in (1,4):
                            result=dest/f'http-{count}-{length}-c{concurrency}.json'
                            run([sys.executable,'scripts/benchmark.py','--url','http://127.0.0.1:18086','--request',str(req),'--requests','100','--concurrency',str(concurrency),'--output',str(result)],dest/'http.log')
                            report=json.loads(result.read_text())
                            if report['statuses']!={'200':100}:raise RuntimeError('HTTP benchmark had failed requests')
                    actual=httpx.post('http://127.0.0.1:18086/score',json=json.loads((ROOT/'examples/request.json').read_text()),timeout=60).json()
                    expected=json.loads((dest/'equivalence.json').read_text())['response']
                    if actual!=expected:raise ValueError('Docker/Python response differs')
                    (dest/'http-complete.json').write_text(json.dumps({'startup_to_ready_seconds':startup,'container_id':container_id,'image_id':image_id,'docker_python_equal':True,'scope':'Warm image and backbone caches; one startup observation, not cold-cache cost.'},indent=2)+'\n')
                finally:
                    with (dest/'container.log').open('w') as log:subprocess.run(['docker','logs',container],stdout=log,stderr=subprocess.STDOUT)
                    subprocess.run(['docker','rm','-f',container],check=True,stdout=subprocess.DEVNULL)
            state['completed'].append(name);update()
        update(state='completed',current=None)
    except Exception as e:update(state='failed',error=str(e));raise

if __name__=='__main__':main()
