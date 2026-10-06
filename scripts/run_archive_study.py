"""Exploratory machine-label archive study; human audit remains separate.

A local teacher annotates exact prepared text. Invalid/uncertain records remain
explicit exclusions. This runner never turns machine agreement into correctness.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/archive-machine-v1'
ART=ROOT/'artifacts/archive-machine-v1'
DATA=ROOT/'data/archive-machine-v1'
JUDGE='annotation/local-judge-v1'
BASE=ROOT/'artifacts/longer-v1/4b/main/seed-11/continued_sft/artifact'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'results/longer-v1/4b/main/seed-11/continued_sft/config.json').read_text())
    cfg.update(updates=400,data_offset=0,purpose='exploratory archive machine-label adaptation; no human review')
    plan={'teacher_plan_sha256':digest(ROOT/JUDGE/'frozen-plan.json'),
          'student_initial_manifest_sha256':digest(BASE/'manifest.json'),'config':cfg,'seed':11,'pilot_updates':100,'main_updates':400,
          'labels':'Local 9B NF4 machine judgments; require at least 80% structurally valid development outputs before expansion.',
          'minimum_rows_per_split':12,'evaluation':'Full retained archive test; thresholds from archive calibration. Before/after native scalar confidence. Full BANKING77 forgetting after adaptation.',
          'limits':['No human ground truth; metrics measure teacher agreement.', 'Single adaptation seed and fixed budget; no convergence or production error claim.',
                    'Same model family for teacher and student may share systematic errors.', 'Existing provisional post groups are used; semantic series grouping has not been human audited.',
                    'Temperature scaling on mixed candidate-count archive tasks is not included in this first study.'],
          'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'scripts/local_archive_judge.py',ROOT/'src/myjev/judging.py',ROOT/'scripts/finalize_machine_annotations.py']}}
    path=OUT/'frozen-plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Archive protocol inputs changed')
    path.write_text(json.dumps(plan,indent=2)+'\n');(OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['v1','v2'],default='v1');p.add_argument('--freeze',action='store_true');p.add_argument('--gpu',default='1');p.add_argument('--wait-for',type=Path);a=p.parse_args()
    global OUT,ART,DATA,JUDGE
    if a.version=='v2':
        OUT=ROOT/'results/archive-machine-v2';ART=ROOT/'artifacts/archive-machine-v2';DATA=ROOT/'data/archive-machine-v2'
        JUDGE='annotation/local-judge-v2-expanded'
    plan=freeze()
    if a.freeze:print('Frozen local-judge and exploratory archive adaptation protocol');return
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'state':'waiting','completed':[],'started':time.time(),'gpu':a.gpu}
    def status(**kw):
        state.update(kw,updated=time.time());tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'status.json')
    status()
    if a.wait_for:
        while True:
            try:s=json.loads(a.wait_for.read_text())
            except (FileNotFoundError,json.JSONDecodeError):s={}
            if s.get('state') in ('completed','failed','blocked'):
                status(preceding_gpu_job_state=s['state']);break
            time.sleep(15)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu,'HF_HOME':str(ROOT/'.cache/huggingface'),'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MYJEV_TRACKING_MODE':'offline'}
    if (ROOT/'.env.wandb').exists():
        from sync_wandb_study import settings
        settings();env.update({k:os.environ[k] for k in ('WANDB_API_KEY','MYJEV_WANDB_ENTITY','MYJEV_WANDB_PROJECT')});env['MYJEV_TRACKING_MODE']='online'
    def run(cmd,name):
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than 8 GiB free disk')
        status(state='running',current=name)
        with (OUT/f'{name}.log').open('a') as log:subprocess.run([sys.executable,*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(name);status()
    def evaluate(artifact,data,cal,out,name):
        if not (out/'metrics.json').exists():run(['-m','myjev.evaluate','--artifact',str(artifact),'--data',str(data),'--calibration',str(cal),'--output',str(out)],name)
    try:
        if not (ROOT/JUDGE/'complete.json').exists():run(['scripts/local_archive_judge.py','--execute',*(['--span-expansion'] if a.version=='v2' else [])],'local-judge')
        if not DATA.exists():run(['scripts/finalize_machine_annotations.py','--labels',str(Path(JUDGE)/'labels.jsonl'),'--output',str(DATA)],'freeze-machine-splits')
        manifest=json.loads((DATA/'manifest.json').read_text())
        if any(n<plan['minimum_rows_per_split'] for n in manifest['counts'].values()):raise ValueError('Too few retained machine labels for the planned split protocol')
        shutil.copy2(DATA/'manifest.json',OUT/'data-manifest.json')
        hashes={str(p.relative_to(ROOT)):digest(p) for p in DATA.glob('*.jsonl')}
        (OUT/'frozen-label-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
        evaluate(BASE,DATA/'test.jsonl',DATA/'calibration.jsonl',OUT/'unadapted','unadapted-evaluation')
        for name,updates in [('pilot',100),('adapted',400)]:
            output=ART/name;path=output/'artifact/manifest.json'
            finished=path.exists() and json.loads(path.read_text()).get('training',{}).get('steps')==updates
            if not finished:
                if (output/'resume.pt').exists():raise ValueError('Interrupted archive stage retained; explicit resume audit required before replay')
                run(['-m','myjev.train','--config',str(OUT/'config.json'),'--data',str(DATA/'train.jsonl'),'--output',str(output),'--initial',str(BASE),'--method','continued_sft','--seed','11','--updates',str(updates)],f'{name}-training')
            record=OUT/name;record.mkdir(exist_ok=True);shutil.copy2(path,record/'artifact-manifest.json');shutil.copy2(output/'training.jsonl',record/'training.jsonl')
            if any(not math.isfinite(json.loads(line)['loss']) for line in (record/'training.jsonl').read_text().splitlines()):
                raise RuntimeError('Nonfinite training loss; stop archive adaptation')
        adapted=ART/'adapted/artifact'
        evaluate(adapted,DATA/'test.jsonl',DATA/'calibration.jsonl',OUT/'adapted/evaluation','adapted-evaluation')
        evaluate(adapted,ROOT/'data/banking77/test.jsonl',ROOT/'data/banking77/calibration.jsonl',OUT/'forgetting','forgetting-evaluation')
        run(['scripts/summarize_archive_study.py','--version',a.version],'summary')
        status(state='completed',current=None)
    except Exception as error:status(state='failed',error=str(error));raise

if __name__=='__main__':main()
