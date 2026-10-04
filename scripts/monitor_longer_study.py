"""Print and retain live progress for the longer matched study."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def snapshot():
    jobs=[]
    for size in ('0.8b','4b','9b'):
        root=Path(f'results/longer-v1/{size}')
        path=root/'status.json'
        status=json.loads(path.read_text()) if path.exists() else {'state':'starting'}
        if status['state']=='running':
            try:
                os.kill(status['pid'],0)
                proc=Path(f'/proc/{status["pid"]}/cmdline')
                if proc.exists() and 'run_longer_study.py' not in proc.read_bytes().decode():
                    status['state']='stale-pid'
            except ProcessLookupError:
                status['state']='stopped-unexpectedly'
        job={'size':size,**status,'tuning_validations':len(list(root.glob('tuning/*/*/validation/metrics.json'))),
             'full_evaluations':len(list(root.glob('main/seed-*/*/evaluation/metrics.json')))}
        stage=status.get('stage','')
        if stage.endswith('/training'):
            name=stage.removesuffix('/training')
            log=Path(f'artifacts/longer-v1/{size}')/name/'training.jsonl'
            config=root/name/'config.json'
            if log.exists() and config.exists():
                try:
                    last=json.loads(log.read_text().splitlines()[-1])
                    job['step']=last['step'];job['target_updates']=json.loads(config.read_text())['updates']
                    job['last_loss']=last['loss'];job['session_seconds']=last['session_seconds']
                except (IndexError,json.JSONDecodeError):
                    pass  # tolerate a concurrent partial log write
        jobs.append(job)
    return {'utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--once',action='store_true')
    p.add_argument('--interval',type=float,default=60)
    p.add_argument('--duration',type=float,default=172800)
    a=p.parse_args()
    if a.interval<=0 or a.duration<=0:p.error('positive interval/duration required')
    end=time.monotonic()+a.duration
    while True:
        state=snapshot()
        if a.once:
            print(json.dumps(state,indent=2));return
        out=Path('results/longer-v1/monitor-history.jsonl');out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('a') as f:f.write(json.dumps(state)+'\n')
        for job in state['jobs']:
            print(f"{state['utc']} | {job['size']} | {job['state']} | {job.get('stage','')} | "
                  f"step {job.get('step','-')}/{job.get('target_updates','-')} | "
                  f"validation {job['tuning_validations']}/8 | full evaluation {job['full_evaluations']}/12",flush=True)
        if all(j['state']=='completed' for j in state['jobs']):return
        if any(j['state'] not in ('starting','running','completed') for j in state['jobs']):
            raise SystemExit('A runner needs attention; see status.json')
        if time.monotonic()>=end:raise SystemExit('Monitoring window ended')
        time.sleep(min(a.interval,end-time.monotonic()))


if __name__=='__main__':main()
