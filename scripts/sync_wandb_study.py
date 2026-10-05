"""Read-only W&B mirror of saved evidence; never changes training files."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'results/wandb-sync'
RUN_PREFIX = ''


def settings(env_file=None):
    path = Path(env_file) if env_file else ROOT / '.env.wandb'
    for line in (path.read_text().splitlines() if path.exists() else []):
        if line.strip() and not line.lstrip().startswith('#'):
            key, sep, value = line.partition('=')
            if sep and key.strip() in {'WANDB_API_KEY', 'WANDB_BASE_URL', 'MYJEV_WANDB_ENTITY', 'MYJEV_WANDB_PROJECT'}:
                os.environ[key.strip()] = value.strip().strip('\"').strip("'")
    if not os.environ.get('WANDB_API_KEY'):
        raise ValueError('Missing local W&B credential')
    for name in ('MYJEV_WANDB_ENTITY', 'MYJEV_WANDB_PROJECT'):
        if not os.environ.get(name):
            raise ValueError(f'Missing {name}')
    os.environ['WANDB_SILENT'] = 'true'


def records(path):
    data = path.read_bytes()
    # Ignore only the unfinished final line of an actively written JSONL file.
    lines = data.splitlines() if data.endswith(b'\n') else data.splitlines()[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def atomic(path, value):
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def init(name, identity, **kwargs):
    import wandb
    return wandb.init(entity=os.environ['MYJEV_WANDB_ENTITY'],
        project=os.environ['MYJEV_WANDB_PROJECT'], mode='online',
        id=hashlib.sha256((RUN_PREFIX + identity).encode()).hexdigest()[:20], resume='allow', name=name,
        dir=str(STATE), save_code=False,
        settings=wandb.Settings(console='off', disable_git=True, x_disable_stats=True), **kwargs)


def epoch_progress(step, target, accumulation, offset, dataset_size):
    if dataset_size <= 0:
        raise ValueError('training dataset must not be empty')
    return {
        'stage_epochs_done': step * accumulation / dataset_size,
        'stage_epochs_remaining': max(0, target - step) * accumulation / dataset_size,
        'lineage_epochs_done': (offset + step * accumulation) / dataset_size,
        'lineage_epochs_target': (offset + target * accumulation) / dataset_size,
    }


def live(interval, duration):
    from monitor_longer_study import snapshot
    from study_progress import progress
    run = init('Live study: all three GPUs', 'myjev-longer-v1-live-dashboard',
        job_type='live-monitor', tags=['live-monitor', 'one-minute-poll'],
        config={'source':'saved local logs', 'interval_seconds':interval,
                'note':'Loss changes meaning across methods; inspect stage. Monitor lifetime is not training duration.'})
    atomic(STATE/'live.json', {'pid':os.getpid(), 'url':run.url, 'interval_seconds':interval})
    end = time.monotonic() + duration
    dataset_size = sum(1 for line in Path('data/banking77/train.jsonl').open() if line.strip())
    try:
        while time.monotonic() < end:
            snap = snapshot(); summary = progress(snap)
            metrics = {'batch/percent':summary['batch_percent'],
                'batch/training_steps':summary['training_updates_done']}
            for job in snap['jobs']:
                prefix = job['size']
                for key in ('step','target_updates','last_loss','session_seconds','full_evaluations','tuning_validations'):
                    if key in job: metrics[f'{prefix}/{key}'] = job[key]
                stage = job.get('stage', '')
                if stage.endswith('/training') and 'step' in job:
                    config = json.loads((Path('results/longer-v1') / prefix / stage.removesuffix('/training') / 'config.json').read_text())
                    for key, value in epoch_progress(job['step'], job['target_updates'],
                            config['accumulation'], config.get('data_offset', 0), dataset_size).items():
                        metrics[f'{prefix}/{key}'] = value
                else:
                    # Clear stage-only values rather than displaying stale training progress during evaluation.
                    for key in ('stage_epochs_done', 'stage_epochs_remaining', 'lineage_epochs_done', 'lineage_epochs_target'):
                        run.summary[f'{prefix}/{key}'] = None
                run.summary[f'{prefix}/stage'] = job.get('stage','')
                run.summary[f'{prefix}/state'] = job['state']
            for job in summary['jobs']:
                metrics[f"{job['size']}/batch_percent"] = job['batch_percent']
            for path in sorted(Path('results/longer-v1').glob('gpu-*/gpu.csv')):
                # Last complete observation per GPU, with source UTC retained.
                latest = {}
                with path.open() as f:
                    for row in csv.DictReader(f):
                        if row.get('temperature.gpu'): latest[row['index']] = row
                for gpu, row in latest.items():
                    for key in ('utilization.gpu','memory.used','memory.total','power.draw','temperature.gpu'):
                        try: metrics[f'gpu{gpu}/{key}'] = float(row[key])
                        except ValueError: pass
                    run.summary[f'gpu{gpu}/source_utc'] = row['utc']
            run.log(metrics)
            run.summary['last_observation_utc'] = snap['utc']
            atomic(STATE/'heartbeat.json', {'utc':snap['utc'],'url':run.url,'metrics':metrics})
            if all(j['state']=='completed' for j in snap['jobs']): break
            time.sleep(interval)
    finally:
        run.finish()


def history():
    statepath = STATE/'history.json'
    state = json.loads(statepath.read_text()) if statepath.exists() else {}
    for path in sorted(Path('artifacts').rglob('training.jsonl')):
        relative = path.as_posix()
        if relative in state: continue
        manifest = path.parent/'artifact/manifest.json'
        if not manifest.exists(): continue
        meta = json.loads(manifest.read_text())
        training = meta.get('training',{})
        rows = records(path)
        if not rows: continue
        target = training.get('config',{}).get('updates',training.get('steps',0))
        if not target or training.get('steps',0) < target or rows[-1]['step'] != training.get('steps'): continue
        steps = [r['step'] for r in rows]
        if any(b <= a for a,b in zip(steps,steps[1:])):
            print('Skipped non-monotonic history:',relative,flush=True); continue
        name = path.parent.relative_to('artifacts').as_posix()
        run = init(name, 'myjev-saved-training:'+relative, job_type='training-history',
            tags=['historical-import','legacy-v1' if 'legacy-v1' in name else 'current-prompt'],
            config={'source':relative,'backbone':meta.get('backbone'),'precision':meta.get('precision'),
                'artifact_revision':meta['artifact_revision'],'training':training,
                'scope':'Saved training metrics; W&B wall clock is upload time, not original training time.'})
        last = int(run.summary.get('imported_step',0))
        try:
            for row in rows:
                if row['step'] > last: run.log(row,step=row['step'])
            run.summary['imported_step'] = rows[-1]['step']
            run.summary['source_complete'] = True
            url = run.url
        finally:
            run.finish()
        state[relative] = {'url':url,'steps':rows[-1]['step']}
        atomic(statepath,state)
        print('Uploaded history:',name,flush=True)


def evidence_files():
    # Explicit experiment roots only. No credentials, notifications, archive text or weights.
    prefixes = ('longer-v1','pilot-','three-seed-','gpu-','docker-','http-',
                'artifact-','untouched-','precision-','tfidf','gliclass','robustness','clinc-')
    allowed = {'.json','.jsonl','.csv','.log','.txt','.png','.svg'}
    for path in sorted(Path('results').rglob('*')):
        rel = path.relative_to('results')
        if not rel.parts[0].startswith(prefixes): continue
        if any(p in {'tracking','wandb','temperature-artifact'} for p in rel.parts): continue
        if path.is_file() and not path.is_symlink() and path.suffix in allowed:
            yield path


def evidence():
    import wandb
    import tempfile
    import shutil
    run = init('Experiment evidence archive', 'myjev-experiment-evidence',job_type='evidence-archive',
        config={'scope':'Public-task experiment evidence; excludes credentials, archive annotations and weights.'})
    try:
        with tempfile.TemporaryDirectory(prefix='myjev-wandb-') as temp:
            artifact = wandb.Artifact('myjev-experiment-evidence',type='evaluation-evidence')
            count=0
            for path in evidence_files():
                dest=Path(temp)/path;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,dest)
                artifact.add_file(str(dest),name=path.as_posix());count+=1
            for path in sorted(Path('artifacts').rglob('training.jsonl')):
                dest=Path(temp)/path;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,dest);artifact.add_file(str(dest),name=path.as_posix());count+=1
            run.log_artifact(artifact).wait()
            run.summary['file_count']=count
            atomic(STATE/'evidence.json',{'files':count,'artifact':artifact.qualified_name,'url':run.url})
    finally: run.finish()


def main():
    global STATE, RUN_PREFIX
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['live','backfill'])
    p.add_argument('--interval',type=int);p.add_argument('--duration',type=int,default=172800)
    p.add_argument('--env-file',default='.env.wandb')
    p.add_argument('--state-dir',default='results/wandb-sync')
    p.add_argument('--run-prefix',default='',help='Stable namespace; use a new prefix and state directory for a new independent study')
    p.add_argument('--once',action='store_true',help='One backfill pass, then exit')
    args=p.parse_args();os.chdir(ROOT)
    if args.interval is None: args.interval=60 if args.mode=='live' else 1800
    if args.interval<=0 or args.duration<=0: p.error('interval and duration must be positive')
    if args.once and args.mode!='backfill': p.error('--once is for backfill')
    STATE=Path(args.state_dir);RUN_PREFIX=args.run_prefix
    STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/f'{args.mode}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    settings(args.env_file)
    target={'entity':os.environ['MYJEV_WANDB_ENTITY'],'project':os.environ['MYJEV_WANDB_PROJECT'],'run_prefix':RUN_PREFIX}
    target_path=STATE/'target.json'
    if target_path.exists() and json.loads(target_path.read_text())!=target:
        raise ValueError('Upload destination changed; use a separate --state-dir')
    if not target_path.exists(): atomic(target_path,target)
    if args.mode=='live':live(args.interval,args.duration)
    else:
        end=time.monotonic()+args.duration
        while time.monotonic()<end:
            try:
                history();evidence()
                print('Backfill pass complete',flush=True)
            except Exception as exc:
                # SDK diagnostics remain local; never print credential-bearing exception text.
                print('Backfill pass failed:',type(exc).__name__,flush=True)
                if args.once: raise SystemExit(1)
            if args.once: break
            time.sleep(args.interval)


if __name__=='__main__': main()
