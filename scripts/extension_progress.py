"""Human-readable progress for the post-training work; no whole-project percentage."""
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def snapshot():
    jobs=[]
    for size in ('0.8b','4b','9b'):
        s=read(ROOT/f'results/generalization-v1/{size}-status.json',{})
        finished=s.get('jobs',[]);remaining=None
        means={k:statistics.mean([j['seconds'] for j in finished if j['kind']==k]) for k in ('transfer','robustness') if any(j['kind']==k for j in finished)}
        if len(means)==2:
            remaining=sum((6-sum(j['kind']==kind for j in finished))*means[kind] for kind in means)
            remaining=max(0,remaining-(time.time()-s.get('updated',time.time()))) if s.get('state')=='running' else (0 if s.get('state')=='completed' else None)
        jobs.append({'size':size,'state':s.get('state','not-started'),'complete':len(finished),'total':12,'current':s.get('current'),'remaining_seconds_estimate':remaining})
    precision=read(ROOT/'results/precision-v1/status.json',{'state':'not-started'})
    updates=0
    for name,target in [('pilot',100),('seed-11',4000),('seed-22',4000),('seed-33',4000)]:
        path=ROOT/f'artifacts/precision-v1/{name}/training.jsonl'
        if path.exists():
            for line in reversed(path.read_text().splitlines()):
                try:updates+=min(target,json.loads(line)['step']);break
                except json.JSONDecodeError:continue
    release=read(ROOT/'results/release-validation-v1/status.json',{'state':'not-started'})
    return {'utc_seconds':time.time(),'generalization':jobs,'precision':precision,'precision_updates':updates,'precision_updates_total':12100,'release':release}


def message(s):
    done=sum(j['complete'] for j in s['generalization'])
    lines=[f'Transfer/robustness: {done}/36 evaluation jobs complete ({100*done/36:.1f}% by job count; job durations differ).']
    for j in s['generalization']:
        eta=j['remaining_seconds_estimate']
        estimate=f"estimated {eta*.8/3600:.1f}-{eta*1.5/3600:.1f} hours remaining" if eta is not None else 'ETA awaits measured transfer and robustness timings'
        lines.append(f"  {j['size']}: {j['state']}, {j['complete']}/12; {j['current']}; {estimate}.")
    lines += [f"Precision control: {s['precision']['state']}; {s['precision_updates']:,}/12,100 new training updates. It also needs full-test evaluation and calibration controls.",
              f"Release verification: {s['release']['state']}; {len(s['release'].get('completed',[]))}/6 candidates verified and benchmarked. It waits for all other GPU jobs to finish.",
              '', 'Already done: Qwen main training and paired findings; scratch teaching study; four draft blog bundles; operational uncertainty report; local candidate packaging.',
              'Still left after these jobs: interpret and write new findings, archive labels/audit and adaptation/forgetting, final release choice/licensing/uploads, clean-release environment and Hugo-theme checks.',
              'No whole-project percentage or total finish time is claimed. Queued stages have not yet supplied their own throughput measurements.']
    return '\n'.join(lines)

if __name__=='__main__':print(message(snapshot()))
