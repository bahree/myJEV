"""Explain batch completion using frozen work counts and measured throughput."""
from datetime import datetime
import json
from pathlib import Path
import statistics

METHOD_NAMES = {'sft':'supervised learning', 'continued_sft':'continued supervised learning',
                'exact':'exact RL', 'sampled':'sampled RL'}


def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default


def read_lines(path):
    try: lines=path.read_text().splitlines()
    except FileNotFoundError: return []
    out=[]
    for line in lines:
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass  # a writer may be appending the final line
    return out


def explain_stage(stage):
    parts=stage.split('/')
    if len(parts)<3:return 'Preparing the next task'
    method=METHOD_NAMES.get(parts[1] if parts[0]=='tuning' else parts[2], 'comparison')
    if parts[0]=='tuning':
        try: trial=int(parts[2].removeprefix('lr-'))+1
        except ValueError: return 'Preparing a tuning trial'
        verb='checking on validation data' if parts[-1]=='validation' else 'training'
        return f'Tuning {method}: {verb} candidate {trial} of 2'
    seed=parts[1].removeprefix('seed-')
    action={'training':'training','full-evaluation':'evaluating all test/calibration examples',
            'posthoc':'fitting calibration controls'}.get(parts[-1],parts[-1])
    return f'Main comparison, seed {seed}, {method}: {action}'


def summarize_job(job, plan, root=Path('.')):
    size=job['size'];methods=plan['methods'];seeds=plan['seeds'];grid=plan['learning_rates']
    evidence=root/f'results/longer-v1/{size}'
    artifacts=root/f'artifacts/longer-v1/{size}'
    tasks=[]
    for method in methods:
        tasks += [(f'tuning/{method}/lr-{i}',method,plan['tuning_updates']) for i in range(len(grid))]
        tasks += [(f'main/seed-{seed}/{method}',method,plan['main_updates']) for seed in seeds]
    done_by_method={m:0 for m in methods};planned_by_method={m:0 for m in methods}
    samples={m:[] for m in methods};completed=0
    for name,method,target in tasks:
        path=artifacts/name
        rows=read_lines(path/'training.jsonl')
        step=min(target,max((r.get('step',0) for r in rows),default=0))
        manifest=read_json(path/'artifact/manifest.json',{})
        finished=manifest.get('training',{}).get('steps')==target
        if finished:step=target;completed+=1
        done_by_method[method]+=step;planned_by_method[method]+=target
        for previous,current in zip(rows,rows[1:]):
            ds=current.get('step',0)-previous.get('step',0)
            dt=current.get('session_seconds',0)-previous.get('session_seconds',0)
            if ds>0 and dt>0 and current.get('step',0)>10:samples[method].append(dt/ds)
    # Fallbacks are disclosed estimates until this batch supplies method timings.
    fallback={'0.8b':.8,'4b':1.05,'9b':1.35}.get(size,1.35)
    sft=statistics.median(samples.get('sft',[])) if samples.get('sft') else fallback
    rates={m:statistics.median(samples[m]) if len(samples[m])>=20 else sft*(1.3 if m in ('exact','sampled') else 1)
           for m in methods}
    estimated_methods=[m for m in methods if len(samples[m])<20]
    validation_paths=list(evidence.glob('tuning/*/*/validation/metrics.json'))
    eval_paths=list(evidence.glob('main/seed-*/*/evaluation/metrics.json'))
    posthoc_paths=list(evidence.glob('main/seed-*/*/posthoc/temperature-metrics.json'))
    timings=[]
    for path in validation_paths+eval_paths:
        for record in read_lines(path.parent/'predictions.jsonl'):
            if record.get('seconds',0)>0:timings.append(record['seconds'])
    if timings:infer=statistics.mean(timings)
    else:
        pilot=read_json(root/f'results/pilot-{size}-sft-evaluation/metrics.json',{})
        infer=pilot.get('latency_p50_p95_seconds',[{'0.8b':.2,'4b':.22,'9b':.32}.get(size,.32)])[0]
    validations_total=len(methods)*len(grid);evals_total=len(methods)*len(seeds)
    posthoc_total=len([m for m in methods if m in ('sft','continued_sft')])*len(seeds)
    validation_seconds=997*infer+20;eval_seconds=4080*infer+60;posthoc_seconds=30
    total=sum(planned_by_method[m]*rates[m] for m in methods)+validations_total*validation_seconds+evals_total*eval_seconds+posthoc_total*posthoc_seconds
    done=sum(done_by_method[m]*rates[m] for m in methods)+len(validation_paths)*validation_seconds+len(eval_paths)*eval_seconds+len(posthoc_paths)*posthoc_seconds
    # Credit a bounded estimate for a validation/evaluation currently in flight.
    stage=job.get('stage','')
    try:elapsed=max(0,(datetime.fromisoformat(job['snapshot_utc'])-datetime.fromisoformat(job['utc'])).total_seconds())
    except (KeyError,ValueError):elapsed=0
    if stage.endswith('/validation') and not (evidence/stage/'metrics.json').exists():
        done+=min(elapsed,validation_seconds*.95)
    elif stage.endswith('/full-evaluation') and not (evidence/stage.removesuffix('/full-evaluation')/'evaluation/metrics.json').exists():
        done+=min(elapsed,eval_seconds*.95)
    if job['state']=='completed':done=total
    done=min(done,total)
    blocked=job['state'] not in ('running','starting','completed')
    return {**job,'stage_description':explain_stage(stage),'training_updates_done':sum(done_by_method.values()),
            'training_updates_total':sum(planned_by_method.values()),'training_runs_done':completed,'training_runs_total':len(tasks),
            'validations_done':len(validation_paths),'validations_total':validations_total,
            'evaluations_done':len(eval_paths),'evaluations_total':evals_total,
            'posthoc_done':len(posthoc_paths),'posthoc_total':posthoc_total,
            'estimated_work_done_seconds':done,'estimated_work_total_seconds':total,
            'batch_percent':min(99.9,100*done/total) if job['state']!='completed' else 100.,
            'remaining_seconds':None if blocked else max(0,total-done),
            'rate_seconds_per_update':rates,'estimated_method_rates':estimated_methods,
            'inference_seconds_per_example':infer,'inference_timing_source':'current batch' if timings else 'pilot/fallback'}


def progress(record, root=Path('.')):
    plan=read_json(root/'configs/study-v1.json')
    if not plan:raise ValueError('Frozen study plan unavailable')
    jobs=[summarize_job({**j,'snapshot_utc':record['utc']},plan,root) for j in record['jobs']]
    total=sum(j['estimated_work_total_seconds'] for j in jobs)
    done=sum(j['estimated_work_done_seconds'] for j in jobs)
    healthy=all(j['remaining_seconds'] is not None for j in jobs)
    remaining=max(j['remaining_seconds'] for j in jobs) if healthy else None
    return {'utc':record['utc'],'jobs':jobs,'batch_percent':100*done/total,
            'training_updates_done':sum(j['training_updates_done'] for j in jobs),
            'training_updates_total':sum(j['training_updates_total'] for j in jobs),
            'remaining_hours_low':remaining*.85/3600 if remaining is not None else None,
            'remaining_hours_high':remaining*1.4/3600 if remaining is not None else None,
            'definition':'Approximate compute-time-weighted progress for this batch, including tuning, training, validation, full evaluation, and calibration controls. Not whole-project completion.',
            'eta_basis':'Measured update/inference speeds where available; unmeasured RL assumed 1.3x supervised update time. The range is a planning allowance, not a confidence interval. Parallel completion uses the slowest remaining job.'}
