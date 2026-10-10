"""Recompute the local/hosted comparison from saved text-free score records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from myjev.calibration import fit_temperature
from myjev.metrics import summarize, thresholds_from_calibration, cluster_interval
from myjev.decision_comparison import VIEWS


def scaled(rows, temperature):
    result=[]
    for row in rows:
        keys=list(row['selection_scores']);values=np.array([row['selection_scores'][k] for k in keys])
        logits=np.log(np.clip(values,1e-12,1.))/temperature
        values=np.exp(logits-logits.max());values/=values.sum()
        scores=dict(zip(keys,values.tolist(),strict=True))
        result.append({**row,'selection_scores':scores,'confidence':scores[row['selected_id']]})
    return result


def analyze(directory):
    rows=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    complete=json.loads((directory/'complete.json').read_text())
    if hashlib.sha256((directory/'records.jsonl').read_bytes()).hexdigest()!=complete['records_sha256']:
        raise ValueError('Recorded responses changed')
    run=json.loads((directory/'run.json').read_text());cal=[r for r in rows if r['phase']=='calibration']
    tests=[r for r in rows if r['phase']=='test'];demo=[r for r in rows if r['phase']=='demo']
    if len(cal)!=1000 or len(tests)!=384 or len(demo)!=7:raise ValueError('Incomplete frozen evaluation')
    if {r['group'] for r in cal}&{r['group'] for r in tests}:raise ValueError('Calibration/test overlap')
    logits=[np.log(np.clip(list(r['selection_scores'].values()),1e-12,1.)).tolist() for r in cal]
    labels=[list(r['selection_scores']).index(r['label']) for r in cal]
    temperature=fit_temperature(logits,labels);modes={}
    for mode,t in [('native-selection',None),('temperature',temperature)]:
        calibration=cal if t is None else scaled(cal,t)
        points=thresholds_from_calibration([r['selected_id']==r['label'] for r in calibration],[r['confidence'] for r in calibration])
        records=tests if t is None else scaled(tests,t)
        base={r['id']:r for r in records if r['view']=='original'};views={}
        for name in VIEWS:
            group=[r for r in records if r['view']==name]
            if len(group)!=64 or {r['id'] for r in group}!=set(base):raise ValueError('Unpaired perturbation rows')
            m=summarize(group,points)
            flips=[float(r['selected_id']!=base[r['id']]['selected_id']) for r in group]
            m.update(changed_count=int(sum(flips)),changed_fraction=float(np.mean(flips)),
                     flip_cluster_ci95=cluster_interval(flips,[r['group'] for r in group]),
                     max_probability_change=max(abs(r['selection_scores'][k]-base[r['id']]['selection_scores'][k]) for r in group for k in r['selection_scores']))
            views[name]=m
        modes[mode]={'temperature':t,'thresholds':points,'views':views}
    timed=[r for r in rows if r['phase']=='test' and r['view']=='original']
    usage=[r['service_metadata'].get('usage',{}) for r in rows]
    costs=[u['cost'] for u in usage if 'cost' in u]
    tokens=[u.get('prompt_tokens',u.get('input_tokens')) for u in usage]
    observed=sorted({(str(r['service_metadata'].get('model')),str(r['service_metadata'].get('provider')),
                      str(r['service_metadata'].get('model_version')),str(r['service_metadata'].get('system_fingerprint'))) for r in rows})
    return {'identity':run['identity'],'modes':modes,'demos':demo,
            'calls':len(rows),'attempts':complete['attempts'],
            'latency_ms':np.quantile([r['seconds']*1000 for r in timed],[.5,.95]).tolist(),
            'latency_scope':run['timing_scope'],'first_observation_utc':rows[0]['started_utc'],
            'last_observation_utc':rows[-1]['finished_utc'],'observed_model_provider_versions':observed,
            'reported_cost_usd':sum(costs) if costs else None,'reported_cost_calls':len(costs),
            'reported_input_tokens':sum(x for x in tokens if x is not None) if any(x is not None for x in tokens) else None,
            'reported_input_token_calls':sum(x is not None for x in tokens),
            'peak_allocated_bytes':complete['peak_allocated_bytes']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/decision-comparison-v1'));a=p.parse_args()
    reports={};sources={}
    for marker in sorted(a.root.glob('*/complete.json')):
        d=marker.parent;reports[d.name]=analyze(d)
        for name in ('records.jsonl','complete.json','run.json'):
            path=d/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if not reports:raise ValueError('No completed evaluations')
    lines=['# Decision-1 and local myJEV: one frozen diagnostic','',
        'Each completed model uses the same 1,000 calibration requests, 64 test requests under six views, and seven authored demos. The sample is diagnostic and does not reproduce Microsoft\'s benchmark. Native selection probabilities and an equally fitted post-hoc temperature are separate views. Vendor confidence is retained in the records without assuming it estimates selected-answer correctness.','',
        '## Original request quality','',
        '| Model | Confidence | Accuracy | Correctness Brier | ECE | Test coverage at 80% calibration threshold | Accepted-case error |',
        '|---|---|---:|---:|---:|---:|---:|']
    pct=lambda x:'undefined' if x is None else f'{100*x:.2f}%'
    for name,run in reports.items():
        for mode,m in run['modes'].items():
            r=m['views']['original'];op=r['operating_points']['coverage_0.8']
            lines.append(f"| {name} | {mode} | {pct(r['accuracy'])} | {r['correctness_brier']:.4f} | {r['ece']:.4f} | {pct(op['coverage'])} | {pct(op['error'])} |")
    lines+=['','Accuracy, macro-F1, Brier metrics, correctness AUROC, 15-bin ECE, group intervals and accepted counts are in `summary.json`. Thresholds use calibration labels only. Their observed error targets provide no deployment guarantee. Temperature is fitted to log probabilities after clipping zeros at 1e-12, and does not change the returned choice. No model or setting is selected using these test outcomes.','',
        '## Changes under each perturbation','',
        'Each row uses the original model settings and threshold. Repeated views share test requests and are never pooled as independent samples. Renamed IDs are mapped back before comparison. The instruction paraphrase is authored and has no independent equivalence audit.','',
        '| Model | View | Changed answers / 64 | Accuracy | Accepted-case error at original 80% threshold |',
        '|---|---|---:|---:|---:|']
    for name,run in reports.items():
        for view,r in run['modes']['native-selection']['views'].items():
            lines.append(f"| {name} | {view} | {r['changed_count']} | {pct(r['accuracy'])} | {pct(r['operating_points']['coverage_0.8']['error'])} |")
    lines+=['','## Sequential latency and billed usage','',
        'Latency uses 64 original test requests after the five warmups and calibration calls. Local scoring includes tokenization and conversion, with GPU synchronization; hosted scoring also includes network and provider queue time. This is a system-level observation across different hardware and request rendering. No matched-hardware speed claim is made.','',
        '| Model | p50 ms | p95 ms | Reported cost USD | Calls with cost / all calls |',
        '|---|---:|---:|---:|---:|']
    for name,run in reports.items():
        cost='not returned' if run['reported_cost_usd'] is None else f"{run['reported_cost_usd']:.6f}"
        lines.append(f"| {name} | {run['latency_ms'][0]:.2f} | {run['latency_ms'][1]:.2f} | {cost} | {run['reported_cost_calls']} / {run['calls']} |")
    lines+=['','Local GPU cost is not estimated here. Missing provider usage is reported as unknown, never zero. A reported total covers only calls returning cost metadata and may exclude failed attempts. Model/service identifiers and observation timestamps are retained; an unchanged API name does not establish unchanged hosted weights.','',
        '## All seven authored demos','',
        '| Model | Request | Expected | Selected | Selected-option probability |','|---|---|---|---|---:|']
    for name,run in reports.items():
        for r in run['demos']:lines.append(f"| {name} | {r['id']} | {r['label']} | {r['selected_id']} | {r['confidence']:.4f} |")
    lines+=['','These examples are unselected diagnostics outside BANKING77. Their confidence has no demonstrated calibration on these new tasks.','',
        '## Replay','',
        'Run `python scripts/summarize_decision_comparison.py` from the repository root. No model download, credentials or GPU is required. The [comparison guide](../../docs/decision-1.md) describes the protocol, inference commands, cost assumptions and external sources.','']
    (a.root/'summary.json').write_text(json.dumps({'models':reports,'source_sha256':sources},indent=2)+'\n')
    (a.root/'report.md').write_text('\n'.join(lines));print(a.root/'report.md')


if __name__=='__main__':main()
