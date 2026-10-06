"""Report machine-reference agreement separately from public-task forgetting."""
import hashlib
import json
from pathlib import Path
from myjev.data import read_jsonl
from myjev.metrics import summarize

ROOT=Path('results/archive-machine-v1')
paths={'unadapted':ROOT/'unadapted/metrics.json','adapted':ROOT/'adapted/evaluation/metrics.json',
       'banking_before':Path('results/longer-v1/4b/main/seed-11/continued_sft/evaluation/metrics.json'),'banking_after':ROOT/'forgetting/metrics.json'}
if not all(p.exists() for p in paths.values()):raise ValueError('Archive/forgetting evidence is incomplete')
records={k:json.loads(p.read_text()) for k,p in paths.items()}
report={'human_reviewed':False,'semantics':'Archive accuracy means agreement with an unreviewed local machine teacher; BANKING accuracy uses its existing dataset labels.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
        'results':records,'per_rubric':{},'banking_accuracy_change':records['banking_after']['accuracy']-records['banking_before']['accuracy']}
for condition,predpath in [('unadapted',ROOT/'unadapted/predictions.jsonl'),('adapted',ROOT/'adapted/evaluation/predictions.jsonl')]:
    rows=read_jsonl(predpath)
    source={r['id']:r['rubric'] for r in read_jsonl('data/archive-machine-v1/test.jsonl')}
    thresholds={k:v['threshold'] for k,v in records[condition]['operating_points'].items()}
    report['per_rubric'][condition]={name:summarize([r for r in rows if source[r['id']]==name],thresholds) for name in sorted(set(source.values()))}
(ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Exploratory machine-label archive adaptation','',report['semantics'],'','| Condition | N | Reference agreement / accuracy | Correctness Brier |','|---|---:|---:|---:|']
for name,m in records.items():lines.append(f"| {name} | {m['n']} | {100*m['accuracy']:.2f}% | {m['correctness_brier']:.4f} |")
lines+=['','Archive confidence and thresholds are evaluated against machine labels, with group-aware intervals retained in JSON. Human label reliability, related-post grouping and production error are not established. Teacher/student family overlap can inflate agreement. The fixed one-seed adaptation budget does not establish convergence.', '',
        'Inspect `data-manifest.json` for excluded annotations and split counts, and `summary.json` for per-rubric metrics, exact source hashes and forgetting. All raw annotations stay private.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'completed':True,'human_reviewed':False,'banking_accuracy_change':report['banking_accuracy_change']}))
