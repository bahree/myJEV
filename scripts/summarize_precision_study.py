"""Compare replicated NF4 SFT with the completed matched BF16 controls."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path('results/precision-v1')
rows=[];sources={}
for seed in (11,22,33):
    paths={'bf16':Path(f'results/longer-v1/4b/main/seed-{seed}/sft/evaluation/metrics.json'),
           'nf4':ROOT/f'seed-{seed}/evaluation/metrics.json'}
    if not all(p.exists() for p in paths.values()):continue
    metrics={k:json.loads(p.read_text()) for k,p in paths.items()}
    for p in paths.values():sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append({'seed':seed,**{k:{metric:m[metric] for metric in ('n','accuracy','correctness_brier')} for k,m in metrics.items()},
                 'nf4_minus_bf16_accuracy':metrics['nf4']['accuracy']-metrics['bf16']['accuracy'],
                 'nf4_minus_bf16_brier':metrics['nf4']['correctness_brier']-metrics['bf16']['correctness_brier']})
report={'complete':len(rows)==3,'completed_seed_pairs':len(rows),'rows':rows,'sources':sources,
        'scope':'4B SFT, same seeds, 4000 examples, LR, adapter targets and readout. BF16 versus NF4 with FP32 nonquantized modules. Training configuration comparison, not an inference-only quantization toggle.',
        'limitations':'Three seeds; final-checkpoint test. No new tuning. Does not identify the precision effect on RL, continued supervision or 9B. Capacity-only conclusions still require care.'}
if rows:report['mean_accuracy_difference']=statistics.mean(r['nf4_minus_bf16_accuracy'] for r in rows)
(ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Replicated 4B precision study','',f"Completed seed pairs: {len(rows)}/3. "+('Final descriptive table.' if len(rows)==3 else 'Partial evidence; do not rank configurations yet.'),'',report['scope'],'',report['limitations'],'',
       '| Seed | BF16 accuracy | NF4 accuracy | NF4 minus BF16 | BF16 Brier | NF4 Brier |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['seed']} | {100*r['bf16']['accuracy']:.2f}% | {100*r['nf4']['accuracy']:.2f}% | {100*r['nf4_minus_bf16_accuracy']:+.2f} pp | {r['bf16']['correctness_brier']:.4f} | {r['nf4']['correctness_brier']:.4f} |")
lines+=['','Regenerate: `python scripts/summarize_precision_study.py`. The frozen plan and source hashes identify the exact controls. Further paired uncertainty analysis should precede strong claims from small differences.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'seed_pairs':len(rows),'complete':len(rows)==3}))
