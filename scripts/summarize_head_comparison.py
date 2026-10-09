"""Regenerate the optional readout report from saved, text-free observations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from myjev.metrics import cluster_interval


def read(path): return json.loads(Path(path).read_text())
def compressed(path): return json.loads(gzip.decompress(Path(path).read_bytes()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/unsloth-head-v1'))
    args=p.parse_args(); root=args.root
    plan=read('configs/head-comparison-v1.json'); results=[]; sources={}
    for phase in ('pilot',):
        for arm in plan['arms']:
            base=root/phase/arm
            for name in ('complete.json','initialization.json','encoding.json','run.json'):
                path=base/name
                sources[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    lines=['# Alias readout and Clef head on the same Qwen backbone','',
        'This is a separate answer-only supervised experiment on Qwen3.5-0.8B. Both arms use the same BF16 backbone revision and rank-8 adapters, but their prompts and answer readouts differ. The completed supervision/RL study and released default are unchanged.','',
        '## Feasibility pilot','',
        'Each pilot completed 100 updates with eight examples per update. Memory is the PyTorch peak allocated on an A30; it excludes CUDA context memory. Time includes first-step kernel preparation and optimizer checkpointing.','',
        '| Readout | Trainable parameters | Peak allocated GiB | Training seconds | Mean input tokens |',
        '|---|---:|---:|---:|---:|']
    for arm in plan['arms']:
        base=root/'pilot'/arm; c=read(base/'complete.json'); i=read(base/'initialization.json'); e=read(base/'encoding.json')
        lines.append(f"| {arm} | {i['trainable_parameters']:,} | {c['peak_vram_bytes']/2**30:.3f} | {c['session_seconds']:.1f} | {e['mean_tokens']:.1f} |")
    lines+=['','These are runtime feasibility observations, not accuracy results. Initial adapter hashes and example/order hashes agree across the paired pilots. A prior compiled attempt failed on its first backward pass. Both reported pilots use eager execution, with no silent input truncation.','']
    required=[root/f'main/seed-{s}/{a}/temperature-metrics.json' for s in plan['seeds'] for a in plan['arms']]
    if not all(path.exists() for path in required):
        lines += ['## Accuracy and calibration','',
                  'The matched main comparison is not yet complete. No test-based architecture recommendation is made from the feasibility pilot.','']
    else:
        lines+=['## All three seeds','',
            'Temperatures were fitted on calibration only, after separate learning-rate selection on validation. Confidence here is the selected option probability. Lower Brier is better; ECE uses fifteen equal-width bins.','',
            '| Seed | Readout | Accuracy % | Raw correctness Brier | Calibrated correctness Brier | Calibrated multiclass Brier | Calibrated ECE | Correctness AUROC |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        raw_inputs={}; paired=[]
        for seed in plan['seeds']:
            for arm in plan['arms']:
                base=root/f'main/seed-{seed}/{arm}'
                raw=read(base/'raw-metrics.json'); c=read(base/'temperature-metrics.json')
                values={'seed':seed,'arm':arm,'raw':raw,'calibrated':c}; results.append(values)
                for path in (base/'raw-metrics.json',base/'temperature-metrics.json',base/'inputs.json.gz'):
                    sources[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
                raw_inputs[seed,arm]=compressed(base/'inputs.json.gz')['test']
                auroc='undefined' if c['correctness_auroc'] is None else f"{c['correctness_auroc']:.4f}"
                lines.append(f"| {seed} | {arm} | {100*c['accuracy']:.2f} | {raw['correctness_brier']:.4f} | {c['correctness_brier']:.4f} | {c['selection_multiclass_brier']:.4f} | {c['ece']:.4f} | {auroc} |")
            a,b=raw_inputs[seed,'alias'],raw_inputs[seed,'clef']
            if [(r['id'],r['group'],r['label']) for r in a] != [(r['id'],r['group'],r['label']) for r in b]:
                raise ValueError('Unpaired test inputs')
            correct=lambda rows:np.array([r['keys'][int(np.argmax(r['logits']))]==r['label'] for r in rows],dtype=float)
            paired.append(correct(b)-correct(a))
        values=np.stack(paired); groups=[r['group'] for r in raw_inputs[plan['seeds'][0],'alias']]
        ci=cluster_interval(values.mean(axis=0),groups)
        deltas=values.mean(axis=1)
        comparison={'clef_minus_alias_accuracy':deltas.tolist(),'mean':float(deltas.mean()),
                    'sample_seed_sd':float(deltas.std(ddof=1)), 'group_bootstrap_ci95_conditional_on_seeds':ci,
                    'semantics':'Resample test groups paired across arms, holding the three trained seed pairs fixed. This interval does not include model-training variance.'}
        lines+=['','## Paired accuracy difference','',
                'Clef minus alias, in percentage points, for seeds 11 / 22 / 33: '+ ' / '.join(f'{100*d:+.2f}' for d in deltas)+'.', '',
                f"The mean difference is {100*deltas.mean():+.2f} points. The conditional test-group interval is [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] points; the sample seed standard deviation is {100*deltas.std(ddof=1):.2f} points. The interval holds these three trained seed pairs fixed and does not include training variance.", '',
                '## Accepting decisions or asking for review','',
                'Thresholds are selected on calibration, then applied unchanged to test. These are empirical operating points, not certified error guarantees. Zero accepted cases have undefined error.','',
                '| Seed | Readout | Test coverage at calibration 80% threshold | Accepted-case error | Accepted cases |',
                '|---|---|---:|---:|---:|']
        for r in results:
            op=r['calibrated']['operating_points']['coverage_0.8']
            error='undefined' if op['error'] is None else f"{100*op['error']:.2f}%"
            lines.append(f"| {r['seed']} | {r['arm']} | {100*op['coverage']:.2f}% | {error} | {op['accepted']} |")
        output={'runs':results,'paired_accuracy':comparison,'source_sha256':sources}
        (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines+=['','## Reproduce','',
            'Use the [comparison guide](../../docs/unsloth.md) for dependencies, training commands and scope. Regenerate this report with `python scripts/summarize_head_comparison.py`. The configuration fixes two learning-rate trials per arm and three fresh main seeds. Candidate-order tests keep calibration fixed and are reported separately.','']
    (root/'report.md').write_text('\n'.join(lines))
    (root/'report-sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(root/'report.md')


if __name__=='__main__':main()
