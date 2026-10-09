"""Regenerate the frozen-backbone head report and figures from saved observations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/candidate-head-v1'))
    args=parser.parse_args();root=args.root;sources={}
    def read(name):
        path=root/name;sources[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    plan_path=Path('configs/candidate-head-v1.json')
    plan=json.loads(plan_path.read_text())
    sources[plan_path.as_posix()]=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    tiny=read('diagnostic/tiny-fit.json');pilot=read('pilot/complete.json');reload=read('pilot/reload.json')
    done=read('main/complete.json');init=read('main/initialization.json')
    raw=read('main/raw-metrics.json');cal=read('main/temperature-metrics.json')
    validation=read('main/validation.json');demos=read('main/demos.json');latency=read('main/latency.json')
    calibrated_reload=read('main/calibrated-reload.json')
    if not tiny['passed'] or not reload['equal'] or not calibrated_reload['all_seven_equal']:
        raise ValueError('A required preflight or reload check failed')
    lines=['# Candidate-attention head on frozen Qwen','',
        'One original 216,193-parameter head reads the frozen Qwen3.5-0.8B text representations. This is a one-seed teaching experiment with fixed settings. It has a different prompt, runtime and training setup from the matched alias/Clef comparison, so differences cannot be attributed to the head alone.','',
        '## Checks before the main run','',
        f"The disposable head fit {tiny['rows']} training rows in {tiny['updates']} updates: {100*tiny['training_accuracy']:.2f}% training accuracy and loss {tiny['training_loss']:.5f}. These rows were used only for a memorization check; the head was discarded.",'',
        f"The {pilot['updates']}-update pilot used {pilot['peak_vram_bytes']/2**30:.3f} GiB peak PyTorch allocation and took {pilot['session_seconds']:.1f} seconds. Its saved/reloaded fixture was identical. The backbone remained frozen.",'',
        '## Main run and held-out decisions','',
        f"Seed {plan['seed']} trained for {done['updates']:,} updates and {done['examples']:,} example exposures. AdamW learning rate was {plan['learning_rate']}, without tuning or validation-selected checkpoints. The head has {init['head_parameters']:,} trainable parameters and the backbone has {init['backbone_parameters']:,} frozen parameters. Mean rendered training length was {init['mean_tokens']:.1f} tokens.",'',
        f"Training session time was {done['session_seconds']:.1f} seconds, with {done['peak_vram_bytes']/2**30:.3f} GiB peak allocated memory. These measurements exclude later evaluation. Descriptive validation accuracy was {100*validation['accuracy']:.2f}% on {validation['n']} examples.",'',
        '| Confidence | Test accuracy | Macro-F1 | Correctness Brier | Multiclass Brier | ECE | Correctness AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in [('Raw selected probability',raw),('Temperature-scaled selected probability',cal)]:
        auroc='undefined' if m['correctness_auroc'] is None else f"{m['correctness_auroc']:.4f}"
        lines.append(f"| {name} | {100*m['accuracy']:.2f}% | {m['macro_f1']:.4f} | {m['correctness_brier']:.4f} | {m['selection_multiclass_brier']:.4f} | {m['ece']:.4f} | {auroc} |")
    lines+=['',f"All {cal['n']:,} official BANKING77 test examples are included. Temperature {cal['temperature']:.6f} was fitted only on the 1,000 calibration examples. ECE uses 15 equal-width bins. The accuracy group-bootstrap interval [{100*cal['accuracy_cluster_ci95'][0]:.2f}%, {100*cal['accuracy_cluster_ci95'][1]:.2f}%] conditions on this one trained checkpoint; it does not include training-seed variation.",'',
        '## Acceptance under the frozen calibration thresholds','',
        'Each threshold was selected on calibration data and applied unchanged to test. The empirical error targets do not certify deployment risk. Empty accepted sets have undefined error.','',
        '| Calibration target | Test coverage | Accepted | Accepted-case error | One-sided 95% error upper bound |',
        '|---|---:|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{100*v:.2f}%'
    for key,op in cal['operating_points'].items():
        lines.append(f"| {key} | {fmt(op['coverage'])} | {op['accepted']} | {fmt(op['error'])} | {fmt(op['error_upper_95'])} |")
    lines+=['','## Seven authored requests','',demos['scope'],'',
            '| Request | Expected | Selected | Confidence | Correct |','|---|---|---|---:|---|']
    for case in demos['cases']:
        r=case['response'];lines.append(f"| {case['id']} | {case['expected_id']} | {r['selected_id']} | {r['confidence']:.4f} | {case['correct']} |")
    lines+=['','Every calibrated response matched after a fresh artifact reload. The implementation checks one backbone invocation for each scoring batch. The expected demo answers are never included in model inputs.','',
        '## Warm scoring on the A30','',
        f"The original three-candidate request measured {latency['p50_ms']:.2f}/{latency['p95_ms']:.2f} ms p50/p95 over {latency['measured_requests']} calls after {latency['warmup_requests']} warmups. Peak allocated memory was {latency['peak_allocated_bytes']/2**30:.3f} GiB. {latency['scope']}",'',
        'The small saved head still needs the Qwen backbone. These local research artifacts use `CandidateHeadModel`; they are separate from the six published adapter releases and the Docker service.','',
        '## Reproduce','',
        'Follow [the candidate-head guide](../../docs/candidate-head.md) for training and scoring. Run `python scripts/summarize_candidate_head.py` to regenerate this report and both figures from the tracked JSON and training log. Recomputing metrics from saved logits uses the optional evidence bundle and `scripts/replay_head_evaluation.py`.','']
    path=root/'main/training.jsonl';trace=[json.loads(x) for x in path.read_text().splitlines()]
    if len(trace)!=plan['main_updates']:raise ValueError('Incomplete main trace')
    sources[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    fig,ax=plt.subplots(figsize=(9,4.3));y=np.array([r['loss'] for r in trace]);x=[r['examples'] for r in trace]
    ax.plot(x,y,color='#7194ae',alpha=.25,lw=.7)
    ax.plot(x[24:],np.convolve(y,np.ones(25)/25,'valid'),color='#225c88',lw=1.8,label='25-update mean')
    ax.set(xlabel='Training example exposures',ylabel='Answer cross-entropy',title='Train the candidate head while Qwen stays frozen')
    ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(root/'training-curve.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4.3))
    for name,m,color in [('Raw',raw,'#b75e31'),('Temperature',cal,'#225c88')]:
        bins=[b for b in m['reliability'] if b['n']]
        axes[0].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],marker='o',ms=3,label=name,color=color)
        axes[1].plot([(b['lo']+b['hi'])/2 for b in m['reliability']],[b['n'] for b in m['reliability']],marker='o',ms=3,label=name,color=color)
    axes[0].plot([0,1],[0,1],ls='--',color='#777777');axes[0].set(xlabel='Mean confidence',ylabel='Fraction correct',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Confidence bin midpoint',ylabel='Test examples in bin')
    for ax in axes:ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('BANKING77: 3,080 test rows, 15 bins, seed 11')
    fig.tight_layout();fig.savefig(root/'reliability.png',dpi=160);plt.close(fig)
    (root/'report.md').write_text('\n'.join(lines))
    (root/'summary.json').write_text(json.dumps({'plan':plan,'tiny_fit':tiny,'pilot':pilot,'main':done,
        'raw':raw,'calibrated':cal,'demos':demos,'latency':latency,'source_sha256':sources},indent=2)+'\n')
    print(root/'report.md')


if __name__=='__main__':main()
