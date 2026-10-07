"""Render measured training/reliability evidence without loading a model."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/review-teaching-v1'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = []
    trace = []
    colors = ['#245b82', '#bc652b', '#28806b']
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout='constrained')
        for ax, method, title in zip(axes, ['sft', 'exact', 'sampled'],
                                    ['Initial supervised', 'Exact expected reward', 'Sampled REINFORCE']):
            for seed, color in zip([11, 22, 33], colors):
                source = ROOT / f'results/longer-v1/4b/main/seed-{seed}/{method}/training.jsonl'
                sources.append(source)
                rows = [json.loads(line) for line in source.read_text().splitlines()]
                assert len(rows) == 4000
                smooth = np.convolve([row['loss'] for row in rows], np.ones(100)/100, mode='valid')
                ax.plot([row['step'] for row in rows][99:], smooth, label=f'Seed {seed}', color=color)
                trace.extend({'method': method, 'seed': seed, **rows[i]} for i in [0, 1999, 3999])
            ax.set_title(title)
            ax.set_xlabel('Updates in this stage')
            ax.set_ylabel('Logged loss, trailing mean of 100 updates')
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False)
        fig.suptitle('Recorded 4B training traces: different objectives, separate axes')
        fig.supxlabel('Each stage has 4,000 updates. RL starts from its seed-matched supervised checkpoint.\nLoss scales differ; these traces do not measure held-out accuracy or establish convergence.', fontsize=9)
        fig.savefig(OUTPUT/'training-traces.png', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        for (method, view, label), color in zip([
            ('continued_sft','selection_temperature','Continued SFT: selection temperature'),
            ('exact','native','Exact RL: native policy'),
            ('exact','selection_temperature','Exact RL: selection temperature')], colors):
            source=ROOT/f'results/review-calibration-v1/4b/seed-11/{method}/{view}-metrics.json'
            sources.append(source)
            rows=json.loads(source.read_text())['reliability']
            occupied=[r for r in rows if r['n']]
            axes[0].plot([r['confidence'] for r in occupied],[r['accuracy'] for r in occupied], 'o-', markersize=4, label=label, color=color)
            axes[1].step([(r['lo']+r['hi'])/2 for r in rows],[r['n'] for r in rows],where='mid',label=label,color=color)
        axes[0].plot([0,1],[0,1],'--',color='#888888',linewidth=1)
        axes[0].set(xlabel='Mean confidence in occupied bin',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1),title='Reliability')
        axes[1].set(xlabel='Confidence bin midpoint',ylabel='Test examples (log scale)',yscale='symlog',title='Support behind each bin')
        axes[1].legend(fontsize=8,frameon=False,loc='upper left')
        for ax in axes: ax.grid(alpha=.2)
        fig.suptitle('Actual 4B seed-11 decisions under three confidence sources')
        fig.supxlabel('3,080 official test examples; 15 equal-width bins; all fits use calibration only.\nConnecting occupied bins does not supply evidence about empty intervals. Seed 11 is the fixed release seed.',fontsize=9)
        fig.savefig(OUTPUT/'review-reliability.png',dpi=180)
        plt.close(fig)
    telemetry=ROOT/'results/longer-v1/gpu-telemetry-20261004T224916Z/gpu.csv'
    sources.append(telemetry)
    subprocess.run([sys.executable,str(ROOT/'scripts/plot_gpu_telemetry.py'),'--input',str(telemetry),'--output',str(OUTPUT/'longer-gpu-activity.png')],check=True)
    (OUTPUT/'training-trace-excerpts.json').write_text(json.dumps(trace,indent=2)+'\n')
    (OUTPUT/'manifest.json').write_text(json.dumps({
        'generator':'scripts/plot_review_evidence.py',
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Measured traces and test reliability; no synthetic outcomes or new trained weights.',
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.iterdir() if p.suffix in ['.png'] or p.name=='training-trace-excerpts.json'}
    },indent=2)+'\n')

if __name__=='__main__': main()
