"""Draw the readout study's measured curves; no model or dataset download needed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'alias':'#226c9a','clef':'#bd652e'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/unsloth-head-v1'))
    args=p.parse_args(); root=args.root
    paths=sorted(root.glob('main/seed-*/*/training.jsonl'))
    if len(paths)!=6 or any(len(p.read_text().splitlines())!=1000 for p in paths):
        raise SystemExit('All six 1,000-update traces are required; no partial figure is written.')
    fig,ax=plt.subplots(figsize=(10,4.8)); sources={}
    for path in paths:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        arm=path.parent.name; seed=path.parent.parent.name
        y=np.array([r['loss'] for r in rows]); x=np.array([r['examples'] for r in rows])
        ax.plot(x[24:],np.convolve(y,np.ones(25)/25,'valid'),color=COLORS[arm],alpha=.6,lw=1.2,label=f'{arm}, {seed}')
        sources[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    ax.set(xlabel='Training example exposures',ylabel='Answer cross-entropy (25-update mean)',
           title='Two readouts, the same answer-only training budget')
    ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.18);ax.legend(ncol=2,fontsize=9)
    fig.text(.1,.02,'Three fresh seeds per readout. Training losses do not measure held-out accuracy.',fontsize=9,color='#333333')
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(root/'training-curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for arm in COLORS:
        for seed in (11,22,33):
            path=root/f'main/seed-{seed}/{arm}/temperature-metrics.json'
            data=json.loads(path.read_text()); sources[path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            x=0 if arm=='alias' else 1
            offset={11:-.08,22:0,33:.08}[seed]
            axes[0].scatter(x+offset,100*data['accuracy'],color=COLORS[arm],s=40)
            axes[0].annotate(str(seed),(x+offset,100*data['accuracy']),xytext=(3,4),textcoords='offset points',fontsize=8)
            bins=[b for b in data['reliability'] if b['n']]
            axes[1].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],
                         color=COLORS[arm],alpha=.6,lw=1,label=arm if seed==11 else None)
    axes[0].set(xticks=[0,1],xticklabels=['Alias','Clef'],ylabel='Test accuracy (%)',title='Keep seed differences visible')
    axes[0].margins(x=.5)
    axes[1].plot([0,1],[0,1],color='#777777',ls='--',lw=1)
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean selected-option probability',ylabel='Fraction correct',title='After calibration-only temperature fitting')
    axes[1].legend()
    for ax in axes:ax.spines[['right','top']].set_visible(False);ax.grid(alpha=.18)
    fig.text(.1,.01,'BANKING77 official test, 3,080 examples per run. Reliability: 15 equal-width bins; empty bins omitted.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(root/'accuracy-reliability.png',dpi=160);plt.close(fig)
    (root/'figure-sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Wrote training-curves.png, accuracy-reliability.png and source checksums.')


if __name__=='__main__':main()
