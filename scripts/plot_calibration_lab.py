"""Plot stored seed-11 toy reliability bins without rerunning training."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/calibration-lab-v1'))
    args=parser.parse_args()
    source=args.results/'summary.json'
    data=json.loads(source.read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,6.4),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    colors=['#2568a0','#bd5c24']
    for col,method in enumerate(('ce','brier')):
        row=next(r for r in data['runs'] if r['seed']==11 and r['method']==method)
        ax,count_ax=axes[:,col]
        ax.plot([0,1],[0,1],color='#777777',linestyle='--',linewidth=1,label='Perfect reliability')
        for i,(key,label) in enumerate((('raw','Raw'),('temperature_scaled','Temperature scaled'))):
            bins=row[key]['reliability_bins']
            populated=[b for b in bins if b['n']]
            ax.plot([b['confidence'] for b in populated],[b['accuracy'] for b in populated],marker='o',color=colors[i],label=label,linewidth=1.8)
            count_ax.bar(np.arange(10)/10+.025+i*.045,[b['n'] for b in bins],width=.042,color=colors[i],alpha=.85)
        ax.set(title=f"{'Cross-entropy' if method=='ce' else 'Multiclass Brier'} | fitted T = {row['temperature']:.2f}",xlim=(0,1),ylim=(0,1),ylabel='Observed selected-answer accuracy')
        ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9,loc='upper left')
        count_ax.set(xlim=(0,1),ylabel='Bin count',xlabel='Confidence (10 equal-width bins)')
        count_ax.set_xticks(np.arange(0,1.01,.2))
        for panel in (ax,count_ax):
            panel.spines[['top','right']].set_visible(False)
    fig.suptitle('Synthetic calibration lab: fixed toy data, seed 11',fontsize=15)
    fig.supxlabel('2,048 test examples. Empty bins omitted from curves; lines join populated bins.\nCalibration fitted on a separate 512-example split. This is not a natural-data benchmark.',fontsize=10)
    target=args.results/'reliability.png';fig.savefig(target,dpi=180);plt.close(fig)
    provenance={'output':str(target),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':11,'bins':10,'scope':'Stored synthetic test bins; illustrative one-seed view; empty bins omitted, no interpolated observations.'}
    (args.results/'figure-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(target)


if __name__=='__main__':main()
