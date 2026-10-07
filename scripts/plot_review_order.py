"""Plot the frozen candidate-order results; dots are permutations, not training seeds."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('results/review-order-v1');source=root/'summary.json'
    rows=json.loads(source.read_text())['groups']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for i,r in enumerate(rows):
        color='#245b82' if r['method']=='continued_sft' else '#bc652b'
        offsets=[-.10,0,.10]
        for d,acc,changed in zip(offsets,r['shuffled_accuracy'],r['prediction_disagreement']):
            axes[0].scatter([100*(acc-r['original_accuracy'])],[i+d],color=color,s=32)
            axes[1].scatter([100*changed],[i+d],color=color,s=32)
    labels=[r['size'].upper()+(' supervised + T' if r['method']=='continued_sft' else ' exact RL') for r in rows]
    for ax in axes:
        ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].axvline(0,color='#777777',linestyle='--',linewidth=1)
    axes[0].set(xlabel='Accuracy change from original order (percentage points)',title='Aggregate accuracy can look stable')
    axes[1].set(xlabel='Requests whose selected ID changed (%)',title='Individual decisions still change')
    fig.suptitle('Three fixed candidate permutations of each released checkpoint')
    fig.supxlabel('Same 3,080 examples, seed-11 weights and original calibration settings.\nDots are order seeds 101/202/303, with vertical offsets for readability; they are not independent test samples.',fontsize=9)
    out=root/'order-sensitivity.png';fig.savefig(out,dpi=180);plt.close(fig)
    (root/'figure-manifest.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generator':'scripts/plot_review_order.py','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output':str(out),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scope':'Measured candidate permutations; position and alias reassignment change together. No training seed or population uncertainty represented by these dots.'},indent=2)+'\n')

if __name__=='__main__':main()
