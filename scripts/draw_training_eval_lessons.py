"""Regenerate original, explicitly illustrative training/evaluation diagrams."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('results/training-eval-diagrams-v1')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    probability=np.array([.15,.45,.30,.10])
    correctness=np.array([1,1,0,0])
    confidence=np.array([.25,.75,.25,.75])
    reward=correctness-(confidence-correctness)**2
    reference=np.full(4,.25)
    beta=.1
    indices=np.array([1,1,0,2,1,3,2,1])  # Fixed illustrative realization, not a random experiment.
    sample_reward=reward[indices]
    baseline=(sample_reward.sum()-sample_reward)/7
    exact=float(probability@reward)
    kl=float(np.sum(probability*np.log(probability/reference)))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.2,1]})
    axes[0].axis('off')
    rows=[[name,f'{p:.2f}',f'{r:.4f}',f'{p*r:.4f}'] for name,p,r in zip(['Correct, q=.25','Correct, q=.75','Wrong, q=.25','Wrong, q=.75'],probability,reward)]
    table=axes[0].table(cellText=rows,colLabels=['Action','p(action)','Reward','p × reward'],bbox=[0,.38,1,.46],cellLoc='center',colWidths=[.39,.19,.20,.22])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
    axes[0].set_title('Exact: add all four weighted rewards',pad=18)
    axes[0].text(.5,.12,f'Expected reward = {exact:.4f}\nKL to uniform reference = {kl:.4f} nats\nLoss = −reward + 0.1 × KL = {-exact+beta*kl:.4f}',ha='center',transform=axes[0].transAxes)
    axes[1].bar(np.arange(1,9),sample_reward,color=['#216e68' if r>0 else '#b45e46' for r in sample_reward])
    axes[1].axhline(exact,color='#23334a',linestyle='--',label=f'Exact mean {exact:.4f}')
    axes[1].axhline(float(sample_reward.mean()),color='#946a19',linestyle=':',label=f'This sample mean {sample_reward.mean():.4f}')
    axes[1].set(xlabel='Eight joint-action draws (fixed illustration)',ylabel='Sampled reward',title='Sampled: draws repeat or miss actions',ylim=(-.72,1.12))
    axes[1].legend(loc='lower left',fontsize=9)
    fig.suptitle('Constructed two-answer / two-confidence example, not model results',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'joint-actions-worked.png',dpi=170);plt.close(fig)
    q=np.array([.95,.80,.75,.40]);c=np.array([1,1,0,0]);thresholds=[.95,.80,.70,.40]
    coverage=[];error=[]
    for threshold in thresholds:
        accepted=c[q>=threshold];coverage.append(len(accepted)/4);error.append(1-float(accepted.mean()))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].bar(['A','B','C','D'],q,color=['#216e68','#216e68','#b45e46','#b45e46'])
    axes[0].axhline(.8,linestyle='--',color='#23334a');axes[0].text(3.4,.81,'threshold .80',ha='right')
    for i,correct in enumerate(c):axes[0].text(i,.06,'correct' if correct else 'wrong',ha='center',color='white',fontsize=10)
    axes[0].set(ylim=(0,1.05),ylabel='Reported correctness confidence',title='Same four decisions; only acceptance changes')
    axes[1].plot(np.array(coverage)*100,np.array(error)*100,'o-',color='#23334a',markersize=7)
    for x,y,t in zip(coverage,error,thresholds):axes[1].annotate(f't={t:.2f}',(x*100,y*100),xytext=(0,10),textcoords='offset points',ha='center')
    axes[1].set(xlim=(10,110),ylim=(-5,65),xlabel='Coverage: accepted / all requests (%)',ylabel='Accepted-case error (%)',title='Full-set accuracy stays at 50%')
    axes[1].grid(alpha=.2)
    fig.suptitle('Constructed four-request ledger, not a measured deployment curve',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'ledger-coverage-worked.png',dpi=170);plt.close(fig)
    copies={'joint-actions-worked.png':'myjev-part2-qwen-training-and-lessons','ledger-coverage-worked.png':'myjev-part3-evaluation-and-transfer'}
    summary={'scope':'Original analytical illustrations, no trained model, not benchmarks. Fixed sample indices demonstrate one possible realization; not a Monte Carlo gradient validation.', 'joint':{'probability':probability.tolist(),'confidence':confidence.tolist(),'correctness':correctness.tolist(),'reward':reward.tolist(),'reference':reference.tolist(),'beta':beta,'expected_reward':exact,'kl_nats':kl,'loss':-exact+beta*kl,'fixed_sample_indices':indices.tolist(),'sample_reward_mean':float(sample_reward.mean()),'leave_one_out_reward_baselines':baseline.tolist()},'ledger':{'confidence':q.tolist(),'correctness':c.tolist(),'thresholds':thresholds,'coverage':coverage,'accepted_error':error,'brier':float(np.mean((q-c)**2))},'source':'scripts/draw_training_eval_lessons.py','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'images':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in copies}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
