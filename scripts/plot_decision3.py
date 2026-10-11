"""Draw figures from the completed comparison; no model execution."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('results/decision3-v1')
    source = root / 'summary.json'
    data = json.loads(source.read_text())
    order = ['myjev-0.8b', 'myjev-0.8b-rl', 'd3-lite', 'myjev-4b', 'myjev-4b-rl', 'd3-mini']
    labels = ['myJEV 0.8B\nsupervised', 'myJEV 0.8B\nRL', 'D3 lite', 'myJEV 4B\nsupervised', 'myJEV 4B\nRL', 'D3 mini']
    colors = ['#31688e', '#7851a9', '#168a78'] * 2
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.facecolor':'white', 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(13,5.2), layout='constrained')
    x = np.arange(len(order))
    native = [data['models'][key]['modes']['native-selection']['phases']['test'] for key in order]
    calibrated = [data['models'][key]['modes']['temperature']['phases']['test'] for key in order]
    accuracy = np.array([m['accuracy'] for m in native]) * 100
    bounds = np.array([m['accuracy_cluster_ci95'] for m in native]) * 100
    axes[0].bar(x,accuracy,color=colors,width=.62)
    axes[0].errorbar(x,accuracy,yerr=[accuracy-bounds[:,0],bounds[:,1]-accuracy],fmt='none',color='#222222',capsize=3)
    axes[0].set_ylabel('Correct answers (%)');axes[0].set_ylim(0,105)
    axes[0].set_title('BANKING77: all 3,080 test requests')
    for i,value in enumerate(accuracy): axes[0].text(i,value+3,f'{value:.2f}',ha='center',fontsize=9)
    axes[1].bar(x-.18,[m['correctness_brier'] for m in native],width=.34,color=colors,alpha=.35,label='Released selected probability')
    axes[1].bar(x+.18,[m['correctness_brier'] for m in calibrated],width=.34,color=colors,label='Equal temperature fitting')
    axes[1].set_ylabel('Correctness Brier (lower is better)')
    axes[1].set_title('Fit temperatures on calibration only');axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=9);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.axvline(2.5,color='#aaaaaa',linewidth=.7)
    fig.suptitle('Released systems with different training histories',fontsize=14)
    fig.savefig(root/'quality.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.1),layout='constrained')
    for i,key in enumerate(order):
        m=data['models'][key];b=m['benchmark']
        if not b: raise ValueError('Complete isolated benchmark before drawing figures')
        for ax,workload,title in zip(axes,['short-demos','banking'],['Seven short demonstrations','77-candidate banking requests']):
            med,p95=b['workloads'][workload]['local_scoring_p50_p95_ms']
            ax.bar(i,med,color=colors[i],width=.62)
            ax.vlines(i,med,p95,color='#222222');ax.hlines(p95,i-.1,i+.1,color='#222222')
            ax.text(i,p95+2,f'{med:.1f}',ha='center',fontsize=9)
            ax.set_title(title);ax.set_ylabel('Local scoring milliseconds')
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=9);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('NVIDIA A30, one loaded model; bars p50, whiskers p95',fontsize=14)
    fig.savefig(root/'latency.png',dpi=170);plt.close(fig)
    receipt={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
             'scope':'Original test accuracy with conditional group intervals, native/equally fitted selection confidence, and isolated local scoring latency. No training-seed or HTTP inference.',
             'figures':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('quality.png','latency.png')}}
    (root/'figure-sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    destinations={'quality.png':'07_blog/myjev-part3-evaluation-and-transfer/images/decision3-quality.png',
                  'latency.png':'07_blog/myjev-part4-inference-and-hosting/images/decision3-latency.png'}
    for name,target in destinations.items():
        path=Path(target)
        if path.parent.is_dir():shutil.copyfile(root/name,path)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
