"""Regenerate original teaching diagrams; these are schematics, not benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = '#15334b'
BLUE = '#e8f1fa'
TEAL = '#e0f3ed'
ORANGE = '#fff0da'
GRAY = '#edf0f3'


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.patch.set_facecolor('#fbfcfe')
    ax.set(xlim=(0, 12), ylim=(0, 6.6))
    ax.axis('off')
    ax.text(.35, 6.15, title, fontsize=18 if len(title)>65 else 20, weight='bold', color=INK)
    ax.text(.35, 5.75, subtitle, fontsize=11, color=INK)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def box(ax, x, y, w, h, title, body='', color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=.12',
                               facecolor=color, edgecolor='#9caebd', linewidth=.9))
    ax.text(x+w/2, y+h-.28, title, ha='center', va='center', fontsize=12,
            weight='bold', color=INK)
    if body:
        ax.text(x+w/2, y+(h-.4)/2, body, ha='center', va='center', fontsize=11,
                linespacing=1.45, color=INK)


def arrow(ax, start, end, text=None, bend=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                               connectionstyle=f'arc3,rad={bend}', color=INK, linewidth=1.3))
    if text:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.13, text,
                ha='center', fontsize=10, color=INK)


def footer(ax, text):
    ax.text(.35, .22, text, fontsize=10, color=INK, va='bottom', linespacing=1.35)


def decoding():
    fig, ax = canvas('The same backbone can do two different jobs',
                     'Pretraining ancestry does not determine whether inference must generate text.')
    ax.text(.4, 5.13, 'GENERATING AN ANSWER', fontsize=11, weight='bold', color=INK)
    box(ax,.4,3.65,2.2,1.2,'Read prompt','Initial context pass')
    box(ax,3.3,3.65,2.2,1.2,'Emit token','For example: billing')
    box(ax,6.2,3.65,2.2,1.2,'Decode again','Condition on prior output')
    box(ax,9.1,3.65,2.4,1.2,'Stop + interpret','Text or structured output')
    arrow(ax,(2.62,4.25),(3.28,4.25)); arrow(ax,(5.52,4.25),(6.18,4.25))
    arrow(ax,(8.42,4.25),(9.08,4.25))
    arrow(ax,(7.3,3.63),(4.4,3.63),bend=-.35)
    ax.text(5.85,3.18,'Repeat as needed',ha='center',fontsize=10,color=INK,
            bbox={'facecolor':'#fbfcfe','edgecolor':'none','pad':2})
    ax.text(.4, 2.85, 'myJEV QWEN READOUT', fontsize=11, weight='bold', color=INK)
    box(ax,.4,1.25,3.1,1.25,'Read combined prompt','Context + instructions\n+ candidate descriptions',TEAL)
    box(ax,4.35,1.25,3.1,1.25,'Project final hidden state','Alias scores +\ncandidate-conditioned heads',TEAL)
    box(ax,8.3,1.25,3.2,1.25,'Select + return numbers','Argmax answer\n+ artifact-specific confidence',TEAL)
    arrow(ax,(3.52,1.88),(4.33,1.88)); arrow(ax,(7.47,1.88),(8.28,1.88))
    footer(ax,'No output-token loop in the lower path. The input pass still costs time and memory.\nSchematic only: box widths and arrow lengths do not represent latency.')
    return fig


def tensors():
    fig, ax = canvas('Inside the scratch scorer: two text paths meet',
                     'B = requests, K = candidates, L = padded sequence length, 64 = learned feature width.')
    box(ax,.35,3.7,3.05,1.5,'Context + instructions','Byte IDs [B, Lc]\nEmbedding + position\nEncoder -> [B, Lc, 64]')
    box(ax,.35,1.7,3.05,1.5,'Candidate descriptions','Byte IDs [B, K, Lo]\nSame encoder, B x K sequences\nMean pool -> [B, K, 64]')
    box(ax,4.15,2.6,3.2,1.65,'Find supporting evidence','Candidate queries attend\nto context keys and values\nThen candidate interaction',TEAL)
    arrow(ax,(3.42,4.42),(4.13,3.83)); arrow(ax,(3.42,2.43),(4.13,3.03))
    ax.text(5.75,2.15,'Candidate features [B, K, 64]',ha='center',fontsize=11,color=INK)
    box(ax,8.15,3.65,3.45,1.45,'Selection projection','One logit per candidate\nMask padding, then softmax\nScores [B, K]',TEAL)
    box(ax,8.15,1.5,3.45,1.6,'Correctness heads','Scalar: [B, K]\nConfidence policy: [B, K, 21]\nRead the selected candidate',ORANGE)
    arrow(ax,(7.37,3.66),(8.13,4.3)); arrow(ax,(7.37,2.92),(8.13,2.4))
    footer(ax,'One decision evaluation includes two shared-encoder calls, then attention and heads.\nIt is non-autoregressive. It is not one encoder invocation for all text.')
    return fig


def lora():
    fig, ax = canvas('LoRA changes a layer through a small trainable branch',
                     'For an input vector x: y = W x + s B A x. The configured adapter rank is r = 8.')
    box(ax,.4,2.65,1.7,1.15,'Input x','d_in features',GRAY)
    box(ax,3.25,3.85,4.8,1.1,'Frozen pretrained weights W','Shape [d_out, d_in]',BLUE)
    box(ax,3.05,1.85,2.4,1.25,'Trainable A','[8, d_in]\nCompress to rank 8',TEAL)
    box(ax,6.2,1.85,2.4,1.25,'Trainable B','[d_out, 8]\nProject back + scale s',TEAL)
    box(ax,9.45,2.65,2.05,1.15,'Add branches','Output y',ORANGE)
    arrow(ax,(2.12,3.45),(3.23,4.35)); arrow(ax,(2.12,2.9),(3.03,2.5))
    arrow(ax,(5.47,2.48),(6.18,2.48)); arrow(ax,(8.07,4.35),(9.43,3.4))
    arrow(ax,(8.62,2.48),(9.43,2.95))
    ax.text(.5,1.1,'Training updates A and B, plus the custom correctness heads.\nInference still needs W. Small adapter files do not replace the backbone.',
            fontsize=13, color=INK, linespacing=1.5)
    footer(ax,'Qwen adapters target attention q/k/v/o projections. This drawing shows one adapted linear layer.\nQLoRA stores frozen backbone weights in lower precision; it does not eliminate that branch.')
    return fig


def partitions():
    fig, ax = canvas('Give each data partition one job',
                     'Decisions flow to the right. Held-out test results never flow back into fitting.')
    items=[(.4,'1. Training','Update model\nweights',BLUE),(3.3,'2. Validation','Select configuration\nand checkpoint',BLUE),
           (6.2,'3. Calibration','Freeze temperature\nand acceptance threshold',TEAL),(9.1,'4. Test','Measure error, coverage\nand uncertainty',ORANGE)]
    for x,t,b,c in items:box(ax,x,3.7,2.5,1.35,t,b,c)
    for x in (2.92,5.82,8.72):arrow(ax,(x,4.38),(x+.36,4.38))
    box(ax,.65,1.35,3.0,1.35,'New request','Frozen model produces\nanswer + confidence q',GRAY)
    box(ax,4.5,1.35,3.0,1.35,'Frozen policy','Accept if q >= threshold\nOtherwise request review',TEAL)
    box(ax,8.35,1.35,3.0,1.35,'Monitor actual outcomes','Task shift can change error\nReviewers can make mistakes',ORANGE)
    arrow(ax,(3.67,2.03),(4.48,2.03)); arrow(ax,(7.52,2.03),(8.33,2.03))
    footer(ax,'Grouped examples stay together across partitions. BANKING77 retains its official test split.\nObserved coverage and error describe the measured sample; they are not a future per-request guarantee.')
    return fig


def artifact():
    fig, ax = canvas('An adapter release is a recipe plus learned additions',
                     'Pin the artifact revision and its backbone revision; reuse the loaded model across requests.')
    box(ax,.4,3.45,3.25,1.8,'Hugging Face release','Manifest + prompt + aliases\nAdapter + custom heads\nCalibration + hashes',TEAL)
    box(ax,.4,1.05,3.25,1.75,'Separate backbone cache','Pinned tokenizer + weights\nBF16 or NF4 runtime\nThe large resident component',BLUE)
    box(ax,4.5,2.35,3.0,1.8,'Shared Python loader','Verify hashes and limits\nLoad weights, attach adapter\nMove to GPU, warm up',GRAY)
    arrow(ax,(3.67,4.3),(4.48,3.65)); arrow(ax,(3.67,1.95),(4.48,2.85))
    box(ax,8.35,3.55,3.1,1.35,'Python / CLI','score(request)\nSame response fields',TEAL)
    box(ax,8.35,1.3,3.1,1.6,'HTTP / Docker','One model per process\nBounded queue + readiness\nSame loader and scoring code',TEAL)
    arrow(ax,(7.52,3.6),(8.33,4.15)); arrow(ax,(7.52,2.85),(8.33,2.15))
    footer(ax,'First download, cached process startup, and warm request latency measure different work.\nA container packages the software. Publishing weights does not start a managed endpoint.')
    return fig


def queue():
    fig, axes = plt.subplots(2,1,figsize=(11,6.3),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    ax, note = axes
    for row, (label,start) in enumerate([('A',0),('B',80),('C',160)]):
        if start:
            ax.barh(row,start,left=0,color='#e0e5eb',height=.5)
        ax.barh(row,80,left=start,color='#278775',height=.5)
        if start:ax.text(start/2,row,f'Wait {start} ms',ha='center',va='center',color=INK)
        ax.text(start+40,row,'Run 80 ms',ha='center',va='center',color='white')
        ax.text(start+83,row,f'Return at {start+80} ms',va='center',fontsize=10,color=INK)
    ax.set(yticks=[0,1,2],yticklabels=['Request A','Request B','Request C'],xlim=(0,325),
           xlabel='Time since all three requests arrived (ms)',title='A burst can increase latency without slowing the model')
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.12)
    ax.set_axisbelow(True)
    note.axis('off')
    note.text(0, .95,'Constructed example: one worker, three simultaneous arrivals, exactly 80 ms per score call.',fontsize=11,color=INK)
    note.text(0, .58,'Each call still needs 80 ms of processing. The last caller waits 240 ms in total.\nThis is a scheduling illustration, not measured myJEV timing; network and serialization costs are omitted.',
              fontsize=11,color=INK,linespacing=1.5)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/teaching-diagrams-v1'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,draw in [('generation-and-readout',decoding),('scratch-tensor-path',tensors),
                      ('lora-branches',lora),('data-roles-and-actions',partitions),('artifact-loading',artifact),
                      ('request-queue',queue)]:
        fig=draw()
        for suffix in ('png','svg'):
            path=args.output/f'{name}.{suffix}'
            fig.savefig(path,dpi=180,metadata={'Creator':'myJEV diagram generator'} if suffix=='svg' else None)
            records.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)
    manifest={'scope':'Original explanatory schematics based on myJEV code. No measured timing encoded.',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sources':['src/myjev/model.py','src/myjev/scratch/model.py','configs/4b.json',
                         'src/myjev/inference.py','docs/protocol.md'],'files':records}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'images':len(records)}))


if __name__=='__main__':main()
