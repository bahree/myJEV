"""Render recorded GPU samples; this is a chart, not a fabricated screenshot."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
with open(a.input) as f:
    rows=list(csv.DictReader(f))
if not rows:
    raise ValueError('no telemetry samples')
start=datetime.fromisoformat(rows[0]['utc'])
fig, axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
for gpu in sorted({r['index'] for r in rows}):
    group=[r for r in rows if r['index']==gpu]
    t=[(datetime.fromisoformat(r['utc'])-start).total_seconds()/60 for r in group]
    for ax,key,scale in [(axes[0],'utilization.gpu',1),(axes[1],'memory.used',1024)]:
        points=[(x,float(r[key])/scale) for x,r in zip(t,group) if r[key] not in {'[N/A]','N/A','[Not Supported]'}]
        if points:
            ax.plot(*zip(*points),label=f'GPU {gpu}')
axes[0].set_ylabel('GPU utilization (%)'); axes[0].set_ylim(0,105)
axes[1].set_ylabel('Device memory (GiB)'); axes[1].set_xlabel('Minutes since '+start.strftime('%Y-%m-%d %H:%M:%S UTC'))
for ax in axes:
    ax.legend(); ax.grid(alpha=.2)
fig.suptitle('A30 study activity — concurrent training and evaluation')
fig.text(.5,.01,'Device-wide telemetry; utilization does not measure computational efficiency.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.03,1,.95])
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=180)
