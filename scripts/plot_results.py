import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument("metrics", nargs="+")
p.add_argument("--output",default="results/reliability.png")
a = p.parse_args()
fig, ax = plt.subplots(figsize=(6,5))
for path in a.metrics:
    m = json.loads(Path(path).read_text())
    bins = [b for b in m["reliability"] if b["n"]]
    ax.plot([b["confidence"] for b in bins],[b["accuracy"] for b in bins],"o-",label=Path(path).stem)
ax.plot([0,1],[0,1],"--",color="gray")
ax.set(xlabel="Reported confidence",ylabel="Observed correctness",xlim=(0,1),ylim=(0,1))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(a.output,dpi=160)
