"""Complete main-method pilot continuations and explicitly limited evaluations."""
import argparse
import subprocess
import sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("size",choices=["0.8b","4b","9b"])
p.add_argument("--limit",type=int,default=256)
a=p.parse_args()
base=Path(f"artifacts/pilot-{a.size}")
if not (base/"artifact/manifest.json").exists():
    raise ValueError("complete supervised pilot first")
for method in ("sampled","continued_sft","exact","sft_brier","exact_correctness","sampled_correctness"):
    dest=Path(f"artifacts/pilot-{a.size}-{method}")
    if not (dest/"artifact/manifest.json").exists():
        cmd=[sys.executable,"-m","myjev.train","--config",f"configs/{a.size}.json","--data","data/banking77/train.jsonl",
             "--output",str(dest),"--method",method.replace("_correctness","")]
        if method != "sft_brier":
            cmd += ["--initial",str(base/"artifact")]
        if method.endswith("_correctness"):
            cmd += ["--reward","correctness"]
        with open(f"results/pilot-{a.size}-{method}.log","w") as f:
            subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
for method in ("sft","sampled","continued_sft","exact","sft_brier","exact_correctness","sampled_correctness"):
    dest=base if method=="sft" else Path(f"artifacts/pilot-{a.size}-{method}")
    output=f"results/pilot-{a.size}-{method}-evaluation"
    if (Path(output) / "metrics.json").exists():
        continue
    with open(f"{output}.log","a") as f:
        subprocess.run([sys.executable,"-m","myjev.evaluate","--artifact",str(dest/"artifact"),
            "--data","data/banking77/test.jsonl","--calibration","data/banking77/calibration.jsonl", "--limit",str(a.limit),
            "--output",output],check=True,stdout=f,stderr=subprocess.STDOUT)
