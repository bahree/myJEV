"""Predeclared short three-seed comparison; all main methods receive matched exposure.
This remains a feasibility study, not evidence of convergence or broad RL benefits.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser()
p.add_argument("size",choices=["0.8b","4b","9b"])
p.add_argument("--limit",type=int,default=256)
a=p.parse_args()
root=Path(f"artifacts/three-seed-{a.size}")
root.mkdir(parents=True,exist_ok=True)
(root/"plan.json").write_text(json.dumps({"seeds":[11,22,33],"initial_updates":100,"continuation_updates":100,
    "accumulation":1,"methods":["sft","continued_sft","exact","sampled"],"evaluation_limit":a.limit,
    "test_sample_seed":42,"calibration_sample_seed":43,"status":"short fixed-exposure comparison; no convergence claim"},indent=2))
# Seed 11 artifacts are produced by pilot_suite; run additional seeds only.
for seed in (22,33):
    for method in ("sft","continued_sft","exact","sampled"):
        dest=root/f"seed-{seed}"/method
        dest.mkdir(parents=True,exist_ok=True)
        if not (dest/"artifact/manifest.json").exists():
            cmd=[sys.executable,"-m","myjev.train","--config",f"configs/{a.size}.json","--data","data/banking77/train.jsonl",
                 "--output",str(dest),"--seed",str(seed),"--method",method]
            if method != "sft":
                cmd += ["--initial",str(root/f"seed-{seed}/sft/artifact")]
            with open(dest/"train.log","w") as f:
                subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT)
        output=Path(f"results/three-seed-{a.size}/seed-{seed}/{method}")
        output.mkdir(parents=True,exist_ok=True)
        if (output/"metrics.json").exists():
            continue
        with open(output/"evaluation.log","a") as f:
            subprocess.run([sys.executable,"-m","myjev.evaluate","--artifact",str(dest/"artifact"),
                "--data","data/banking77/test.jsonl","--calibration","data/banking77/calibration.jsonl",
                "--limit",str(a.limit),"--output",str(output)],check=True,stdout=f,stderr=subprocess.STDOUT)
