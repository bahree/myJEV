"""Run the frozen three-seed matrix sequentially on one GPU. No cloud resources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--data",default="data/banking77")
p.add_argument("--output",required=True)
p.add_argument("--updates",type=int,default=1000)
p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33])
p.add_argument("--evaluation-limit",type=int)
a=p.parse_args()
root=Path(a.output)
root.mkdir(parents=True,exist_ok=True)
plan={**vars(a),"methods":["sft","sft_brier","continued_sft","exact","sampled","exact_correctness","sampled_correctness"],
      "note":"SFT exposure N; all continuations start from same SFT and receive additional N. SFT+Brier has N exposure."}
(root/"plan.json").write_text(json.dumps(plan,indent=2))
def run(command,log):
    with open(log,"w") as f:
        subprocess.run([sys.executable,*command],check=True,stdout=f,stderr=subprocess.STDOUT)
for seed in a.seeds:
    base=root/f"seed-{seed}"
    base.mkdir(exist_ok=True)
    initial=base/"sft"/"artifact"
    for method in plan["methods"]:
        dest=base/method
        cmd=["-m","myjev.train","--config",a.config,"--data",f"{a.data}/train.jsonl",
             "--output",str(dest),"--seed",str(seed),"--updates",str(a.updates),"--method",method.replace("_correctness","")]
        if method not in ("sft","sft_brier"):
            cmd += ["--initial",str(initial)]
        if method.endswith("_correctness"):
            cmd += ["--reward","correctness"]
        if not (dest/"artifact/manifest.json").exists():
            run(cmd,base/f"{method}.log")
        evalcmd=["-m","myjev.evaluate","--artifact",str(dest/"artifact"),"--data",f"{a.data}/test.jsonl",
                 "--calibration",f"{a.data}/calibration.jsonl","--output",str(dest/"evaluation")]
        if a.evaluation_limit:
            evalcmd += ["--limit",str(a.evaluation_limit)]
        if not (dest/"evaluation/metrics.json").exists():
            run(evalcmd,base/f"{method}-evaluation.log")
