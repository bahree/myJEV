import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from myjev.calibration import fit_temperature
from myjev.data import read_jsonl, request_from_row
from myjev.inference import DecisionModel
from myjev.artifacts import save_artifact
p = argparse.ArgumentParser()
p.add_argument("--artifact", required=True)
p.add_argument("--calibration", required=True)
p.add_argument("--output", required=True)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()
if Path(a.output).exists():
    raise ValueError("calibration output must be a new directory")
model = DecisionModel.load(a.artifact, device=a.device)
logits, targets = [], []
with torch.inference_mode():
    for row in read_jsonl(a.calibration):
        request, target = request_from_row(row)
        _, (answer, _, _) = model.logits(request)
        logits.append(answer[0].cpu())
        targets.append(target)
t = fit_temperature(torch.stack(logits), targets)
m = dict(model.manifest)
m.pop("checksums",None)
m.pop("artifact_revision",None)
m.update(temperature=t, confidence_mode="selection", calibration_revision=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest())
save_artifact(a.output, model.network, m)
print(json.dumps({"temperature":t,"output":a.output}))
