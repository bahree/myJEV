import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from fastapi.testclient import TestClient
from myjev.inference import DecisionModel
from myjev.server import create_app
p=argparse.ArgumentParser()
p.add_argument("--artifact",required=True)
p.add_argument("--output",default="results/artifact-equivalence.json")
a=p.parse_args()
request=json.loads(Path("examples/request.json").read_text())
model=DecisionModel.load(a.artifact)
calls=[]
base=model.network.backbone.get_base_model()
hook=base.model.register_forward_hook(lambda *args: calls.append(1))
py=model.score(request)
assert len(calls)==1
hook.remove()
with TestClient(create_app(loader=lambda:model)) as client:
    http=client.post("/score",json=request).json()
assert py==http
large={**request,"context":" ".join(["example"]*5000)}
try:
    model.score(large)
    raise AssertionError("oversized input was accepted")
except ValueError:
    pass
many={**request,"candidates":[{"id":str(i),"description":f"Option number {i}"} for i in range(160)]}
assert len(model.score(many)["selection_scores"])==160
# Release the first loaded model before the independent CLI reload so 9B fits.
client.app.state.model = None
del model, base, client
import gc
gc.collect()
torch.cuda.empty_cache()
cli=json.loads(subprocess.check_output([sys.executable,"-m","myjev.cli","score","--artifact",a.artifact,
                                      "--input","examples/request.json"]))
# __main__ entry point and independent reload are both exercised here.
assert py==cli
Path(a.output).write_text(json.dumps({"python_cli_http_equal":True,"backbone_calls_per_request":1,
    "oversized_rejected":True,"candidate_160_smoke":True,"response":py,
    "scope":"short 160-candidate smoke; not a comprehensive length or calibration validation"},indent=2))
