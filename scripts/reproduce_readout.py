"""Check JevK5's published small-option prompt/readout on the pinned 0.8B backbone.
This is a readout reproduction, not a reproduction of JevK5's trained checkpoint.
Upstream code must be checked out at the revision in docs/attribution.md.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from myjev.model import load_backbone
p=argparse.ArgumentParser()
p.add_argument("--upstream",required=True)
p.add_argument("--output",default="results/readout-reproduction.json")
a=p.parse_args()
spec=importlib.util.spec_from_file_location("upstream_prompt",Path(a.upstream)/"jevk5/prompt.py")
prompt=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt)
cfg=json.loads(Path("configs/0.8b.json").read_text())
tok=AutoTokenizer.from_pretrained(cfg["backbone"],revision=cfg["revision"])
model=load_backbone(cfg["backbone"],cfg["revision"])
model.eval()
messages=prompt.messages("The policy requires a receipt. The customer has no receipt.","Is a refund allowed?",["Yes","No"])
text=prompt.prompt_text("The policy requires a receipt. The customer has no receipt.","Is a refund allowed?",["Yes","No"])
assert text == tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
inputs=tok(text,return_tensors="pt").to("cuda")
ids=[tok.encode(s,add_special_tokens=False)[0] for s in "AB"]
assert all(len(tok.encode(s,add_special_tokens=False))==1 for s in "AB")
with torch.inference_mode():
    full=model(**inputs,use_cache=False).logits[:,-1,ids].float()
    hidden=model.model(**inputs,use_cache=False).last_hidden_state[:,-1]
    restricted=torch.nn.functional.linear(hidden,model.get_output_embeddings().weight[ids]).float()
torch.testing.assert_close(full,restricted,atol=.125,rtol=.01)
Path(a.output).write_text(json.dumps({"model":cfg,"prompt_equivalent":True,
    "full_logits":full.tolist(),"restricted_logits":restricted.tolist(),
    "max_abs_logit_difference":float((full-restricted).abs().max()),
    "scope":"untouched 0.8B token-readout mechanics; not JevK5 trained-weight reproduction"},indent=2))
