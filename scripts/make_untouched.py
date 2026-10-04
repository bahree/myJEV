import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from myjev.artifacts import save_artifact
from myjev.model import load_backbone, DecisionNetwork
from myjev.prompt import discover_aliases, PROMPT_VERSION
p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--output",required=True)
a=p.parse_args()
c=json.loads(Path(a.config).read_text())
tokenizer=AutoTokenizer.from_pretrained(c["backbone"],revision=c["revision"])
net=DecisionNetwork(load_backbone(c["backbone"],c["revision"],c["precision"]))
aliases,ids=discover_aliases(tokenizer,c["max_candidates"])
save_artifact(a.output,net,{"schema_version":1,"backbone":c["backbone"],"backbone_revision":c["revision"],
    "tokenizer_revision":c["revision"],"precision":c["precision"],"adapter":False,"aliases":aliases,"alias_ids":ids,
    "max_candidates":c["max_candidates"],"max_tokens":c["max_tokens"],"prompt_version":PROMPT_VERSION,
    "temperature":1.,"confidence_mode":"selection","calibration_revision":"uncalibrated",
    "note":"Untouched backbone. Random heads are saved for schema compatibility but never used for reported confidence."})
