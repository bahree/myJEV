import argparse
import json
from pathlib import Path
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from myjev.data import read_jsonl, write_jsonl
from myjev.metrics import summarize, thresholds_from_calibration
p=argparse.ArgumentParser()
p.add_argument("--model",default="knowledgator/gliclass-small-v1.0")
p.add_argument("--revision")
p.add_argument("--data",default="data/banking77")
p.add_argument("--output",default="results/gliclass")
p.add_argument("--limit",type=int)
a=p.parse_args()
revision=a.revision or HfApi().model_info(a.model).sha
local=snapshot_download(a.model,revision=revision)
model=GLiClassModel.from_pretrained(local)
tokenizer=AutoTokenizer.from_pretrained(local)
pipeline=ZeroShotClassificationPipeline(model,tokenizer,classification_type="single-label",device="cuda:0",max_length=2048)
preds={}
for split in ("calibration","test"):
    rows=read_jsonl(Path(a.data)/f"{split}.jsonl")
    if a.limit:
        rows=rows[:a.limit]
    preds[split]=[]
    for row in rows:
        labels=[c["description"] for c in row["candidates"]]
        with torch.inference_mode():
            inputs=pipeline.pipe.prepare_inputs([row["context"]],labels,same_labels=True)
            logits=model(**inputs,max_num_classes=len(labels)).logits[0,:len(labels)]
            probabilities=logits.float().softmax(-1).cpu().tolist()
        by_description=dict(zip(labels,probabilities,strict=True))
        scores={c["id"]:by_description[c["description"]] for c in row["candidates"]}
        selected=max(scores,key=scores.get)
        preds[split].append({"id":row["id"],"group":row["group"],"label":row["label"],
                            "selected_id":selected,"selection_scores":scores,"confidence":scores[selected]})
thresholds=thresholds_from_calibration([r["selected_id"]==r["label"] for r in preds["calibration"]],[r["confidence"] for r in preds["calibration"]])
out=Path(a.output)
out.mkdir(parents=True,exist_ok=True)
write_jsonl(out/"predictions.jsonl",preds["test"])
metrics=summarize(preds["test"],thresholds)
metrics.update(model=a.model,revision=revision,subset_limit=a.limit,confidence_semantics="maximum selection score proxy")
(out/"metrics.json").write_text(json.dumps(metrics,indent=2))
