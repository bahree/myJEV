"""Transfer only. No output from this script is an eligible training/tuning input."""
import argparse
import json
import urllib.request
from pathlib import Path
from myjev.data import text_group, write_jsonl
p = argparse.ArgumentParser()
p.add_argument("--revision")
p.add_argument("--output", default="data/clinc150")
a = p.parse_args()
def fetch(url):
    with urllib.request.urlopen(url) as f:
        return json.load(f)
revision = a.revision or fetch("https://api.github.com/repos/clinc/oos-eval/commits/master")["sha"]
base = f"https://raw.githubusercontent.com/clinc/oos-eval/{revision}/data"
data, domains = fetch(f"{base}/data_full.json"), fetch(f"{base}/domains.json")
labels = sorted(label for ls in domains.values() for label in ls)
near = set(domains["banking"] + domains["credit_cards"])
# Hold out five near-domain classes from candidate lists to test unfamiliar but related unsupported intents.
unsupported = set(sorted(near)[::6])
supported = [l for l in labels if l not in unsupported]
candidates = [{"id": l,"description": l.replace("_"," ")} for l in supported]
parts = {"near": [], "distant": [], "unsupported-near-none": [], "unsupported-near-deferral": [],
         "oos-none": [], "oos-deferral": []}
for i, (text,label) in enumerate(data["test"] + data["oos_test"]):
    row = {"id": f"clinc-test-{i}", "group":text_group(text), "context":text,
           "instructions":"Select the appropriate intent.", "label":label,"candidates":candidates}
    if label in unsupported or label == "oos":
        prefix = "oos" if label == "oos" else "unsupported-near"
        parts[prefix+"-none"].append({**row,"label":"none","candidates":candidates+[{"id":"none","description":"None of the listed intents applies"}]})
        parts[prefix+"-deferral"].append({**row,"label":"__unsupported__"})
    else:
        parts["near" if label in near else "distant"].append(row)
out=Path(a.output)
for name, rows in parts.items():
    write_jsonl(out/f"{name}.jsonl",rows)
(out/"manifest.json").write_text(json.dumps({"revision":revision,"purpose":"transfer-only",
    "held_out_near_labels":sorted(unsupported),"counts":{k:len(v) for k,v in parts.items()}},indent=2))
