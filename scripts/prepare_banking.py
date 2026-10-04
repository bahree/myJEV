"""Pin dataset revision; preserve official test and quarantine train/test duplicates."""
import argparse
import hashlib
import json
from pathlib import Path
import csv
import io
import urllib.request
from myjev.data import stratified_groups, text_group, validate_isolation, write_jsonl

p = argparse.ArgumentParser()
p.add_argument("--output", default="data/banking77")
p.add_argument("--revision")
a = p.parse_args()
def fetch(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode()
revision = a.revision or json.loads(fetch("https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/master"))["sha"]
base = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data"
labels = json.loads(fetch(f"{base}/categories.json"))
ds = {part: list(csv.DictReader(io.StringIO(fetch(f"{base}/{part}.csv")))) for part in ("train", "test")}
candidates = [{"id": name, "description": name.replace("_", " ")} for name in labels]

def rows(part):
    return [{"id": f"banking77-{part}-{i}", "group": text_group(r["text"]),
             "context": r["text"], "instructions": "Select the banking support intent.",
             "candidates": candidates, "label": r["category"]} for i, r in enumerate(ds[part])]

test = rows("test")
train = rows("train")
test_groups = {r["group"] for r in test}
excluded = [r for r in train if r["group"] in test_groups]
parts = stratified_groups([r for r in train if r["group"] not in test_groups])
parts["test"] = test
validate_isolation(parts)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
for name, records in parts.items():
    write_jsonl(out / f"{name}.jsonl", records)
write_jsonl(out / "excluded.jsonl", excluded)
manifest = {"dataset": "PolyAI-LDN/task-specific-datasets/banking_data", "revision": revision, "split_seed": 42,
            "counts": {k: len(v) for k, v in parts.items()}, "excluded_train_test_overlap": len(excluded),
            "sha256": {k: hashlib.sha256((out / f"{k}.jsonl").read_bytes()).hexdigest() for k in parts}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
