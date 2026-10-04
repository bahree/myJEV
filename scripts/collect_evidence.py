"""Copy compact manifests and full update logs; never copy private weights into Git."""
import json
import shutil
from pathlib import Path
out=Path("results/artifact-manifests")
out.mkdir(parents=True,exist_ok=True)
summary=[]
roots = list(Path("artifacts").glob("pilot-*")) + list(Path("artifacts").glob("three-seed-*/seed-*/*"))
for root in sorted(roots):
    manifest=root/"artifact/manifest.json"
    if not manifest.exists():
        continue
    name = "--".join(root.relative_to("artifacts").parts)
    m=json.loads(manifest.read_text())
    shutil.copy2(manifest,out/f"{name}.json")
    if (root/"training.jsonl").exists():
        shutil.copy2(root/"training.jsonl",out/f"{name}-updates.jsonl")
        records=[json.loads(l) for l in (root/"training.jsonl").read_text().splitlines()]
        summary.append({"run":name,"artifact_revision":m["artifact_revision"],"precision":m["precision"],
                        "backbone":m["backbone"],"training":m["training"],"last_update":records[-1]})
Path("results/pilot-summary.json").write_text(json.dumps(summary,indent=2))
if Path("data/clinc150/manifest.json").exists():
    shutil.copy2("data/clinc150/manifest.json","results/clinc-manifest.json")
