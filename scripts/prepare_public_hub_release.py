"""Prepare checksummed packages with owner-approved public names; no network writes."""
import json,hashlib,shutil
from pathlib import Path
old=json.loads(Path('results/release-readiness-v1/publication-manifest-v3.json').read_text())
root=Path('artifacts/hub-public-v1'); root.mkdir(exist_ok=False)
for p in old['packages']:
 size=p['name'].split('-')[1].upper()
 name=f'myJEV-{size}'+('-RL' if '-exact-' in p['name'] else '')
 dest=root/name
 shutil.copytree(p['path'],dest)
 card=(dest/'README.md').read_text().replace('# '+p['name']+': research checkpoint','# '+name)
 card=card.replace('## Load locally','## Load from Hugging Face')
 card=card.replace(f'model = DecisionModel.load("{p["path"]}")',f'from huggingface_hub import HfApi\n# Resolve once and retain this immutable revision in your deployment configuration.\nrevision = HfApi().model_info("bahree/{name}").sha\nmodel = DecisionModel.load("bahree/{name}", revision=revision)')
 (dest/'README.md').write_text(card)
 p.update(path=str(dest),repo_id='bahree/'+name,public_name=name)
 p['files']={f.relative_to(dest).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.rglob('*') if f.is_file()}
old.update(namespace='bahree',visibility='public',scope='Owner-authorized public adapter/head packages; no backbone, training data or optimizer state.')
manifest=Path('results/release-readiness-v1/publication-manifest-public.json');manifest.write_text(json.dumps(old,indent=2)+'\n')
