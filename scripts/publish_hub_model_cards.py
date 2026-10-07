"""Prepare and optionally publish reviewed model cards without changing runtime files."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import shutil
from huggingface_hub import HfApi, ModelCard, snapshot_download


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=Path('results/release-readiness-v1/publication-manifest-public-v2.json'))
    p.add_argument('--cards',type=Path,default=Path('model_cards') if Path('model_cards').is_dir() else Path('publishing/huggingface'))
    p.add_argument('--packages',type=Path,default=Path('artifacts/hub-reader-v1'))
    p.add_argument('--output',type=Path,default=Path('results/model-card-refresh-v1/publication-manifest.json'))
    p.add_argument('--receipts',type=Path,default=Path('results/release-readiness-v1/hub'))
    p.add_argument('--source-code-revision',help='Public loader commit shown by this card edition')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    initial=json.loads(a.manifest.read_text());updated=copy.deepcopy(initial)
    updated.update(uploaded=False,previous_publication_manifest=str(a.manifest),amendment='Reader-facing README and model metadata only; runtime files unchanged')
    if a.source_code_revision:
        if not re.fullmatch('[0-9a-f]{40}',a.source_code_revision):raise ValueError('Immutable public source commit required')
        updated['reader_source_code_revision']=a.source_code_revision
    a.output.parent.mkdir(parents=True,exist_ok=True)
    backup=a.output.parent/'prior-receipts';backup.mkdir(exist_ok=True)
    prepared=[]
    # Validate every card and package before the first remote change.
    for row in updated['packages']:
        original=Path(row['path']);card_path=a.cards/(row['public_name']+'.md');card=card_path.read_text()
        if '\u2014' in card:raise ValueError('Em dash found in card')
        ModelCard(card).validate()
        for name,expected in row['files'].items():
            if (original/name).is_symlink() or sha(original/name)!=expected:raise ValueError('Original package changed')
        destination=a.packages/row['public_name']
        if not destination.exists():
            shutil.copytree(original,destination)
            (destination/'README.md').write_text(card)
        expected={**row['files'],'README.md':hashlib.sha256(card.encode()).hexdigest()}
        actual={str(f.relative_to(destination)):sha(f) for f in destination.rglob('*') if f.is_file()}
        if actual!=expected:raise ValueError('Prepared snapshot differs from reviewed card/runtime package')
        row.update(path=str(destination),files=expected)
        prepared.append(row)
    api=HfApi() if a.apply else None
    for row in prepared:
        if not a.apply:continue
        receipt_path=a.receipts/(row['public_name']+'.json');receipt=json.loads(receipt_path.read_text())
        preserved=backup/receipt_path.name
        if not preserved.exists():shutil.copyfile(receipt_path,preserved)
        new_hash=row['files']['README.md']
        if receipt.get('reader_card_sha256')!=new_hash:
            commit=api.upload_file(repo_id=row['repo_id'],path_in_repo='README.md',
               path_or_fileobj=(Path(row['path'])/'README.md').read_bytes(),parent_commit=receipt['revision'],
               commit_message='Explain model choices, measured results and how to run inference')
            receipt.update(previous_revision=receipt['revision'],revision=commit.oid,reader_card_sha256=new_hash,
                           runtime_files_unchanged=True,pinned_download_hashes_equal=False,
                           amendment='Reader-facing README and metadata; runtime files unchanged')
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        downloaded=Path(snapshot_download(row['repo_id'],revision=receipt['revision']))
        for name,expected in row['files'].items():
            if sha(downloaded/name)!=expected:raise ValueError(f'Downloaded file mismatch: {name}')
        receipt.update(pinned_download_hashes_equal=True,
           next_check='Runtime bytes match the six-model verified release. Card example pins that verified prior revision; latest default reload checked separately.')
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');row['hub_revision']=receipt['revision']
        print(f"{row['repo_id']} {receipt['revision']}: reader card published; every file hash verified",flush=True)
    updated['uploaded']=a.apply
    a.output.write_text(json.dumps(updated,indent=2)+'\n')


if __name__=='__main__':main()
