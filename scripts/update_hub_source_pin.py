"""Publish a README-only source revision correction with optimistic concurrency."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
from huggingface_hub import HfApi, snapshot_download


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-commit', required=True)
    p.add_argument('--manifest', type=Path, default=Path('results/release-readiness-v1/publication-manifest-public.json'))
    p.add_argument('--output', type=Path, default=Path('results/release-readiness-v1/publication-manifest-public-v2.json'))
    p.add_argument('--packages', type=Path, default=Path('artifacts/hub-public-v2'))
    p.add_argument('--receipts', type=Path, default=Path('results/release-readiness-v1/hub'))
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('Full immutable source commit required')
    original = json.loads(args.manifest.read_text())
    updated = copy.deepcopy(original)
    updated.update(source_code_revision=args.source_commit, uploaded=False,
                   previous_publication_manifest=str(args.manifest),
                   amendment='README source pin only; all runtime files unchanged')
    initial = args.receipts.with_name('hub-initial')
    initial.mkdir(exist_ok=True)
    api = HfApi() if args.apply else None
    for package in updated['packages']:
        source = Path(package['path'])
        if any(sha(source / name) != expected for name, expected in package['files'].items()):
            raise ValueError('Original package content changed')
        destination = args.packages / package['public_name']
        old_card = (source / 'README.md').read_text()
        if original['source_code_revision'] not in old_card:
            raise ValueError('Expected old source pin missing')
        new_card = old_card.replace(original['source_code_revision'], args.source_commit)
        if not destination.exists():
            shutil.copytree(source, destination)
            (destination / 'README.md').write_text(new_card)
        if (destination / 'README.md').read_text() != new_card:
            raise ValueError('Existing new snapshot differs')
        for name, expected in package['files'].items():
            if name != 'README.md' and sha(destination / name) != expected:
                raise ValueError('Runtime file changed during metadata amendment')
        package['path'] = str(destination)
        package['files']['README.md'] = sha(destination / 'README.md')
        if not args.apply:
            continue
        receipt_path = args.receipts / (package['public_name'] + '.json')
        receipt = json.loads(receipt_path.read_text())
        backup = initial / receipt_path.name
        if not backup.exists():
            shutil.copyfile(receipt_path, backup)
        if receipt.get('source_code_revision') != args.source_commit:
            commit = api.upload_file(repo_id=receipt['repo_id'], path_or_fileobj=new_card.encode(),
                                     path_in_repo='README.md', parent_commit=receipt['revision'],
                                     commit_message='Pin source release with Hub cache support')
            receipt.update(previous_revision=receipt['revision'], revision=commit.oid,
                           source_code_revision=args.source_commit, amendment='README source pin only',
                           runtime_files_unchanged=True, pinned_download_hashes_equal=False)
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        downloaded = Path(snapshot_download(receipt['repo_id'], revision=receipt['revision']))
        for name, expected in package['files'].items():
            if sha(downloaded / name) != expected:
                raise ValueError(f'Updated Hub snapshot mismatch: {name}')
        receipt.update(pinned_download_hashes_equal=True,
                       next_check='Runtime files unchanged; initial six-model output equality retained; newest default revision smoke checked separately.')
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        package['hub_revision'] = receipt['revision']
        print(f"{receipt['repo_id']} {receipt['revision']}: card updated, all hashes verified", flush=True)
    updated['uploaded'] = args.apply
    args.output.write_text(json.dumps(updated, indent=2) + '\n')


if __name__ == '__main__':
    main()
