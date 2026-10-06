"""Review or upload exactly one checksummed package to an explicit Hub destination."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('results/release-readiness-v1/publication-manifest.json'))
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--repo-id', required=True, help='Owner-selected namespace/repository')
    parser.add_argument('--visibility', choices=('private', 'public'), required=True)
    parser.add_argument('--apply', action='store_true', help='Upload; omission performs local checks only')
    parser.add_argument('--output', type=Path, default=Path('results/release-readiness-v1/hub-upload.json'))
    args = parser.parse_args()
    if args.repo_id.count('/') != 1 or not all(args.repo_id.split('/')):
        raise ValueError('An explicit namespace/repository is required')
    package = next(p for p in json.loads(args.manifest.read_text())['packages'] if p['name'] == args.candidate)
    folder = Path(package['path'])
    actual = {}
    for path in folder.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed in publication packages')
        if path.is_file():
            actual[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != package['files']:
        raise ValueError('Package file list or hashes changed after preparation')
    result = {'candidate': args.candidate, 'repo_id': args.repo_id, 'visibility': args.visibility, 'files': len(actual), 'uploaded': False}
    if args.apply:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        try:
            info = api.repo_info(args.repo_id, repo_type='model')
        except Exception as exc:
            from huggingface_hub.errors import RepositoryNotFoundError
            if not isinstance(exc, RepositoryNotFoundError):
                raise
            api.create_repo(args.repo_id, repo_type='model', private=args.visibility == 'private')
        else:
            if info.private != (args.visibility == 'private'):
                raise ValueError('Existing repository visibility differs; no automatic visibility change')
            if set(api.list_repo_files(args.repo_id)) - {'.gitattributes'}:
                raise ValueError('Destination contains files; use a new repository for this immutable first release')
        commit = api.upload_folder(repo_id=args.repo_id, folder_path=folder, commit_message=f'Release {args.candidate}', repo_type='model')
        with tempfile.TemporaryDirectory(prefix='myjev-hub-verify-') as download:
            copied = Path(snapshot_download(args.repo_id, revision=commit.oid, local_dir=download))
            for name, expected in actual.items():
                if hashlib.sha256((copied / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Pinned download checksum mismatch: {name}')
        result.update(uploaded=True, revision=commit.oid, pinned_download_hashes_equal=True,
                      next_check='Load this pinned Hub revision and compare outputs with the local package.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
