"""Restore optional study inputs from pinned GitHub release assets (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name
            or ':' in name or relative.as_posix() != name
            or len(relative.parts) < 2 or relative.parts[0] != 'results'):
        raise ValueError(f'Unsafe evidence path: {name}')
    path = root / name
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink in evidence destination: {name}')
    if path.exists() and not path.is_file():
        raise ValueError(f'Expected a regular file: {name}')
    return path


def matches(path, record):
    return (path.is_file() and path.stat().st_size == record['bytes']
            and sha256(path) == record['sha256'])


def require_evidence(bundle, root=Path('.')):
    """Stop replay scripts before writing partial results when inputs are absent."""
    manifest_path = root / 'results/evidence-manifest.json'
    if not manifest_path.exists():
        return  # Older or independent experiment trees have no release manifest.
    manifest = json.loads(manifest_path.read_text())
    if any(not (root / name).is_file() for name in manifest['bundles'][bundle]['files']):
        raise SystemExit(f'Missing {bundle} evidence. Run: python scripts/fetch_evidence.py --bundle {bundle}')


def install_archive(archive, record, root):
    """Validate all members and local conflicts before installing any files."""
    root = root.resolve()
    if not matches(archive, record):
        raise ValueError(f'Archive checksum or size mismatch: {archive.name}')
    expected = record['files']
    destinations = {name: checked_path(root, name) for name in expected}
    for name, path in destinations.items():
        if path.exists() and not matches(path, expected[name]):
            raise ValueError(f'Refusing to overwrite different local evidence: {name}')
    with tempfile.TemporaryDirectory(prefix='myjev-evidence-') as temp:
        staged = Path(temp)
        seen = set()
        with tarfile.open(archive, 'r:gz') as bundle:
            for member in bundle:
                name = member.name
                if name not in expected or name in seen or not member.isfile():
                    raise ValueError(f'Unexpected, repeated or non-regular member: {name}')
                if member.size != expected[name]['bytes']:
                    raise ValueError(f'Member size mismatch: {name}')
                checked_path(root, name)
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                if not matches(target, expected[name]):
                    raise ValueError(f'Member checksum mismatch: {name}')
                seen.add(name)
        if seen != set(expected):
            raise ValueError('Archive is missing manifest files')
        for name, path in destinations.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            # Same-directory temporary file keeps each replacement atomic.
            fd, temporary = tempfile.mkstemp(prefix='.evidence-', dir=path.parent)
            try:
                with os.fdopen(fd, 'wb') as output, (staged / name).open('rb') as source:
                    shutil.copyfileobj(source, output)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bundle', action='append', help='pilot, calibration, order or controls; repeatable; default all')
    parser.add_argument('--archive-dir', type=Path, help='Use downloaded archives without network access')
    parser.add_argument('--check', action='store_true', help='Verify installed files; never download or write')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'results/evidence-manifest.json').read_text())
    if manifest['schema_version'] != 1:
        parser.error('Unsupported evidence manifest version')
    selected = args.bundle or list(manifest['bundles'])
    if set(selected) - manifest['bundles'].keys():
        parser.error('Unknown bundle; choose from ' + ', '.join(manifest['bundles']))
    checked = 0
    try:
        for key in dict.fromkeys(selected):
            record = manifest['bundles'][key]
            missing = [name for name, info in record['files'].items()
                       if not matches(checked_path(root, name), info)]
            if missing and args.check:
                raise ValueError(f'{key}: {len(missing)} missing or changed files; run python scripts/fetch_evidence.py --bundle {key}')
            if missing:
                if args.archive_dir:
                    archive = args.archive_dir / record['archive']
                    if not archive.is_file():
                        raise ValueError(f'Missing local archive: {archive}')
                else:
                    cache = root / '.cache/evidence'
                    cache.mkdir(parents=True, exist_ok=True)
                    archive = cache / record['archive']
                    if not matches(archive, record):
                        print(f'Downloading {key}: {record["bytes"] / 1024**2:.1f} MiB', flush=True)
                        fd, partial = tempfile.mkstemp(prefix='.download-', dir=cache)
                        try:
                            with os.fdopen(fd, 'wb') as output, urlopen(record['url'], timeout=60) as response:
                                remaining = record['bytes']
                                while remaining:
                                    chunk = response.read(min(1024**2, remaining))
                                    if not chunk:
                                        break
                                    output.write(chunk)
                                    remaining -= len(chunk)
                                if response.read(1):
                                    raise ValueError('Download exceeded the recorded archive size')
                            if not matches(Path(partial), record):
                                raise ValueError(f'Download checksum or size mismatch: {key}')
                            os.replace(partial, archive)
                        finally:
                            if os.path.exists(partial):
                                os.unlink(partial)
                install_archive(archive, record, root)
            checked += len(record['files'])
            print(f'{key}: verified {len(record["files"])} files', flush=True)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Verified {checked} evidence files. No model download or GPU required.')


if __name__ == '__main__':
    main()
