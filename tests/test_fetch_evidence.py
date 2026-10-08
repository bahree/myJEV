import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/fetch_evidence.py'
spec = importlib.util.spec_from_file_location('fetch_evidence', SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def bundle(tmp_path, members, expected=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, data, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            member.linkname = '/tmp/outside' if kind == tarfile.SYMTYPE else ''
            archive.addfile(member, io.BytesIO(data) if member.isfile() else None)
    path = tmp_path / 'evidence.tar.gz'
    path.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    files = expected if expected is not None else {
        name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for name, data, kind in members}
    return path, {'bytes': path.stat().st_size, 'sha256': fetch.sha256(path),
                  'archive': path.name, 'files': files}


def test_verified_restore_and_repeat_preserve_existing_files(tmp_path):
    archive, record = bundle(tmp_path, [('results/run/inputs.json', b'[]', tarfile.REGTYPE)])
    root = tmp_path / 'checkout'
    root.mkdir()
    fetch.install_archive(archive, record, root)
    restored = root / 'results/run/inputs.json'
    before = restored.stat().st_mtime_ns
    fetch.install_archive(archive, record, root)
    assert restored.read_bytes() == b'[]'
    assert restored.stat().st_mtime_ns == before


@pytest.mark.parametrize('problem', ['hash', 'size', 'missing', 'duplicate', 'extra', 'symlink'])
def test_bad_bundle_never_partially_installs(tmp_path, problem):
    members = [('results/a.json', b'a', tarfile.REGTYPE), ('results/b.json', b'b', tarfile.REGTYPE)]
    archive, record = bundle(tmp_path, members)
    if problem == 'hash':
        record['files']['results/b.json']['sha256'] = '0' * 64
    elif problem == 'size':
        record['files']['results/b.json']['bytes'] = 3
    else:
        bad = members[:1] if problem == 'missing' else members + [members[0]] if problem == 'duplicate' else members + [('results/c.json', b'c', tarfile.REGTYPE)] if problem == 'extra' else [members[0], ('results/b.json', b'', tarfile.SYMTYPE)]
        archive, record = bundle(tmp_path, bad, record['files'])
    root = tmp_path / 'checkout'
    root.mkdir()
    with pytest.raises(ValueError):
        fetch.install_archive(archive, record, root)
    assert not (root / 'results').exists()


def test_conflict_and_corrupt_archive_leave_files_intact(tmp_path):
    archive, record = bundle(tmp_path, [('results/a.json', b'a', tarfile.REGTYPE)])
    root = tmp_path / 'checkout'
    (root / 'results').mkdir(parents=True)
    existing = root / 'results/a.json'
    existing.write_bytes(b'new experiment')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        fetch.install_archive(archive, record, root)
    archive.write_bytes(b'broken')
    with pytest.raises(ValueError, match='Archive checksum'):
        fetch.install_archive(archive, record, root)
    assert existing.read_bytes() == b'new experiment'


@pytest.mark.parametrize('name', ['../escape', '/results/a', 'results/../escape', 'results\\a', 'results//a', '.git/config', 'results/C:/x'])
def test_paths_cannot_escape_results(tmp_path, name):
    with pytest.raises(ValueError, match='Unsafe'):
        fetch.checked_path(tmp_path, name)


def test_symlink_destination_rejected(tmp_path):
    (tmp_path / 'results').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        fetch.checked_path(tmp_path, 'results/a.json')


def test_cli_offline_restore_and_check(tmp_path):
    archive, record = bundle(tmp_path, [('results/a.json', b'a', tarfile.REGTYPE)])
    root = tmp_path / 'checkout'
    (root / 'results').mkdir(parents=True)
    (root / 'results/evidence-manifest.json').write_text(json.dumps({'schema_version': 1, 'bundles': {'pilot': record}}))
    args = [sys.executable, str(SCRIPT), '--root', str(root)]
    missing = subprocess.run(args + ['--check'], capture_output=True, text=True)
    assert missing.returncode == 1 and 'missing or changed' in missing.stderr
    subprocess.run(args + ['--archive-dir', str(archive.parent)], check=True, capture_output=True)
    subprocess.run(args + ['--check'], check=True, capture_output=True)
