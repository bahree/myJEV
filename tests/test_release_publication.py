"""Publication must fail closed on changed or injected files before network access."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / 'scripts/publish_hub_artifact.py'


def fixture(tmp_path):
    folder = tmp_path / 'package'
    folder.mkdir()
    (folder / 'manifest.json').write_text('{}')
    manifest = tmp_path / 'publication.json'
    manifest.write_text(json.dumps({'packages': [{'name': 'candidate', 'path': str(folder), 'files': {'manifest.json': hashlib.sha256(b'{}').hexdigest()}}]}))
    command = [sys.executable, str(SCRIPT), '--manifest', str(manifest), '--candidate', 'candidate', '--repo-id', 'owner/model', '--visibility', 'private', '--output', str(tmp_path / 'report.json')]
    return folder, command


def test_dry_run_checks_only_local_package(tmp_path):
    _, command = fixture(tmp_path)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['uploaded'] is False


def test_changed_package_refused_before_upload(tmp_path):
    folder, command = fixture(tmp_path)
    (folder / 'unexpected.txt').write_text('not reviewed')
    result = subprocess.run(command + ['--apply'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Package file list or hashes changed' in result.stderr


def test_symlink_refused(tmp_path):
    folder, command = fixture(tmp_path)
    (folder / 'other').symlink_to('/etc/passwd')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Symlinks are not allowed' in result.stderr
