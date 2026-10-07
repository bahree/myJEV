"""Pinned SDK cache links work without weakening local artifact boundaries."""
import hashlib
import json
from pathlib import Path
import pytest
from myjev import inference

REVISION = 'a' * 40


class ReachedTokenizer(Exception):
    pass


def make_snapshot(tmp_path, monkeypatch):
    repository = tmp_path / 'models--owner--model'
    snapshot = repository / 'snapshots' / REVISION
    blobs = repository / 'blobs'
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    payload = b'adapter metadata'
    blob = blobs / hashlib.sha256(payload).hexdigest()
    blob.write_bytes(payload)
    (snapshot / 'adapter').mkdir()
    (snapshot / 'adapter/README.md').symlink_to(blob)
    manifest = {'schema_version': 1, 'prompt_version': inference.PROMPT_VERSION,
                'backbone_revision': 'b' * 40, 'tokenizer_revision': 'b' * 40,
                'backbone': 'owner/backbone',
                'checksums': {'adapter/README.md': hashlib.sha256(payload).hexdigest()}}
    manifest_blob = blobs / 'manifest-content'
    manifest_blob.write_text(json.dumps(manifest))
    (snapshot / 'manifest.json').symlink_to(manifest_blob)
    monkeypatch.setattr(inference, 'snapshot_download', lambda *args, **kwargs: str(snapshot))
    def reached(*args, **kwargs):
        raise ReachedTokenizer()
    monkeypatch.setattr(inference.AutoTokenizer, 'from_pretrained', reached)
    return snapshot, blobs, manifest, manifest_blob


def test_pinned_hub_snapshot_validates_symlinked_manifest_and_adapter(tmp_path, monkeypatch):
    make_snapshot(tmp_path, monkeypatch)
    # Reaching tokenizer loading proves the real Hub loader accepted all hashes.
    with pytest.raises(ReachedTokenizer):
        inference.DecisionModel.load('owner/model', revision=REVISION)


def test_hub_link_still_requires_matching_checksum(tmp_path, monkeypatch):
    snapshot, _, _, _ = make_snapshot(tmp_path, monkeypatch)
    (snapshot / 'adapter/README.md').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        inference.DecisionModel.load('owner/model', revision=REVISION)


@pytest.mark.parametrize('filename', ['../outside', '/etc/passwd', 'adapter/../../outside', r'..\outside', 'C:/outside'])
def test_manifest_traversal_rejected_before_reading(tmp_path, monkeypatch, filename):
    _, _, manifest, manifest_blob = make_snapshot(tmp_path, monkeypatch)
    manifest['checksums'] = {filename: '0' * 64}
    manifest_blob.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe artifact path'):
        inference.DecisionModel.load('owner/model', revision=REVISION)


def test_hub_link_cannot_escape_into_other_repository(tmp_path, monkeypatch):
    snapshot, _, _, _ = make_snapshot(tmp_path, monkeypatch)
    outside = tmp_path / 'another-repository' / 'blobs' / 'value'
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b'adapter metadata')
    link = snapshot / 'adapter/README.md'
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='unsafe artifact path'):
        inference.DecisionModel.load('owner/model', revision=REVISION)


def test_local_path_does_not_gain_hub_symlink_exception(tmp_path, monkeypatch):
    snapshot, _, _, _ = make_snapshot(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='unsafe artifact path'):
        inference.DecisionModel.load(snapshot, revision=REVISION)


def test_redirected_blobs_directory_is_not_trusted(tmp_path, monkeypatch):
    snapshot, blobs, _, _ = make_snapshot(tmp_path, monkeypatch)
    moved = tmp_path / 'external-blobs'
    blobs.rename(moved)
    blobs.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe artifact path'):
        inference.DecisionModel.load('owner/model', revision=REVISION)


def test_sdk_shared_sharded_blob_store(tmp_path, monkeypatch):
    snapshot, _, _, _ = make_snapshot(tmp_path, monkeypatch)
    # The storage object ID need not equal SHA-256(file); manifest hashes do.
    shared = tmp_path / 'blobs' / 'cc' / ('c' * 64)
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b'adapter metadata')
    link = snapshot / 'adapter/README.md'
    link.unlink()
    link.symlink_to(shared)
    with pytest.raises(ReachedTokenizer):
        inference.DecisionModel.load('owner/model', revision=REVISION)
    shared.write_bytes(b'changed shared content')
    with pytest.raises(ValueError, match='checksum mismatch'):
        inference.DecisionModel.load('owner/model', revision=REVISION)


@pytest.mark.parametrize('shard,object_id', [('cc', 'not-a-blob'), ('ab', 'c' * 64)])
def test_shared_blob_store_requires_exact_sharded_address(tmp_path, monkeypatch, shard, object_id):
    snapshot, _, _, _ = make_snapshot(tmp_path, monkeypatch)
    shared = tmp_path / 'blobs' / shard / object_id
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b'adapter metadata')
    link = snapshot / 'adapter/README.md'
    link.unlink()
    link.symlink_to(shared)
    with pytest.raises(ValueError, match='unsafe artifact path'):
        inference.DecisionModel.load('owner/model', revision=REVISION)
