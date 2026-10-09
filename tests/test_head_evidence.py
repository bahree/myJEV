import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('head_evidence', Path(__file__).parents[1] / 'scripts/sync_head_evidence.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_head_evidence_excludes_secrets_setup_weights_and_tracking(tmp_path):
    wanted = 'results/candidate-head-v1/main/training.jsonl'
    names = [wanted, '.env.wandb', 'results/candidate-head-v1/setup/credentials.json',
             'results/candidate-head-v1/main/tracking/run.json',
             'results/candidate-head-v1/main/credentials.json',
             'results/candidate-head-v1/main/head.safetensors',
             'results/candidate-head-v1/main/main-console.log',
             'results/archive-machine-v2/private-annotations.json']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    assert module.evidence_files(tmp_path) == [Path(wanted)]


def test_head_evidence_refuses_symlinked_files_and_directories(tmp_path):
    secret = tmp_path / 'secret.json'
    secret.write_text('{}')
    base = tmp_path / 'results/candidate-head-v1/main'
    base.mkdir(parents=True)
    (base / 'metrics.json').symlink_to(secret)
    (tmp_path / 'results/candidate-head-v1/pilot').symlink_to(base, target_is_directory=True)
    assert module.evidence_files(tmp_path) == []
