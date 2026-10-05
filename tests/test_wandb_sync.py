"""The upload bridge must tolerate live writes and exclude private material."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('sync_wandb_study',Path(__file__).parents[1]/'scripts/sync_wandb_study.py')
sync=importlib.util.module_from_spec(spec);spec.loader.exec_module(sync)


def test_records_ignore_only_partial_tail(tmp_path):
    path=tmp_path/'training.jsonl'
    path.write_text('{"step":1}\n{"step":')
    assert sync.records(path)==[{'step':1}]
    path.write_text('invalid\n{"step":2}\n')
    with pytest.raises(ValueError): sync.records(path)


def test_evidence_excludes_secrets_weights_and_private_text(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    good='results/longer-v1/0.8b/train.log'
    names=[good,'results/notifications/state.json','results/archive-preparation.json',
           'results/longer-v1/.env','results/longer-v1/model.safetensors',
           'results/pilot-0.8b/temperature-artifact/manifest.json',
           'results/longer-v1/tracking/config.json']
    for name in names:
        path=Path(name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    Path('results/longer-v1/linked.json').symlink_to(Path(good).resolve())
    assert list(sync.evidence_files())==[Path(good)]


def test_epoch_progress_includes_initial_exposure_without_adding_sibling_branches():
    result=sync.epoch_progress(2000,4000,1,4000,7999)
    assert result['stage_epochs_done']==pytest.approx(2000/7999)
    assert result['stage_epochs_remaining']==pytest.approx(2000/7999)
    assert result['lineage_epochs_done']==pytest.approx(6000/7999)
    assert result['lineage_epochs_target']==pytest.approx(8000/7999)
    assert sync.epoch_progress(100,100,4,0,800)['stage_epochs_remaining']==0


def test_settings_accept_custom_env_file_without_executing_shell(tmp_path,monkeypatch):
    path=tmp_path/'local.env'
    path.write_text('WANDB_API_KEY=test-only\nMYJEV_WANDB_ENTITY=example\nMYJEV_WANDB_PROJECT=study\nUNRELATED=ignored\n')
    monkeypatch.delenv('UNRELATED',raising=False)
    for key in ('WANDB_API_KEY','MYJEV_WANDB_ENTITY','MYJEV_WANDB_PROJECT','WANDB_SILENT'):
        monkeypatch.delenv(key,raising=False)
    sync.settings(path)
    assert sync.os.environ['MYJEV_WANDB_PROJECT']=='study'
    assert 'UNRELATED' not in sync.os.environ


def test_export_template_has_no_key():
    example=Path(__file__).parents[1]/'configs/wandb.env.example'
    values=dict(line.split('=',1) for line in example.read_text().splitlines() if line and not line.startswith('#'))
    assert values['WANDB_API_KEY']==''
    assert values['MYJEV_WANDB_ENTITY']=='your-account-or-team'
