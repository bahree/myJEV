import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('study_progress',Path(__file__).parents[1]/'scripts/study_progress.py')
progress=importlib.util.module_from_spec(spec);spec.loader.exec_module(progress)


def setup(root):
    (root/'configs').mkdir()
    plan={'methods':['sft','continued_sft','exact','sampled'],'seeds':[11,22,33],
          'learning_rates':[.00003,.0001],'tuning_updates':1000,'main_updates':4000}
    (root/'configs/study-v1.json').write_text(json.dumps(plan))
    return {'utc':'2026-10-04T23:00:00+00:00','jobs':[{'size':s,'state':'running','stage':'tuning/sft/lr-0/training'} for s in ('0.8b','4b','9b')]}


def test_zero_progress_denominator_and_parallel_eta(tmp_path):
    report=progress.progress(setup(tmp_path),tmp_path)
    assert report['training_updates_total']==168000
    assert report['batch_percent']==0
    assert report['training_updates_done']==0
    assert all(j['training_runs_total']==20 for j in report['jobs'])
    longest=max(j['remaining_seconds'] for j in report['jobs'])
    assert report['remaining_hours_high']==longest*1.4/3600


def test_partial_logs_do_not_count_as_complete_and_failure_hides_eta(tmp_path):
    record=setup(tmp_path)
    root=tmp_path/'artifacts/longer-v1/0.8b/tuning/sft/lr-0';root.mkdir(parents=True)
    (root/'training.jsonl').write_text(json.dumps({'step':100,'session_seconds':70})+'\n{"step":')
    report=progress.progress(record,tmp_path)
    assert report['training_updates_done']==100
    assert report['jobs'][0]['training_runs_done']==0
    assert 0<report['batch_percent']<1
    record['jobs'][1]['state']='failed'
    report=progress.progress(record,tmp_path)
    assert report['remaining_hours_high'] is None


def test_completion_is_explicit_and_stage_labels_cover_main_work(tmp_path):
    record=setup(tmp_path)
    for job in record['jobs']:job['state']='completed'
    report=progress.progress(record,tmp_path)
    assert report['batch_percent']==100
    assert report['remaining_hours_high']==0
    assert 'candidate 2 of 2' in progress.explain_stage('tuning/exact/lr-1/training')
    assert 'seed 22' in progress.explain_stage('main/seed-22/sampled/full-evaluation')
    assert 'calibration' in progress.explain_stage('main/seed-11/sft/posthoc')
