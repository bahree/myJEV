import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('archive_majority_control',Path(__file__).parents[1]/'scripts/archive_majority_control.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def rows(labels):return [{'rubric':'format','label':label} for label in labels]

def test_train_selects_and_calibration_sets_confidence():
    control=m.fit(rows(['a','a','b']),rows(['a','b','b','b']))
    assert control['format']['selected_label']=='a'
    assert control['format']['confidence']==.25
    metrics=m.evaluate(rows(['b','b']),control)['overall']
    assert metrics=={'n':2,'reference_agreement':0.,'correctness_brier':.0625}
    assert control['format']['selected_label']=='a'

def test_tie_is_deterministic_not_calibration_selected():
    control=m.fit(rows(['b','a']),rows(['b','b']))
    assert control['format']['selected_label']=='a'
    assert control['format']['confidence']==0.

def test_missing_calibration_and_unseen_test_rejected():
    with pytest.raises(ValueError):m.fit(rows(['a']),[])
    control=m.fit(rows(['a']),rows(['a']))
    with pytest.raises(ValueError):m.evaluate([{'rubric':'unknown','label':'a'}],control)
