import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('archive_changes',Path(__file__).parents[1]/'scripts/analyze_archive_changes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_paired_groups_not_individual_rubrics():
    a=[{'id':str(i),'group':str(i//3),'label':'a','selected_id':'b','confidence':.5} for i in range(6)]
    b=[{**r,'selected_id':'a'} for r in reversed(a)]
    result=m.contrast(a,b,repeats=100)
    assert result['groups']==2 and result['n']==6
    assert result['metrics']['agreement']['after_minus_before']==1.
    assert result['metrics']['agreement']['paired_group_ci95']==[1.,1.]
    assert result['metrics']['correctness_brier']['after_minus_before']==0.
    b[0]['label']='different'
    with pytest.raises(AssertionError):m.contrast(a,b,repeats=10)
