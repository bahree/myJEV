import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('scratch_ladder_v2',Path(__file__).parents[1]/'scripts/run_scratch_ladder_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_nonce_cannot_predict_label_and_splits_are_disjoint():
    parts=m.fixtures()
    assert [len(parts[s]) for s in ('train','test')]==[128,64]
    for rows in parts.values():
        for group in {r['group'] for r in rows}:
            subset=[r for r in rows if r['group']==group]
            assert {r['label'] for r in subset}==set(m.COLORS)
            assert len(subset)==4
            assert len({r['context'].replace('signal: '+r['label'],'signal: <cue>') for r in subset})==1
    assert not {r['group'] for r in parts['train']}&{r['group'] for r in parts['test']}
    assert m.fixtures()==parts
