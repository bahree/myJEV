import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('review_calibration',Path(__file__).parents[1]/'scripts/review_calibration.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_binary_temperature_reduces_calibration_nll_without_reordering():
    q=np.repeat([.1,.3,.7,.9],100)
    y=np.concatenate([np.r_[np.ones(n),np.zeros(100-n)] for n in (30,40,60,70)])
    t,loss=m.fit_binary_temperature(q,y)
    assert t>1
    assert loss['fitted_calibration_nll'] < loss['identity_calibration_nll']
    mapped=m.binary_temperature([0,.1,.3,.7,.9,1],t)
    assert np.all(np.diff(mapped)>0) and np.isfinite(mapped).all()
    assert np.allclose(m.binary_temperature([.2,.8],1),[.2,.8])


def test_policy_map_does_not_replace_selection_or_change_answer():
    rows=[dict(id='a',group='a',label='left',selected_id='left',selection_logits=[3.,1.],selection_scores={'left':.88,'right':.12},confidence=.6,learned_confidence=.6)]
    native=m.view(rows,'native',2,3)[0]
    learned=m.view(rows,'learned_temperature',2,3)[0]
    selection=m.view(rows,'selection_temperature',2,3)[0]
    assert learned['selection_scores']==native['selection_scores']
    assert learned['confidence']!=selection['confidence']
    assert {r['selected_id'] for r in (native,learned,selection)}=={'left'}
    with pytest.raises(ValueError):m.binary_temperature([.5],0)
