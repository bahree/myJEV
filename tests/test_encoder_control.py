import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

spec=importlib.util.spec_from_file_location('encoder_control',Path(__file__).parents[1]/'scripts/run_encoder_control.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_temperature_reduces_overconfident_calibration_nll():
    logits=np.array([[8.,0.]]*20)
    labels=np.array([0,1]*10)
    temperature=module.fit_temperature(logits,labels)
    def nll(t):
        z=logits/t
        return (logsumexp(z,axis=1)-z[np.arange(20),labels]).mean()
    assert .05<=temperature<=10
    assert nll(temperature)<nll(1.)


def test_fixed_taxonomy_predictions_preserve_mapping_and_argmax():
    rows=[{'id':'a','group':'ga','label':'second'},{'id':'b','group':'gb','label':'first'}]
    logits=[[0.,3.],[2.,0.]]
    raw=module.make_predictions(rows,logits,['first','second'],1.)
    calibrated=module.make_predictions(rows,logits,['first','second'],2.)
    assert [r['selected_id'] for r in raw]==['second','first']
    assert [r['selected_id'] for r in calibrated]==['second','first']
    for a,b in zip(raw,calibrated):
        assert set(b['selection_scores'])=={'first','second'}
        assert abs(sum(b['selection_scores'].values())-1)<1e-12
        assert b['confidence']<a['confidence']
