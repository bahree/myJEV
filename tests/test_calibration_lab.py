import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('calibration_lab',Path(__file__).parents[1]/'scripts/calibration_lab.py')
lab=importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_temperature_scales_logits_without_reclassifying_or_mutating_base():
    base=torch.nn.Linear(3,2).double()
    state={k:v.clone() for k,v in base.state_dict().items()}
    x=torch.tensor([[1.,2.,3.],[-1.,0.,2.],[3.,2.,1.]],dtype=torch.float64)
    logits=base(x)
    original=logits.detach().clone()
    scaler=lab.fit_temperature(logits,torch.tensor([0,1,0]))
    assert scaler.temperature.item()>0
    assert scaler.temperature.device==logits.device
    assert scaler.temperature.dtype==logits.dtype
    scaled=scaler(logits)
    assert scaled.shape==(3,2)  # Feeding 2-wide logits into 3-wide base would fail.
    assert torch.allclose(scaled,logits/scaler.temperature)
    assert torch.equal(scaled.argmax(-1),logits.argmax(-1))
    assert torch.equal(logits.detach(),original)
    assert all(torch.equal(v,state[k]) for k,v in base.state_dict().items())
    assert all(p.grad is None for p in base.parameters())


def test_noise_is_uniform_replacement_not_forced_class_change():
    clean=torch.eye(4)
    observed=lab.observed_probabilities(clean,.08)
    assert torch.allclose(observed.diag(),torch.full((4,),.94))
    assert torch.allclose(observed.sum(-1),torch.ones(4))
    assert torch.allclose(observed[0,1:],torch.full((3,),.02))
    assert torch.equal(lab.observed_probabilities(clean,0),clean)
    assert torch.equal(lab.observed_probabilities(clean,1),torch.full((4,4),.25))
    a,b=lab.make_data(),lab.make_data()
    assert set(a)=={'train','validation','calibration','test'}
    for key in a:
        assert all(torch.equal(x,y) for x,y in zip(a[key],b[key]))
        assert a[key][2].shape[-1]==4
        assert torch.allclose(a[key][2].sum(-1),torch.ones(len(a[key][0])))
    assert not torch.equal(a['train'][0][:256],a['validation'][0])


def test_frozen_output_guard_preserves_existing_files(tmp_path):
    import pytest
    output=tmp_path/'results'
    lab.prepare_output(output)
    saved=output/'summary.json'
    saved.write_text('original')
    with pytest.raises(FileExistsError,match='--output results/calibration-lab-rerun'):
        lab.prepare_output(output)
    assert saved.read_text()=='original'
