import importlib.util
from pathlib import Path
import pytest
import torch
spec=importlib.util.spec_from_file_location('numeric_readout_probe',Path(__file__).parents[1]/'scripts/numeric_readout_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_full_vocab_suffix_score_includes_closing_token_and_correct_shift():
    # Prefix token0, then candidate token1, then closing token2. Wrong positions differ.
    prob=torch.tensor([[.1,.6,.3],[.2,.3,.5],[.8,.1,.1]])
    score=probe.sequence_log_score(prob.log(),torch.tensor([0,1,2]),1)
    assert torch.allclose(score,torch.tensor(.6*.5).log())
    # Global completion scores .30 and .20 normalize to .60 and .40, retaining full-vocabulary conditional mass.
    other=probe.sequence_log_score(torch.tensor([[.1,.2,.7],[.0,.0,1.],[.8,.1,.1]]).clamp_min(1e-20).log(),torch.tensor([0,1,2]),1)
    assert torch.allclose(torch.stack([score,other]).softmax(0),torch.tensor([.6,.4]))


def test_boundary_check_uses_full_string_tokenization_and_rejects_merging():
    class CharacterTokenizer:
        def encode(self,text,add_special_tokens=False):return list(text.encode())
    prefix,suffix=probe.suffix_tokens(CharacterTokenizer(),'Best answer: [','10]')
    assert bytes(prefix).decode()=='Best answer: ['
    assert bytes(suffix).decode()=='10]'
    class MergingTokenizer:
        def encode(self,text,add_special_tokens=False):return [1] if text=='[' else [2]
    with pytest.raises(ValueError,match='boundary'):
        probe.suffix_tokens(MergingTokenizer(),'[','1]')


def test_suffix_scores_are_summed_not_length_normalized():
    short=probe.sequence_log_score(torch.tensor([[.6,.4]]).log(),torch.tensor([0,1]),1)
    long=probe.sequence_log_score(torch.tensor([[.4,.6],[.4,.6]]).log(),torch.tensor([0,1,1]),1)
    assert short>long  # .4 > .6*.6; averaging log scores would reverse this.
    assert torch.allclose(long,torch.tensor(.36).log())
