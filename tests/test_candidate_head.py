import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from myjev.candidate_head import (CandidateAttentionHead,HeadConfig,CandidateHeadModel,
                                 encode_spans,render_spans,collate_spans)

REQUEST={'context':'A quoted "OPTION 0:" is just customer text.','instructions':'Choose the route.',
         'candidates':[{'id':'secret-label-one','description':'Billing'}, {'id':'secret-label-two','description':'Technical'}]}


class CharTokenizer:
    pad_token_id=0
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c)%127+1 for c in text], 'offset_mapping':[(i,i+1) for i in range(len(text))]}


class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('table',torch.sin(torch.arange(129*8).reshape(129,8).float()))
    def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.table[input_ids])


class Backbone(nn.Module):
    base_model_prefix='decoder'
    def __init__(self):super().__init__();self.decoder=Decoder()


def dummy():
    return CandidateHeadModel(Backbone(),CharTokenizer(),CandidateAttentionHead(HeadConfig(8,8,2)),
        {'backbone':'fixture','backbone_revision':'fixture','width':8,'heads':2,'max_tokens':4096})


def test_spans_are_exact_and_caller_ids_are_not_model_input():
    text,spans=render_spans(REQUEST)
    assert 'secret-label-one' not in text
    assert json.loads(text[slice(*spans['context'])])==REQUEST['context']
    assert json.loads(text[slice(*spans['candidate_0'])])=='Billing'
    e=encode_spans(CharTokenizer(),REQUEST)
    assert sum(e['context'])==len(json.dumps(REQUEST['context']))
    assert not any(a and b for a,b in zip(e['context'],e['candidates'][0]))
    with pytest.raises(ValueError,match='no truncation'):encode_spans(CharTokenizer(),REQUEST,max_tokens=10)


def tensors():
    torch.manual_seed(7)
    hidden=torch.randn(2,10,8);context=torch.zeros(2,10,dtype=torch.bool);context[:,:4]=True
    question=torch.zeros_like(context);question[:,4:6]=True
    candidates=torch.zeros(2,3,10,dtype=torch.bool)
    candidates[:,0,6]=True;candidates[:,1,7]=True;candidates[:,2,8]=True
    valid=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    return hidden,context,question,candidates,valid


def test_masked_tokens_and_candidates_cannot_change_valid_scores():
    head=CandidateAttentionHead(HeadConfig(8,8,2)).eval();items=list(tensors())
    expected=head(*items);items[0]=items[0].clone();items[0][:,9]=1e5
    actual=head(*items)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.isneginf(actual[1,2])
    assert actual.softmax(-1)[1,2]==0


def test_candidate_permutation_is_equivariant_for_fixed_representations():
    head=CandidateAttentionHead(HeadConfig(8,8,2)).eval();h,c,q,cs,v=tensors();order=[2,0,1]
    expected=head(h,c,q,cs,v)[:,order]
    actual=head(h,c,q,cs[:,order],v[:,order])
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)


def test_trainable_head_has_finite_nonzero_gradients():
    head=CandidateAttentionHead(HeadConfig(8,8,2));loss=nn.functional.cross_entropy(head(*tensors()),torch.tensor([1,0]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.cross.in_proj_weight.grad.abs().sum()>0
    h,c,q,cs,v=tensors();c[:]=False
    with pytest.raises(ValueError,match='Context'):head(h,c,q,cs,v)


def test_variable_candidate_collation_and_one_pass():
    model=dummy();a=encode_spans(CharTokenizer(),REQUEST)
    longer={**REQUEST,'candidates':REQUEST['candidates']+[{'id':'other','description':'Neither'}]}
    b=encode_spans(CharTokenizer(),longer)
    inputs,masks=collate_spans([a,b],0,'cpu')
    assert masks[-1].tolist()==[[True,True,False],[True,True,True]]
    result=model.predict([REQUEST,longer])
    assert [len(r['logits']) for r in result]==[2,3]
    assert model.calls==1
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_artifact_reload_and_integrity(tmp_path,monkeypatch):
    model=dummy();before=model.score(REQUEST);model.save(tmp_path/'model')
    monkeypatch.setattr(CandidateHeadModel,'fresh',classmethod(lambda cls,plan,device='cpu':dummy()))
    restored=CandidateHeadModel.load(tmp_path/'model','cpu')
    after=restored.score(REQUEST)
    assert before['selection_scores']==after['selection_scores']
    assert after['confidence'] is None
    path=tmp_path/'model'/'manifest.json';value=json.loads(path.read_text());value['temperature']=.5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Manifest checksum'):CandidateHeadModel.load(tmp_path/'model','cpu')


def test_invalid_dimensions():
    with pytest.raises(ValueError):HeadConfig(8,7,2)
