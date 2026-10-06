import json
import pytest
import torch
from myjev.scratch.model import ScratchConfig, ScratchNetwork, ScratchDecisionModel, collate


def request(colors=('red','blue')):
    return dict(context='red', instructions='Match the color.',
                candidates=[dict(id=f'id-{c}',description=c) for c in colors])


def model():
    torch.manual_seed(7)
    return ScratchNetwork(ScratchConfig(width=16,layers=1,heads=2)).eval()


def test_permutation_padding_and_ids():
    net=model()
    a=request();b=request(('blue','red'))
    x=net(**collate([a],net.config))
    y=net(**collate([b],net.config))
    torch.testing.assert_close(x['answer'],y['answer'].flip(1),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(x['scalar'],y['scalar'].flip(1),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(x['policy'],y['policy'].flip(1),atol=1e-6,rtol=1e-5)
    z=net(**collate([a,request(('red','green','blue'))],net.config))
    torch.testing.assert_close(x['answer'][0],z['answer'][0,:2],atol=1e-6,rtol=1e-5)
    assert torch.isneginf(z['answer'][0,2])
    assert torch.isfinite(z['policy']).all()
    changed=request();changed['candidates'][0]['id']='unseen-id'
    torch.testing.assert_close(x['answer'],net(**collate([changed],net.config))['answer'])


def test_reload_and_corruption(tmp_path):
    m=ScratchDecisionModel(model(),confidence_mode='policy')
    m.save(tmp_path)
    assert ScratchDecisionModel.load(tmp_path).score(request())==m.score(request())
    with pytest.raises(ValueError,match='overwrite'):m.save(tmp_path)
    p=tmp_path/'manifest.json';v=json.loads(p.read_text());v['temperature']=2.;p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='checksum'):ScratchDecisionModel.load(tmp_path)


def test_limits_and_untrained_confidence():
    net=model();m=ScratchDecisionModel(net)
    assert m.score(request())['confidence'] is None
    r=request();r['context']='x'*257
    with pytest.raises(ValueError,match='byte limit'):m.score(r)
    r=request();r['candidates'][1]['id']=r['candidates'][0]['id']
    with pytest.raises(ValueError):m.score(r)
    r=request(('red','blue'));r['candidates'][0]['description']='é'*33
    with pytest.raises(ValueError,match='byte limit'):m.score(r)


def test_overfit_clean_fixture():
    torch.set_num_threads(1)
    net=model().train();rs=[request(),request()];rs[1]['context']='blue'
    batch=collate(rs,net.config);target=torch.tensor([0,1]);opt=torch.optim.AdamW(net.parameters(),lr=.005)
    for _ in range(160):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(net(**batch)['answer'],target);loss.backward();opt.step()
    assert net.eval()(**batch)['answer'].argmax(-1).tolist()==[0,1]


def test_generator_isolation_and_targets(tmp_path):
    from myjev.scratch.data import build
    from myjev.data import read_jsonl, validate_isolation
    counts=dict(train=32,validation=16,calibration=16,test=16)
    build(tmp_path/'a',counts=counts);build(tmp_path/'b',counts=counts)
    parts={k:read_jsonl(tmp_path/'a'/f'{k}.jsonl') for k in counts}
    validate_isolation(parts)
    assert (tmp_path/'a'/'manifest.json').read_bytes()==(tmp_path/'b'/'manifest.json').read_bytes()
    for rows in parts.values():
        for r in rows:
            assert sum(r['known_probabilities'].values())==1
            assert r['known_probabilities'][r['label']]>0
            from myjev.data import request_from_row
            inputs,_=request_from_row(r)
            assert 'known_probabilities' not in inputs
    assert {r['context'] for r in parts['train']}.isdisjoint(r['context'] for r in parts['test'])


def test_single_network_call_and_finite_rl():
    from myjev.objectives import exact_loss, sampled_loss, joint_log_probs
    net=model();seen=[]
    hook=net.register_forward_hook(lambda *args:seen.append(1))
    ScratchDecisionModel(net,confidence_mode='scalar').score(request());hook.remove()
    assert len(seen)==1
    batch=collate([request(),request(('red','blue','green'))],net.config)
    for objective in (exact_loss,sampled_loss):
        net.zero_grad();out=net(**batch);loss=0
        for i,k in enumerate((2,3)):
            a,p=out['answer'][i:i+1,:k],out['policy'][i:i+1,:k]
            c=torch.zeros_like(a);c[:,0]=1
            loss=loss+objective(a,p,c,joint_log_probs(a,p).detach(),.05)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_shared_loader_and_http(tmp_path):
    from myjev.inference import DecisionModel
    from myjev.server import create_app
    from fastapi.testclient import TestClient
    m=ScratchDecisionModel(model(),confidence_mode='scalar');m.save(tmp_path)
    loaded=DecisionModel.load(tmp_path,device='cpu')
    assert loaded.score(request())==m.score(request())
    with TestClient(create_app(loader=lambda:loaded)) as client:
        assert client.get('/readyz').status_code==200
        result=client.post('/score',json=request())
        assert result.status_code==200
        assert result.json()==m.score(request())
        bad=request();bad['context']='x'*300
        assert client.post('/score',json=bad).status_code==422


def test_exact_byte_and_candidate_boundaries():
    from myjev.scratch.model import encode
    assert len(encode('x'*64,64))==65  # BOS is additional to the declared byte budget.
    with pytest.raises(ValueError):encode('x'*65,64)
    net=model();m=ScratchDecisionModel(net)
    r=request();r['candidates']=[dict(id=str(i),description='x') for i in range(32)]
    assert len(m.score(r)['selection_scores'])==32
    r['candidates'].append(dict(id='32',description='x'))
    with pytest.raises(ValueError,match='too many'):m.score(r)
