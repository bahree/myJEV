import torch
from myjev.objectives import exact_loss, sampled_loss, rewards, joint_log_probs


def test_reward_arithmetic():
    r = rewards(torch.tensor([[0., 1.]]))
    assert r[0, 0, 0] == 0 and r[0, 0, 20] == -1
    assert r[0, 1, 0] == 0 and r[0, 1, 20] == 1
    assert r[0, 1, 10] == .75


def test_exact_gradient_finite_difference():
    torch.manual_seed(3)
    a = torch.randn(1, 3, dtype=torch.double, requires_grad=True)
    q = torch.randn(1, 3, 21, dtype=torch.double, requires_grad=True)
    c = torch.tensor([[0., 1., 0.]], dtype=torch.double)
    ref = joint_log_probs(torch.zeros_like(a), torch.zeros_like(q))
    assert torch.autograd.gradcheck(lambda x,y: exact_loss(x,y,c,ref,.1), (a,q))


def test_sampled_gradient_matches_exact_with_loo():
    torch.manual_seed(11)
    a = torch.tensor([[.4, -.2]], dtype=torch.double, requires_grad=True)
    q = torch.randn(1, 2, 21, dtype=torch.double, requires_grad=True)
    c = torch.tensor([[1., 0.]], dtype=torch.double)
    ref = joint_log_probs(torch.zeros_like(a), torch.zeros_like(q))
    exact = torch.autograd.grad(exact_loss(a,q,c,ref,.13), (a,q))
    # Vectorized independent groups: each uses exactly eight samples and a LOO baseline.
    n = 40000
    loss = sampled_loss(a.expand(n,-1), q.expand(n,-1,-1), c.expand(n,-1), ref.expand(n,-1,-1), .13)
    sampled = torch.autograd.grad(loss, (a,q))
    for e,s in zip(exact, sampled):
        torch.testing.assert_close(s, e, atol=.002, rtol=.06)


def test_known_uncertainty_optimal_confidence():
    # Bernoulli correctness probability .7: expected quadratic score peaks at q=.7.
    expected = .7 * rewards(torch.ones(1,1)) + .3 * rewards(torch.zeros(1,1))
    assert expected.argmax().item() == 14


def test_kl_zero_at_reference():
    a, q = torch.randn(2,3), torch.randn(2,3,21)
    c = torch.zeros_like(a)
    ref = joint_log_probs(a,q)
    torch.testing.assert_close(exact_loss(a,q,c,ref,.5), exact_loss(a,q,c))


def test_shared_frozen_reference_adapter_matches_independent_reference():
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model
    from myjev.model import DecisionNetwork
    import copy
    torch.manual_seed(8)
    base=GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=2,n_embd=16,vocab_size=32,n_positions=32,
                                   resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,bos_token_id=1,eos_token_id=2))
    backbone=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=["c_attn"],task_type="CAUSAL_LM"))
    policy=DecisionNetwork(backbone).eval()
    independent=copy.deepcopy(policy).eval()
    backbone.add_adapter("reference",LoraConfig(r=2,lora_alpha=4,target_modules=["c_attn"],task_type="CAUSAL_LM"))
    # Both LoRA B matrices initialize at zero, so their backbone policy is identical.
    reference=DecisionNetwork(backbone).eval()
    reference.heads.load_state_dict(policy.heads.state_dict())
    reference.heads.requires_grad_(False)
    inputs={"input_ids":torch.tensor([[1,2,3]])}
    backbone.set_adapter("reference")
    with torch.no_grad():
        actual=reference(inputs,[4,5])
        expected=independent(inputs,[4,5])
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e)
    backbone.set_adapter("default")
    answer,q,_=policy(inputs,[4,5])
    exact_loss(answer,q,torch.tensor([[1.,0.]]),joint_log_probs(actual[0],actual[1]),.1).backward()
    assert any(p.grad is not None for n,p in backbone.named_parameters() if ".default." in n)
    assert all(p.grad is None for n,p in backbone.named_parameters() if ".reference." in n)
    assert all(p.grad is None for p in reference.heads.parameters())
