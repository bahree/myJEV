"""Identical finite joint policy for exact and sampled optimization."""
import torch
import torch.nn.functional as F


def joint_log_probs(answer_logits, confidence_logits):
    return F.log_softmax(answer_logits, -1).unsqueeze(-1) + F.log_softmax(confidence_logits, -1)


def rewards(correctness, confidence_aware=True):
    q = torch.linspace(0, 1, 21, device=correctness.device, dtype=correctness.dtype)
    c = correctness.unsqueeze(-1)
    return c - (q - c).square() if confidence_aware else c.expand(*c.shape[:-1], 21)


def exact_loss(answer_logits, confidence_logits, correctness, reference=None, beta=0.0,
               confidence_aware=True):
    logp = joint_log_probs(answer_logits, confidence_logits)
    p = logp.exp()
    loss = -(p * rewards(correctness, confidence_aware)).sum((-2, -1))
    if reference is not None:
        loss = loss + beta * (p * (logp - reference.detach())).sum((-2, -1))
    return loss.mean()


def sampled_loss(answer_logits, confidence_logits, correctness, reference=None, beta=0.0,
                 confidence_aware=True, samples=8):
    if samples < 2:
        raise ValueError("leave-one-out requires at least two samples")
    logp = joint_log_probs(answer_logits, confidence_logits).flatten(1)
    indices = torch.multinomial(logp.exp(), samples, replacement=True)
    chosen_logp = logp.gather(1, indices)
    r = rewards(correctness, confidence_aware).flatten(1).gather(1, indices)
    if reference is not None:
        r = r - beta * (chosen_logp - reference.detach().flatten(1).gather(1, indices))
    # For KL, the omitted explicit derivative has zero expectation (score identity).
    baseline = (r.sum(1, keepdim=True) - r) / (samples - 1)
    return -((r - baseline).detach() * chosen_logp).mean()


def supervised_loss(answer_logits, confidence_logits, scalar_logits, target, brier_weight=0.0):
    c = F.one_hot(target, answer_logits.shape[-1]).to(answer_logits)
    q = torch.linspace(0, 1, 21, device=answer_logits.device)
    # Pretrain the same candidate-conditioned policy subsequently used by both RL methods.
    policy_brier = (confidence_logits.softmax(-1) * (q - c.unsqueeze(-1)).square()).sum(-1).mean()
    chosen = answer_logits.detach().argmax(-1)
    scalar = scalar_logits.gather(1, chosen[:, None]).squeeze(1)
    correct = (chosen == target).to(scalar)
    loss = F.cross_entropy(answer_logits, target) + F.binary_cross_entropy_with_logits(scalar, correct) + policy_brier
    if brier_weight:
        loss = loss + brier_weight * (answer_logits.softmax(-1) - c).square().sum(-1).mean()
    return loss
