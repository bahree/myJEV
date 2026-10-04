import torch
import torch.nn.functional as F


def fit_temperature(logits, targets):
    """Fit a positive selection temperature on reserved calibration labels only."""
    logits = torch.as_tensor(logits, dtype=torch.float64).detach()
    targets = torch.as_tensor(targets, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp().clamp(.05, 100), targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(.05, 100))
