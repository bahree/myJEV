"""CPU-only, fixed-protocol synthetic CE/Brier/temperature teaching experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def observed_probabilities(clean: torch.Tensor, noise: float) -> torch.Tensor:
    """Uniform replacement can redraw the original class; change rate is noise*(1-1/K)."""
    if not 0 <= noise <= 1:
        raise ValueError('noise must be between zero and one')
    return (1 - noise) * clean + noise / clean.shape[-1]


def make_data(sizes=(1024, 256, 512, 2048), noise=.08):
    gen = torch.Generator().manual_seed(20261006)
    weights = torch.randn(12, 4, generator=gen) / 2
    nonlinear = torch.randn(12, 4, generator=gen) / 3
    output = {}
    for name, n in zip(('train', 'validation', 'calibration', 'test'), sizes, strict=True):
        x = torch.randn(n, 12, generator=gen)
        clean = (x @ weights + torch.sin(x) @ nonlinear).softmax(-1)
        observed = observed_probabilities(clean, noise)
        clean_y = torch.multinomial(clean, 1, generator=gen).squeeze(-1)
        replace = torch.rand(n, generator=gen) < noise
        uniform = torch.randint(4, (n,), generator=gen)
        y = torch.where(replace, uniform, clean_y)
        output[name] = (x, y, observed)
    return output


class Temperature(nn.Module):
    """Consumes already-computed logits, never sends logits through the classifier."""
    def __init__(self, device='cpu', dtype=torch.float32):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros((), device=device, dtype=dtype))

    @property
    def temperature(self):
        # Explicit fixed optimization range, not chosen from test performance.
        return self.log_temperature.clamp(-2.995732273553991, 2.995732273553991).exp()

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits, labels):
    logits = logits.detach()
    scaler = Temperature(logits.device, logits.dtype)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=.2, max_iter=80, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return scaler


def metrics(prob, y, truth):
    predicted = prob.argmax(-1)
    confidence = prob.max(-1).values
    correctness = predicted.eq(y).float()
    ece = 0.
    bins = []
    for i in range(10):
        selected = (confidence >= i / 10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        n = int(selected.sum())
        gap = abs(float(confidence[selected].mean() - correctness[selected].mean())) if n else 0.
        ece += n / len(y) * gap
        bins.append({'lower': i/10, 'upper': (i+1)/10, 'n': n, 'accuracy': float(correctness[selected].mean()) if n else None,
                     'confidence': float(confidence[selected].mean()) if n else None})
    return {'n': len(y), 'accuracy': float(correctness.mean()),
        'nll': float(F.nll_loss(prob.clamp_min(1e-12).log(), y)),
        'multiclass_brier_sum': float(((prob-F.one_hot(y, prob.shape[-1]))**2).sum(-1).mean()),
        'selected_correctness_brier': float(((confidence-correctness)**2).mean()),
        'oracle_probability_mse_mean_over_classes': float(((prob-truth)**2).mean()),
        'ece_10_equal_width': ece, 'reliability_bins': bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/calibration-lab-v1'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    protocol = {'scope': 'Synthetic teaching experiment, not a new natural-data finding', 'seeds': [11,22,33],
        'data_seed': 20261006, 'sizes': {'train':1024,'validation':256,'calibration':512,'test':2048},
        'epochs':60,'batch_size':128,'optimizer':'Adam','learning_rate':.01,'noise_uniform_replacement':.08,
        'expected_label_change_rate':.08*(1-1/4),'temperature_range':[.05,20],
        'student':'12 -> 32 ReLU -> 4; fixed nonlinear generating map',
        'selection':'No checkpoint/hyperparameter selection; validation monitored only. Final epoch for every run.',
        'temperature_fit':'Held-out calibration NLL only, LBFGS max80; test never used in fitting',
        'matching':'Within each seed, identical initial state, batches, updates and examples; same Adam hyperparameters without objective-specific tuning',
        'uncertainty':'Three initialization/order seeds conditional on one fixed synthetic dataset. No population CI.',
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch_version':torch.__version__}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')  # frozen before evaluation
    data = make_data()
    dataset_hash = hashlib.sha256()
    for name, tensors in data.items():
        dataset_hash.update(name.encode())
        for tensor in tensors:dataset_hash.update(tensor.numpy().tobytes())
    rows=[]
    with (out/'training.jsonl').open('w') as log:
        for seed in protocol['seeds']:
            torch.manual_seed(seed)
            initial=nn.Sequential(nn.Linear(12,32),nn.ReLU(),nn.Linear(32,4))
            generator=torch.Generator().manual_seed(seed+1000)
            orders=[torch.randperm(len(data['train'][0]),generator=generator) for _ in range(60)]
            for method in ('ce','brier'):
                model=copy.deepcopy(initial)
                optimizer=torch.optim.Adam(model.parameters(),lr=.01)
                run_start=time.perf_counter()
                x,y,_=data['train']
                for epoch,order in enumerate(orders,1):
                    for batch in order.split(128):
                        logits=model(x[batch])
                        loss=F.cross_entropy(logits,y[batch]) if method=='ce' else ((logits.softmax(-1)-F.one_hot(y[batch],4))**2).sum(-1).mean()
                        optimizer.zero_grad();loss.backward();optimizer.step()
                    if epoch==1 or epoch%10==0:
                        with torch.no_grad():
                            vx,vy,vt=data['validation'];v=metrics(model(vx).softmax(-1),vy,vt)
                        log.write(json.dumps({'seed':seed,'method':method,'epoch':epoch,'optimizer_updates':epoch*8,'last_batch_loss':float(loss.detach()),'validation':v,'elapsed_seconds':time.perf_counter()-run_start})+'\n');log.flush()
                model.eval()
                with torch.no_grad():
                    cx,cy,_=data['calibration'];cal_logits=model(cx)
                    tx,ty,truth=data['test'];test_logits=model(tx)
                scaler=fit_temperature(cal_logits,cy)
                with torch.no_grad():
                    raw=metrics(test_logits.softmax(-1),ty,truth)
                    calibrated=metrics(scaler(test_logits).softmax(-1),ty,truth)
                    assert raw['accuracy']==calibrated['accuracy']
                rows.append({'seed':seed,'method':method,'temperature':float(scaler.temperature.detach()),'raw':raw,'temperature_scaled':calibrated,'runtime_seconds':time.perf_counter()-run_start})
    result={'protocol':protocol,'dataset_sha256':dataset_hash.hexdigest(),'runs':rows,'oracle':metrics(data['test'][2],data['test'][1],data['test'][2]),'runtime_seconds':time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Calibration lab: synthetic results','','Fixed nonlinear map, four classes, separate training/validation/calibration/test. Three matched initialization/order seeds. No test tuning. This is a teaching experiment, not evidence about natural-data model quality.','',
           'Uniform replacement occurs with probability 8%; a replacement redraws the original class one quarter of the time. Expected changed labels: 6%. The true observed distribution is `(1 - 0.08) * p_clean + 0.08 / 4`.','',
           '| Seed | Objective | Temperature | Accuracy | Multiclass Brier raw / scaled | Selected Brier raw / scaled | ECE raw / scaled | Oracle MSE raw / scaled |','|---|---|---|---|---|---|---|---|']
    for row in rows:
        a,b=row['raw'],row['temperature_scaled']
        pairs=[' / '.join(f'{z[k]:.4f}' for z in (a,b)) for k in ('multiclass_brier_sum','selected_correctness_brier','ece_10_equal_width','oracle_probability_mse_mean_over_classes')]
        lines.append(f"| {row['seed']} | {row['method']} | {row['temperature']:.3f} | {a['accuracy']:.4f} | "+' | '.join(pairs)+' |')
    lines.extend(['','Multiclass Brier sums squared errors across the four classes. Selected-correctness Brier compares the largest selection probability with whether its selected class is correct; it is not a separately learned confidence head. Oracle MSE averages across classes against the known observed probability vector. ECE uses 10 equal-width bins, left closed and right open except the last bin. Full bins and NLL appear in JSON.','',
                  'Positive scalar temperature preserves argmax accuracy. It is fitted by calibration NLL, so it does not guarantee improved test Brier or ECE. CE and Brier receive equal exposure and hyperparameters, not individually optimized tuning budgets. Neither loss is guaranteed to win.','',f"CPU runtime: {result['runtime_seconds']:.2f} seconds; one Torch thread. Regenerate with `.venv/bin/python scripts/calibration_lab.py`. Protocol is written before fitting; training/validation logs and dataset/code hashes are retained."])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
