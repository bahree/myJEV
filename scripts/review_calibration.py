"""Exploratory matched post-hoc controls from saved predictions, without model loading."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, logsumexp
from myjev.calibration import fit_temperature
from myjev.metrics import summarize, thresholds_from_calibration


def binary_temperature(q, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    return expit(logit(np.clip(np.asarray(q, dtype=float), 1e-6, 1-1e-6)) / temperature)


def fit_binary_temperature(q, correct):
    z = logit(np.clip(np.asarray(q, dtype=float), 1e-6, 1-1e-6))
    y = np.asarray(correct, dtype=float)
    if z.shape != y.shape or z.ndim != 1 or not len(y) or not np.all(np.isfinite(z)) or not np.isin(y, [0,1]).all():
        raise ValueError('finite paired probabilities and binary labels required')
    def loss(log_t):
        a = z / math.exp(log_t)
        return float(np.mean(np.logaddexp(0, a) - y*a))
    fitted = minimize_scalar(loss, bounds=(math.log(.05), math.log(100)), method='bounded')
    if not fitted.success:
        raise RuntimeError('binary temperature fit failed')
    # Identity fallback uses only calibration loss; never selects by test outcomes.
    chosen = fitted.x if fitted.fun < loss(0) else 0.0
    return math.exp(chosen), {'identity_calibration_nll': loss(0), 'fitted_calibration_nll': loss(chosen)}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compact(rows, learned_source):
    return [{k: r[k] for k in ('id','group','label','selected_id','selection_logits','selection_scores')}
            | {'confidence': r['confidence'], 'learned_confidence': r['confidence_controls'][learned_source]}
            for r in rows]


def view(rows, name, selection_t, learned_t):
    result = []
    for r in rows:
        logits = np.asarray(r['selection_logits'], dtype=float)
        z = logits / (selection_t if name == 'selection_temperature' else 1.0)
        scores = np.exp(z-logsumexp(z))
        if name == 'native':
            confidence = r['confidence']
        elif name == 'learned_temperature':
            confidence = float(binary_temperature(r['learned_confidence'], learned_t))
        else:
            confidence = float(scores.max())
        # The study's temperatures are positive; answer choice never changes.
        assert list(r['selection_scores'])[int(scores.argmax())] == r['selected_id']
        result.append({**r,'confidence':confidence,'selection_scores':dict(zip(r['selection_scores'],scores.tolist()))})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/review-calibration-v1.json'))
    p.add_argument('--source',type=Path,default=Path('results/longer-v1'))
    p.add_argument('--output',type=Path,default=Path('results/review-calibration-v1'))
    p.add_argument('--from-compact',action='store_true',help='Use previously exported text-free inputs')
    a=p.parse_args(); protocol=json.loads(a.protocol.read_text()); a.output.mkdir(parents=True,exist_ok=True)
    aggregate=[]; sources={}
    for size in protocol['sizes']:
      for seed in protocol['seeds']:
       for method in protocol['methods']:
        dest=a.output/size/f'seed-{seed}'/method;dest.mkdir(parents=True,exist_ok=True)
        source=a.source/size/'main'/f'seed-{seed}'/method/'evaluation'
        input_path=dest/'inputs.json.gz'
        if a.from_compact:
            data=json.loads(gzip.decompress(input_path.read_bytes()))
        else:
            paths={part:source/(('calibration-' if part=='calibration' else '')+'predictions.jsonl') for part in ('calibration','test')}
            data={part:compact(read_rows(path),protocol['learned_source'][method]) for part,path in paths.items()}
            data['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
            input_path.write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
        sources[str(input_path)]=hashlib.sha256(input_path.read_bytes()).hexdigest()
        cal,test=data['calibration'],data['test']
        assert len(cal)==1000 and len(test)==3080
        assert not ({r['group'] for r in cal}&{r['group'] for r in test})
        targets=[list(r['selection_scores']).index(r['label']) for r in cal]
        selection_t=fit_temperature([r['selection_logits'] for r in cal],targets)
        learned_t,fit=fit_binary_temperature([r['learned_confidence'] for r in cal],[r['selected_id']==r['label'] for r in cal])
        fits={'selection_temperature':selection_t,'learned_temperature':learned_t,'learned_source':protocol['learned_source'][method],**fit}
        (dest/'fit.json').write_text(json.dumps(fits,indent=2)+'\n')
        for name in protocol['views']:
            c=view(cal,name,selection_t,learned_t);t=view(test,name,selection_t,learned_t)
            thresholds=thresholds_from_calibration([r['selected_id']==r['label'] for r in c],[r['confidence'] for r in c])
            metrics=summarize(t,thresholds)
            q=np.clip([r['confidence'] for r in t],1e-6,1-1e-6);correct=np.array([r['selected_id']==r['label'] for r in t],dtype=float)
            metrics['correctness_nll']=float(np.mean(-correct*np.log(q)-(1-correct)*np.log1p(-q)))
            metrics.update(size=size,seed=seed,method=method,confidence_view=name,fit=fits,scope=protocol['classification'])
            (dest/f'{name}-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
            aggregate.append({k:metrics[k] for k in ('size','seed','method','confidence_view','accuracy','correctness_brier','ece','correctness_auroc','correctness_nll')})
        print(json.dumps({'completed':str(dest),'fits':fits}),flush=True)
    means=[]
    for size in protocol['sizes']:
      for method in protocol['methods']:
       for name in protocol['views']:
        rows=[r for r in aggregate if (r['size'],r['method'],r['confidence_view'])==(size,method,name)]
        means.append(dict(size=size,method=method,confidence_view=name,**{key+'_mean':mean(r[key] for r in rows) for key in ('accuracy','correctness_brier','ece','correctness_auroc','correctness_nll')}))
    report={'protocol':protocol,'protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest(),'per_seed':aggregate,'means':means,'compact_input_sha256':sources}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Exploratory matched post-hoc confidence controls','','All four views were specified before this analysis. Original test results had already been inspected; this is an exploratory follow-up, not a preregistered experiment. Fits use only the original calibration partition. No trained weights or released artifacts were changed.','','Native confidence is the SFT scalar head or RL policy expectation. Selection temperature calibrates the answer distribution; learned temperature applies the same one-parameter binary log-odds map to the trained scalar/policy correctness estimate. Positive temperature preserves answer argmax. No view is selected using test results.','','| Size | Method | Confidence view | Accuracy | Correctness Brier | ECE (15 bins) | AUROC |','|---|---|---|---:|---:|---:|---:|']
    for r in means:lines.append(f"| {r['size']} | {r['method']} | {r['confidence_view']} | {r['accuracy_mean']:.2%} | {r['correctness_brier_mean']:.4f} | {r['ece_mean']:.4f} | {r['correctness_auroc_mean']:.4f} |")
    lines+=['','Means describe seeds 11, 22, 33. Per-seed metrics, reliability bins and empirical operating points remain alongside the summary. Error targets do not provide guarantees. The one-parameter confidence map cannot correct every calibration defect.','', 'Reproduce on exported text-free inputs: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/review_calibration.py --from-compact`. Full-source mode requires original saved predictions. Public inputs include labels, scores and IDs, but no request text.','']
    (a.output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
