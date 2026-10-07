"""Regenerate reward arithmetic and an explicitly simulated reviewer-cost example."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = Path('results/decision-lessons-v1'); out.mkdir(parents=True, exist_ok=True)
    # Outcomes and confidence draws are independent conditional on this fixed case.
    r = .7
    policies = {'fixed_0.70': ([.7], [1.]), 'mixed_0.40_1.00': ([.4, 1.], [.5, .5])}
    arithmetic = {}
    for name, (q, weights) in policies.items():
        q, weights = np.array(q), np.array(weights)
        mean = float(weights @ q)
        variance = float(weights @ ((q-mean)**2))
        squared = float(weights @ (r*(q-1)**2+(1-r)*q**2))
        assert np.isclose(squared, (mean-r)**2+r*(1-r)+variance)
        arithmetic[name] = {'mean_confidence': mean, 'variance': variance,
                            'expected_squared_error': squared, 'expected_reward': r-squared}
    source = out/'inputs.jsonl.gz'
    provenance = json.loads((out/'inputs-provenance.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance['compact_sha256']
    rows = [json.loads(line) for line in gzip.decompress(source.read_bytes()).decode().splitlines()]
    correct = np.array([x['correct'] for x in rows])
    confidence = np.array([x['confidence'] for x in rows])
    scenarios = []
    for reviewer_error in (0., .02, .05):
        error_cost, review_cost = 100., 2.
        review_total = review_cost + reviewer_error*error_cost
        threshold = 1-review_total/error_cost
        accepted = confidence >= threshold
        scenarios.append({'reviewer_error_assumption': reviewer_error, 'error_cost_units': error_cost,
            'review_cost_units': review_cost, 'threshold_from_cost_assumptions': threshold,
            'test_coverage': float(accepted.mean()), 'accepted_count': int(accepted.sum()),
            'observed_accepted_error': float((~correct[accepted]).mean()) if accepted.any() else None,
            'simulated_cost_per_request': float((accepted & ~correct).mean()*error_cost + (~accepted).mean()*review_total),
            'always_accept_cost_per_request': float((~correct).mean()*error_cost),
            'always_review_simulated_cost_per_request': review_total})
    result = {'semantics': 'Teaching analysis, post-hoc. Reviewer errors/costs are assumptions, not measured human or LLM performance. Thresholds derive from declared costs; none is selected by test performance. Treating confidence as probability can fail under shift.',
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'n': len(rows), 'reward_arithmetic': arithmetic, 'cost_scenarios': scenarios}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with plt.rc_context({'axes.spines.top':False,'axes.spines.right':False,'font.size':11}):
        fig, ax = plt.subplots(figsize=(7,4), layout='constrained')
        q = np.linspace(0,1,201)
        ax.plot(q, r-(r*(q-1)**2+(1-r)*q**2),label='Fixed confidence q',color='#2864a5')
        for name, row in arithmetic.items():
            ax.scatter([row['mean_confidence']],[row['expected_reward']],s=65,label=name.replace('_',' '))
        ax.set(xlabel='Reported mean confidence',ylabel='Expected reward',title='Same mean confidence, different training reward')
        ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.15)
        fig.supxlabel('Constructed correctness probability 0.70. No KL; not a measured model result.',fontsize=9)
        p=Path('07_blog/myjev-part2-qwen-training-and-lessons/images/confidence-variance.png')
        p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=180);plt.close(fig)
        (out/'figure.json').write_text(json.dumps({'output':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':str(out/'summary.json'),'source_sha256':hashlib.sha256((out/'summary.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps(scenarios,indent=2))


if __name__ == '__main__':
    main()
