"""Freeze one diagnostic sample and source hashes before any model is evaluated."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = {'study': 'decision-comparison-v1', 'test_limit': 64, 'sample_seed': 42,
            'calibration_rows': 1000, 'warmup_requests': 5,
            'views': ['original', 'reverse', 'shuffle', 'renamed-ids', 'whitespace', 'instruction-paraphrase'],
            'confidence': 'Selected-option probability is the common primary readout. Vendor/native confidence is retained separately.',
            'posthoc': 'Secondary positive temperature on log option probabilities, fit on calibration only for each model; thresholds fit only on calibration and frozen across perturbations.',
            'scope': 'One fixed 64-request BANKING77 diagnostic, all 1000 calibration requests, six separately reported views, and seven unselected authored demos. No tuning or model selection.',
            'limitations': ['Instruction paraphrase is authored, without independent semantic-equivalence review.',
                'Public benchmark exposure in external model training is unknown.',
                'Hosted latency includes network/queue time; local scoring latency excludes HTTP.',
                'A service model name and observation date do not establish an immutable weight revision.',
                'Repeated views of the same request are not independent test observations.',
                'Calibration thresholds are empirical rules, without certified risk guarantees.'],
            'hosted_model': 'microsoft/microsoft-decision-1', 'hosted_context_tokens': 32768,
            'input_usd_per_million_assumption': .042, 'price_observed_date': '2026-10-09',
            'sources': {}, 'code_sha256': {}}
    for name in ('data/banking77/calibration.jsonl', 'data/banking77/test.jsonl',
                 'examples/demo-requests.jsonl', 'examples/demo-expectations.json'):
        plan['sources'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in ('src/myjev/decision_comparison.py', 'scripts/run_decision_comparison.py', 'scripts/summarize_decision_comparison.py'):
        plan['code_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    tests = [json.loads(x) for x in (ROOT/'data/banking77/test.jsonl').read_text().splitlines()]
    plan['test_ids'] = [r['id'] for r in random.Random(42).sample(tests, 64)]
    path = ROOT/'configs/decision-comparison-v1.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Frozen protocol differs; use a new study version')
    path.write_text(json.dumps(plan, indent=2)+'\n')
    print('Frozen: 1000 calibration + 384 test-view + 7 demo + 5 warmup calls per model.')


if __name__ == '__main__':
    main()
