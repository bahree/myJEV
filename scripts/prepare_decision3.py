"""Freeze a full-test, local-hardware comparison before evaluating either new model."""
import json
from pathlib import Path

from myjev.decision3 import comparison_items, file_sha
from myjev.decision_comparison import digest


def main():
    target = Path('configs/decision3-v1.json')
    if target.exists():
        raise ValueError('Protocol already exists; do not overwrite a frozen study')
    models = {
        'd3-lite': ['d3', 'vllm-sr/d3-lite', 'a8d25a91ef75552803f7009d25b7bd435991abd9', '0.8b'],
        'd3-mini': ['d3', 'vllm-sr/d3-mini', 'a475367de0d8133a3c1a2a0dc12303d89cfa2ab4', '4b'],
        'myjev-0.8b': ['myjev', 'bahree/myJEV-0.8B', '1c956c89d21c0ab136e98ffe66a16752fa37d823', '0.8b'],
        'myjev-0.8b-rl': ['myjev', 'bahree/myJEV-0.8B-RL', '44b2ab8e78cb176ed01a63c6fee2146b2d91cc95', '0.8b'],
        'myjev-4b': ['myjev', 'bahree/myJEV-4B', '38f7cca5a8530483309f576b0c3dd1756bc27c33', '4b'],
        'myjev-4b-rl': ['myjev', 'bahree/myJEV-4B-RL', '0a9413105fc84cb500850059b641755890c4201b', '4b'],
    }
    prior = json.loads(Path('configs/decision-comparison-v1.json').read_text())
    cohorts = ['near', 'distant', 'unsupported-near-none', 'unsupported-near-deferral', 'oos-none', 'oos-deferral']
    source_paths = ['data/banking77/calibration.jsonl', 'data/banking77/test.jsonl',
                    'examples/demo-requests.jsonl', 'examples/demo-expectations.json']
    source_paths += [f'data/clinc150/{name}.jsonl' for name in cohorts]
    plan = {
        'study': 'decision3-v1', 'models': {k: dict(zip(['backend', 'repo', 'revision', 'size'], v)) for k, v in models.items()},
        'sources': {p: file_sha(p) for p in source_paths},
        'sample_seed': 42, 'calibration_rows': 1000, 'test_rows': 3080,
        'perturbation_ids': prior['test_ids'], 'transfer_cohorts': cohorts,
        'transfer_rows_per_cohort': 256, 'warmup_requests': 5,
        'max_input_tokens': 4096, 'd3_batch_size': 1, 'd3_permutation_average': False,
        'temperature_probability_floor': 1e-12,
        'calibration': 'Report released selected probabilities and a positive temperature fit on log probabilities from calibration only for every model. Use 15 equal-width ECE bins. Preserve the released selected ID. Record clipped probabilities. No test-based checkpoint, prompt or temperature selection.',
        'confidence': 'Primary confidence is selected-option probability. Native myJEV correctness confidence is a separate view. D3 entropy confidence is retained and used only for ranking/AUROC, not labelled as a correctness probability or assigned a correctness Brier.',
        'thresholds': 'Select coverage 0.5, 0.8, 0.9 and empirical calibration error targets 0.01, 0.05. Freeze for test, perturbations and all CLINC cohorts. No certified risk claim.',
        'uncertainty': 'Paired group bootstrap on original test rows, 1000 repeats, seed 42. Report intervals conditional on these released checkpoints, with no training-seed inference. Do not pool perturbations.',
        'benchmark': {'banking_ids': prior['test_ids'], 'banking_repeats': 3, 'demo_repeats': 5,
                      'scope': 'After quality evaluation, run one model at a time on GPU 0 with no other GPU jobs. Five warmups per workload; 192 banking calls and 35 short demo calls. Measure synchronized local scoring, backbone CUDA-event time, cached load, and peak allocated/reserved VRAM. Excludes HTTP and model download.'},
        'environment': 'D3 uses pinned Transformers 5.17.0; myJEV retains 5.18.0. Both use the existing PyTorch 2.14.1, BF16 backbones and PyTorch reference DeltaNet kernels. No FLA or AMD fast path. D3 retains the vision tower but only text is evaluated.',
        'limitations': ['External training data/exposure are not matched; this compares released systems, not training methods.',
                        'Possible public benchmark exposure in D3 training is unknown.',
                        'One published checkpoint per release; repeated views and latency repetitions are not training seeds.',
                        'AMD publisher benchmark numbers are not reproduced; measurements here use NVIDIA A30.',
                        'CLINC is evaluation only. Missing-answer deferral and explicit none choices are reported separately.',
                        'No images/videos, CPU, HTTP throughput or vLLM engine integration are evaluated.'],
    }
    items = comparison_items(plan)
    plan['request_count_per_model'] = len(items)
    plan['requests_sha256'] = digest(items)
    target.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'protocol': str(target), 'requests_per_model': len(items), 'sha256': file_sha(target)}, indent=2))


if __name__ == '__main__':
    main()
