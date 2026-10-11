"""Regenerate the frozen Decision 3.0 comparison from saved predictions."""
import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from myjev.calibration import fit_temperature
from myjev.decision3 import file_sha, rows
from myjev.metrics import risk_at_threshold, thresholds_from_calibration


def interval(values, groups, repeats=1000):
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(groups, return_inverse=True)
    n = int(inverse.max()) + 1
    sizes = np.bincount(inverse, minlength=n)
    totals = np.bincount(inverse, weights=values, minlength=n)
    sampled = np.random.default_rng(42).integers(n, size=(repeats, n))
    means = totals[sampled].sum(1) / sizes[sampled].sum(1)
    return np.quantile(means, [.025, .975]).tolist()


def metrics(predictions, thresholds):
    correct = np.array([r['selected_id'] == r['label'] for r in predictions], dtype=float)
    confidence = np.array([r['confidence'] for r in predictions])
    reliability = []
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else confidence <= hi)
        reliability.append({'lo': float(lo), 'hi': float(hi), 'n': int(mask.sum()),
                            'confidence': float(confidence[mask].mean()) if mask.any() else None,
                            'accuracy': float(correct[mask].mean()) if mask.any() else None})
    points = {}
    for name, threshold in thresholds.items():
        point = risk_at_threshold(correct, confidence, threshold)
        selected = [r for r in predictions if r['confidence'] >= threshold]
        point['error_cluster_ci95'] = interval([float(r['selected_id'] != r['label']) for r in selected],
                                               [r['group'] for r in selected]) if selected else None
        points[name] = point
    supported = all(r['label'] in r['selection_scores'] for r in predictions)
    return {'n': len(predictions), 'accuracy': float(correct.mean()),
            'accuracy_cluster_ci95': interval(correct, [r['group'] for r in predictions]),
            'macro_f1': float(f1_score([r['label'] for r in predictions], [r['selected_id'] for r in predictions],
                                      average='macro', zero_division=0)),
            'correctness_brier': float(np.mean((correct - confidence) ** 2)),
            'selection_multiclass_brier': float(np.mean([sum((v - float(k == r['label'])) ** 2
                                                            for k, v in r['selection_scores'].items())
                                                        for r in predictions])) if supported else None,
            'correctness_auroc': float(roc_auc_score(correct, confidence)) if len(set(correct)) > 1 else None,
            'ece': sum(b['n'] / len(predictions) * abs(b['confidence'] - b['accuracy']) for b in reliability if b['n']),
            'ece_bins': '15 equal-width bins; last includes 1', 'reliability': reliability,
            'operating_points': points,
            'scope': 'All labels absent from candidates; coverage is false acceptance.' if not supported else 'Reference-label agreement.'}


def transformed(predictions, mode, temperature, floor):
    output = []
    for row in predictions:
        r = dict(row)
        if mode == 'temperature':
            keys, values = zip(*r['selection_scores'].items())
            x = np.log(np.maximum(values, floor)) / temperature
            p = np.exp(x - x.max()); p /= p.sum()
            r['selection_scores'] = dict(zip(keys, p.tolist()))
            r['confidence'] = r['selection_scores'][r['selected_id']]
        elif mode == 'native-correctness':
            r['confidence'] = r['native_confidence']
        output.append(r)
    return output


def percent(value):
    return 'n/a (no accepted requests)' if value is None else f'{100 * value:.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/decision3-v1'))
    parser.add_argument('--require-benchmark', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan_path = Path('configs/decision3-v1.json')
    plan = json.loads(plan_path.read_text())
    floor = plan['temperature_probability_floor']
    summary = {'protocol_sha256': file_sha(plan_path), 'models': {}, 'paired_accuracy': {},
               'scope': 'Frozen released systems, one checkpoint each; intervals resample test groups and exclude training variation.',
               'limitations': plan['limitations']}
    source_predictions = {}
    for name, spec in plan['models'].items():
        folder = args.root / name / 'quality'
        done = json.loads((folder / 'complete.json').read_text())
        path = folder / 'records.jsonl'
        if not path.exists():
            path = path.with_suffix('.jsonl.gz')
            data = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
            import hashlib
            observed = hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
        else:
            data, observed = rows(path), file_sha(path)
        if observed != done['records_sha256'] or len(data) != plan['request_count_per_model']:
            raise ValueError('Incomplete or changed predictions: ' + name)
        run = json.loads((folder / 'run.json').read_text())
        if run['identity']['protocol_sha256'] != file_sha(plan_path):
            raise ValueError('Protocol differs: ' + name)
        cal = [r for r in data if r['phase'] == 'calibration']
        test = [r for r in data if r['phase'] == 'test']
        if {r['group'] for r in cal} & {r['group'] for r in test}:
            raise ValueError('Calibration/test group overlap')
        source_predictions[name] = test
        logits = [np.log(np.maximum(list(r['selection_scores'].values()), floor)).tolist() for r in cal]
        targets = [list(r['selection_scores']).index(r['label']) for r in cal]
        temperature = fit_temperature(logits, targets)
        modes = ['native-selection', 'temperature']
        if spec['backend'] == 'myjev': modes.append('native-correctness')
        result = {'spec': spec, 'temperature': temperature, 'temperature_fit_rows': len(cal),
                  'calibration_probabilities_clipped': sum(v < floor for r in cal for v in r['selection_scores'].values()),
                  'modes': {}, 'source_records_sha256': observed, 'benchmark': None}
        for mode in modes:
            converted = transformed(data, mode, temperature, floor)
            cal2 = [r for r in converted if r['phase'] == 'calibration']
            thresholds = thresholds_from_calibration([r['selected_id'] == r['label'] for r in cal2],
                                                       [r['confidence'] for r in cal2])
            phases = {}
            for phase in ['test'] + ['clinc/' + cohort for cohort in plan['transfer_cohorts']]:
                phases[phase] = metrics([r for r in converted if r['phase'] == phase], thresholds)
            original = {r['id']: r for r in converted if r['phase'] == 'test'}
            perturbations = {}
            for view in ['reverse', 'shuffle', 'renamed-ids', 'whitespace', 'instruction-paraphrase']:
                selected = [r for r in converted if r['phase'] == 'perturbation' and r['view'] == view]
                changed = sum(r['selected_id'] != original[r['id']]['selected_id'] for r in selected)
                perturbations[view] = {'changed_answers': changed, **metrics(selected, thresholds)}
            perturbations['original'] = {'changed_answers': 0, **metrics([original[key] for key in plan['perturbation_ids']], thresholds)}
            result['modes'][mode] = {'thresholds': thresholds, 'phases': phases, 'perturbations': perturbations}
        correct = [r['selected_id'] == r['label'] for r in test]
        result['native_confidence'] = {'definition': test[0]['native_confidence_mode'],
                                      'correctness_auroc': float(roc_auc_score(correct, [r['native_confidence'] for r in test]))
                                      if len(set(correct)) > 1 else None,
                                      'scope': 'Entropy is evaluated as a ranking signal, not a correctness probability.' if spec['backend'] == 'd3' else 'Released confidence definition.'}
        result['demos'] = [{k: r[k] for k in ('id', 'label', 'selected_id', 'confidence', 'native_confidence', 'native_confidence_mode')}
                           for r in data if r['phase'] == 'demo']
        benchdir = args.root / name / 'benchmark'
        if (benchdir / 'complete.json').exists():
            bdone = json.loads((benchdir / 'complete.json').read_text())
            bpath = benchdir / 'records.jsonl'
            if file_sha(bpath) != bdone['records_sha256']: raise ValueError('Benchmark file changed')
            bench = rows(bpath)
            brun = json.loads((benchdir / 'run.json').read_text())
            result['benchmark'] = {'cached_load_seconds': brun['cached_load_seconds'],
                                   'peak_allocated_gib': bdone['peak_allocated_bytes'] / 1024**3,
                                   'peak_reserved_gib': bdone['peak_reserved_bytes'] / 1024**3,
                                   'workloads': {}}
            for workload in ['banking', 'short-demos']:
                selected = [r for r in bench if r['phase'] == workload]
                result['benchmark']['workloads'][workload] = {
                    'n': len(selected), 'local_scoring_p50_p95_ms': (np.quantile([r['seconds'] for r in selected], [.5,.95]) * 1000).tolist(),
                    'backbone_gpu_p50_p95_ms': np.quantile([r['backbone_forward_ms'] for r in selected], [.5,.95]).tolist(),
                    'sequential_requests_per_second': len(selected) / sum(r['seconds'] for r in selected)}
        elif args.require_benchmark:
            raise ValueError('Missing benchmark: ' + name)
        summary['models'][name] = result
    for d3, comparisons in [('d3-lite', ['myjev-0.8b', 'myjev-0.8b-rl']), ('d3-mini', ['myjev-4b', 'myjev-4b-rl'])]:
        left = source_predictions[d3]
        for other in comparisons:
            right = source_predictions[other]
            if [r['id'] for r in left] != [r['id'] for r in right]: raise ValueError('Pair IDs differ')
            delta = [float(a['selected_id'] == a['label']) - float(b['selected_id'] == b['label']) for a,b in zip(left,right)]
            summary['paired_accuracy'][d3 + '-minus-' + other] = {'mean_delta': float(np.mean(delta)),
                                                                  'conditional_group_ci95': interval(delta,[r['group'] for r in left])}
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    lines = ['# Decision 3.0 and myJEV on the same A30 hardware', '',
             'All six pinned releases completed the full 3,080-row BANKING77 test, 1,000 calibration rows, six CLINC150 cohorts, five perturbations of the same 64 cases and seven authored demos. These are released-system comparisons with different training exposure. No checkpoint, prompt or temperature was selected from test results.', '',
             '## Original BANKING77 requests', '',
             '| Model | Confidence calculation | Accuracy | Correctness Brier | ECE | Test coverage at 80% calibration coverage | Accepted-case error |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, result in summary['models'].items():
        for mode in ['native-selection','temperature']:
            m = result['modes'][mode]['phases']['test']; point = m['operating_points']['coverage_0.8']
            lines.append(f"| {name} | {mode} | {percent(m['accuracy'])} | {m['correctness_brier']:.4f} | {m['ece']:.4f} | {percent(point['coverage'])} | {percent(point['error'])} |")
    lines += ['', 'Temperature fits use calibration labels only, with a 1e-12 floor before taking log probabilities. Native myJEV correctness-head results, macro-F1, multiclass Brier, AUROC, 15-bin reliability, every threshold and accepted count are in [summary.json](summary.json). D3 native confidence measures normalized entropy and is reported separately as a ranking signal. Calibration thresholds are empirical rules; their selected error targets do not certify future error.', '',
              '## Paired accuracy differences', '', '| Pair | Difference in percentage points | Conditional 95% group interval |', '|---|---:|---:|']
    for name, m in summary['paired_accuracy'].items():
        lo,hi = m['conditional_group_ci95']; lines.append(f"| {name} | {100*m['mean_delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', 'These intervals resample test groups for the published checkpoints. They exclude variation from retraining and cannot isolate an architecture or training-method effect.', '',
              '## Transfer with the same BANKING thresholds', '', '| Model | CLINC cohort | Rows | Accuracy | Coverage at BANKING 80% threshold | Accepted-case error |', '|---|---|---:|---:|---:|---:|']
    for name,result in summary['models'].items():
        for cohort in plan['transfer_cohorts']:
            m=result['modes']['temperature']['phases']['clinc/'+cohort]; pt=m['operating_points']['coverage_0.8']
            lines.append(f"| {name} | {cohort} | {m['n']} | {percent(m['accuracy'])} | {percent(pt['coverage'])} | {percent(pt['error'])} |")
    lines += ['', 'This table uses the equally fitted selection temperatures. Deferral cohorts omit the correct option, so acceptance is a false acceptance. Explicit none-of-these cohorts supply a selectable answer. No CLINC labels were used for fitting.', '',
              '## Candidate and wording changes', '', '| Model | View | Changed answers / 64 | Accuracy |', '|---|---|---:|---:|']
    for name,result in summary['models'].items():
        for view,m in result['modes']['temperature']['perturbations'].items(): lines.append(f"| {name} | {view} | {m['changed_answers']} | {percent(m['accuracy'])} |")
    lines += ['', 'The views share requests and are never pooled as independent samples. Each shuffle is determined by the request ID. Temperatures and thresholds remain those fitted for original requests.', '',
              '## Local scoring latency and memory', '', '| Model | BANKING p50 / p95 ms | Short-demo p50 / p95 ms | Peak allocated GiB | Cached load s |', '|---|---:|---:|---:|---:|']
    for name,result in summary['models'].items():
        b=result['benchmark']
        if b:
            bank=b['workloads']['banking']['local_scoring_p50_p95_ms'];short=b['workloads']['short-demos']['local_scoring_p50_p95_ms']
            lines.append(f"| {name} | {bank[0]:.2f} / {bank[1]:.2f} | {short[0]:.2f} / {short[1]:.2f} | {b['peak_allocated_gib']:.3f} | {b['cached_load_seconds']:.2f} |")
    lines += ['', 'The separate benchmark runs one model at a time on GPU 0. Timing includes tokenization, scoring and response conversion, with CUDA synchronization, and excludes HTTP and downloading. Each workload has five warmups. D3 retains its vision tower and verifies the full package during loading. myJEV uses its release loader. Backbone CUDA-event timings and reserved memory are also saved. Quality-run timings may overlap other GPU work and are not used for this table.', '',
              '## Seven authored demonstrations', '', '| Model | Case | Expected | Selected | Selected probability | Native confidence |', '|---|---|---|---|---:|---:|']
    for name,result in summary['models'].items():
        for r in result['demos']: lines.append(f"| {name} | {r['id']} | {r['label']} | {r['selected_id']} | {r['confidence']:.4f} | {r['native_confidence']:.4f} |")
    lines += ['', 'The two confidence columns have different meanings for D3 and RL releases. These examples were written for teaching and do not estimate population accuracy.', '', '## Reproduce', '',
              'Run `python scripts/summarize_decision3.py --require-benchmark` after restoring the prediction files. [The guide](../../docs/decision3.md) gives the model pins, setup, commands, architecture comparison and evidence locations.', '']
    (args.root / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'models': len(summary['models']), 'report': str(args.root / 'report.md')}))


if __name__ == '__main__': main()
