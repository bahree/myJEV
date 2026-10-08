"""Summarize the complete frozen candidate-order check; never pool permutations as new examples."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path('results/review-order-v1')
    protocol=json.loads(Path('configs/review-order-v1.json').read_text())
    rows=[];sources={};groups=[]
    for size in protocol['sizes']:
      for method in protocol['methods']:
        items=[]
        for seed in protocol['permutation_seeds']:
            path=root/size/method/f'order-{seed}-metrics.json'
            r=json.loads(path.read_text())
            assert r['n']==3080 and r['order_seed']==seed
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            item={'size':size,'method':method,'order_seed':seed,**r}
            items.append(item);rows.append(item)
        base=items[0]['source_order_accuracy']
        assert all(r['source_order_accuracy']==base for r in items)
        baseline_path=Path('results/longer-v1')/size/'main'/'seed-11'/method/('posthoc/temperature-metrics.json' if method=='continued_sft' else 'evaluation/metrics.json')
        baseline=json.loads(baseline_path.read_text())
        sources[str(baseline_path)]=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        groups.append({'size':size,'method':method,'original_accuracy':base,
                      'original_brier':baseline['correctness_brier'],
                      'original_coverage80':baseline['operating_points']['coverage_0.8'],
                      'shuffled_coverage80':[r['operating_points']['coverage_0.8'] for r in items],
                      'shuffled_accuracy':[r['accuracy'] for r in items],
                      'prediction_disagreement':[r['prediction_disagreement'] for r in items],
                      'correctness_brier':[r['correctness_brier'] for r in items]})
    output={'protocol':protocol,'scope':'Six released seed-11 checkpoints, three per-request seeded permutations for each of the same 3080 examples. No calibration refit, retraining, or test-driven order selection. Repeated permutations are not independent test examples.','groups':groups,'per_permutation':rows,'source_sha256':sources}
    (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# Candidate-order sensitivity of the released models','',output['scope'],'','The protocol was fixed in-session and committed seven minutes after the first run started; it was not preregistered in Git. Evaluations loaded local copies of the released artifacts, with matching artifact revisions and manifest hashes. The --hub command below is the public reproduction route, not the original loading path. Change rates are per permutation, not the fraction that can ever change under arbitrary orders.', '', 'Training randomizes candidates, but the Qwen readout binds their current positions to token aliases. Order changes both placement and alias assignment. This diagnostic measures their combined sensitivity; it does not isolate position from token identity.','','| Size | Release | Original accuracy | Shuffled accuracy, seeds 101 / 202 / 303 | Changed selected ID per permutation, same order seeds | Brier range |','|---|---|---:|---|---|---|']
    for r in groups:
        acc=' / '.join(f'{v:.2%}' for v in r['shuffled_accuracy'])
        flip=' / '.join(f'{v:.2%}' for v in r['prediction_disagreement'])
        lines.append(f"| {r['size']} | {r['method']} | {r['original_accuracy']:.2%} | {acc} | {flip} | {min(r['correctness_brier']):.4f}-{max(r['correctness_brier']):.4f} |")
    lines+=['','## The original 80% calibration-coverage threshold stays fixed','','| Size | Release | Original test coverage / error | Order 101 coverage / error | Order 202 coverage / error | Order 303 coverage / error |','|---|---|---|---|---|---|']
    for r in groups:
        points=[r['original_coverage80']]+r['shuffled_coverage80']
        assert all(p['threshold']==points[0]['threshold'] for p in points)
        cells=' | '.join(f"{p['coverage']:.2%} / {p['error']:.2%}" for p in points)
        lines.append(f"| {r['size']} | {r['method']} | {cells} |")
    lines+=['','These are achieved test coverages and accepted-case errors at a threshold previously selected for 80% calibration coverage. They are not equal-coverage comparisons or certified error targets. Per-file reports retain accepted counts and uncertainty.']
    lines+=['','All 18 metrics files preserve reliability bins, group-bootstrap accuracy intervals and achieved coverage/error at the original calibration thresholds. Those thresholds and each release\'s confidence source stay fixed. Different orders can exchange correct and incorrect cases even when aggregate accuracy barely changes. Do not interpret a small mean change as individual prediction invariance.','','Reproduce the summary with `python scripts/summarize_review_order.py`. For new inference, prepare the pinned BANKING77 data, then run `python scripts/review_candidate_order.py --size 0.8b --hub --output results/my-order-check`; repeat for 4b and 9b with an appropriate GPU. The released revisions are in `model_cards/facts.json`; public text-free calibration inputs supply the original-order reference. Preserve the published evidence directory when collecting new runs. These six-checkpoint tests do not establish robustness over new training seeds or arbitrary candidate descriptions.','']
    (root/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
