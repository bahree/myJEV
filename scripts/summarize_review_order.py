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
        groups.append({'size':size,'method':method,'original_accuracy':base,
                      'shuffled_accuracy':[r['accuracy'] for r in items],
                      'prediction_disagreement':[r['prediction_disagreement'] for r in items],
                      'correctness_brier':[r['correctness_brier'] for r in items]})
    output={'protocol':protocol,'scope':'Six released seed-11 checkpoints, three fixed permutations of the same 3080 examples. No calibration refit, retraining, or test-driven order selection. Repeated permutations are not independent test examples.','groups':groups,'per_permutation':rows,'source_sha256':sources}
    (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# Candidate-order sensitivity of the released models','',output['scope'],'','Training randomizes candidates, but the Qwen readout binds their current positions to token aliases. Order changes both placement and alias assignment. This diagnostic measures their combined sensitivity; it does not isolate position from token identity.','','| Size | Release | Original accuracy | Shuffled accuracy, seeds 101 / 202 / 303 | Changed selected ID, same order seeds | Brier range |','|---|---|---:|---|---|---|']
    for r in groups:
        acc=' / '.join(f'{v:.2%}' for v in r['shuffled_accuracy'])
        flip=' / '.join(f'{v:.2%}' for v in r['prediction_disagreement'])
        lines.append(f"| {r['size']} | {r['method']} | {r['original_accuracy']:.2%} | {acc} | {flip} | {min(r['correctness_brier']):.4f}-{max(r['correctness_brier']):.4f} |")
    lines+=['','All 18 metrics files preserve reliability bins, group-bootstrap accuracy intervals and achieved coverage/error at the original calibration thresholds. Those thresholds and each release\'s confidence source stay fixed. Different orders can exchange correct and incorrect cases even when aggregate accuracy barely changes. Do not interpret a small mean change as individual prediction invariance.','','Reproduce the summary with `python scripts/summarize_review_order.py`. For new inference, prepare the pinned BANKING77 data, then run `python scripts/review_candidate_order.py --size 0.8b --hub --output results/my-order-check`; repeat for 4b and 9b with an appropriate GPU. The released revisions are in `model_cards/facts.json`; public text-free calibration inputs supply the original-order reference. Preserve the published evidence directory when collecting new runs. These six-checkpoint tests do not establish robustness over new training seeds or arbitrary candidate descriptions.','']
    (root/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
