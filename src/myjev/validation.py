"""Validation-only model selection; never fit decisions on test or calibration labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import read_jsonl, write_jsonl


def checked_validation(path):
    path = Path(path)
    if path.name != 'validation.jsonl':
        raise ValueError('model selection requires validation.jsonl')
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest['sha256']['validation'] != digest:
        raise ValueError('validation partition hash mismatch')
    rows = read_jsonl(path)
    train = read_jsonl(path.parent / 'train.jsonl')
    if not rows or {r['group'] for r in rows} & {r['group'] for r in train}:
        raise ValueError('empty validation or training-group leakage')
    return rows, digest


def select_trial(trials):
    if not trials or len({t['validation_sha256'] for t in trials}) != 1:
        raise ValueError('trials must use the same nonempty validation partition')
    # Fixed rule: accuracy, then selection NLL, then lower learning rate.
    for trial in trials:
        if not all(np.isfinite(trial[k]) for k in ('accuracy', 'selection_nll', 'learning_rate')):
            raise ValueError('nonfinite selection metric')
    return min(trials, key=lambda t: (-t['accuracy'], t['selection_nll'], t['learning_rate']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows, digest = checked_validation(a.data)
    from .inference import DecisionModel
    from .evaluate import predict
    model = DecisionModel.load(a.artifact)
    predictions = predict(model, rows)
    nll = []
    for row, pred in zip(rows, predictions):
        logits = np.asarray(pred['selection_logits'], dtype=float)
        target = next(i for i, c in enumerate(row['candidates']) if c['id'] == row['label'])
        peak = logits.max()
        nll.append(float(peak + np.log(np.exp(logits - peak).sum()) - logits[target]))
    report = {'partition': 'validation', 'examples': len(rows), 'validation_sha256': digest,
              'artifact_revision': model.manifest['artifact_revision'],
              'accuracy': float(np.mean([r['selected_id'] == r['label'] for r in predictions])),
              'selection_nll': float(np.mean(nll)),
              'selection_rule': 'highest accuracy; lowest selection NLL; lowest learning rate'}
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'predictions.jsonl', predictions)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
