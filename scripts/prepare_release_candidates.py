"""Prepare local, checksummed release candidates. Never uploads or selects a default."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/release-candidates-v1'));p.add_argument('--report',type=Path,default=Path('results/release-candidates-v1/inventory.json'));a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite release candidates')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    records=[]
    for size in ('0.8b','4b','9b'):
        for method in ('continued_sft','exact'):
            base=Path(f'results/longer-v1/{size}/main/seed-11/{method}')
            if method=='continued_sft':
                source=base/'posthoc/temperature-artifact';metric_path=base/'posthoc/temperature-metrics.json'
            else:
                source=Path(f'artifacts/longer-v1/{size}/main/seed-11/{method}/artifact');metric_path=base/'evaluation/metrics.json'
            manifest=json.loads((source/'manifest.json').read_text());metrics=json.loads(metric_path.read_text())
            files=['manifest.json',*manifest['checksums']]
            for name in files:
                raw=source/name
                src=raw.resolve()
                if not src.is_relative_to(source.resolve()) or raw.is_symlink():raise ValueError('Unsafe artifact path')
                if name!='manifest.json' and sha(src)!=manifest['checksums'][name]:raise ValueError(f'Checksum mismatch: {src}')
            label=f'myjev-{size}-{method}-seed11';dest=a.output/label
            for name in files:
                target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
            mode='temperature-scaled selected-answer probability' if method=='continued_sft' else 'candidate-conditioned 21-bin policy expected confidence'
            body=f'''---
base_model: {manifest['backbone']}
tags: [myjev, lora, decision-model, research]
---

# {label}: local release candidate

**Draft packaging only. Not a production recommendation or a published Hub release.**

This candidate preserves seed 11 by convention rather than selecting the best test seed. Compare the full three-seed study before interpreting this individual checkpoint. The same artifact must still pass final representative serving and clean-environment release checks.

- Backbone revision: `{manifest['backbone_revision']}`.
- Artifact revision: `{manifest['artifact_revision']}`.
- Precision: `{manifest['precision']}`; nonquantized dtype: `{manifest.get('nonquantized_dtype')}`.
- Method: `{method}`, 4,000 initial supervised examples plus 4,000 continuation examples, BANKING77 train partition only.
- Confidence: {mode}. Selection scores and selected-answer correctness confidence have different semantics.
- Calibration revision: `{manifest.get('calibration_revision','uncalibrated')}`. Temperature uses reserved BANKING77 calibration data only.
- This seed's official test accuracy: {100*metrics['accuracy']:.2f}% on {metrics['n']} examples; correctness Brier: {metrics['correctness_brier']:.4f}.

The public evidence includes seeds 11, 22 and 33, [paired findings](https://github.com/bahree/myJEV/blob/main/docs/qwen-findings.md) and fixed-threshold operational uncertainty. Transfer evaluation and final release selection remain separate gates. Neither a low Brier score nor a calibration empirical-error target establishes low deployment error.

## Load locally

Install the pinned myJEV environment from source commit `{revision}`. Use the custom shared loader; a generic text-generation widget does not implement this contract.

```python
from myjev import DecisionModel
model = DecisionModel.load("{dest}")
result = model.score({{
    "context": "I was charged twice.",
    "instructions": "Select the support route.",
    "candidates": [
        {{"id": "billing", "description": "Charges and refunds"}},
        {{"id": "other", "description": "Neither listed route applies"}}
    ]
}})
```

This example demonstrates the request shape, not accuracy on a novel support taxonomy. The separately pinned Qwen backbone must be downloaded or cached. The package contains adapters, heads and manifest; no backbone, optimizer state or training text is included. Declared limits are {manifest['max_candidates']} candidates and {manifest['max_tokens']} prompt tokens; synthetic pilot boundary tests do not establish quality or latency at those limits for this final checkpoint.

## Scope and licensing

BANKING77 is one intent family. Archive adaptation has not been performed; transfer and robustness evidence must be consulted separately. The 9B configuration uses NF4, while smaller original configurations use BF16, limiting capacity-only conclusions. Request-supplied candidate descriptions are model input and can change behavior.

The repository source code is MIT-licensed, copyright Amit Bahree. Artifact release terms remain subject to review. Backbone and dataset licenses remain separate and must be preserved. Do not upload this draft as a finalized licensed release. No managed endpoint is running.
'''
            (dest/'README.md').write_text(body)
            record=dict(name=label,source=str(source),destination=str(dest),artifact_revision=manifest['artifact_revision'],source_code_revision=revision,
                        metric_source=str(metric_path),metric_sha256=sha(metric_path),files={str(f.relative_to(dest)):sha(f) for f in sorted(dest.rglob('*')) if f.is_file()},
                        state='local candidate; final verification and licensing pending')
            records.append(record)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps({'default_selected':False,'uploaded':False,'candidates':records},indent=2)+'\n')
    print(json.dumps({'candidates':len(records),'uploaded':False,'report':str(a.report)}))

if __name__=='__main__':main()
