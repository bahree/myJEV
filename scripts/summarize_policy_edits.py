"""Regenerate the bounded policy-edit diagnostic table from saved predictions."""
import json
from pathlib import Path
from evaluate_policy_edits import summarize

root=Path('results/policy-edits-v1')
rows=[]
for size in ['0.8b','4b','9b']:
    for method in ['continued_sft','exact']:
        path=root/f'myjev-{size}-{method}-seed11'
        saved=json.loads((path/'report.json').read_text())
        predictions=[json.loads(line) for line in (path/'predictions.jsonl').read_text().splitlines()]
        actual=summarize(predictions)
        if actual != saved['overall']: raise ValueError('saved summary disagrees with predictions')
        rows.append({'size':size,'method':method,'confidence_mode':predictions[0]['confidence_mode'],**actual})
summary={'scope':json.loads((root/'frozen-plan.json').read_text())['limits'],'rows':rows}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Frozen policy-edit diagnostic','',
       'This prospective diagnostic contains 24 original paired fixtures, 48 requests from six templates. Six existing seed-11 release candidates were evaluated without tuning or recalibration. Sixteen pairs require a changed answer; eight require invariance. Context and candidate order remain fixed within each pair. Candidate order alternates between pairs. Expectations and file hashes were frozen before inference.','',
       '| Size | Method | Accuracy | Both variants correct | Required changes followed | Required invariances followed | Confidence Brier |',
       '|---|---|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['size']} | {row['method']} | {row['accuracy']:.1%} | {row['both_variants_correct']:.1%} | {row['changed_when_required']:.1%} | {row['unchanged_when_required']:.1%} | {row['confidence_brier']:.4f} |")
lines += ['', 'Accuracy counts 48 answers; both-correct counts 24 pairs. Change and invariance rates alone do not establish correctness: a model can change to the wrong answer or remain consistently wrong. Confidence Brier uses the deployed correctness confidence, not the full selection distribution. Continued-SFT release candidates report temperature-scaled selected probability; exact-RL candidates report expected grid confidence. No calibration is fitted on these fixtures.', '',
          'These are correlated synthetic templates, not 48 independent natural-language tasks. One checkpoint per method/size cannot establish a general model ranking, a scaling law, or the effect of RL. Only the six myJEV release candidates were evaluated. Performance here does not establish production policy compliance. All predictions, reported confidence, selection scores, frozen hashes, and raw runtime logs are retained.', '',
          '## Reproduce', '', '```bash', 'python scripts/evaluate_policy_edits.py --artifact artifacts/release-candidates-v1/myjev-4b-continued_sft-seed11 --output /tmp/policy-edit-replay', '```', '', 'For a replay, first copy the committed `frozen-plan.json` into the output directory. Its fixture and evaluator hashes must match. Run each candidate once in a fresh output subdirectory. Set `HF_HOME` to the shared backbone cache if applicable. To regenerate this table from the committed predictions: `python scripts/summarize_policy_edits.py`.', '']
(root/'report.md').write_text('\n'.join(lines))
