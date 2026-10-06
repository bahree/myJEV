# Frozen policy-edit diagnostic

This prospective diagnostic contains 24 original paired fixtures, 48 requests from six templates. Six existing seed-11 release candidates were evaluated without tuning or recalibration. Sixteen pairs require a changed answer; eight require invariance. Context and candidate order remain fixed within each pair. Candidate order alternates between pairs. Expectations and file hashes were frozen before inference.

| Size | Method | Accuracy | Both variants correct | Required changes followed | Required invariances followed | Confidence Brier |
|---|---|---:|---:|---:|---:|---:|
| 0.8b | continued_sft | 52.1% | 33.3% | 31.2% | 100.0% | 0.3731 |
| 0.8b | exact | 56.2% | 37.5% | 56.2% | 62.5% | 0.2089 |
| 4b | continued_sft | 100.0% | 100.0% | 100.0% | 100.0% | 0.0062 |
| 4b | exact | 81.2% | 79.2% | 93.8% | 100.0% | 0.1488 |
| 9b | continued_sft | 93.8% | 87.5% | 81.2% | 100.0% | 0.0499 |
| 9b | exact | 100.0% | 100.0% | 100.0% | 100.0% | 0.1866 |

Accuracy counts 48 answers; both-correct counts 24 pairs. Change and invariance rates alone do not establish correctness: a model can change to the wrong answer or remain consistently wrong. Confidence Brier uses the deployed correctness confidence, not the full selection distribution. Continued-SFT release candidates report temperature-scaled selected probability; exact-RL candidates report expected grid confidence. No calibration is fitted on these fixtures.

These are correlated synthetic templates, not 48 independent natural-language tasks. One checkpoint per method/size cannot establish a general model ranking, a scaling law, or the effect of RL. No PolicyLM or Aplomb model was run. Performance here does not establish production policy compliance. All predictions, reported confidence, selection scores, frozen hashes, and raw runtime logs are retained.

## Reproduce

```bash
python scripts/evaluate_policy_edits.py --artifact artifacts/release-candidates-v1/myjev-4b-continued_sft-seed11 --output /tmp/policy-edit-replay
```

For a replay, first copy the committed `frozen-plan.json` into the output directory. Its fixture and evaluator hashes must match. Run each candidate once in a fresh output subdirectory. Set `HF_HOME` to the shared backbone cache if applicable. To regenerate this table from the committed predictions: `python scripts/summarize_policy_edits.py`.
