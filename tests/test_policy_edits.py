import importlib.util
import json
from pathlib import Path

from myjev.schema import ScoreRequest

spec = importlib.util.spec_from_file_location('policy_edits', Path(__file__).parents[1]/'scripts/evaluate_policy_edits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_pairs_have_valid_requests_and_declared_changes():
    root=Path(__file__).parents[1]
    fixture=root/'fixtures/policy-edits-v1/pairs.json'
    plan=json.loads((root/'results/policy-edits-v1/frozen-plan.json').read_text())
    assert module.digest(fixture)==plan['fixture_sha256']
    pairs=json.loads(fixture.read_text())
    assert len(pairs)==24
    assert sum(pair['expected_change'] for pair in pairs)==16
    for pair in pairs:
        a,b=pair['variants']
        assert a['request']['context']==b['request']['context']
        assert a['request']['candidates']==b['request']['candidates']
        assert a['request']['instructions']!=b['request']['instructions']
        for row in pair['variants']:
            request=ScoreRequest.model_validate(row['request'])
            assert row['label'] in {c.id for c in request.candidates}


def test_change_rate_does_not_mistake_wrong_changes_for_success():
    rows=[dict(pair_id='x',expected_change=True,label='0',selected_id='2',confidence=.9),
          dict(pair_id='x',expected_change=True,label='1',selected_id='0',confidence=.8)]
    result=module.summarize(rows)
    assert result['changed_when_required']==1
    assert result['both_variants_correct']==0
    assert result['accuracy']==0
    assert abs(result['confidence_brier']-.725)<1e-9
    assert result['mean_confidence_correct'] is None
