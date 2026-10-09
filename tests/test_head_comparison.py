import copy
import pytest
from myjev.head_comparison import clef_request, exposure, exposure_digest, permute_request, selected_trial


def rows():
    return [{'id': str(i), 'group': str(i), 'context': str(i), 'instructions': 'Route',
             'candidates': [{'id': 'secret_intent_z', 'description': 'Charges'},
                            {'id': 'secret_intent_a', 'description': 'Errors'}],
             'label': 'secret_intent_z'} for i in range(5)]


def test_neutral_criteria_preserve_order_and_hide_label_identifiers():
    request = {k: rows()[0][k] for k in ('context', 'instructions', 'candidates')}
    record, mapping = clef_request(request)
    criteria = record['questions']['decision']['criteria']
    assert list(criteria) == sorted(criteria)
    assert [mapping[k] for k in sorted(criteria)] == [c['id'] for c in request['candidates']]
    assert 'secret_intent' not in str(record)
    many = {**request, 'candidates': [{'id': str(i), 'description': str(i)} for i in range(160)]}
    record, mapping = clef_request(many)
    assert [mapping[k] for k in sorted(record['questions']['decision']['criteria'])] == list(map(str, range(160)))


def test_shared_exposure_replays_candidate_permutations_across_epochs():
    original = rows(); saved = copy.deepcopy(original)
    a, b = exposure(original, 11, 13), exposure(original, 11, 13)
    assert exposure_digest(a) == exposure_digest(b)
    assert exposure_digest(a) != exposure_digest(exposure(original, 22, 13))
    assert len(a) == 13 and original == saved
    assert all(request['candidates'][target]['id'] == row['label'] for row, request, target in a)
    assert {r['id'] for r, _, _ in a[:5]} == set(map(str, range(5)))


def test_order_shuffle_is_recoverable_and_does_not_mutate_request():
    request = {k: rows()[0][k] for k in ('context', 'instructions', 'candidates')}
    before = copy.deepcopy(request)
    assert permute_request(request, 'x', 101) == permute_request(request, 'x', 101)
    assert request == before
    assert sorted(c['id'] for c in permute_request(request, 'x', 101)['candidates']) == sorted(c['id'] for c in request['candidates'])


def test_selection_uses_only_declared_validation_fields():
    a = {'accuracy': .7, 'selection_nll': .8, 'learning_rate': 1e-4, 'test_accuracy': .9}
    b = {**a, 'learning_rate': 3e-5, 'test_accuracy': .1}
    assert selected_trial([a,b]) == b
    assert selected_trial([a, {**b, 'accuracy': .8, 'selection_nll': 1.2}])['accuracy'] == .8
    with pytest.raises(ValueError): selected_trial([])
