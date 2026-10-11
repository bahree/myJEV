import math
import pytest
from myjev.decision3 import normalize_d3, normalized_entropy_confidence


def test_entropy_is_not_selected_probability():
    assert normalized_entropy_confidence([.9, .1]) == pytest.approx(.5310044064)
    assert normalized_entropy_confidence([.5, .5]) == pytest.approx(0)
    assert normalized_entropy_confidence([1., 0.]) == pytest.approx(1)


def test_d3_preserves_both_confidence_definitions():
    request = {'candidates': [{'id': 'a'}, {'id': 'b'}]}
    raw = {'answers': {'route': {'type': 'choice', 'choice': 'a', 'probabilities': {'a': .9, 'b': .1},
                                'confidence': normalized_entropy_confidence([.9, .1])}},
           'usage': {'output_tokens': 0}}
    result = normalize_d3(raw, request)
    assert result['confidence'] == .9
    assert result['native_confidence'] != result['confidence']
    raw['answers']['route']['confidence'] = .9
    with pytest.raises(ValueError, match='entropy'):
        normalize_d3(raw, request)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -.2])
def test_rejects_invalid_external_probabilities(bad):
    request = {'candidates': [{'id': 'a'}, {'id': 'b'}]}
    raw = {'answers': {'route': {'type': 'choice', 'choice': 'a', 'probabilities': {'a': bad, 'b': .1},
                                'confidence': .5}}, 'usage': {'output_tokens': 0}}
    with pytest.raises(ValueError): normalize_d3(raw, request)
