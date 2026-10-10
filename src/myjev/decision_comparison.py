"""Shared request perturbations and strict hosted-choice normalization."""
import hashlib
import json
import math
import random

VIEWS = ('original', 'reverse', 'shuffle', 'renamed-ids', 'whitespace', 'instruction-paraphrase')
FIELDS = ('context', 'instructions', 'candidates')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def request_views(row):
    original = {k: row[k] for k in FIELDS}
    identity = {c['id']: c['id'] for c in row['candidates']}
    yield 'original', original, identity
    yield 'reverse', {**original, 'candidates': list(reversed(original['candidates']))}, identity
    shuffled = list(original['candidates'])
    random.Random('decision-comparison-v1:' + row['id']).shuffle(shuffled)
    yield 'shuffle', {**original, 'candidates': shuffled}, identity
    renamed = [{**c, 'id': 'option_' + hashlib.sha256((row['id'] + ':' + c['id']).encode()).hexdigest()[:16]}
               for c in original['candidates']]
    mapping = {new['id']: old['id'] for new, old in zip(renamed, original['candidates'], strict=True)}
    yield 'renamed-ids', {**original, 'candidates': renamed}, mapping
    yield 'whitespace', {**original, 'context': '\n  ' + original['context'] + '  \n',
                          'instructions': '\n' + original['instructions'] + '\n',
                          'candidates': [{**c, 'description': '  ' + c['description'] + '  '} for c in original['candidates']]}, identity
    yield 'instruction-paraphrase', {**original, 'instructions': "Choose the banking support category that best describes the customer's request."}, identity


def systemone_payload(request, model):
    return {'model': model, 'state': request['context'], 'questions': {
        'route': {'type': 'choice', 'instructions': request['instructions'],
                  'criteria': {c['id']: c['description'] for c in request['candidates']}}}}


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid response probability')
    return float(value)


def validate_scores(selected, scores, candidates):
    keys = [c['id'] for c in candidates]
    if not isinstance(scores, dict) or set(scores) != set(keys) or selected not in keys:
        raise ValueError('response candidate IDs do not match the request')
    scores = {key: probability(scores[key]) for key in keys}
    if not math.isclose(sum(scores.values()), 1., abs_tol=1e-5):
        raise ValueError('response probabilities do not sum to one')
    if scores[selected] + 1e-7 < max(scores.values()):
        raise ValueError('selected choice is not a maximum-probability option')
    return scores


def normalize_hosted(payload, request):
    if not isinstance(payload, dict):
        raise ValueError('invalid response object')
    answers = payload.get('answers')
    answer = answers.get('route') if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get('type') != 'choice':
        raise ValueError('missing typed route answer')
    choice = answer.get('choice')
    scores = validate_scores(choice, answer.get('probabilities'), request['candidates'])
    confidence = answer.get('confidence')
    if confidence is not None:
        confidence = probability(confidence)
    # Vendor confidence is retained separately: its event/definition may differ.
    return {'selected_id': choice, 'selection_scores': scores, 'confidence': scores[choice],
            'confidence_mode': 'selected-option-probability', 'provider_confidence': confidence,
            'provider_confidence_semantics': 'Unspecified by this comparison; not treated as learned correctness.'}


def canonical_response(response, request, mapping):
    scores = validate_scores(response['selected_id'], response['selection_scores'], request['candidates'])
    if set(mapping) != set(scores) or len(set(mapping.values())) != len(mapping):
        raise ValueError('invalid candidate ID inverse mapping')
    return {**response, 'selected_id': mapping[response['selected_id']],
            'selection_scores': {mapping[k]: v for k, v in scores.items()}}


def budget_reservation(requests, context_tokens=32768, price_per_million=.042):
    if isinstance(requests, bool) or not isinstance(requests, int) or requests < 1:
        raise ValueError('request count must be positive')
    if context_tokens < 1 or not math.isfinite(price_per_million) or price_per_million < 0:
        raise ValueError('invalid cost assumptions')
    return requests * context_tokens * price_per_million / 1_000_000
