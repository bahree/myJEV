"""Pinned Decision 3.0 comparison helpers; no changes to the release loader."""
import hashlib
import json
import math
from pathlib import Path
import random

from .decision_comparison import FIELDS, canonical_response, digest, request_views, validate_scores


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def comparison_items(plan):
    for path, expected in plan['sources'].items():
        if file_sha(path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    calibration = rows('data/banking77/calibration.jsonl')
    test = rows('data/banking77/test.jsonl')
    if len(calibration) != 1000 or len(test) != 3080:
        raise ValueError('Unexpected BANKING77 partition size')
    if {r['group'] for r in calibration} & {r['group'] for r in test}:
        raise ValueError('Calibration and test groups overlap')
    items = []

    def add(phase, row, view='original', request=None, mapping=None):
        request = request or {k: row[k] for k in FIELDS}
        mapping = mapping or {c['id']: c['id'] for c in request['candidates']}
        items.append({'key': f"{phase}/{row['id']}/{view}", 'phase': phase,
                      'id': row['id'], 'group': row['group'], 'label': row['label'],
                      'view': view, 'request': request, 'mapping': mapping})

    for row in calibration:
        add('calibration', row)
    for row in test:
        add('test', row)
    lookup = {r['id']: r for r in test}
    for identifier in plan['perturbation_ids']:
        for view, request, mapping in request_views(lookup[identifier]):
            if view != 'original':
                add('perturbation', lookup[identifier], view, request, mapping)
    for cohort in plan['transfer_cohorts']:
        available = rows(f'data/clinc150/{cohort}.jsonl')
        selected = random.Random(plan['sample_seed']).sample(
            available, min(len(available), plan['transfer_rows_per_cohort']))
        for row in selected:
            add('clinc/' + cohort, row)
    demos = rows('examples/demo-requests.jsonl')
    expected = json.loads(Path('examples/demo-expectations.json').read_text())['cases']
    for case in expected:
        add('demo', {**demos[case['line'] - 1], 'id': case['id'], 'group': case['id'],
                     'label': case['expected_id']})
    return items


def normalized_entropy_confidence(values):
    if len(values) < 2:
        return 1.0
    return max(0., min(1., 1 + sum(p * math.log(p) for p in values if p > 0) / math.log(len(values))))


def normalize_d3(raw, request):
    answer = raw['answers']['route']
    if answer.get('type') != 'choice':
        raise ValueError('Expected a choice answer')
    selected = answer['choice']
    scores = validate_scores(selected, answer['probabilities'], request['candidates'])
    native = answer.get('confidence')
    expected = normalized_entropy_confidence(list(scores.values()))
    if not isinstance(native, (int, float)) or isinstance(native, bool) or not math.isfinite(native) or abs(native - expected) > 1e-6:
        raise ValueError('Decision 3.0 confidence no longer matches the reviewed entropy definition')
    if raw.get('usage', {}).get('output_tokens') != 0:
        raise ValueError('Decision comparison requires zero generated tokens')
    return {'selected_id': selected, 'selection_scores': scores, 'confidence': scores[selected],
            'confidence_mode': 'selected-option-probability', 'native_confidence': float(native),
            'native_confidence_mode': 'one-minus-normalized-entropy', 'usage': raw.get('usage')}


def public_record(item, response):
    response = canonical_response(response, item['request'], item['mapping'])
    return {**{key: item[key] for key in ('key', 'phase', 'id', 'group', 'label', 'view')},
            **response, 'request_sha256': digest(item['request'])}
