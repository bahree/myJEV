"""Data contracts for the optional alias/Clef research comparison."""
import hashlib
import json
import random
from pathlib import Path
from .data import read_jsonl, request_from_row, training_order, validate_isolation


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frozen_splits(plan):
    root = Path(plan['data'])
    parts = {}
    for split, expected in plan['data_sha256'].items():
        path = root / (split + '.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen partition changed: ' + split)
        parts[split] = read_jsonl(path)
    validate_isolation(parts)
    return parts


def exposure(rows, seed, count):
    rng = random.Random(seed)
    order, cursor = training_order(rows, rng)
    examples = []
    for _ in range(count):
        if cursor == len(order):
            rng.shuffle(order)
            cursor = 0
        row = rows[order[cursor]]
        cursor += 1
        request, target = request_from_row(row, rng)
        examples.append((row, request, target))
    return examples


def exposure_digest(examples):
    return digest([{'id': row['id'], 'candidates': [c['id'] for c in request['candidates']]}
                   for row, request, _ in examples])


def clef_request(request):
    """Neutral IDs preserve caller ordering through Clef's lexical key sorting."""
    candidates = request['candidates']
    criteria = {f'c{i:04d}': c['description'] for i, c in enumerate(candidates)}
    mapping = {key: c['id'] for key, c in zip(criteria, candidates)}
    return {'state': request['context'], 'questions': {'decision': {
        'type': 'choice', 'instructions': request['instructions'], 'criteria': criteria}}}, mapping


def permute_request(request, row_id, seed):
    value = json.loads(json.dumps(request))
    rng = random.Random(int(hashlib.sha256(f'{seed}:{row_id}'.encode()).hexdigest(), 16))
    rng.shuffle(value['candidates'])
    return value


def selected_trial(trials):
    if not trials:
        raise ValueError('No validation trials')
    return min(trials, key=lambda r: (-r['accuracy'], r['selection_nll'], r['learning_rate']))
