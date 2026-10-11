"""Account for native prompt lengths without loading model weights."""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from myjev.decision3 import comparison_items, file_sha, rows
from myjev.prompt import encode
from myjev.schema import ScoreRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/decision3-v1'))
    args = parser.parse_args()
    plan = json.loads(Path('configs/decision3-v1.json').read_text())
    items = comparison_items(plan)
    items = [r for r in items if (r['phase'] == 'test' and r['id'] in plan['benchmark']['banking_ids'])
             or r['phase'] == 'demo']
    result = {'scope': 'Descriptive input-length accounting for the frozen benchmark requests. D3 lengths come from saved usage; myJEV lengths are recomputed with its pinned tokenizer and release prompt. No model weights are loaded.',
              'protocol_sha256': file_sha('configs/decision3-v1.json'), 'models': {}}
    for name, spec in plan['models'].items():
        folder = args.root / name / 'quality'
        measured = []
        if spec['backend'] == 'd3':
            lookup = {r['key']: r for r in rows(folder / 'records.jsonl')}
        else:
            manifest = json.loads((folder / 'run.json').read_text())['loader']['artifact_manifest']
            tokenizer = AutoTokenizer.from_pretrained(manifest['backbone'], revision=manifest['tokenizer_revision'])
        for item in items:
            if spec['backend'] == 'd3':
                count = lookup[item['key']]['usage']['input_tokens']
            else:
                request = ScoreRequest.model_validate(item['request'])
                count = encode(tokenizer, request, manifest['aliases'], manifest['max_tokens'])['input_ids'].shape[1]
            measured.append({'key': item['key'], 'workload': 'banking' if item['phase'] == 'test' else 'short-demos',
                             'input_tokens': int(count)})
        stats = {}
        for workload in ('banking', 'short-demos'):
            values = [r['input_tokens'] for r in measured if r['workload'] == workload]
            stats[workload] = {'requests': len(values), 'min': min(values),
                               'median': float(np.median(values)), 'max': max(values)}
        result['models'][name] = {'workloads': stats, 'requests': measured}
    (args.root / 'prompt-tokens.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: data['workloads'] for name, data in result['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
