"""Compare each pinned public Hub release with its local package on one GPU."""
import argparse
import gc
import json
from pathlib import Path
import torch
from myjev import DecisionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('results/release-readiness-v1/publication-manifest-public-v2.json'))
    parser.add_argument('--only', help='Check one exact repository ID')
    parser.add_argument('--output', type=Path, default=Path('results/release-readiness-v1/hub-output-equivalence.json'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    request = json.loads(Path('examples/request.json').read_text())
    records = []
    for package in manifest['packages']:
        if args.only and package['repo_id'] != args.only:
            continue
        receipt = json.loads((Path('results/release-readiness-v1/hub') / (package['public_name'] + '.json')).read_text())
        model = DecisionModel.load(package['path'])
        expected = model.score(request)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = DecisionModel.load(receipt['repo_id'], revision=receipt['revision'])
        actual = model.score(request)
        assert actual == expected, f"Output mismatch: {receipt['repo_id']}"
        records.append({'repo_id': receipt['repo_id'], 'revision': receipt['revision'],
                        'local_hub_output_equal': True, 'response': actual})
        del model
        gc.collect()
        torch.cuda.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({'scope': 'Exact output equality for examples/request.json on the same GPU and runtime; package-wide checksums verified separately.',
                        'results': records}, indent=2) + '\n')
        print(receipt['repo_id'] + ': exact output equality', flush=True)
    if not records:
        raise ValueError('No matching repository in publication manifest')


if __name__ == '__main__':
    main()
