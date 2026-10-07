"""Compare each pinned public Hub release with its local package on one GPU."""
import gc
import json
from pathlib import Path
import torch
from myjev import DecisionModel


def main():
    manifest = json.loads(Path('results/release-readiness-v1/publication-manifest-public.json').read_text())
    request = json.loads(Path('examples/request.json').read_text())
    records = []
    for package in manifest['packages']:
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
        Path('results/release-readiness-v1/hub-output-equivalence.json').write_text(
            json.dumps({'scope': 'Exact output equality for examples/request.json on the same GPU and runtime; package-wide checksums verified separately.',
                        'results': records}, indent=2) + '\n')
        print(receipt['repo_id'] + ': exact output equality', flush=True)


if __name__ == '__main__':
    main()
