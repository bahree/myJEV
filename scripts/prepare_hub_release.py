"""Create immutable adapter-only upload packages; never contacts the Hub."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/hub-ready-v1'))
    parser.add_argument('--report', type=Path, default=Path('results/release-readiness-v1/publication-manifest.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing release snapshot')
    records = []
    for source in sorted(Path('artifacts/release-candidates-v1').glob('myjev-*')):
        manifest = json.loads((source / 'manifest.json').read_text())
        for name, expected in manifest['checksums'].items():
            path = source / name
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or sha(path) != expected:
                raise ValueError(f'Unsafe or changed artifact file: {name}')
        destination = args.output / source.name
        shutil.copytree(source, destination)
        shutil.copyfile('LICENSE', destination / 'LICENSE')
        size = source.name.split('-')[1]
        shutil.copyfile(f'results/release-readiness-v1/upstream/{size}-LICENSE', destination / 'BACKBONE_LICENSE')
        (destination / 'NOTICE.md').write_text(
            f"# Attribution\n\nOriginal myJEV adapter/head contributions: copyright Amit Bahree, MIT.\n\n"
            f"Based on {manifest['backbone']} at revision {manifest['backbone_revision']}, provided by the Qwen team under Apache-2.0. "
            "The upstream license is preserved in BACKBONE_LICENSE. Backbone weights are not included. "
            "This package does not relicense those upstream weights.\n\n"
            "Training source: BANKING77, Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders, "
            "https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data, CC BY 4.0. "
            "No dataset records or optimizer states are included.\n")
        card = (destination / 'README.md').read_text()
        card = card.replace('---\nbase_model:', '---\nlicense: mit\nbase_model:', 1)
        card = card.replace(': local release candidate', ': research checkpoint')
        card = card.replace('**Draft packaging only. Not a production recommendation or a published Hub release.**', '**Locally validated research checkpoint. Not production validated.**')
        card = card.replace('The same artifact must still pass final representative serving and clean-environment release checks.', 'This artifact passed Python/CLI/HTTP/Docker equivalence and the recorded serving workloads. Consult docs/hosting.md in the source repository for tested limits and workload conditions.')
        card = card.replace('Transfer evaluation and final release selection remain separate gates.', 'The three-seed transfer and robustness evaluation is complete. Consult the public model comparison; 4B continued SFT with temperature is the local default recommendation.')
        card = card.replace('Artifact release terms remain subject to review. Backbone and dataset licenses remain separate and must be preserved. Do not upload this draft as a finalized licensed release.', 'Original adapter/head contributions are MIT-licensed. The pinned Qwen backbone is Apache-2.0 and retains its separate terms; see NOTICE.md and BACKBONE_LICENSE. BANKING77 provenance and its CC BY 4.0 license are documented separately.')
        card = card.replace(str(source), str(destination))
        (destination / 'README.md').write_text(card)
        records.append({'name': destination.name, 'path': str(destination), 'artifact_revision': manifest['artifact_revision'],
                        'local_default': destination.name == 'myjev-4b-continued_sft-seed11',
                        'files': {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}})
    report = {'schema_version': 1, 'uploaded': False, 'namespace': None, 'visibility': None,
              'scope': 'Adapter/head-only research packages; no backbone, training data or optimizer state. Destination and visibility require owner selection.',
              'packages': records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(args.report)


if __name__ == '__main__':
    main()
