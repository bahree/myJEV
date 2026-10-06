"""Summarize recorded checks without re-running or changing frozen benchmarks."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('results/release-validation-v1'))
    parser.add_argument('--output', type=Path, default=Path('results/release-readiness-v1/serving-summary.json'))
    args = parser.parse_args()
    rows = []
    for folder in sorted(args.source.glob('myjev-*')):
        equivalence = json.loads((folder / 'equivalence.json').read_text())
        container = json.loads((folder / 'http-complete.json').read_text())
        paths = json.loads((folder / 'paths/report.json').read_text())
        cells = []
        for row in paths['results']:
            outputs = row['outputs']
            cells.append({key: row[key] for key in ('candidates', 'context_repeats', 'path', 'direct_prompt_tokens', 'p50_p95_seconds', 'peak_allocated_vram_bytes')} | {
                'format_valid': sum(bool(o['format_valid']) for o in outputs),
                'truncated': sum(bool(o.get('truncated')) for o in outputs),
                'requests': len(outputs),
            })
        rows.append({'candidate': folder.name, 'python_cli_http_equal': equivalence['python_cli_http_equal'],
                     'docker_python_equal': container['docker_python_equal'],
                     'startup_to_ready_seconds': container['startup_to_ready_seconds'],
                     'short_http_c1': {k: v for k, v in json.loads((folder / 'http-3-0-c1.json').read_text()).items() if k != 'timings'},
                     'paths': cells})
    report = {'source': str(args.source), 'environment': json.loads((args.source / 'environment.json').read_text()),
              'scope': 'Six seed-11 candidates; isolated GPU-compute host at start; warm image/backbone caches. Short three-candidate and 32-candidate workloads. Generation format validity is not decision accuracy. Allocated VRAM excludes allocator reserve and driver/runtime memory.',
              'candidates': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
