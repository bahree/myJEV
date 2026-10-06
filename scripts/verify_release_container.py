"""Check a selected local image against an independently recorded Python result."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import httpx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--expected', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--port', type=int, default=18087)
    p.add_argument('--output', type=Path, default=Path('results/release-readiness-v1/container-smoke.json'))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    name = 'myjev-release-smoke-' + str(time.time_ns())
    image_id = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    start = time.perf_counter()
    subprocess.run(['docker', 'run', '-d', '--name', name, '--gpus', f'device={args.gpu}',
                    '-p', f'127.0.0.1:{args.port}:8000', '-v', f'{args.artifact.resolve()}:/artifact:ro',
                    '-v', f'{args.cache.resolve()}:/hf:ro', '-e', 'HF_HOME=/hf', '-e', 'HF_HUB_OFFLINE=1',
                    '-e', 'MYJEV_ARTIFACT=/artifact', '-e', 'OMP_NUM_THREADS=2', '-e', 'MKL_NUM_THREADS=2', args.image], check=True, stdout=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', timeout=60) as client:
            while True:
                try:
                    ready = client.get('/readyz')
                    if ready.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - start > 300:
                    raise TimeoutError('Container failed readiness deadline')
                time.sleep(1)
            startup = time.perf_counter() - start
            request = json.loads(Path('examples/request.json').read_text())
            expected = json.loads(args.expected.read_text())['response']
            assert client.get('/healthz').status_code == 200
            assert client.get('/health').status_code == 200
            response = client.post('/score', json=request)
            response.raise_for_status()
            assert response.json() == expected, 'Container differs from independent Python result'
            alternate = client.post('/generate', json=request)
            alternate.raise_for_status()
            assert alternate.json() == expected
            duplicate = {**request, 'candidates': [request['candidates'][0]] * 2}
            assert client.post('/score', json=duplicate).status_code == 422
            result = {'image_tag': args.image, 'image_id': image_id, 'artifact': str(args.artifact),
                      'startup_to_ready_seconds': startup, 'python_docker_equal': True, 'managed_aliases_equal': True,
                      'duplicate_ids_rejected': True,
                      'scope': 'Clean image, cached pinned backbone, concurrent work on other GPUs. Readiness/contract verification, not isolated latency.'}
            args.output.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        with args.output.with_suffix('.log').open('w') as log:
            subprocess.run(['docker', 'logs', name], stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(['docker', 'rm', '-f', name], check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
