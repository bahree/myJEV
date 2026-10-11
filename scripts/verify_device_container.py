"""Record real CPU/GPU container responses, startup, validation and device logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18093)
    parser.add_argument('--gpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--platform')
    parser.add_argument('--emulator', type=Path, help='Optional static QEMU binary, mounted read-only for an emulated architecture check')
    parser.add_argument('--demos', action='store_true')
    parser.add_argument('--cli', action='store_true')
    parser.add_argument('--expected', type=Path, help='Optional saved response for exact GPU regression comparison')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    name = 'myjev-device-check-' + str(time.time_ns())
    command = ['docker', 'run', '-d', '--name', name,
               '-p', f'127.0.0.1:{args.port}:8000',
               '-v', f'{args.cache.resolve()}:/hf:ro',
               '-e', 'HF_HOME=/hf', '-e', 'HF_HUB_OFFLINE=1',
               '-e', f'MYJEV_ARTIFACT={args.repo}', '-e', f'MYJEV_REVISION={args.revision}',
               '-e', f'MYJEV_DEVICE={args.device}', '-e', f'OMP_NUM_THREADS={args.threads}',
               '-e', f'MKL_NUM_THREADS={args.threads}', '-e', 'MYJEV_TIMEOUT=600']
    if args.gpu is not None:
        command += ['--gpus', f'device={args.gpu}']
    if args.platform:
        command += ['--platform', args.platform]
    if args.emulator:
        command += ['-v', f'{args.emulator.resolve()}:/emulator:ro', '--entrypoint', '/emulator']
    command.append(args.image)
    if args.emulator:
        command += ['/usr/local/bin/python', '-m', 'uvicorn', 'myjev.server:app_factory',
                    '--factory', '--host', '0.0.0.0', '--port', '8000', '--no-access-log', '--workers', '1']
    fixture = json.loads(Path('examples/request.json').read_text())
    requests = [fixture]
    if args.demos:
        requests += [json.loads(line) for line in Path('examples/demo-requests.jsonl').read_text().splitlines() if line.strip()]
    started = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', timeout=600) as client:
            while True:
                try:
                    if client.get('/readyz').status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                running = subprocess.check_output(['docker', 'inspect', name, '--format', '{{.State.Running}}'], text=True).strip()
                if running != 'true':
                    raise RuntimeError('Container exited before readiness; inspect the saved log')
                if time.perf_counter() - started > 900:
                    raise TimeoutError('Readiness deadline exceeded')
                time.sleep(1)
            readiness = time.perf_counter() - started
            responses = []
            for request in requests:
                begin = time.perf_counter()
                response = client.post('/score', json=request)
                response.raise_for_status()
                responses.append({'request': request, 'response': response.json(), 'seconds': time.perf_counter() - begin})
            if args.expected:
                expected = json.loads(args.expected.read_text())
                assert responses[0]['response'] == expected.get('response', expected)
            alias = client.post('/generate', json=fixture)
            alias.raise_for_status()
            assert alias.json() == responses[0]['response']
            rejections = {}
            for route in ['/score', '/generate']:
                for value in ['NaN', 'Infinity', '-Infinity']:
                    response = client.post(route, content=json.dumps(fixture).replace('"I was charged twice."', value), headers={'Content-Type': 'application/json'})
                    assert response.status_code == 422
                    rejections[route + ':' + value] = response.status_code
            assert client.post('/score', content=b'x' * 1_048_577).status_code == 413
            python = ['/emulator', '/usr/local/bin/python'] if args.emulator else ['python']
            environment = json.loads(subprocess.check_output(['docker', 'exec', name, *python, '-c',
                'import json,torch,platform; print(json.dumps({"machine":platform.machine(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"threads":torch.get_num_threads()}))'], text=True))
            if args.cli:
                cli = subprocess.run(['docker', 'exec', name, *python, '-m', 'myjev.cli', 'score',
                    '--artifact', args.repo, '--revision', args.revision, '--device', args.device,
                    '--input', '/app/examples/request.json'], capture_output=True, text=True, check=True)
                args.output.with_suffix('.cli.log').write_text(cli.stderr + '\n' + cli.stdout)
                assert json.loads(cli.stdout) == responses[0]['response']
            stats = json.loads(subprocess.check_output(['docker', 'stats', '--no-stream', '--format', '{{json .}}', name], text=True))
            result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
                      'image_inspect': json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]['Image'],
                      'environment': environment, 'readiness_seconds': readiness, 'responses': responses,
                      'managed_alias_equal': True, 'cli_equal': True if args.cli else None,
                      'expected_response_equal': True if args.expected else None,
                      'nonfinite_statuses': rejections, 'oversize_status': 413,
                      'container_stats_after_requests': stats, 'emulated': args.emulator is not None,
                      'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/myjev').rglob('*.py')},
                      'scope': 'Short request smoke checks with cached weights and the recorded thread count; no full accuracy/calibration evaluation or maximum-input memory claim. Timings include concurrent host activity and are not isolated performance benchmarks.', 'passed': True}
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'passed': True, 'image': args.image, 'model': args.repo, 'requests': len(requests), 'ready_seconds': readiness}), flush=True)
    finally:
        with args.output.with_suffix('.log').open('w') as log:
            subprocess.run(['docker', 'logs', name], stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(['docker', 'rm', '-f', name], check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
