"""Import saved metrics honestly as a historical W&B run (offline by default)."""
import argparse
import json
from pathlib import Path
from myjev.tracking import start_run

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, help='Completed training output containing artifact/ and training.jsonl')
p.add_argument('--evaluation', help='Optional evaluation directory with metrics.json')
p.add_argument('--output', required=True)
p.add_argument('--mode', choices=['offline', 'online'], default='offline')
a = p.parse_args()
root = Path(a.run)
manifest = json.loads((root/'artifact/manifest.json').read_text())
records = [json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
steps = [r['step'] for r in records]
if not steps or any(right <= left for left, right in zip(steps, steps[1:])):
    raise ValueError('training steps must increase strictly; separate restarted sessions before import')
config = {'artifact_revision': manifest['artifact_revision'],
          'scope': 'historical metrics; no reconstructed system telemetry'}
if manifest.get('format') == 'myjev-scratch-v1':
    config.update(backbone='randomly initialized scratch encoder', precision='FP32',
                  architecture=manifest['config'], training=json.loads((root/'run.json').read_text()))
else:
    config.update(backbone=manifest['backbone'], precision=manifest['precision'], training=manifest['training'])
run = start_run(a.output, config, mode=a.mode, historical=True)
Path(a.output).mkdir(parents=True, exist_ok=True)
(Path(a.output)/'run-url.json').write_text(json.dumps({'url': run.url, 'id': run.id, 'historical': True}, indent=2)+'\n')
try:
    for record in records:
        run.log(record, step=record['step'])
    if a.evaluation:
        result = json.loads((Path(a.evaluation)/'metrics.json').read_text())
        if result['artifact_revision'] != manifest['artifact_revision']:
            raise ValueError('evaluation artifact revision mismatch')
        for key, value in result.items():
            if isinstance(value, (int, float, str)) or value is None:
                run.summary['evaluation/'+key] = value
    run.summary['imported_updates'] = len(records)
finally:
    run.finish()
