"""Capture authentic, timestamped NVIDIA telemetry; no cloud dependency."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

FIELDS = ['index', 'uuid', 'name', 'utilization.gpu', 'memory.used', 'memory.total', 'power.draw', 'temperature.gpu']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--interval', type=float, default=5)
    p.add_argument('--duration', type=float, default=7200)
    a = p.parse_args()
    if a.interval <= 0 or a.duration <= 0:
        p.error('interval and duration must be positive')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
    (out/'nvidia-smi-start.txt').write_text(snapshot.stdout)
    (out/'metadata.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
        'interval_seconds':a.interval, 'duration_seconds':a.duration,
        'scope':'Device-wide samples during concurrent study jobs; not per-run GPU efficiency or isolated latency.'}, indent=2))
    end = time.monotonic() + a.duration
    with (out/'gpu.csv').open('w') as f, (out/'processes.jsonl').open('w') as processes:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['utc', *FIELDS])
        while time.monotonic() < end:
            utc = datetime.now(timezone.utc).isoformat()
            result = subprocess.run(['nvidia-smi', '--query-gpu='+','.join(FIELDS), '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            for row in csv.reader(result.stdout.splitlines()):
                writer.writerow([utc, *[v.strip() for v in row]])
            f.flush()
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            processes.write(json.dumps({'utc':utc,'compute_processes':result.stdout.splitlines()})+'\n')
            processes.flush()
            time.sleep(min(a.interval, max(0, end-time.monotonic())))
    (out/'nvidia-smi-end.txt').write_text(subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout)

if __name__ == '__main__':
    main()
