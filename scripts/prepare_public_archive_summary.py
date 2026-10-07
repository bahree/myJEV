"""Amend only pooled archive macro-F1; retain the immutable original study output."""
import hashlib
import json
from pathlib import Path

source = Path('results/archive-machine-v2/summary.json')
report = json.loads(source.read_text())
for stage in ('unadapted', 'adapted'):
    report['results'][stage]['macro_f1'] = None
    report['results'][stage]['macro_f1_status'] = 'not_applicable: candidate IDs denote different classes across rubrics; use per-rubric macro-F1'
report['reporting_amendment'] = {
    'original_summary_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'reason': 'Suppress pooled macro-F1 across incompatible archive rubrics. Original experiment outputs preserved; per-rubric results unchanged.',
    'changed_fields': ['results.unadapted.macro_f1', 'results.adapted.macro_f1'],
}
Path('results/archive-machine-v2/public-summary.json').write_text(json.dumps(report, indent=2) + '\n')
