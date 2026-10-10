"""Run the frozen local/OpenRouter diagnostic, with an explicit live-call budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from myjev.decision_comparison import (FIELDS, budget_reservation, canonical_response, digest,
                                      normalize_hosted, request_views, systemone_payload)

ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'
DEFAULT_REVISION = '38f7cca5a8530483309f576b0c3dd1756bc27c33'


def utc(): return datetime.now(timezone.utc).isoformat()
def read_rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_key(path):
    key = os.environ.get('OPENROUTER_API_KEY')
    if path:
        for line in Path(path).read_text().splitlines():
            name, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and name.strip() == 'OPENROUTER_API_KEY':
                key = value.strip().strip('\"\'')
    if not key or '\n' in key or '\r' in key:
        raise ValueError('Configure OPENROUTER_API_KEY in the environment or --env-file')
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None


def hosted_call(payload, key, timeout):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    req = urllib.request.Request(ENDPOINT, data=encoded, headers={
        'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.build_opener(NoRedirect).open(req, timeout=timeout) as response:
        body = response.read(2*1024*1024+1)
        if len(body) > 2*1024*1024:
            raise ValueError('response body exceeds the recorded limit')
        result = json.loads(body, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non-finite response JSON')))
        headers = {name: response.headers[name] for name in ('x-request-id', 'apim-request-id', 'x-model-version') if name in response.headers}
    # No credentials, arbitrary headers, state echoes or free-form error bodies are logged.
    metadata = {k: result[k] for k in ('id', 'model', 'provider', 'created', 'model_version', 'system_fingerprint')
                if isinstance(result.get(k), (str, int, float)) and not isinstance(result.get(k), bool)}
    usage = result.get('usage', {})
    if isinstance(usage, dict):
        metadata['usage'] = {k: v for k, v in usage.items() if k in (
            'prompt_tokens', 'input_tokens', 'completion_tokens', 'output_tokens', 'total_tokens', 'cost')
            and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0}
    metadata.update(headers=headers, response_sha256=hashlib.sha256(body).hexdigest(),
                    immutable_weight_revision=None)
    return result, metadata


def calls(plan):
    for path, expected in plan['sources'].items():
        if sha(path) != expected:
            raise ValueError('Frozen input changed: '+path)
    cal = read_rows('data/banking77/calibration.jsonl')
    test_all = {r['id']: r for r in read_rows('data/banking77/test.jsonl')}
    test = [test_all[key] for key in plan['test_ids']]
    if len(cal) != plan['calibration_rows'] or {r['group'] for r in cal} & {r['group'] for r in test}:
        raise ValueError('Calibration/test partition mismatch')
    items = []
    def add(phase, row, view, request, mapping, suffix=''):
        items.append({'key': f"{phase}/{row['id']}/{view}{suffix}", 'phase': phase, 'id': row['id'],
                      'group': row['group'], 'label': row['label'], 'view': view,
                      'request': request, 'mapping': mapping})
    for i in range(plan['warmup_requests']):
        row = cal[0];name, request, mapping = next(request_views(row))
        add('warmup', row, name, request, mapping, str(i))
    for row in cal:
        name, request, mapping = next(request_views(row));add('calibration', row, name, request, mapping)
    for row in test:
        for name, request, mapping in request_views(row):add('test', row, name, request, mapping)
    demos = read_rows('examples/demo-requests.jsonl')
    expectations = json.loads(Path('examples/demo-expectations.json').read_text())['cases']
    for expected in expectations:
        request = demos[expected['line']-1]
        row = {**request, 'id': expected['id'], 'group': expected['id'], 'label': expected['expected_id']}
        add('demo', row, 'original', request, {c['id']: c['id'] for c in request['candidates']})
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('local', 'openrouter'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--artifact', default='bahree/myJEV-4B')
    p.add_argument('--revision')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--execute', action='store_true', help='Required to make billable API calls; otherwise a dry run')
    p.add_argument('--max-cost-usd', type=float)
    p.add_argument('--max-requests', type=int, default=1400)
    p.add_argument('--timeout', type=float, default=60)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not math.isfinite(a.timeout) or a.timeout <= 0: p.error('timeout must be positive')
    plan_path=Path('configs/decision-comparison-v1.json');plan=json.loads(plan_path.read_text())
    for name, expected in plan['code_sha256'].items():
        if sha(name) != expected: raise ValueError('Frozen implementation changed: '+name)
    items=calls(plan);reserve=budget_reservation(len(items), plan['hosted_context_tokens'],plan['input_usd_per_million_assumption'])
    if a.backend=='openrouter':
        if not a.execute:
            print(json.dumps({'dry_run':True, 'calls':len(items), 'full_context_input_reservation_usd':reserve,
                              'endpoint':ENDPOINT, 'model':plan['hosted_model'],
                              'scope':'No network calls. Reservation uses published context and input price; excludes provider price changes or additional fees.'},indent=2));return
        if a.max_cost_usd is None or not math.isfinite(a.max_cost_usd) or a.max_cost_usd < reserve or a.max_requests < len(items):
            p.error('Live calls require sufficient --max-cost-usd and --max-requests for the entire frozen plan')
        key=load_key(a.env_file)
    else: key=None
    revision=a.revision or (DEFAULT_REVISION if a.artifact=='bahree/myJEV-4B' else None)
    identity={'backend':a.backend,'plan_sha256':sha(plan_path),'requests_sha256':digest(items),
              'artifact':a.artifact if a.backend=='local' else None,'revision':revision if a.backend=='local' else None,
              'model':plan['hosted_model'] if a.backend=='openrouter' else None,
              'endpoint':ENDPOINT if a.backend=='openrouter' else None,
              'max_requests':a.max_requests if key else None,'max_cost_usd':a.max_cost_usd if key else None,
              'input_usd_per_million_assumption':plan['input_usd_per_million_assumption'] if key else None}
    run_path=a.output/'run.json';records_path=a.output/'records.jsonl';attempts_path=a.output/'attempts.jsonl'
    if a.output.exists():
        if not a.resume:raise ValueError('Output exists; choose a new directory or explicit --resume')
        run=json.loads(run_path.read_text())
        if run['identity']!=identity:raise ValueError('Resume configuration differs')
        if (a.output/'complete.json').exists():raise ValueError('Run already complete')
    else:
        a.output.mkdir(parents=True)
        run={'identity':identity,'started_utc':utc(),'expected_calls':len(items),
             'timing_scope':'Sequential calls, five explicit warmups. Local includes tokenization and result conversion; hosted also includes network and provider queue. No concurrent-load benchmark.',
             'service_revision_scope':'Returned identifiers are recorded per call. A model name or service version may not identify fixed weights.'}
        run_path.write_text(json.dumps(run,indent=2)+'\n')
    previous=read_rows(records_path) if records_path.exists() else []
    if [r['key'] for r in previous] != [r['key'] for r in items[:len(previous)]]:raise ValueError('Saved records are not a valid prefix')
    attempts=read_rows(attempts_path) if attempts_path.exists() else []
    model=None
    if a.backend=='local':
        import torch
        from myjev.inference import DecisionModel
        start=time.perf_counter();model=DecisionModel.load(a.artifact,revision=revision,device=a.device)
        run.update(load_seconds=time.perf_counter()-start,artifact_manifest=model.manifest,
                   hardware=torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else 'cpu')
        run_path.write_text(json.dumps(run,indent=2)+'\n')
    per_call_reserve=budget_reservation(1,plan['hosted_context_tokens'],plan['input_usd_per_million_assumption'])
    start=time.perf_counter()
    with records_path.open('a') as records, attempts_path.open('a') as ledger:
        for item in items[len(previous):]:
            if key and (len(attempts)+1>a.max_requests or (len(attempts)+1)*per_call_reserve>a.max_cost_usd+1e-12):
                raise ValueError('Recorded API attempt budget exhausted; no further call made')
            attempt={'key':item['key'],'utc':utc(),'attempt':len(attempts)+1}
            ledger.write(json.dumps(attempt)+'\n');ledger.flush();attempts.append(attempt)
            request=item['request'];payload=systemone_payload(request,plan['hosted_model']) if key else request
            try:
                if model is not None and a.device.startswith('cuda'):torch.cuda.synchronize(a.device)
                begun=utc();before=time.perf_counter()
                if key:
                    raw,metadata=hosted_call(payload,key,a.timeout);response=normalize_hosted(raw,request)
                else:
                    raw=model.score(request)
                    response={**raw,'native_confidence':raw['confidence'],'native_confidence_mode':raw['confidence_mode'],
                              'confidence':raw['selection_scores'][raw['selected_id']], 'confidence_mode':'selected-option-probability'}
                    metadata={}
                    if a.device.startswith('cuda'):torch.cuda.synchronize(a.device)
                seconds=time.perf_counter()-before
                response=canonical_response(response,request,item['mapping'])
                record={**{k:item[k] for k in ('key','phase','id','group','label','view')},**response,
                        'request_sha256':digest(request),'wire_payload_sha256':digest(payload),'started_utc':begun,
                        'finished_utc':utc(),'seconds':seconds,'service_metadata':metadata}
                records.write(json.dumps(record,allow_nan=False)+'\n');records.flush()
                if len(previous)%100==0:print(json.dumps({'completed':len(previous)+1,'total':len(items),'phase':item['phase']}),flush=True)
                previous.append(record)
            except Exception as error:
                failure={'utc':utc(),'key':item['key'],'exception_class':type(error).__name__,
                         'http_status':error.code if isinstance(error,urllib.error.HTTPError) else None,
                         'scope':'Failed attempt retained; no automatic retry, no error body or credential recorded.'}
                (a.output/f'failure-{len(attempts)}.json').write_text(json.dumps(failure,indent=2)+'\n')
                print(json.dumps(failure),file=sys.stderr);raise SystemExit(1) from None
    complete={'completed_utc':utc(),'calls':len(previous),'attempts':len(attempts),'session_seconds':time.perf_counter()-start,
              'records_sha256':sha(records_path),'peak_allocated_bytes':torch.cuda.max_memory_allocated(a.device) if model is not None and a.device.startswith('cuda') else None}
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(complete),flush=True)


if __name__=='__main__': main()
