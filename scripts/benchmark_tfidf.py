"""Measure the saved fixed-taxonomy baseline on deterministic BANKING77 requests."""
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    output=Path('results/tfidf-serving-v1');output.mkdir(parents=True,exist_ok=True)
    artifact=Path('results/tfidf/model.joblib')
    data=Path('data/banking77/test.jsonl')
    rows=[json.loads(line) for line in data.read_text().splitlines()]
    indices=np.random.default_rng(42).choice(len(rows),512,replace=False)
    config={'model_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'test_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
            'request_ids':[rows[i]['id'] for i in indices], 'warmup':20,'requests':512,
            'threads':2,'concurrency':1,'taxonomy':'fixed 77 BANKING intents',
            'scope':'Local Python end-to-end vectorizer plus classifier, not HTTP. Inputs differ from Qwen short three-candidate HTTP workload; do not calculate cross-path speedup.'}
    (output/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    start=time.perf_counter();model=joblib.load(artifact);load=time.perf_counter()-start
    timings=[]
    with threadpool_limits(limits=2):
        for i in indices[:20]: model.predict_proba([rows[i]['context']])
        for i in indices:
            start=time.perf_counter();model.predict_proba([rows[i]['context']]);timings.append(time.perf_counter()-start)
    result={'load_seconds_warm_filesystem':load,'latency_p50_p95_ms':(1000*np.quantile(timings,[.5,.95])).tolist(),
        'sequential_requests_per_second':len(timings)/sum(timings),'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifact_bytes':artifact.stat().st_size,'cpu':platform.processor(),'platform':platform.platform(),
        'memory_scope':'Whole Python process peak RSS including dependencies; differs from CUDA allocated-memory accounting.',
        'protocol_sha256':hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'timings.json').write_text(json.dumps(timings)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
