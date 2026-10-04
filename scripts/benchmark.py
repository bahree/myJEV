import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import httpx
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--request",default="examples/request.json")
p.add_argument("--requests",type=int,default=100)
p.add_argument("--concurrency",type=int,default=1)
p.add_argument("--output",required=True)
a=p.parse_args()
request=json.loads(Path(a.request).read_text())
with httpx.Client(timeout=120) as client:
    client.get(a.url+"/readyz").raise_for_status()
    client.post(a.url+"/score",json=request).raise_for_status()
    def run(_):
        start=time.perf_counter()
        r=client.post(a.url+"/score",json=request)
        return {"seconds":time.perf_counter()-start,"status":r.status_code}
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results=list(pool.map(run,range(a.requests)))
    elapsed=time.perf_counter()-started
successful=[r["seconds"] for r in results if r["status"] == 200]
Path(a.output).write_text(json.dumps({"concurrency":a.concurrency,"requests":a.requests,
    "warm_p50_p95_seconds":np.quantile(successful,[.5,.95]).tolist() if successful else None,
    "successful_throughput_per_second":len(successful)/elapsed,"elapsed_seconds":elapsed,
    "statuses":{str(s):sum(r["status"]==s for r in results) for s in set(r["status"] for r in results)},
    "timings":results},indent=2))
