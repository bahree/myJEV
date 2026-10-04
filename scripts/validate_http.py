import argparse
import json
import time
from pathlib import Path
import httpx
p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:18000')
p.add_argument('--output',default='results/docker-equivalence.json')
a=p.parse_args()
request=json.loads(Path('examples/request.json').read_text())
started=time.monotonic()
with httpx.Client(timeout=30) as client:
    while True:
        try:
            r=client.get(a.url+'/readyz')
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if time.monotonic()-started>55:
                raise
            time.sleep(1)
    result=client.post(a.url+'/score',json=request)
    result.raise_for_status()
    actual=result.json()
    expected=json.loads(Path('results/artifact-equivalence.json').read_text())['response']
    assert actual==expected,(actual,expected)
    invalid={**request,'candidates':[request['candidates'][0]]*2}
    assert client.post(a.url+'/score',json=invalid).status_code==422
Path(a.output).write_text(json.dumps({'equal':True,'response':actual,'invalid_request_status':422,
    'scope':'actual GPU container HTTP response equals direct Python/CLI/HTTP'},indent=2))
print('Docker response equivalence passed')
