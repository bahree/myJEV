"""Score a JSONL demo batch; stdout contains only ordinary API responses."""
import argparse
import json
from pathlib import Path
from myjev import DecisionModel

DEFAULT_ARTIFACT='bahree/myJEV-4B'
DEFAULT_REVISION='38f7cca5a8530483309f576b0c3dd1756bc27c33'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',default=DEFAULT_ARTIFACT)
    p.add_argument('--revision',help='Immutable Hub commit; the default model has a tested built-in pin')
    p.add_argument('--input',type=Path,default=Path('examples/demo-requests.jsonl'))
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    revision=a.revision
    if revision is None and a.artifact==DEFAULT_ARTIFACT:revision=DEFAULT_REVISION
    model=DecisionModel.load(a.artifact,revision=revision,device=a.device)
    with a.input.open() as rows:
        for row in rows:
            if row.strip():print(json.dumps(model.score(json.loads(row))),flush=True)


if __name__=='__main__':main()
