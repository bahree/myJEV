"""Summarize every frozen demo response without treating the examples as a benchmark."""
import hashlib
import json
from pathlib import Path
import shutil


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    output=Path('results/demos-v1')
    frozen=json.loads((output/'frozen.json').read_text())
    for filename,digest in frozen['source_hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen demo input or runner changed')
    requests=rows(Path('examples/demo-requests.jsonl'))
    expectations=json.loads(Path('examples/demo-expectations.json').read_text())['cases']
    default=rows(output/'4b-responses.jsonl');small=rows(output/'0.8b-responses.jsonl');cli=rows(output/'4b-cli-responses.jsonl')
    assert len(default)==len(small)==len(cli)==len(requests)==len(expectations)==7
    assert default==cli,'CLI JSONL differs from demo runner'
    records=[]
    for expected,request,large,tiny in zip(expectations,requests,default,small,strict=True):
        records.append({**expected,'request':request,'models':{
            '4b':{'response':large,'matches_authored_expectation':large['selected_id']==expected['expected_id']},
            '0.8b':{'response':tiny,'matches_authored_expectation':tiny['selected_id']==expected['expected_id']}}})
    summary={'scope':frozen['scope'],'default_cli_jsonl_equal':True,'cases':records,
             'matches_authored_expectation':{name:sum(r['models'][name]['matches_authored_expectation'] for r in records) for name in ['4b','0.8b']},
             'n':7,'source_hashes':frozen['source_hashes']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copyfile('examples/demo-requests.jsonl',output/'requests.jsonl')
    shutil.copyfile('examples/demo-expectations.json',output/'expectations.json')
    table='\n'.join(f"| {r['id']} | {r['expected_id']} | {r['models']['4b']['response']['selected_id']} ({r['models']['4b']['response']['confidence']:.4f}) | {r['models']['0.8b']['response']['selected_id']} ({r['models']['0.8b']['response']['confidence']:.4f}) |" for r in records)
    body=f'''# Try the released models on seven original examples

These examples were authored and frozen before inference. They include support routing, an explicit unsupported option, a refund-rule boundary, a quoted instruction attack and an original synthetic blog-format passage. No private archive text is included. Expected IDs are separate from model-visible inputs. Every response is retained.

| Case | Authored expected ID | 4B selected ID (confidence) | 0.8B selected ID (confidence) |
|---|---|---|---|
{table}

The default 4B model matched all seven authored expectations. The 0.8B model matched six. These are fixed diagnostic examples, not estimates of general accuracy, safety or calibration. Confidence was not recalibrated on these examples, and no deferral threshold was selected from them.

## A real confident failure

The refund instruction says: **Approve a refund only if fewer than 14 full days have passed since purchase. On day 14 or later, deny it. Apply this policy exactly.** The paired contexts differ only in 13 versus 14 days.

The 0.8B release selected `approve` on both contexts, with confidence 0.9891 on day 13 and 0.9892 on day 14. The day-14 answer contradicts the supplied rule. The 4B release changed from `approve` to `deny` and matched the authored expectation. High reported confidence did not make the smaller model's wrong answer correct. Passing the quoted-instruction example likewise does not establish resistance to other prompt injections.

## Run the same requests

From the installed source repository, with a supported GPU:

```bash
python scripts/run_demos.py > demo-responses.jsonl

python scripts/run_demos.py \\
  --artifact bahree/myJEV-0.8B \\
  --revision 1c956c89d21c0ab136e98ffe66a16752fa37d823 \\
  > small-model-responses.jsonl

myjev score --artifact bahree/myJEV-4B \\
  --revision 38f7cca5a8530483309f576b0c3dd1756bc27c33 \\
  --input examples/demo-requests.jsonl --jsonl \\
  > cli-responses.jsonl
```

`run_demos.py` defaults to the pinned 4B revision shown above. `--artifact` also accepts a local package; use `--revision` for another Hub release. `--input` selects your own JSONL requests. Standard output contains ordinary API responses only; loader messages go to standard error.

The independent CLI batch exactly matched all seven default-model responses, including scores, confidence and revisions. The recorded execution used one A30 on GPU index 2, with two CPU threads and the existing pinned environment. These runs were for behavior and interface equivalence, not timing comparisons.

## Inspect and regenerate the evidence

- `requests.jsonl`: exact seven requests sent to each model.
- `expectations.json`: authored IDs, expected answers and explanations, never passed to the model.
- `4b-responses.jsonl`, `0.8b-responses.jsonl`, `4b-cli-responses.jsonl`: all unedited API responses.
- `summary.json`: requests, expectations and responses joined for inspection.
- `frozen.json`: pre-inference fixture/runner hashes and immutable model revisions.

Run `python scripts/summarize_demos.py` to regenerate this report from the retained outputs. It rejects changed frozen inputs and requires exact default Python/CLI equality.
'''
    (output/'report.md').write_text(body)
    print(json.dumps({'cases':7,'default_cli_jsonl_equal':True,'matches':summary['matches_authored_expectation']}))


if __name__=='__main__':main()
