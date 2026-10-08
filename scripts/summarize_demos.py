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

These are seven requests to the same trained model. They walk from ordinary support routing to a rule with a precise boundary, then change the task to classifying a short post. Each request supplies its own instructions and candidate answers. There is no retraining between requests.

If you have not installed myJEV, start with the [quick start](../../docs/quickstart.md#install-and-test). You can read all the saved requests and responses below without installing anything or downloading weights.

## What each request asks

The line numbers below match `examples/demo-requests.jsonl` and every saved response file.

| Line and case | Text the model reads | Why we expect this answer |
|---|---|---|
| 1. `billing` | “I was charged twice.” | The supplied billing description covers charges, invoices and refunds. |
| 2. `technical` | “The mobile app crashes every time I open the settings screen.” | An application error belongs to the supplied technical route. |
| 3. `none-of-these` | “Please recommend a trail for a weekend hike.” | Neither support route applies; the instruction explicitly says to choose `other` in that situation. |
| 4. `refund-day-13` | A refund requested 13 full days after purchase. | The rule permits refunds only before 14 full days, so choose `approve`. |
| 5. `refund-day-14` | The same request after 14 full days. | The rule explicitly denies refunds on day 14 or later, so choose `deny`. |
| 6. `quoted-instruction` | A quote says to choose technical, followed by the actual issue: a duplicate charge. | The instruction says to treat the quote as untrusted text and route the actual issue, so choose `billing`. |
| 7. `synthetic-blog-format` | “How to back up a folder,” with four steps for copying and checking a backup. | The available formats are tutorial, opinion and announcement. A procedure for completing a task fits `tutorial`. |

The first two requests change the customer issue while keeping the routes fixed. The third introduces an outside category: choosing `other` is a decision the model can be confident about, distinct from asking for review because it is unsure. Requests four and five change only the number of elapsed days, so the answer should change at the rule's boundary.

Request six puts an instruction inside the text being inspected. This is a small prompt-injection example: that text tries to redirect the model. The final request changes both the instruction and the candidate list. It shows how to use the same interface for another task. The short post was written for this demo; it is not an item from the archive study.

I wrote down the requests and expected answers before running inference. The expectations are in a separate file and are never passed to the model. Every response, including the mistake, is retained.

## Recorded answers

Each result below gives the selected ID and its reported correctness confidence, rounded to four decimal places. For example, 0.9834 means the model reports about 98.34% confidence in its chosen answer; the next section shows why that number still needs checking.

| Case | Authored expected ID | 4B selected ID (confidence) | 0.8B selected ID (confidence) |
|---|---|---|---|
{table}

The default 4B model matched all seven authored expectations. The 0.8B model matched six. These are fixed diagnostic examples, not estimates of general accuracy, safety or calibration. Confidence was not recalibrated on these examples, and no deferral threshold was selected from them.

## A real confident failure

The refund instruction says: **Approve a refund only if fewer than 14 full days have passed since purchase. On day 14 or later, deny it. Apply this policy exactly.** The paired contexts differ only in 13 versus 14 days.

The 0.8B release selected `approve` on both contexts, with confidence 0.9891 on day 13 and 0.9892 on day 14. The day-14 answer contradicts the supplied rule. The 4B release changed from `approve` to `deny` and matched the authored expectation. High reported confidence did not make the smaller model's wrong answer correct. Passing the quoted-instruction example likewise does not establish resistance to other prompt injections.

## Run the same requests

From the installed source repository, with a supported GPU, run the pinned 4B release:

```bash
python scripts/run_demos.py > demo-responses.jsonl
```

The runner loads the model once, then scores the seven requests in file order. JSONL means one JSON object per line. The output file has one response for each input line, so line 1 answers the duplicate-charge request and line 5 answers the day-14 refund. To inspect the first response with indentation:

```bash
head -n 1 demo-responses.jsonl | python -m json.tool
```

To compare with the smaller release, run:

```bash
python scripts/run_demos.py \\
  --artifact bahree/myJEV-0.8B \\
  --revision 1c956c89d21c0ab136e98ffe66a16752fa37d823 \\
  > small-model-responses.jsonl
```

The regular CLI can read the same request file:

```bash
myjev score --artifact bahree/myJEV-4B \\
  --revision 38f7cca5a8530483309f576b0c3dd1756bc27c33 \\
  --input examples/demo-requests.jsonl --jsonl \\
  > cli-responses.jsonl
```

`run_demos.py` defaults to the pinned 4B revision shown above. `--artifact` also accepts a local package; use `--revision` to select a specific version of another release on the Hugging Face Hub. `--input` selects your own JSONL requests. Standard output contains only API responses, so `>` saves those responses to the named file. Loading messages go to standard error and can still appear in the terminal.

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
