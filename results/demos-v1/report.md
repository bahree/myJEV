# Try the released models on seven original examples

These examples were authored and frozen before inference. They include support routing, an explicit unsupported option, a refund-rule boundary, a quoted instruction attack and an original synthetic blog-format passage. No private archive text is included. Expected IDs are separate from model-visible inputs. Every response is retained.

| Case | Authored expected ID | 4B selected ID (confidence) | 0.8B selected ID (confidence) |
|---|---|---|---|
| billing | billing | billing (0.9834) | billing (0.9764) |
| technical | technical | technical (0.9669) | technical (0.9835) |
| none-of-these | other | other (0.9828) | other (0.8359) |
| refund-day-13 | approve | approve (0.9672) | approve (0.9891) |
| refund-day-14 | deny | deny (0.9835) | approve (0.9892) |
| quoted-instruction | billing | billing (0.9383) | billing (0.9688) |
| synthetic-blog-format | tutorial | tutorial (0.9975) | tutorial (0.9499) |

The default 4B model matched all seven authored expectations. The 0.8B model matched six. These are fixed diagnostic examples, not estimates of general accuracy, safety or calibration. Confidence was not recalibrated on these examples, and no deferral threshold was selected from them.

## A real confident failure

The refund instruction says: **Approve a refund only if fewer than 14 full days have passed since purchase. On day 14 or later, deny it. Apply this policy exactly.** The paired contexts differ only in 13 versus 14 days.

The 0.8B release selected `approve` on both contexts, with confidence 0.9891 on day 13 and 0.9892 on day 14. The day-14 answer contradicts the supplied rule. The 4B release changed from `approve` to `deny` and matched the authored expectation. High reported confidence did not make the smaller model's wrong answer correct. Passing the quoted-instruction example likewise does not establish resistance to other prompt injections.

## Run the same requests

From the installed source repository, with a supported GPU:

```bash
python scripts/run_demos.py > demo-responses.jsonl

python scripts/run_demos.py \
  --artifact bahree/myJEV-0.8B \
  --revision 1c956c89d21c0ab136e98ffe66a16752fa37d823 \
  > small-model-responses.jsonl

myjev score --artifact bahree/myJEV-4B \
  --revision 38f7cca5a8530483309f576b0c3dd1756bc27c33 \
  --input examples/demo-requests.jsonl --jsonl \
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
