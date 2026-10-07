# Numeric readout: seven-case untouched-backbone probe

Selection probabilities are not an independently calibrated correctness estimate. No generalization, architecture-superiority or speed claim is supported by these seven previously viewed examples.

| Case | Expected | Existing aliases | Numeric suffix |
|---|---|---|---|
| billing | billing | billing | billing |
| technical | technical | technical | technical |
| none-of-these | other | other | other |
| refund-day-13 | approve | approve | approve |
| refund-day-14 | deny | deny | deny |
| quoted-instruction | billing | billing | billing |
| synthetic-blog-format | tutorial | tutorial | tutorial |

Correct: aliases 7/7; numeric 7/7.

No fine-tuning. The same untouched pinned BF16 4B backbone was used. Prompt format and answer readout both change, so this does not isolate bracket tokens alone. Numeric scoring includes the closing bracket and sums full-vocabulary suffix log probabilities; it does not length-normalize.

The implementation repeats the prompt across candidate continuations and teacher-forces full sequences. It does not reproduce cached prefill/branch execution and is not a latency comparison. Raw execution timings are retained only as logs. Token-boundary checks fail closed.

Recommendation: retain this as an implementation demonstration. A separately frozen broader readout comparison is needed before changing release architecture. Numeric identifiers allow more encodable labels, not infinite context or automatic multimodal support. The paper's tree-local training distribution also need not equal its globally normalized inference distribution.
