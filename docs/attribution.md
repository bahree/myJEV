# Upstream methods and attribution

Sources inspected on 2026-10-04. These are independent implementations, not a reproduction of TypeSafe's unpublished training recipe.

| Project | Pinned source | Architecture and objective |
|---|---|---|
| JevK5, Alibi Serikbay | [f26426d](https://github.com/allebee/jevk5/tree/f26426d16f59e8bbe1470e5b162cc89329e29b29) | Distilled supervised Qwen model; letter-token logits and fitted temperature. The prompt/readout credits TheoLeeCJ/SemIf. Up to sixteen choices use one pass; larger sets use combined passes. |
| JevForge, zwliJay | [6cf0dfe](https://github.com/zwliJay/jev-forge/tree/6cf0dfe3b811d623331c0cc8b1c00ff13614772e) | Causal backbone with a scalar head over candidate paths. Exact expected utility or sampled utility, plus directly differentiated multiclass Brier and forward KL. Its sampled baseline includes each sample in its group mean. |
| myJEV | This repository | Single context containing all descriptions; token aliases select the answer. Separate scalar and candidate-conditioned 21-value confidence heads. Exact and sampled joint answer/confidence policies share reward and KL; sampled training uses a leave-one-out baseline. |

No upstream model code is vendored. `scripts/reproduce_readout.py` imports a pinned external JevK5 prompt to compare its rendered chat template and restricted token projection with an ordinary full-vocabulary forward on the small untouched backbone. It does **not** reproduce JevK5's distilled checkpoint accuracy. The output is recorded separately from natural-data experiments. JevForge's objective was studied from `jevforge/rlcd.py`; its published benchmark is not a myJEV result.

Reference models: [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [4B](https://huggingface.co/Qwen/Qwen3.5-4B), [9B](https://huggingface.co/Qwen/Qwen3.5-9B). Immutable revisions are in `configs/`. Backbone licensing and upstream notices continue to apply to adapters and any future merged release.

OpenJev was subsequently inspected as an external comparison at the user's request. See [the inclusion decision](openjev.md) for its distinct backbone, confidence formula, candidate limit and execution status.

## Related inference tutorial

Avi Chawla, [Build your own Jev (100% local)](https://blog.dailydoseofds.com/p/build-your-own-jev-100-local), September 22, 2026, demonstrates single-token candidate scoring through SGLang. It explicitly separates the inference mechanism from Jev training/calibration and distinguishes restricted selection probabilities from empirical correctness. We cite it as related work, not a reproduction of its benchmark or an independent review of myJEV. It motivates the [planned serving comparison](hosting.md#planned-scoring-versus-generation-benchmark); no tutorial code or timing results are incorporated as myJEV evidence.
