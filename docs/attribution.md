# Upstream methods and attribution

Sources inspected on 2026-10-04. These are independent implementations, not a reproduction of TypeSafe's unpublished training recipe.

| Project | Pinned source | Architecture and objective |
|---|---|---|
| JevK5, Alibi Serikbay | [f26426d](https://github.com/allebee/jevk5/tree/f26426d16f59e8bbe1470e5b162cc89329e29b29) | Distilled supervised Qwen model; letter-token logits and fitted temperature. The prompt/readout credits TheoLeeCJ/SemIf. Up to sixteen choices use one pass; larger sets use combined passes. |
| JevForge, zwliJay | [6cf0dfe](https://github.com/zwliJay/jev-forge/tree/6cf0dfe3b811d623331c0cc8b1c00ff13614772e) | Causal backbone with a scalar head over candidate paths. Exact expected utility or sampled utility, plus directly differentiated multiclass Brier and forward KL. Its sampled baseline includes each sample in its group mean. |
| myJEV | This repository | Single context containing all descriptions; token aliases select the answer. Separate scalar and candidate-conditioned 21-value confidence heads. Exact and sampled joint answer/confidence policies share reward and KL; sampled training uses a leave-one-out baseline. |

No upstream model code is vendored. `scripts/reproduce_readout.py` imports a pinned external JevK5 prompt to compare its rendered chat template and restricted token projection with an ordinary full-vocabulary forward on the small untouched backbone. It does **not** reproduce JevK5's distilled checkpoint accuracy. The output is recorded separately from natural-data experiments. JevForge's objective was studied from `jevforge/rlcd.py`; its published benchmark is not a myJEV result.

Reference models: [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [4B](https://huggingface.co/Qwen/Qwen3.5-4B), [9B](https://huggingface.co/Qwen/Qwen3.5-9B). Immutable revisions are in `configs/`. Backbone licensing and upstream notices continue to apply to adapters and any future merged release.

OpenJev was inspected as related work. See [the inclusion decision](openjev.md) for its distinct backbone, confidence formula, candidate limit and execution status. No local OpenJev inference result is included in this study.

## Related inference tutorial

Avi Chawla, [Build your own Jev (100% local)](https://blog.dailydoseofds.com/p/build-your-own-jev-100-local), September 22, 2026, demonstrates single-token candidate scoring through SGLang. It explicitly separates the inference mechanism from Jev training/calibration and distinguishes restricted selection probabilities from empirical correctness. We cite it as related work, not a reproduction of its benchmark or an independent review of myJEV. It motivates the [planned serving comparison](hosting.md#scoring-versus-generation-benchmark); no tutorial code or timing results are incorporated as myJEV evidence.

## Backbone comparison across related implementations

This follow-up review covers the linked projects and additional Hugging Face and Cloudflare releases. These are source-described architectures, not locally reproduced benchmark results. Current upstream pages can change; this comparison does not replace the pinned sources above or change the running experiment.

A decision interface describes the outputs and how they are computed. It can reuse a pretrained language model while returning scores without autoregressive text generation. Hugging Face hosts several unrelated projects called OpenJev; the full repository name matters.

| Implementation and primary source | Documented backbone | Decision mechanism |
|---|---|---|
| [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Not named in the announcement reviewed | Typed decisions, parallel sampler and RLCD described. An LLM origin cannot be confirmed from this source. |
| [JevK5](https://github.com/allebee/jevk5) | Qwen3.5-4B and 9B | Distilled LoRA adaptation, answer-token logits and temperature calibration. |
| [JevK5-Lite](https://github.com/allebee/jevk5#jevk5-lite) | DeBERTa-v3-large encoder | Separate experimental encoder variant for supplied label sets. This is not the Qwen decoder architecture. |
| [JevForge](https://github.com/zwliJay/jev-forge) | Qwen3.5-0.8B in the documented small configuration | Shared prefix and candidate branches with a scalar scoring head; supervised and exact/sampled optimization. |
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | Qwen3.5-4B reference configuration | Frozen model with option-token readout; task-specific fine-tuning is not required. |
| [openjev/openjev](https://huggingface.co/openjev/openjev/blob/ac97900fd034fdd7e7e536f3d4c21b836cae0750/NOTICE) | Qwen3.8-27B | Fine-tuned backbone with option readout. See the [pinned assessment](openjev.md) for confidence semantics and multi-pass limits. |
| [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) | Qwen3.5 family, including 0.8B, 2B, 4B and a 35B-A3B variant | Cross-entropy-trained three-class NLI head. Scores premise/hypothesis pairs per option; shared-prefix batching does not make it one joint candidate readout. |
| [shenjunhao/OpenJev-4B](https://huggingface.co/shenjunhao/OpenJev-4B) | Qwen3.5-4B text backbone | Option Set Interactor and shared decision head; full-parameter SFT followed by REINFORCE-Analysis. |
| [Cloudflare Clef](https://huggingface.co/Cloudflare/clef) | Qwen3.8-27B, retaining vision encoder | Joint schema head scores the allowed options across questions. |
| [Cloudflare Clef-flash](https://huggingface.co/Cloudflare/clef-flash) | Qwen3.5-9B, retaining vision encoder | Same family of joint schema head, smaller backbone. |
| [Avi Chawla tutorial](https://blog.dailydoseofds.com/p/build-your-own-jev-100-local) | Qwen2.5-0.5B-Instruct | Untouched model with SGLang token scoring; demonstrates inference mechanics. |
| myJEV | Qwen3.5-0.8B, 4B and 9B | LoRA/QLoRA with alias selection and a separate correctness head/policy. |

Cloudflare's [training description](https://blog.cloudflare.com/clef-decision-models/) specifies frozen Qwen backbones, rank-256 adapters and a learned routing head, using label-smoothed cross-entropy plus Brier loss and a secondary RLCD objective. That supports the rationale for adapting a language backbone, but its joint schema architecture and objective differ from myJEV's answer/correctness policy. Published latency and quality claims are not evidence on our A30s.

### Consequences for this study

- Keep the frozen main comparison unchanged. Newly found architectures belong in related work and explicitly scoped follow-up experiments.
- The AlexWortega **4B v5** model card discloses training on BANKING77 and CLINC150 **test splits**. Exclude that checkpoint from held-out comparisons on those datasets. Do not automatically extend that finding to other versions without checking their data manifests.
- Compare the number of candidate branches, backbone reads and supported questions, rather than treating every non-generative API as identical computational work.
- A normalized option distribution, an NLI entailment score and a separately estimated probability of answer correctness have different meanings. Measure calibration for the actual deployed output.
- Audit dataset exposure, licensing, pinned revisions and output equivalence before adding any external checkpoint. None of these newly reviewed weights was downloaded or evaluated for this review.
