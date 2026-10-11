# Compare Decision 3.0 with myJEV

Decision 3.0 supplies another way to turn Qwen representations into choices. The vLLM Semantic Router team publishes its weights and inference code on [Hugging Face](https://huggingface.co/collections/vllm-sr/decision-30). I selected `d3-lite` and `d3-mini` because they start from the Qwen3.5-0.8B and 4B families used in our study. We compare each with the supervised and RL myJEV releases at the corresponding size.

The [frozen protocol](../configs/decision3-v1.json) identifies every checkpoint and input file. These models have different training histories, prompts and output layers. A better result on the same request would establish a difference between the released systems; it would not tell us which individual design choice caused it. Decision 3.0's exposure to these public benchmarks during training is unknown.

## Follow a request through both models

For “I was charged twice,” the application supplies `billing`, `technical` and `other`. Both implementations attach short, single-token codes to the choices inside the prompt. Those codes let a final input representation produce one score per allowed answer, without generating answer text.

Our released myJEV model selects rows from Qwen's existing vocabulary projection. The added heads estimate candidate correctness. The separate [candidate-attention experiment](candidate-head.md) replaces answer selection with a trainable attention layer over frozen Qwen features.

Decision 3.0 instead loads a separate matrix with 255 output rows from `readout.safetensors`. Multiplying the final token's feature vector by that matrix gives answer-code scores. It masks out codes beyond the current candidate count and normalizes the remaining scores. A 255-row output layer allows a request to supply different choices; those rows do not name 255 permanent business categories. Our implementation already projects only the supplied aliases, so a narrower stored output matrix alone does not establish a speed advantage over myJEV.

| Choice | Released myJEV | Candidate-attention experiment | Decision 3.0 |
|---|---|---|---|
| Answer scores | Qwen vocabulary rows for supplied aliases | Shared scorer after candidate-to-context attention | Separate 255-row answer-code projection |
| Backbone | Qwen with LoRA adapters | Frozen Qwen | Released Qwen-derived weights |
| Confidence | Calibrated selected probability or learned RL policy expectation | Temperature-scaled selected probability | API field measures one minus normalized entropy |
| Questions per model input | One | One | One; several questions can be batched |

The reviewed D3 runtime also changes attention. Its full-attention layers can read both earlier and later prompt tokens. The Gated DeltaNet layers retain their causal recurrence, so it would be inaccurate to describe the whole network as a bidirectional encoder. Our released Qwen path retains the backbone's causal behavior. The D3 code does not establish whether its training used LoRA or full weight updates; the published package contains inference weights and a readout.

## Read confidence according to its definition

Suppose a two-option request returns probabilities 0.9 and 0.1. The first answer has selected probability 0.9. D3's `confidence` is about 0.531 because it measures how concentrated the distribution is:

$$c_{\text{entropy}} = 1 - \frac{H(p)}{\ln K},\qquad H(p)=-\sum_{k=1}^{K}p_k\ln p_k.$$

Here $K$ counts options. A uniform distribution gives zero; all probability on one option gives one. For our example, $H(p)\approx0.3251$ and $\ln 2\approx0.6931$, giving $1-0.3251/0.6931\approx0.5310$. The calculation uses only the distribution and never observes whether the answer was correct.

The comparison retains that field as `native_confidence` and reports its ability to rank correct answers above errors. It uses selected-option probability for common Brier and calibration comparisons. A second view fits the same positive temperature procedure to each model's calibration probabilities. Our RL policy confidence is reported separately. Carrying an application threshold from one of these fields to another would change its meaning.

## What runs on an NVIDIA A30

The team name refers to vLLM Semantic Router. The reviewed release runs through Transformers and custom PyTorch code; it does not provide an automatic conversion of our models into the vLLM serving engine. Its AMD fast path checks for ROCm. On the A30, D3 uses its standard CUDA path. Our comparison verifies one backbone invocation per request and zero generated output tokens for D3.

Both backbones use BF16 and the PyTorch reference implementations for causal convolution and the gated delta rule. D3 requires Transformers 5.17.0 for this comparison; myJEV retains the release's 5.18.0 environment. PyTorch is 2.14.1 on both paths. The D3 vision tower remains loaded, although the study sends only text. No image or video accuracy is measured.

Publisher latency numbers use AMD MI325X hardware. The local benchmark measures one model at a time on A30 GPU 0 after the concurrent quality jobs finish. It records synchronized scoring time, backbone GPU time and peak memory for short demonstrations and 77-candidate banking requests. Loading time includes D3's full package hash verification. The measurements exclude HTTP and downloading.

## Run the comparison

Use the [reference environment](quickstart.md#install-and-test) for myJEV. Create a separate Linux environment for D3 with the same requirements except `transformers==5.17.0`:

```bash
python3.12 -m venv .venv-decision3
.venv-decision3/bin/python -m pip install -r requirements.decision3.lock
.venv-decision3/bin/python -m pip install --no-deps .
```

The recorded host reused the reference environment's packages through a `.pth` file and installed that Transformers version locally; it did not change the released runtime. The commands above give a self-contained installation instead. The run receipts list the imported versions. Neither optimized FLA nor `causal-conv1d` was installed.

Prepare BANKING77 and CLINC150 using their [dataset cards](README.md#data-and-provenance). Download the two complete D3 snapshots, including their example assets: upstream integrity verification checks those files too. Model weights and cards remain under their upstream Apache 2.0 license.

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf download vllm-sr/d3-lite --revision a8d25a91ef75552803f7009d25b7bd435991abd9
hf download vllm-sr/d3-mini --revision a475367de0d8133a3c1a2a0dc12303d89cfa2ab4

# Activate the D3 environment:
source .venv-decision3/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_decision3.py \
  --model d3-lite --phase quality --device cuda:0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_decision3.py \
  --model d3-mini --phase quality --device cuda:0

# Switch back to the installed reference environment:
source .venv/bin/activate
python scripts/run_decision3.py --model myjev-0.8b --phase quality --device cuda:0
python scripts/run_decision3.py --model myjev-0.8b-rl --phase quality --device cuda:0
python scripts/run_decision3.py --model myjev-4b --phase quality --device cuda:0
python scripts/run_decision3.py --model myjev-4b-rl --phase quality --device cuda:0
```

Each run validates the frozen data hashes and saves predictions as it goes. `--resume` requires matching protocol and runner hashes. Run each model again with `--phase benchmark` in its corresponding environment while the GPU is otherwise idle, then regenerate the report:

```bash
python scripts/summarize_decision3.py --require-benchmark
```

`scripts/run_decision3_study.py` automates the three-GPU schedule and subsequent isolated benchmarks when the reference environment and `.venv-decision3` are both available. Its optional `--wandb` flag uses the existing [tracking configuration](tracking.md). Local prediction and progress files remain authoritative if cloud tracking is unavailable.

## Sources

- [d3-lite card and pinned files](https://huggingface.co/vllm-sr/d3-lite/tree/a8d25a91ef75552803f7009d25b7bd435991abd9).
- [d3-mini card and pinned files](https://huggingface.co/vllm-sr/d3-mini/tree/a475367de0d8133a3c1a2a0dc12303d89cfa2ab4).
- [Runtime, including entropy confidence and attention masks](https://huggingface.co/vllm-sr/d3-mini/blob/a475367de0d8133a3c1a2a0dc12303d89cfa2ab4/d3_runtime.py).
- [AMD fast-path device checks](https://huggingface.co/vllm-sr/d3-mini/blob/a475367de0d8133a3c1a2a0dc12303d89cfa2ab4/d3_fast.py).
