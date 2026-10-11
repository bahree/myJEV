# Build and run the scratch decision model

I built a small network from random weights to follow how a decision model learns. We start with a controlled task: read a color signal and choose the matching option. The network receives context and candidate descriptions, converts the text to numbers, and learns to score each choice. This guide follows those steps and the failures we found. The [Qwen study](qwen-findings.md) starts with language knowledge learned during pretraining, which the scratch model has to do without.

You can [generate data and train this model on CPU](#generate-and-train-locally) without downloading pretrained weights. First install and activate the environment in the [quick start](quickstart.md#install-and-test). If you only want to inspect a saved result, the [recorded scratch response](../results/scratch-deployment/equivalence-and-model-timings.json) is available without installation.

The model converts text to UTF-8 bytes and gives each byte an integer ID. For `red`, the byte values are 114, 101 and 100. Our encoder adds two to those values and prepends a start marker, giving `[1, 116, 103, 102]`. The byte IDs run from 2 to 257. ID 1 is the beginning-of-sequence marker (BOS); ID 0 is padding (PAD), the empty slots used to make sequences in a batch the same length.

Each ID selects a learned vector, an array of numbers. The **encoder** combines information across text positions to update those vectors. We use them to compare the request with the candidate descriptions, then score each choice. Qwen instead reads scores for the short aliases assigned to candidates, as the [architecture guide](architecture.md#one-forward-pass) shows.

## Trace one decision through the network

The implemented fixture configuration has two encoder blocks, width 64, four attention heads and a feed-forward width of 256. With candidate interaction enabled it has **201,175 parameters**, smaller than the original 5-20M exploratory design target. This size let us inspect failures in padding masks, training and performance on new examples with less computation. A learned subword tokenizer and a larger encoder remain optional extensions.

A **tensor** is an array with several axes. Here B counts requests in a batch, K counts candidates, L is the context length and D is the candidate-description length. The final 64 in a shape is the number of learned features per token or candidate. For one request with three candidates, `1 x 3 x 64` stores three vectors of 64 numbers. More generally, `B x K x 64` stores one such vector for each candidate in each request.

To turn a whole description into one vector, we average its encoded token vectors and leave out the padding. That operation is called **mean pooling**. Each resulting candidate vector then gathers information from the context through attention. Four attention heads perform that calculation in parallel, each using 16 features. These internal attention operations are separate from the output heads that produce scores or confidence.

The arrays change as follows:

| Operation | Tensor shape | What is shared |
|---|---|---|
| Context bytes and learned token positions | B x L x 64 | Byte embedding and encoder weights |
| Candidate bytes, padded and flattened for encoding | (B*K) x D x 64 | Same embedding and encoder |
| Mean-pool each description | B x K x 64 | Padding excluded from pooling |
| Candidates attend to context tokens | B x K x 64 | One cross-attention module |
| Candidates attend to the candidate set | B x K x 64 | No candidate-index positions |
| Scalar score per candidate | B x K | One shared selection head |
| Scalar correctness / 21-bin confidence policy | B x K / B x K x 21 | Separate confidence heads |

One decision calls the same text encoder twice: once for the context and once for all the candidate descriptions together. Both calls use the same learned weights. Qwen reads a combined prompt containing all those fields in one backbone pass. Candidate count and description length still increase computation. Parameter sharing does not make processing cost constant.

Token positions inside a description matter; positions in the candidate list do not receive embeddings. Caller-supplied IDs remain outside the network. The score head operates on each contextualized candidate, so permuting candidates should permute the outputs. Tests check that behavior in evaluation mode, including confidence heads and mixed-length batches.

The fixture limits are 256 UTF-8 bytes for instructions plus newline plus context, 64 bytes per candidate description, and 32 candidates. The scratch encoder counts bytes. The Qwen interface uses its tokenizer to count tokens. Oversized input is rejected. Training and quality evidence initially use four candidates; accepting 32 does not establish useful accuracy at that size.

## Start with selection, then confidence

Selection-only training optimizes cross-entropy. Its saved artifact reports `confidence: null`, because the confidence heads have not been trained. Temperature fitting on reserved calibration data can supply selected-option probability as an explicitly named confidence source.

Supervised confidence training adds a scalar correctness loss for the currently selected answer and a Brier term for the candidate-conditioned confidence policy. The selected-answer target is detached from selection; confidence gradients still update the encoder. Exact and sampled RL start from the same supervised artifact and use the shared finite-action objective with a frozen reference. Padded candidates are sliced out before the objective so zero-probability padding cannot create undefined KL arithmetic.

## Reproduce the first learning failure

The original 100-update selection pilot learned to answer examples arranged like its training inputs. Moving the same fields to a different layout reduced accuracy. Restoring the validation inputs to the training layout recovered some of it:

| Diagnostic | Accuracy |
|---|---:|
| First 256 training rows | 85.5% |
| Held-out validation layout | 20.7% |
| Same validation rows restored to training layout | 68.4% |

The comparison points to text layout as one contributor. In variant 2, I varied the layouts and the lengths of irrelevant record strings, called **nonces**. A 500-update selection pilot reached **78.9% validation accuracy**. I changed both the examples and the amount of training, so the result cannot tell us how much each change contributed. The original failed pilot remains in `results/scratch-v1/`; the revised pilot is in `results/scratch-v2/`.

## Generate and train locally

Use the existing installation in the [quick start](quickstart.md). Commands below access no external model service or pretrained weights. Data and run outputs must be new directories; completed artifacts are not silently overwritten.

```bash
python scripts/run_scratch.py data --version 2 --output data/scratch-routing-v2
python scripts/run_scratch.py train \
  --data data/scratch-routing-v2/train.jsonl \
  --output artifacts/my-scratch-sft --method sft \
  --updates 1000 --batch-size 8 --seed 11 --device cpu
python scripts/run_scratch.py evaluate \
  --artifact artifacts/my-scratch-sft/artifact \
  --data data/scratch-routing-v2/test.jsonl \
  --calibration data/scratch-routing-v2/calibration.jsonl \
  --temperature --output results/my-scratch-temperature --device cpu
```

Use `--device cuda:0` for a GPU. The frozen synthetic comparison is configured in `configs/scratch-study-v1.json` and run with `python scripts/run_scratch_study.py`. It uses two SFT learning-rate trials, then three seeds and matched continuation exposure. Unlike Qwen, this small study shares the selected supervised learning rate across methods, which limits cross-study conclusions. The runner refuses to overwrite an existing study and does not yet resume partial optimizer state; training histories flush every update and full artifacts save at stage completion.

The dataset has explicit conditional probabilities and sampled latent labels. Those probabilities are evaluator-only metadata, never model inputs. A quarter of examples have two equally likely signals. The test partition additionally holds out ambiguous color pairings. The measurement covers transfer between the controlled rules and templates. Natural-language understanding needs a separate test. See the [synthetic card](datasets/scratch-routing.md).

## Score a saved artifact

```python
import json
from myjev import DecisionModel
model = DecisionModel.load("artifacts/my-scratch-sft/artifact", device="cpu")
print(model.score(json.load(open("examples/scratch-request.json"))))
```

```bash
myjev score --artifact artifacts/my-scratch-sft/artifact \
  --device cpu --input examples/scratch-request.json
MYJEV_ARTIFACT=artifacts/my-scratch-sft/artifact MYJEV_DEVICE=cpu \
  OMP_NUM_THREADS=2 uvicorn myjev.server:app_factory --factory \
  --host 127.0.0.1 --port 8000 --no-access-log
curl -s http://127.0.0.1:8000/score -H 'Content-Type: application/json' \
  --data-binary @examples/scratch-request.json
```

The shared loader dispatches by the scratch manifest format. It verifies the manifest and weight hashes, then loads the complete small network. It does not load Qwen or an adapter. The same HTTP queue, readiness and timeout behavior applies. The existing [Docker recipe](inference.md#gpu-docker) can mount a scratch artifact; set `MYJEV_DEVICE=cpu` for CPU service or assign one GPU and set `cuda:0`.

## Completed controlled study and its limits

The frozen study used 13,000 updates in total. SFT labels supervised learning from known answers in the result files. Main runs used three seeds, which control random choices such as initial weights and example order. Each run had 1,000 initial SFT updates and separate 1,000-update continuations at batch size 8. The supervised learning rate was chosen using validation, then shared across methods. The smaller tuning budget limits what can be inferred about each method’s best attainable result.

| Method | Mean test accuracy | Seed SD |
|---|---:|---:|
| SFT | 60.74% | 30.66 percentage points |
| Continued SFT | 66.34% | 33.61 percentage points |
| Exact RL | 51.43% | 22.13 percentage points |
| Sampled RL | 40.62% | 11.45 percentage points |

Continued SFT scored 85.16%, 27.54% and 86.33% across its three seeds. One successful run would give a misleading picture of how reliable this recipe is. RL did not improve the mean. Data coverage, optimizer settings and objective weighting are candidates for another experiment; none was retuned using these test outcomes.

The [complete synthetic report](../results/scratch-study-v1/summary.md) includes scalar, temperature and constant controls, and confidence error against known conditional probabilities. Temperature scaling was helpful in the Qwen study but did not beat the scalar head on this shifted synthetic test. That difference reinforces the need to measure each deployment distribution. Twelve completed training histories were imported into W&B as historical runs, with original step/time axes and no invented system telemetry; URLs are in the evidence folder.

## Natural language exposed a harder failure

The separate [BANKING77 diagnostic](../results/scratch-banking-diagnostic/report.md) trained a randomly initialized 217,559-parameter version with a 512-byte context budget, FP32 and 8,000 examples of exposure. It had one seed, a fixed learning rate and no natural-language tuning. On all 3,080 test examples it selected the same class, scoring **1.30% accuracy**. It is not a usable classifier under this setup.

Its scalar correctness Brier was only **0.0128**. A low score here is misleading if viewed without accuracy: the model was almost always wrong and reported low confidence. A model that rarely succeeds has a different correctness outcome distribution from a competent model. Brier alone cannot rank deployment utility across them. Calibration also cannot repair wrong answer selection.

The failure belongs to this configuration and training budget. Tokenization, capacity, initialization and optimization are all possible contributors, while Qwen also brings language pretraining. The synthetic rule results did not tell us whether this model could learn banking intents; the language test exposed that gap.

## Measured deployment check

The engineering check uses the seed-11 synthetic SFT artifact with 201,175 parameters. Its recorded request asks for a color. The failed 217,559-parameter BANKING77 model above is a separate checkpoint trained on different data. No scratch artifact is recommended for production. It contains 809,244 bytes of weights. Python/CLI outputs were identical on the recorded request; Docker GPU HTTP probabilities differed from CPU by less than 1e-8, and the selected ID and artifact revision matched. Oversized HTTP input returned 422.

| Path | Warm p50 | Warm p95 |
|---|---:|---:|
| CPU model, two threads | 1.97 ms | 2.01 ms |
| A30 model | 2.94 ms | 2.99 ms |
| GPU Docker HTTP, concurrency 1 | 4.58 ms | 5.28 ms |
| GPU Docker HTTP, concurrency 4 | 16.81 ms | 18.09 ms |

Each measurement has 100 requests using one short four-candidate input. Both HTTP runs returned 100/100 HTTP 200. GPU peak allocated memory for the local model check was about 9.6 MiB; this excludes driver/context overhead and is not total process VRAM. GPU startup and kernel overhead can outweigh its benefit on such a small request, so CPU being faster here is plausible and workload-specific. These are not matched Qwen or Jev benchmarks. The container uses the shared research dependencies and is much larger than the tiny model weights; no minimal-runtime image-size claim is made.

The [raw deployment evidence](../results/scratch-deployment/equivalence-and-model-timings.json), HTTP timing records and local image ID are retained. The test container was removed after validation; no persistent public endpoint or registry release was created.

## What remains separate

Controlled synthetic behavior does not establish useful BANKING77 quality, unfamiliar natural-language competence or Jev-equivalent speed. The [design and validation guide](scratch-plan.md) distinguishes the natural-language diagnostic from additional candidate-count/length checks and release requirements. Compare complete scratch weights with the entire Qwen deployment, including its backbone, and keep synthetic results outside the BANKING77 accuracy table.

## Packaged teaching checkpoint

The separate local `artifacts/scratch-teaching-release-v1/` package contains the seed-11 initial supervised network's full weights, manifest, MIT license, dataset provenance and [model card](../results/scratch-teaching-release-v1/model-card.md). Seed 11 follows the packaging convention; its relatively strong individual synthetic score does not replace the unstable three-seed results above. The package holds the synthetic checkpoint. The failed BANKING77 diagnostic and six Qwen adapter releases are separate artifacts.

The [release manifest](../results/scratch-teaching-release-v1/manifest.json) records every package-file checksum. [CPU verification](../results/scratch-teaching-release-v1/verification.json) confirms byte-identical weights/manifest, exact reload response equality with the original artifact, acceptance at 256 context bytes / 64 description bytes / 32 candidates, and rejection of each limit plus one. Those shape checks do not establish accuracy at the bounds. The package remains local; no downloadable scratch model is claimed.

## Diagnostic ladder after the original failure

I tested whether the model could at least memorize a tiny training set before changing the architecture. It reached 100% in all three seeds. An initial deterministic fixture then exposed a data mistake: numeric record indices correlated with labels. A successful run could exploit that shortcut, so the original fixture and failed seeds remain in the records, followed by the corrected test.

The [corrected protocol](../results/scratch-ladder-v2/report.md) pairs each irrelevant record string with all four color labels. Its value therefore cannot identify the answer. Strings used in the test examples are kept out of training. With the same architecture, learning rate and per-run budget, seeds 11/22/33 each recovered the explicit color rule on all 64 held-out decisions across 16 nonce groups. The check tests rule recovery within one template. It does not repair the original noisy/layout-shift task or the 1.30% BANKING result.

The extension used 2,700 total updates including the preserved confounded attempt. The network learned to memorize a small set and recover a simple signal. Learning to handle other layouts, ambiguous examples and natural-language banking requests remains unresolved by this diagnostic.
