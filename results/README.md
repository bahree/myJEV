# Study results

These files record what the experiments measured. You do not need them to run inference: the [quick start](../docs/quickstart.md) downloads model artifacts from Hugging Face separately.

Start with a report, then inspect its measurements if you want to check a claim. Large prediction files and the detailed short-pilot records are optional [evidence downloads](https://github.com/bahree/myJEV/releases/tag/evidence-v1). Reports, plots, training logs and release checks remain in this repository.

## Read the findings

| Question | Start here | What the evidence covers |
|---|---|---|
| What changed with training method and model size? | [Qwen findings](../docs/qwen-findings.md), [longer-study summary](longer-v1/summary.md) | Three sizes and three seeds; the full official BANKING77 test set |
| How much do the seeds matter? | [Seed contrasts](review-seeds-v1/report.md) | Per-seed gains and losses alongside intervals conditional on the observed seeds |
| Does RL still help when calibration is matched? | [Matched calibration](review-calibration-v1/report.md) | The same post-hoc fitting opportunities for all four methods |
| What happens on unfamiliar requests? | [Transfer and robustness](generalization-v1/report.md), [paired contrasts](extension-analysis-v1/report.md) | CLINC150, precision controls and frozen evaluation conditions |
| Does changing candidate order change answers? | [Order report](review-order-v1/report.md), [requests affected across orders](review-order-union-v1/report.md) | Recorded per-request permutations; no claim about every possible order |
| Did the scratch model learn useful rules? | [Scratch study](scratch-study-v1/summary.md), [BANKING failure](scratch-banking-diagnostic/report.md) | Controlled synthetic learning and the failed natural-language diagnostic |
| What did archive adaptation change? | [Archive report](archive-machine-v2/report.md), [paired changes](archive-machine-v2/paired-changes.md) | Agreement with machine labels and forgetting; independent human auditing remains absent |
| Could a simpler classifier do the job? | [TF-IDF](tfidf/), [ModernBERT](encoder-control-v1/report.md) | Fixed-taxonomy controls with different interfaces and training budgets |
| Does a different decision head help? | [Unsloth comparison](../docs/unsloth.md), [recorded results](unsloth-head-v1/report.md) | Matched 0.8B prompt/readout study; pilot measurements separated from quality results |
| What does a real response look like? | [Seven demos](demos-v1/report.md) | Saved requests and responses, including a confident mistake |
| What does serving cost? | [Model and serving comparison](../docs/models.md), [container evidence](container-registry-v3/README.md) | Measured hardware, precision, startup and latency conditions |

For the short pilot, read the [experiment guide](../docs/experiments.md), [summary](study-summary.json) and [figures](figures/). Its 100-update stages and 256-example evaluation differ from the longer study. Its detailed records are in the `pilot` download.

## Restore inputs for a CPU replay

No model download, API key or GPU is needed to retrieve the evidence. From the repository root, using Python 3.12:

```bash
# All four bundles; verifies archive and individual-file SHA-256 hashes.
python3 scripts/fetch_evidence.py

# Or just the inputs needed for matched calibration.
python3 scripts/fetch_evidence.py --bundle calibration

# Check installed evidence without downloading or writing anything.
python3 scripts/fetch_evidence.py --check
```

The command restores original `results/` paths, so analysis scripts keep their input names. It refuses to overwrite different local files. Downloaded inputs are ignored by Git. Keep a separate checkout for new experiments if you want to compare them with these saved results.

| Bundle | Download size | Contents | Example replay after installation |
|---|---:|---|---|
| `pilot` | 65.0 MiB | 384 files: short-pilot metrics, predictions and post-hoc controls | `python scripts/paired_comparisons.py` |
| `calibration` | 240.3 MiB | 36 compact calibration/test inputs for the matched comparison | `python scripts/review_calibration.py --from-compact` |
| `order` | 53.8 MiB | 18 sets of recorded candidate-order predictions | `python scripts/summarize_order_union.py` (also needs `calibration`) |
| `controls` | 17.1 MiB | 46 larger inputs for paired contrasts, archive changes, scratch and encoder controls | `python scripts/analyze_extensions.py` |

The [manifest](evidence-manifest.json) lists every file, size and checksum, with the public source revision. The bundles contain only files already published at that revision. Small demonstration fixtures stay in Git. Archive inputs retain their existing privacy boundary: compact correctness/confidence records, without the private post text or raw annotations.

Analysis commands need the installed project environment described in the [experiment guide](../docs/experiments.md). Reading the saved reports and fetching their inputs do not require installing PyTorch.

For offline use, download the `.tar.gz` assets from the [evidence release](https://github.com/bahree/myJEV/releases/tag/evidence-v1), copy them to a local folder, then run:

```bash
python3 scripts/fetch_evidence.py --archive-dir /path/to/downloads
```

The release assets are checksum-pinned. A changed asset fails verification. Corrections should use a new manifest and release rather than replacing evidence in place.

## What the other files mean

- `metrics.json` and `*-metrics.json` hold numerical evaluations. Read the [confidence-file guide](longer-v1/README.md) before comparing scalar, policy and calibrated selection metrics.
- `training.jsonl`, `train.log` and GPU telemetry record actual training. They remain in Git, including failed experiments. [Tracking](../docs/tracking.md) explains their fields.
- `frozen-plan.json`, configuration files and artifact manifests record experimental conditions and exact model identities.
- `container-*`, `release-*` and model-card publication manifests connect tested runtime behavior with downloadable releases. Older records describe their own versions.
- [Reader validation](reader-validation-v1/) preserves the source of the README response and executed teaching commands. Editorial link checks, email previews and export bookkeeping belong in the private working repository.

The numbered directories distinguish recorded experiments; they are not instructions to run every experiment. The [documentation index](../docs/README.md) gives a shorter route through the project.

Moving bulky inputs out of the working tree does not remove old Git objects. Use `git clone --depth 1 https://github.com/bahree/myJEV.git` for a smaller checkout of the current code and reports. A normal full clone retains the project's original history, including those inputs.
