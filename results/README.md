# Study results

These files record what the experiments measured. You do not need them to run inference: the [quick start](../docs/quickstart.md) downloads model artifacts from Hugging Face separately.

Start with a report from the table below. This directory keeps reports, aggregate summaries, figures, demo responses and the release checks linked from the guides. Per-run metrics, training logs, progress snapshots and older publication records are in optional [evidence downloads](https://github.com/bahree/myJEV/releases). Download them when reproducing a result or inspecting a training run.

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
| Does a different decision head help? | [Unsloth comparison](../docs/unsloth.md), [recorded results](unsloth-head-v1/report.md) | Three seeds, equal tuning/exposure/calibration, order probes and warm timings |
| Can a small new head use frozen language features? | [Candidate-head walkthrough](../docs/candidate-head.md), [saved report](candidate-head-v1/report.md) | One seed, 1,000 updates, full BANKING test and seven unselected demos |
| How does a hosted decision model behave on the same requests? | [Decision-1 comparison](../docs/decision-1.md), [saved report](decision-comparison-v1/report.md) | One fixed 64-case diagnostic, six views, equal calibration fitting and dated service metadata |
| How do other downloadable models compare on this GPU? | [Decision 3.0 comparison](../docs/decision3.md), [saved report](decision3-v1/report.md) | Six pinned releases, full BANKING77 test, CLINC cohorts, matched calibration opportunity and isolated A30 timings |
| What does a real response look like? | [Seven demos](demos-v1/report.md) | Saved requests and responses, including a confident mistake |
| What does serving cost? | [Model and serving comparison](../docs/models.md), [container evidence](container-registry-v3/README.md) | Measured hardware, precision, startup and latency conditions |

For the short pilot, read the [experiment guide](../docs/experiments.md), [summary](study-summary.json) and [figures](figures/). Its 100-update stages and 256-example evaluation differ from the longer study. Its detailed records are in the `pilot` download.

## Restore inputs for a CPU replay

No model download, API key or GPU is needed to retrieve the evidence. From the repository root, using Python 3.12:

```bash
# All nine bundles; verifies archive and individual-file SHA-256 hashes.
python3 scripts/fetch_evidence.py

# Detailed metrics, training traces and recorded run settings.
python3 scripts/fetch_evidence.py --bundle records

# Or just the inputs needed for matched calibration.
python3 scripts/fetch_evidence.py --bundle calibration

# Check installed evidence without downloading or writing anything.
python3 scripts/fetch_evidence.py --check
```

The command restores original `results/` paths, so analysis scripts keep their input names. It refuses to overwrite different local files. Downloaded inputs are ignored by Git. Keep a separate checkout for new experiments if you want to compare them with these saved results.

| Bundle | Download size | Contents | Example replay after installation |
|---|---:|---|---|
| `records` | 6.3 MiB | 1,451 detailed files: per-run metrics, training logs, telemetry, manifests and historical checks | `python scripts/summarize_longer_study.py` |
| `pilot` | 65.0 MiB | 384 files: short-pilot metrics, predictions and post-hoc controls | `python scripts/paired_comparisons.py` |
| `calibration` | 240.3 MiB | 36 compact calibration/test inputs for the matched comparison | `python scripts/review_calibration.py --from-compact` |
| `order` | 53.8 MiB | 18 sets of recorded candidate-order predictions | `python scripts/summarize_order_union.py` (also needs `calibration`) |
| `controls` | 17.1 MiB | 46 larger inputs for paired contrasts, archive changes, scratch and encoder controls | `python scripts/analyze_extensions.py` |
| `readout` | 22.5 MiB | 12 text-free input files for alias/Clef calibration, test and order replay | `python scripts/replay_head_evaluation.py --study readout --output /tmp/readout-replay.json` |
| `candidate` | 3.6 MiB | One text-free calibration/test input for the frozen-backbone head | `python scripts/replay_head_evaluation.py --study candidate --output /tmp/candidate-replay.json` |
| `decision` | 4.0 MiB | Three text-free response files for the local/hosted comparison | `python scripts/summarize_decision_comparison.py` |
| `decision3` | 42.9 MiB | Twelve text-free quality and timing files for six local releases | `python scripts/summarize_decision3.py --require-benchmark` |

The [manifest](evidence-manifest.json) lists every file, size and checksum, with the public source revision. The four `evidence-v1` bundles restore files published at their recorded revision. The `evidence-v2` head-study bundles add text-free logits outside Git; their source revisions identify the public implementation and metric records. The `evidence-v3` decision bundle records the observed hosted/local responses; its revision identifies the frozen public protocol and code. Small demonstration fixtures stay in Git. Archive inputs retain their existing privacy boundary: compact correctness/confidence records, without the private post text or raw annotations.

The `evidence-v4` bundle adds the Decision 3.0 comparison. Its [source mapping](decision3-v1/public-source.json) identifies public files matching the recorded runner, and the [verification](decision3-v1/verification.json) checks request identity, score arithmetic and equality between quality and isolated benchmark outputs. The six models each completed 5,731 quality requests and 227 measured benchmark calls. The JSONL files contain IDs, labels, probabilities and timings, without request text.

The `records` bundle in `evidence-v5` preserves the detailed files removed from the public working tree, without changing their contents. It also supplies three missing precision-control inputs whose hashes were already recorded in the public summary; restoring them reproduces that summary unchanged. Fetching `readout`, `candidate`, `decision` or `decision3` also fetches `records`, because their replay commands use the saved run settings and metrics. For other summaries and training plots, fetch `records` first. The `calibration` bundle contains enough information to regenerate its per-run fits and metrics without `records`.

Analysis commands need the installed project environment described in the [experiment guide](../docs/experiments.md). Reading the saved reports and fetching their inputs do not require installing PyTorch.

For offline use, download the `.tar.gz` assets from the [evidence releases](https://github.com/bahree/myJEV/releases), copy them to a local folder, then run:

```bash
python3 scripts/fetch_evidence.py --archive-dir /path/to/downloads
```

The release assets are checksum-pinned. A changed asset fails verification. Corrections should use a new manifest and release rather than replacing evidence in place.

## What the other files mean

- `metrics.json` and `*-metrics.json` hold numerical evaluations. Read the [confidence-file guide](longer-v1/README.md) before comparing scalar, policy and calibrated selection metrics.
- `training.jsonl`, `train.log` and GPU telemetry record actual training, including failed experiments. Restore them with `--bundle records`; [tracking](../docs/tracking.md) explains their fields.
- `frozen-plan.json`, configuration files and artifact manifests record experimental conditions and exact model identities.
- `container-*`, `release-*` and model-card publication manifests connect tested runtime behavior with downloadable releases. The linked checks stay here; detailed logs and older card receipts are in `records`. Older records describe their own versions.
- [Reader validation](reader-validation-v1/) preserves the source of the README response and executed teaching commands. Editorial link checks, email previews and export bookkeeping belong in the private working repository.

The numbered directories distinguish recorded experiments; they are not instructions to run every experiment. The [documentation index](../docs/README.md) gives a shorter route through the project.

Moving bulky inputs out of the working tree does not remove old Git objects. Use `git clone --depth 1 https://github.com/bahree/myJEV.git` for a smaller checkout of the current code and reports. A normal full clone retains the project's original history, including those inputs.
