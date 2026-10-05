# W&B tracking and saved evidence

W&B is an optional dashboard. Local JSONL logs, evaluation outputs and artifact manifests remain the primary record. Uploading never deletes those files. The integration does not upload model weights, optimizer state, credentials or archive annotations.

## Set up your own project

Create a W&B project with the visibility you intend, then install the pinned optional SDK and create a local configuration:

```bash
.venv/bin/pip install -r requirements.tracking.txt
cp configs/wandb.env.example .env.wandb
chmod 600 .env.wandb
```

Edit `.env.wandb`: fill in `WANDB_API_KEY`, `MYJEV_WANDB_ENTITY` and `MYJEV_WANDB_PROJECT`. Keep the key out of Git, shell command arguments, screenshots and shared logs. The example file has no credentials. The real `.env.wandb` is ignored by Git.

For a new training process, load your locally edited shell-compatible environment file:

```bash
set -a
. ./.env.wandb
set +a
export MYJEV_TRACKING_MODE=online
# Now run a training command from training.md or quickstart.md.
```

`online` streams metrics directly. `offline` records W&B data locally without uploading; `disabled` turns off W&B while ordinary training logs still record results. Changing the environment of your terminal does not change jobs that are already running. Configuration includes model, precision and training settings. Code saving, Git capture and console capture are disabled in the integration.

## Watch an existing longer study

The bridge reads the directory layout created by `scripts/run_longer_study.py`. Prepare BANKING77 and launch the study first, following [the training guide](training.md). This bridge is specific to that three-size study, not a generic monitor for arbitrary training scripts.

```bash
# Terminal 1: live metrics, every minute, for up to 48 hours.
.venv/bin/python scripts/sync_wandb_study.py live \
  --env-file .env.wandb --interval 60 --duration 172800

# Terminal 2: historical training curves and evidence, once.
.venv/bin/python scripts/sync_wandb_study.py backfill \
  --env-file .env.wandb --once

# Alternatively, keep importing new completed runs every 30 minutes.
.venv/bin/python scripts/sync_wandb_study.py backfill \
  --env-file .env.wandb --interval 1800 --duration 172800
```

Run long-lived commands inside your own tmux session if they must survive terminal disconnects. The server must stay running. The bridge uses Linux process locks and the monitor inspects `/proc`; Linux is the tested platform. With only one model size launched, the other sizes remain marked as starting.

The dashboard link is written to `results/wandb-sync/live.json`. Upload progress, heartbeat, historical run links and the evidence artifact version are stored in the same ignored directory. Per-mode locks prevent duplicate bridge processes. Backfill failures retry on the next interval; `--once` exits unsuccessfully if uploading fails. The bridge runs independently of training, so an upload failure does not terminate training.

Stable IDs allow the same study to resume uploading without creating duplicate completed runs. For a new independent study, use both a new `--run-prefix study-2` and a separate `--state-dir results/wandb-sync-study-2`, consistently for live and backfill commands. Ignore that additional state directory locally. A changed entity/project/prefix is rejected when using a state directory that already records another destination.

## Read the dashboard

| Field | Interpretation |
|---|---|
| `batch/percent` | Estimated completion of this study, including evaluation; not overall project completion |
| `batch/training_steps` | Exact updates completed across separate runs |
| `<size>/stage` and `<size>/state` | Current method, seed, operation and runner state, in the run summary |
| `<size>/step` | Current training stage's update count |
| `<size>/stage_epochs_done` | Current stage's processed examples divided by training set size |
| `<size>/stage_epochs_remaining` | Current stage's remaining examples divided by training set size |
| `<size>/lineage_epochs_done` | Exposure including the inherited supervised starting model |
| `<size>/last_loss` | Latest method-specific training loss; not a shared quality metric |
| `gpuN/utilization.gpu` | Device utilization percentage from sampled telemetry |
| `gpuN/memory.used` | Device memory use in MiB, which differs from PyTorch peak allocated bytes |

Select these numeric fields in W&B charts. Use saved `step` or `session_seconds` as the x-axis for imported training curves; their upload timestamps are not historical training timestamps. The live monitor shows one observation per minute. Individual historical runs preserve every saved training step after a stage completes. Earlier GPU measurements remain in the evidence CSVs; the live dashboard begins when the bridge starts.

Epoch fields are cleared in the summary during evaluation. The initial 4,000-step supervised model plus one 4,000-step continuation has about one epoch of exposure on 7,999 examples. Do not add sibling continuation branches or seeds as one model's epochs. See [the budget explanation](training.md#why-168000-training-steps).

The bridge does not yet turn every evaluation file into a W&B scalar chart. Accuracy, Brier, calibration and coverage/error results are available in the versioned evidence artifact as their saved metrics and prediction files. Training loss alone cannot establish improvement or convergence.

## Retain and interpret the evidence

`myjev-experiment-evidence` stores versioned snapshots of public-task training/evaluation logs, predictions, figures and GPU telemetry. Inspect the artifact's Files view to download the underlying evidence. Snapshot files are collected during each pass; the collection is not an atomic model checkpoint. Per-file deduplication can reduce repeated uploads, but monitor your own account's storage usage.

The selected experiment directories are explicit in `evidence_files()`. Review that allowlist before adding other datasets. Live W&B directories and temperature-artifact weight directories are excluded. No source log is deleted or rewritten after an upload. Git checkpointing is separate and is not performed by the uploader.

To record device-wide telemetry for the live dashboard:

```bash
.venv/bin/python scripts/gpu_telemetry.py \
  --output results/longer-v1/gpu-observation --interval 15 --duration 172800
```

Use a fresh output directory. GPU utilization does not measure model quality or FLOP efficiency. A chart regenerated from CSV is a chart, not a screenshot. If publishing an actual W&B or NVIDIA terminal screenshot, retain its source run/time and hide credentials.

## References

- [W&B offline recording and later sync](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues)
- [W&B run logging and lifecycle](https://docs.wandb.ai/ref/python/experiments/run/)
