# Milestone history

This history describes what each reader-facing snapshot contains. It records the state of the project at publication; upcoming work is tracked separately in the [roadmap](docs/roadmap.md).

## Inference pilot: prepared 2026-10-04

### Available

- Single-pass candidate-selection model, separate confidence heads, and exact/sampled RL objectives.
- Pinned 0.8B, 4B, and 9B configurations with a completed short three-seed comparison.
- BANKING77 preparation, classical and generalist controls, and calibration/evaluation tools.
- Python, CLI, HTTP, Docker, and a managed-hosting recipe.
- Saved pilot predictions, numerical metrics, manifests, and regenerable charts.
- Reader guides covering architecture, training, evaluation, hosting, and planned releases.

### Findings and scope

Continued supervision has the highest observed mean accuracy at each size in this short pilot. Temperature scaling is a stronger confidence control than the RL policies in the mean Brier comparison. Neural evaluations use a fixed 256-example test subset; the training schedule does not establish convergence.

### Still planned

Longer matched training, broader transfer/adaptation studies, released adapters, a registry image, source-license selection, and published blog articles. These entries will be filled in as work is completed; this milestone does not imply their availability.
