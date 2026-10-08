# Documentation

myJEV explores how to choose from supplied answers and estimate whether the choice is correct. The [main README](../README.md) introduces Jev, the motivation for this project, and the two approaches we built. These guides take you through using the model, understanding it, training it and checking the results.

If you are new to the project, start with the [quick start](quickstart.md) and [seven example requests](../results/demos-v1/report.md). Then follow the [hands-on walkthrough](walkthrough.md), which connects those outputs to the computation. All commands run from the repository root; the quick start activates the Python environment used by later guides.

## Choose a starting point

| Question | Guide |
|---|---|
| Can I try real requests and see the failures? | [Seven runnable demos](../results/demos-v1/report.md) |
| Can I explore confidence on a CPU? | [Calibration lab](../results/calibration-lab-v1/report.md) and [decision lessons](decision-lessons.md) |
| How do I install and score my first request? | [Quick start](quickstart.md) |
| How do I understand or build the model? | [Architecture](architecture.md), then [scratch walkthrough](scratch.md) or [Qwen training](training.md) |
| Which results should guide my choice? | [Findings](qwen-findings.md), with worked definitions in [decision lessons](decision-lessons.md) |
| How do I call it from another application? | [Python, CLI and HTTP](inference.md), then [hosting](hosting.md) |

## Build and train

The scratch route starts from random weights so we can inspect the whole network. The Qwen route starts from pretrained language knowledge and studies what further training adds. They answer different questions; their training exposure is not matched.

| Guide | What it explains |
|---|---|
| [Architecture](architecture.md) | How a request becomes candidate scores and confidence |
| [Scratch walkthrough](scratch.md) | Build, train and debug the small model, including its language-task failure |
| [Scratch design](scratch-plan.md) | Design constraints and implementation checks |
| [Qwen training](training.md) | Adapters, supervised and reward-based training, hardware and the update budget |
| [W&B tracking](tracking.md) | Read training curves, epochs and GPU measurements while retaining local logs |
| [Engineering lessons](learnings.md) | Why training exposure, hardware and stopping rules matter |

## Evaluate the decisions

Start with the worked examples if confidence and calibration are unfamiliar. The findings interpret the results; the experiment guide and saved reports provide the detail needed to reproduce them.

| Guide | What it explains |
|---|---|
| [Decision lessons](decision-lessons.md) | Accuracy, confidence, asking for review and the cost of mistakes |
| [CPU calibration lab](../results/calibration-lab-v1/report.md) | Train small classifiers where the true probabilities are known |
| [Qwen findings](qwen-findings.md) | Compare methods and sizes while keeping seed variation visible |
| [Results index](../results/README.md) | Find reports, training logs and optional prediction downloads |
| [Experiments and evidence](experiments.md) | Separate the short pilot, longer comparison and other controls |
| [Experimental protocol](protocol.md) | Keep training, validation, calibration and test data separate |
| [Generated comparison](../results/longer-v1/summary.md) | Inspect the numerical summary and its source files |

## Run and host

| Guide | What it explains |
|---|---|
| [Model releases](models.md) | Choose among the six downloadable Qwen releases and identify their exact versions |
| [Inference](inference.md) | Load once, score requests, run a batch, and troubleshoot the local GPU path |
| [Hosting](hosting.md) | Docker, queues, startup, benchmark conditions and the unexecuted managed-endpoint recipe |

## Related work and further experiments

[Attribution](attribution.md) traces the design influences. [System One research](system-one-research.md), the [source-based peer review](system-one-peer-review.md), and [OpenJev notes](openjev.md) compare related interfaces and their limits. [Scope and extensions](roadmap.md) separates measured results from ideas for further study.

## Data and provenance

- [BANKING77](datasets/banking77.md): controlled intent routing, official test split preserved.
- [CLINC150](datasets/clinc150.md): unfamiliar-task and unsupported-request evaluation, excluded from tuning.
- [Blog archive](datasets/blog-archive.md): post-format, instructional-completeness and claim-support rubrics; completed machine-reference adaptation and forgetting study; human correctness remains unaudited.
- [Synthetic fixtures](datasets/synthetic.md): objective arithmetic and controlled uncertainty.
- [Scratch routing](datasets/scratch-routing.md): original rules, known ambiguity, versioned templates and held-out combinations.
- [Attribution](attribution.md): JevK5, SemIf, and JevForge architecture distinctions.
- [OpenJev](openjev.md): a related released model with different scale, confidence semantics, and licensing.

## Follow the project

[Release history](../CHANGELOG.md) records versioned changes. The [blog](https://blog.desigeek.com) accompanies these guides; [scope and extensions](roadmap.md) explains the limits of the evidence and questions for further study.
