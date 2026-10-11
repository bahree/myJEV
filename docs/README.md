# Documentation

myJEV explores how to choose from supplied answers and estimate whether the choice is correct. The [main README](../README.md) introduces Jev, the motivation for this project, and the two approaches we built. These guides take you through using the model, understanding it, training it and checking the results.

If you are new to the project, start with the [quick start](quickstart.md) and [seven example requests](../results/demos-v1/report.md). Then follow the [hands-on walkthrough](walkthrough.md), which connects those outputs to the computation. All commands run from the repository root; the quick start activates the Python environment used by later guides.

## Choose a starting point

| Question | Guide |
|---|---|
| Can I try real requests and see the failures? | [Seven runnable demos](../results/demos-v1/report.md) |
| Can I explore confidence on a CPU? | [Calibration lab](../results/calibration-lab-v1/report.md) and [decision lessons](decision-lessons.md) |
| How do I install and score my first request? | [Quick start](quickstart.md) |
| How do I understand or build the model? | [Architecture](architecture.md), then [build-from-scratch walkthrough](scratch.md) or [Qwen training](training.md) |
| Which results should guide my choice? | [Findings](qwen-findings.md), with worked definitions in [decision lessons](decision-lessons.md) |
| How do I call it from another application? | [Python, CLI and HTTP](inference.md), then [hosting](hosting.md) |

## Build and train

Building a small model from scratch starts its weights with random values so we can inspect how the whole network learns. The Qwen route starts from pretrained language knowledge and studies what further training adds. They answer different questions; their training exposure is not matched.

| Guide | What it explains |
|---|---|
| [Architecture](architecture.md) | How a request becomes candidate scores and confidence |
| [Build-from-scratch walkthrough](scratch.md) | Build, train and debug the small model, including its language-task failure |
| [Small-model design](scratch-plan.md) | Design constraints and implementation checks |
| [Qwen training](training.md) | Adapters, supervised and reward-based training, hardware and the update budget |
| [Unsloth readout comparison](unsloth.md) | Why a joint schema head changes the experiment, and how to run the matched control |
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

The [Decision-1 comparison](decision-1.md) adds a hosted reference and tests option IDs, formatting and instruction wording alongside candidate order. Its cloud requests are optional; local model use needs no hosted account.

The [Decision 3.0 comparison](decision3.md) examines another downloadable Qwen-based model on the same A30 hardware, including its answer-code projection and different definition of confidence.

## Run and host

| Guide | What it explains |
|---|---|
| [Model releases](models.md) | Choose among the six downloadable Qwen releases and identify their exact versions |
| [Inference](inference.md) | Load once, score requests, run a batch, and troubleshoot the local GPU path |
| [Docker on Windows, macOS and Linux](docker.md) | Download a model and run it with GPU access or CPU fallback |
| [Hosting](hosting.md) | Docker, queues, startup, benchmark conditions and the unexecuted managed-endpoint recipe |

## Related work and further experiments

[Attribution](attribution.md) traces the design influences. [System One research](system-one-research.md), the [source-based peer review](system-one-peer-review.md), and [OpenJev notes](openjev.md) compare related interfaces and their limits. [Scope and extensions](roadmap.md) separates measured results from ideas for further study.

## Data and provenance

Each source answers a different question. The dataset cards explain preparation, labels, partitions and limits.

| Dataset and source | Why we chose it | Role in the study |
|---|---|---|
| [BANKING77](datasets/banking77.md), PolyAI’s labelled online-banking queries | Closely related requests test whether a model understands the customer’s intent; existing labels supply training targets and rewards | Main training and evaluation, with the official test split preserved |
| [CLINC150](datasets/clinc150.md), crowdsourced intent and out-of-scope data released by Larson and colleagues | Test unfamiliar categories and requests the supported choices cannot answer | Transfer evaluation, excluded from training and tuning |
| [SST-2](../results/sst2-transfer-v1/report.md), Stanford’s human-labelled movie-review sentiment data | Test positive/negative sentiment instead of another routing task | Evaluation on the public validation split, without sentiment training or calibration |
| [Blog archive](datasets/blog-archive.md), posts from Amit Bahree’s RSS feed | Test longer documents, new rubrics and adaptation | Post format, instructional completeness and claim support; reference labels come from models and remain unaudited |
| [Generated color examples](datasets/scratch-routing.md), original Python-generated rules | Inspect learning where the answer probabilities are known | Small-model training with versioned layouts and held-out color combinations |
| [Synthetic test cases](datasets/synthetic.md), authored for this project | Verify reward arithmetic, gradients and controlled uncertainty | Implementation checks; these are separate from natural-data evaluation |

The [CPU calibration lab](../results/calibration-lab-v1/report.md) also generates its own numerical features and labels from a known probability distribution. That lets us measure probability error directly; it does not measure language understanding.

[Attribution](attribution.md) explains the JevK5, SemIf and JevForge architecture influences. [OpenJev](openjev.md) describes a related released model with different scale, confidence semantics and licensing.

## Follow the project

[Release history](../CHANGELOG.md) records versioned changes. The [blog](https://blog.desigeek.com) accompanies these guides; [scope and extensions](roadmap.md) explains the limits of the evidence and questions for further study.

The [candidate-head walkthrough](candidate-head.md) builds a small original attention head on frozen Qwen. Its one-seed protocol is separate from the [matched alias/Clef comparison](unsloth.md).
