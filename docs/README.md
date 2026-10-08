# Documentation

Read the [completed Qwen findings and lessons](qwen-findings.md), including paired uncertainty, confidence controls and what changes next.


Follow the [hands-on learning route](walkthrough.md) for worked arithmetic, runnable commands, actual outputs and checks.

Start with the question you want to answer. All commands run from the repository root.

| Question | Guide |
|---|---|
| Can I try real requests and see the failures? | [Seven runnable demos](../results/demos-v1/report.md) |
| Can I explore confidence on a CPU? | [Calibration lab](../results/calibration-lab-v1/report.md) and [decision lessons](decision-lessons.md) |
| How do related decision systems differ? | [System One research](system-one-research.md) and [source-based peer review](system-one-peer-review.md) |
| How do I install, train, and run it? | [Quick start](quickstart.md) |
| How should I interpret confidence, costs and simpler baselines? | [Decision lessons and worked examples](decision-lessons.md) |
| What happens in one forward pass? | [Architecture and objectives](architecture.md) |
| What was actually measured? | [Experiments and evidence](experiments.md) |
| How are methods compared fairly? | [Experimental protocol](protocol.md) |
| What did the longer training batch find? | [Completed comparison](../results/longer-v1/summary.md) |
| How do I build and run the scratch model? | [Scratch walkthrough and findings](scratch.md), [implementation gates](scratch-plan.md) |
| How do I watch training and read epochs? | [W&B integration](tracking.md) |
| What have we learned about budgets and convergence? | [Engineering lessons](learnings.md) |
| How do I run Python, CLI, HTTP and Docker inference? | [Inference walkthrough](inference.md) |
| How do I host it? | [Local and managed hosting](hosting.md) |
| Where are released weights? | [Six public Qwen adapter/head releases](models.md), including the 4B default |
| What is complete, and what comes next? | [Roadmap](roadmap.md) |

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
