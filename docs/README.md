# Documentation

Read the [completed Qwen findings and lessons](qwen-findings.md), including paired uncertainty, confidence controls and what changes next.


Start with the question you want to answer. All commands run from the repository root.

| Question | Guide |
|---|---|
| How do I install, train, and run it? | [Quick start](quickstart.md) |
| What happens in one forward pass? | [Architecture and objectives](architecture.md) |
| What was actually measured? | [Experiments and evidence](experiments.md) |
| How are methods compared fairly? | [Experimental protocol](protocol.md) |
| What did the longer training batch find? | [Completed comparison](../results/longer-v1/summary.md) |
| How will we build the scratch model? | [Implementation plan and acceptance gates](scratch-plan.md) |
| How do I watch training and read epochs? | [W&B integration](tracking.md) |
| What have we learned about budgets and convergence? | [Engineering lessons](learnings.md) |
| How do I run Python, CLI, HTTP and Docker inference? | [Inference walkthrough](inference.md) |
| How do I host it? | [Local and managed hosting](hosting.md) |
| Where are released weights? | [Model release tracker](models.md): releases pending |
| What is complete, and what comes next? | [Roadmap](roadmap.md) |

## Data and provenance

- [BANKING77](datasets/banking77.md): controlled intent routing, official test split preserved.
- [CLINC150](datasets/clinc150.md): unfamiliar-task and unsupported-request evaluation, excluded from tuning.
- [Blog archive](datasets/blog-archive.md): post-format, instructional-completeness and claim-support rubrics; 40-post machine-annotation pilot complete, adaptation/evaluation pending.
- [Synthetic fixtures](datasets/synthetic.md): objective arithmetic and controlled uncertainty.
- [Attribution](attribution.md): JevK5, SemIf, and JevForge architecture distinctions.
- [OpenJev](openjev.md): a related released model with different scale, confidence semantics, and licensing.

## Follow the project

[Milestone history](../CHANGELOG.md) records the research and engineering state represented by each snapshot. The [blog](https://blog.desigeek.com) will explain the experiments; article links remain placeholders until published. The roadmap and model tracker are living reader documents, not promises that unfinished work has passed evaluation.
