# Scope, limits and research extensions

myJEV studies one-pass candidate selection and correctness confidence through two architectures: a small randomly initialized scorer and adapted Qwen backbones. [Findings](qwen-findings.md), [scratch lessons](scratch.md), and [experiments](experiments.md) connect the measured outcomes to reproducible evidence.

## What the evidence supports

The Qwen comparison contains 60 runs and 168,000 optimizer updates across three sizes, with three seeds for each main method. Matched calibration controls distinguish the released configurations from training-method comparisons. Candidate-order tests use three per-request seeded permutations with fixed release settings. Conditional test-resampling intervals and seed spread describe different uncertainty sources.

The scratch scorer recovers controlled synthetic rules but fails its BANKING77 natural-language diagnostic. A fixed-taxonomy ModernBERT control and TF-IDF show why a flexible candidate-description interface must justify its resource cost. Different exposure and interfaces prevent treating them as a matched architecture ranking.

The archive adaptation study measures agreement with machine-generated labels. It does not establish human correctness. Adaptation increases that agreement while weakening explicit unsupported-option behavior. Independent label and grouping audits are necessary for stronger archive claims. See the [archive data card](datasets/blog-archive.md).

## Using the released implementation

Six adapter/head packages have immutable Hub revisions. Python, CLI, HTTP and GPU Docker share the reference inference implementation. [Models](models.md), [inference](inference.md), and [hosting](hosting.md) document installation, actual response checks and measured resource use. Publishing weights does not provide a hosted endpoint. The managed-container recipe is unexecuted.

## Questions for another study

- How stable are method differences over more training seeds and longer Brier/correctness-only ablations?
- Can better untouched readout, other backbones or distillation improve quality per unit of memory and latency?
- What data and capacity let the scratch model learn useful language behavior beyond its synthetic rules?
- How do policy edits, candidate wording and order affect operational decisions across genuinely new tasks?
- Can an independently designed certification procedure support selective-risk claims on the deployment distribution?

Phi, SmolLM and MAI were not evaluated. A broader numeric-readout comparison, confidence-policy initialization ablations and optimized backends need separate protocols and equivalence checks. No reconstruction of proprietary Jev or matched vendor-speed claim is made.

## Deployment limits

Empirical thresholds are not production guarantees. Workload-specific validation, authentication, monitoring and memory headroom matter. The research container runs as root; a non-root deployment requires permission testing for caches and runtime kernel compilation. No paid endpoint or additional hardware is assumed by these extensions.
