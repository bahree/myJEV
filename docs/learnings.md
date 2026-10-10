# Engineering and experimental lessons

Several things went wrong before the model could be trained and served reliably. These notes explain what those failures changed in the implementation and how to read the resulting measurements. The [Qwen findings](qwen-findings.md) cover model quality; the [experiment guide](experiments.md) links the runs and their settings.

The short pilot and longer comparison used different training budgets and test sizes. Use the [longer-study results](../results/longer-v1/summary.md) for the main method comparison.

## Count updates across the experiment

The 168,000-step budget comprises 24,000 tuning steps and 144,000 main-comparison steps. It spans three model sizes, four training approaches, two tuning learning rates and three main seeds. All methods use AdamW; exact and sampled identify the two ways of estimating the RL objective. The [training guide](training.md) gives the full arithmetic.

One initial supervised run consumes 4,000 examples. Each continuation consumes another 4,000, starting from that supervised artifact. With 7,999 training rows, each continuation model has roughly one epoch in its ancestry. Reusing the supervised artifact does not mean adding all three continuation branches into one model's history.

## A fixed budget does not establish convergence

The budget supports a matched comparison under equal exposure and tuning opportunity. It does not establish each model's best achievable result. Training loss can fall while held-out performance worsens, and losses from supervised and RL objectives are not interchangeable.

Improvement should be assessed using held-out accuracy/F1, correctness Brier, calibration and accepted-case error at fixed coverage. A future convergence study needs predefined periodic validation, a meaningful improvement threshold and patience across several checks. Its stopping decisions must not use the test set. Current endpoint-only evaluations cannot establish a reliable plateau.

The unattended monitor checks for NaN or infinite loss, fatal process errors and exhausted disk space. Those are reasons to stop a job and inspect its logs. A negative RL loss can be valid, and ordinary fluctuations do not tell the monitor whether a model has stopped improving.

## Hardware is only part of elapsed time

Each of the three 24 GB A30 GPUs ran an independent model size. The schedule processes one example per update, computes a reference policy for RL, saves checkpoints frequently and runs full evaluation passes. That work contributes to elapsed time alongside the hardware. GPU utilization tells us how often a device was busy; finding a faster batch size or backend would need a separate benchmark.

## Small adapters do not imply a small serving model

LoRA saves a compact weight update and reduces training state. Serving still loads the Qwen backbone and processes the full input. The one-pass readout saves answer decoding, but neither adapter size nor a successful short training pilot tells us maximum-context memory use or how we compare with Jev’s service.

The [hardware and model-size rationale](training.md#choosing-model-size-around-the-hardware) distinguishes per-device fit from aggregate GPU memory. The [inference cost explanation](inference.md#what-the-adapter-saves-and-what-inference-still-costs) separates adapter files, backbone requirements, measured HTTP latency and untested optimizations. Final model selection must weigh quality against these costs; 9B is not automatically the default.

## Keep observability separate from training

Local logs remain authoritative even when a dashboard is unavailable. An independent W&B bridge can show live progress and import completed histories without restarting training. Record observation time, original step/time axes and historical-import labels explicitly. Keep secrets, weights and archive text outside telemetry uploads. See [W&B setup and usage](tracking.md).

## Test inference through the real deployment path

The pilot container built successfully and could run a CUDA tensor operation, then failed on its first real model request: a runtime kernel needed a C compiler. Another wrapper resolved the virtualenv Python symlink to system Python and lost the environment’s dependencies. Both failures are in the logs. The deployment checks now load the actual artifact and exercise its confidence output through Python, CLI, HTTP and Docker.

All three pilot sizes passed after those fixes. Final-checkpoint serving checks and the empty-model-cache startup test followed as separate runs. Their request counts, cache conditions and timings are in [inference and Docker](inference.md) and [hosting](hosting.md).

## Scratch models exposed two different failure modes

A small encoder fit the initial routing template but failed a reordered layout. Restoring that layout recovered some accuracy; varied training layouts and more exposure improved validation, but the final three-seed study remained unstable. See the [scratch evidence](scratch.md) rather than a single successful example.

The BANKING77 scratch model chose one class for every request and reached only 1.30% accuracy. Its confidence was low, so its correctness Brier score also looked low. That is why accuracy and the accepted-request counts have to sit beside the probability metrics: the model was correctly warning us about answers we could not use.
## A new readout changes more than the output layer

The alias/Clef study used equal tuning, example exposure and calibration across three seeds. Clef's mean accuracy was 83.50%, versus 81.34% for aliases. Its main training prompt averaged 1,908 tokens, compared with 875, and the training sessions averaged 40.2 rather than 16.9 minutes. Both the head and the prompt contribute to that cost. Compilation failed before any completed update, so the reported comparison used eager execution for both arms.

Clef had worse raw correctness Brier on two seeds, then lower Brier on all three after both arms received the same calibration-only temperature fit. The [full comparison](unsloth.md) shows the per-seed values and a case where temperature slightly worsened test Brier.

The [original candidate-attention head](candidate-head.md) trained 216,193 parameters while leaving Qwen frozen. It passed the tiny-set fit and artifact checks, then reached 58.02% on BANKING77. The duplicate-charge demo selected `other` with 0.9993 confidence. A small trainable head can learn part of a labelled task while failing a new candidate scheme, and its inference still requires the backbone.

## A hosted model needs an observation record

Microsoft Decision-1 shares the Qwen3.5 backbone family with myJEV, but its reviewed announcement does not disclose the head or training objective. An API model name also need not identify fixed weights. The [comparison runner](decision-1.md) records returned model/provider identifiers, observation times and response hashes alongside the option probabilities. It retains vendor confidence separately because the field need not estimate the same event as our correctness head.

The runner reserves the listed full-context input price before every paid attempt and writes that attempt before sending it. Failures stop execution without automatic retries. Provider token and cost fields supply observed usage; absent fields remain unknown. The returned scores can be reanalyzed on CPU without calling the service again.
