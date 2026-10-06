# Scratch BANKING77 diagnostic

One seed (11), random initialization, 217,559 parameters, FP32, 1,000 updates at batch size 8 (8,000 examples processed), fixed learning rate 0.001. A separate 100-update fit/throughput pilot preceded the run. No BANKING77 hyperparameter tuning or early stopping. Complete official test: 3080 examples, separate 1,000-example calibration partition.

Accuracy: **1.30%**, macro-F1 0.000333. The model selected `card_payment_wrong_exchange_rate` on all 3,080 test examples. It collapsed to one class and is not a usable classifier under this configuration and exposure. Its 1.30% accuracy equals the 1/77 uniform-chance rate on this balanced test, but the predictor itself is deterministic, not uniform random.

Scalar correctness Brier: **0.0128**. The low number is not a success: the model is almost always wrong and reports low correctness confidence. Always predicting zero confidence would also have a low Brier on this outcome distribution. Do not rank this artifact above a competent classifier by Brier alone.

The byte input limit is 512 for context/instructions, 64 per description and 160 candidates; the task supplies 77. Extra positional embeddings increase parameters relative to the 201,175-parameter synthetic fixture. The corpus fits those byte limits without truncation. These limits and candidate processing differ from Qwen's tokenizer/readout.

This is a one-seed failure diagnostic, not evidence that scratch models or encoder architectures cannot learn banking intents. Architecture, tokenizer, initialization, optimization, exposure and data diversity are competing factors. No test-driven retraining was performed after observing the result. The original Qwen and TF-IDF results remain separate controls with different pretraining/training histories.

See `configs/scratch-banking-diagnostic.json`, logs, metric files and manifests for the frozen settings. Full predictions remain local/private; the public snapshot carries metrics and training evidence. Temperature calibration changes confidence, not the selected labels, and does not repair the failed classifier.
