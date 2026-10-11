# BANKING77 dataset card

BANKING77 contains English online-banking requests with 77 intent labels. An intent describes what a customer wants, such as checking card arrival or reporting a card that does not work. PolyAI released it with the 2020 paper [Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/).

We chose it because related banking categories require more than recognizing a topic, and the existing labels let us train a classifier and calculate rewards without a model judge. It covers a controlled routing task; it is not a generalist training mixture. Source: [PolyAI authors' dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), Creative Commons Attribution 4.0 as specified by the source repository. The preparation manifest pins its Git commit and hashes generated split files. Labels are existing task labels; ambiguity between related intents remains possible. No labels were synthesized.

Candidate IDs are the 77 original names and descriptions replace underscores with spaces. Text is preserved verbatim for model input. Only grouping normalizes case and whitespace. Identical normalized texts remain together. Seven training examples matching official test groups were quarantined; test data was untouched. Remaining training groups are stratified by label with seeded approximately 80/10/10 group allocation. Rounding operates per label, so row counts need not be exactly 80/10/10. Conflicting labels in one group fail preparation rather than silently resolving them.

Counts from the local preparation: training 7,999; validation 997; calibration 1,000; official test 3,080. Validation chooses hyperparameters; calibration fits temperatures and thresholds; test supplies final evaluation only. Duplicate grouping does not detect all semantic paraphrases. Official train/test provenance is preserved in IDs. Re-run with the recorded revision, not a moving branch.

Cannot support claims about arbitrary documents, subjective quality, unfamiliar rubrics, production error rates or broad generalist competence. Results may reflect pretraining exposure.

## Follow-up overlap audit

Removing ASCII punctuation in addition to case/whitespace normalization finds four additional same-label training/test matches. The original grouping only removed case and whitespace differences, so its isolation claim was narrower. The [audit and sensitivity report](../../results/review-overlap-v1/report.md) lists original IDs and all 36 run results after excluding those four test rows (3,076 remain). Official 3,080-row results stay primary; no historical splits or trained weights were changed. This sensitivity check cannot undo any influence of duplicated training examples, and it does not detect all semantic paraphrases. A new training study should strengthen grouping before fitting.
