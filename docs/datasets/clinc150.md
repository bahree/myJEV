# CLINC150 dataset card

CLINC150 is a crowdsourced English request dataset released by Larson and colleagues for their [2019 study of intent classification and out-of-scope prediction](https://aclanthology.org/D19-1131/). It includes 150 supported intents and requests outside those categories. Crowd workers paraphrased seed phrases or wrote requests for supplied scenarios.

We chose it to check how a BANKING-trained model handles unfamiliar requests, including cases where no supplied choice is suitable. It is used only for transfer and unsupported-intent evaluation, never training or tuning. Source: [CLINC authors' out-of-scope benchmark](https://github.com/clinc/oos-eval), CC BY 3.0 per its source license. Preparation pins the Git revision in a manifest. Existing crowdsourced intent labels are retained; open-ended out-of-scope labels do not identify every possible semantic overlap.

Only official test and out-of-scope test records are used. Banking and credit-card domains are reported as nearby; the remaining domains are more distant. Five deterministically selected banking/credit-card classes are omitted from the candidate list, producing related unsupported requests. The manifest names them. This makes candidate-relative unsupportedness explicit; it is not the same as the dataset's out-of-scope category.

Each unsupported cohort has two views: explicit `none` option and no such option with required deferral. The latter's gold sentinel is absent from candidates, so no accepted answer can be correct. The same text shares a group across views. No CLINC thresholds are fitted: transfer operating thresholds must be frozen on BANKING77 calibration or a separately justified predeclared deployment calibration task. Candidate descriptions expand label names only. Do not conflate 150-way intent identification with a BANKING77 label mapping.

Cannot support universal OOD detection or independent-domain claims for the nearby subset. Generated candidate omissions are diagnostic perturbations, not new natural labels.
