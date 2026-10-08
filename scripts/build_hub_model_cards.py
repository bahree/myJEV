"""Render reader-facing cards from checked study facts; no upload occurs."""
import argparse
import json
from pathlib import Path
import yaml


INTRO = {
 'myJEV-0.8B': ('The smallest myJEV model for choosing a support route and estimating confidence.',
   'Choose this version to start with the lowest measured memory use and latency in the family. It trades some banking-intent accuracy for a smaller backbone. It is useful for learning the API, reproducing the study, and testing whether this approach fits your routing workflow.'),
 'myJEV-0.8B-RL': ('A small decision model for studying whether confidence-aware learning improves routing.',
   'Choose this version to reproduce the small-model reinforcement-learning experiment. In this study it was less accurate than the standard 0.8B model, and its confidence estimates had higher Brier error. The standard 0.8B release is the better starting point if your priority is the measured quality/resource trade-off.'),
 'myJEV-4B': ('The recommended starting point within the published myJEV family: choose a route and return confidence.',
   'Choose this version for the best measured starting balance of banking-intent quality, confidence error and local serving cost within the published myJEV family. It uses continued supervised training followed by temperature calibration. The 4B-RL variant has higher banking accuracy on two of three seeds and in the mean; this standard release has lower confidence Brier error and stronger explicit unsupported-option transfer in the study.'),
 'myJEV-4B-RL': ('The accuracy-oriented myJEV variant, trained to reward correct decisions and useful confidence.',
   'Choose this version to explore the strongest mean BANKING77 accuracy in the released family. It had higher accuracy than standard 4B on two of three seeds and on the mean. Its native-policy Brier was higher than the released standard confidence configuration. Use standard myJEV-4B as the initial default when the measured confidence/resource balance matters most.'),
 'myJEV-9B': ('The larger myJEV model for exploring how capacity changes decisions and transfer.',
   'Choose this version when studying capacity or transfer to unfamiliar intent sets. It reached higher mean accuracy than 4B on the study’s distant CLINC transfer cohort, but did not improve mean BANKING77 accuracy over standard 4B and cost more memory and time. It uses four-bit backbone weights so the measured setup fits one 24 GB A30.'),
 'myJEV-9B-RL': ('The larger confidence-aware myJEV experiment, with one-pass decisions and a learned confidence policy.',
   'Choose this version to compare reinforcement learning at larger capacity. Its mean BANKING77 accuracy gain over standard 9B was small, and the paired uncertainty interval included zero. Confidence Brier error was also higher. It is an experimental comparison, rather than an automatic upgrade over the smaller default.'),
}


def render(row, facts):
    name=row['public_name'];rl=row['method']=='exact';single=row['metrics'][0]
    tagline,choice=INTRO[name]
    metadata={'license':'mit','language':['en'],'base_model':row['backbone'],'base_model_relation':'adapter','datasets':['PolyAI/banking77'],
      'tags':['myjev','intent-classification','single-pass','decision-model','confidence-estimation','custom-inference','qlora' if row['size']=='9b' else 'lora', 'reinforcement-learning' if rl else 'temperature-scaling'],
      'model-index':[{'name':name,'results':[{'task':{'type':'text-classification','name':'Intent classification'},
        'dataset':{'type':'PolyAI/banking77','name':'BANKING77 official test','split':'test'},
        'metrics':[{'type':'accuracy','name':'Accuracy (%) for released seed-11 checkpoint','value':round(single['accuracy']*100,4)}]}]}]}
    header='---\n'+yaml.safe_dump(metadata,sort_keys=False,allow_unicode=True)+'---\n'
    family='\n'.join(f"| [{m['public_name']}](https://huggingface.co/{m['repo_id']}) | {100*m['accuracy_mean']:.2f}% | {m['brier_mean']:.4f} | {m['latency_ms'][0]:.2f} ms | {m['allocated_vram_gib']:.2f} GiB |" for m in facts['models'])
    if rl:
        training='''After 4,000 supervised updates, this model received 4,000 updates using an exact expected-reward objective. The reward combines whether the answer is correct with squared error in its reported confidence: `correct - (confidence - correct)^2`. A penalty keeps the policy close to the frozen supervised reference. The finite action space lets training calculate the expectation directly rather than estimate it by sampling. This release is the exact-RL arm, not the separate sampled-REINFORCE arm.'''
        confidence='''**Reported confidence** comes from a separate candidate-conditioned policy over 21 values, from 0.00 through 1.00 in steps of 0.05. Serving first chooses the highest-scoring answer, then reports the policy’s expected confidence for that answer. It does not sample an answer or confidence at inference. The released native-policy confidence has higher mean Brier than the corresponding temperature-scaled supervised release. That comparison gives the two methods different post-hoc treatment; it does not establish an intrinsic calibration disadvantage of RL.'''
    else:
        training='''This model received 4,000 initial supervised updates and another 4,000 supervised updates. Continuing supervised training is a control for the extra optimization used by the reinforcement-learning variants. A single temperature was then fitted on reserved calibration data to adjust the selected-option probability. Calibration examples were not used to fit the adapter, and the official test partition was not used to choose the temperature.'''
        confidence='''**Reported confidence** is the selected option’s probability after temperature scaling on reserved BANKING77 calibration data. In this standard release it is derived from the selection scores, rather than the separate RL confidence policy. Calibration on banking intents does not establish calibration for arbitrary new tasks or candidate descriptions.'''
    precision='NF4 four-bit QLoRA' if row['size']=='9b' else 'BF16 LoRA'
    request=json.dumps({'context':'I was charged twice.','instructions':'Select the appropriate support route.','candidates':[{'id':'billing','description':'Charges, invoices, and refunds'},{'id':'technical','description':'Errors and configuration'},{'id':'other','description':'Neither listed route applies'}]},indent=4)
    return header+f'''
# {name}

{tagline}

Give it a request and descriptions of the allowed choices. It returns the chosen ID, a score for each choice and an estimate of correctness, without generating an answer paragraph. The main training task is banking support routing. Possible workflows include choosing a support queue, selecting the next workflow branch and handing uncertain cases to a reviewer after setting a threshold on your own calibration data.

By [Amit Bahree](https://huggingface.co/bahree). [Source and study](https://github.com/bahree/myJEV) | [Training findings](https://github.com/bahree/myJEV/blob/main/docs/qwen-findings.md) | [Inference and hosting](https://github.com/bahree/myJEV/blob/main/docs/inference.md)

## Why choose this version?

{choice}

**If your labels are fixed, compare a smaller classifier too.** A separate one-seed ModernBERT-base control reached 90.78% BANKING77 accuracy and 0.0555 correctness Brier after temperature calibration. It saw 23,997 training examples over three epochs, versus 8,000 example presentations in these myJEV runs, so this is a practical control with a different budget, not a matched architecture comparison. Its output head fixes the 77 labels; myJEV accepts candidate descriptions with each request. That flexibility does not establish accuracy on an unfamiliar taxonomy. See [the decision guide](https://github.com/bahree/myJEV/blob/main/docs/decision-lessons.md) and [encoder control](https://github.com/bahree/myJEV/blob/main/results/encoder-control-v1/report.md).

This is a **{row['backbone']} backbone plus a small adapter, confidence heads and calibration settings**. The download here contains the learned additions; the loader also downloads the separately pinned backbone. A small adapter file does not eliminate the backbone’s memory or compute cost. Serving reads the input once and performs no autoregressive generation.

## Results you can compare

The released checkpoint uses seed 11 by a fixed packaging convention. It was not selected for having the best test result. Accuracy and macro-F1 below use all 3,080 examples in BANKING77’s official test split. Brier measures squared error of reported correctness confidence, where lower is better.

| Evaluation | Accuracy | Macro-F1 | Correctness Brier |
|---|---:|---:|---:|
| This released checkpoint, seed 11 | {100*single['accuracy']:.2f}% | {single['macro_f1']:.4f} | {single['correctness_brier']:.4f} |
| Mean of seeds 11, 22 and 33 | {100*row['accuracy_mean']:.2f}% | {row['f1_mean']:.4f} | {row['brier_mean']:.4f} |

Accuracy’s sample standard deviation across those three seeds is {100*row['accuracy_sd']:.2f} percentage points. The [paired analysis](https://github.com/bahree/myJEV/blob/main/results/longer-v1/paired-analysis.md) reports uncertainty for method contrasts. Three observed seeds do not establish performance across all future runs or user tasks.

For this seed, warm HTTP latency was **{row['latency_ms'][0]:.2f} ms p50 / {row['latency_ms'][1]:.2f} ms p95**, using 100 requests, concurrency one and a short three-candidate request on one NVIDIA A30. The maximum allocated GPU memory observed for direct scoring across the three benchmark workloads was **{row['allocated_vram_gib']:.2f} GiB**. That excludes some driver/runtime allocations and is not a maximum-context memory guarantee. Startup to readiness was {row['startup_seconds']:.2f} seconds with cached weights, measured once. All three sizes were validated on 24 GB A30 hardware; precision here is **{precision}**. [Full serving conditions and evidence](https://github.com/bahree/myJEV/blob/main/docs/hosting.md#completed-candidate-validation-and-local-default).

## Run it

The reference environment is Linux, Python 3.12 and an NVIDIA GPU. The pinned requirements include PyTorch, Transformers, PEFT and the four-bit runtime. A working NVIDIA driver and a host C compiler are needed for the tested GPU path. On Debian/Ubuntu, install `gcc` and `libc6-dev` if missing.

```bash
git clone https://github.com/bahree/myJEV.git
cd myJEV
git checkout {facts['source_commit']}
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps .
```

Use the custom `DecisionModel` loader. Loading only the adapter, calling ordinary `generate()`, or using a generic classification widget does not execute this model’s complete decision interface.

```python
from myjev import DecisionModel

model = DecisionModel.load(
    "{row['repo_id']}",
    revision="{row['runtime_revision']}",
    device="cuda:0",
)
request = {request}
result = model.score(request)
print(result["selected_id"], result["confidence"])
```

The recorded pinned-runtime check selected `billing` with confidence approximately `{row['response']['confidence']:.4f}` on that exact fixture. It demonstrates a working call, not a guarantee on a new support taxonomy. The immutable revision above contains the tested runtime files; subsequent card-only edits leave those files unchanged.

The same request is checked into the source repository as `examples/request.json`:

```bash
myjev score --artifact {row['repo_id']} \\
  --revision {row['runtime_revision']} \\
  --input examples/request.json

# Run the local HTTP service in a separate terminal.
myjev serve --artifact {row['repo_id']} \\
  --revision {row['runtime_revision']}

# Once /readyz succeeds:
curl -fsS http://127.0.0.1:8000/score \\
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Python, CLI and HTTP return the same response fields. Keep your artifact revision pinned in deployment configuration. Publishing this package does not start a hosted endpoint. The source repository includes tested local Docker instructions and a separate, unexecuted managed-hosting recipe.

## What the scores mean

**Selection scores** are normalized values used to rank the supplied candidates. The selected ID is the highest-scoring choice.

{confidence}

The API also returns `artifact_revision` and `calibration_revision` so callers can identify the exact behavior they used. Set acceptance/deferral thresholds using representative calibration data. The [threshold protocol](https://github.com/bahree/myJEV/blob/main/docs/protocol.md) explains why the searched empirical operating points are not certified risk guarantees. An explicit `other` candidate is a classification option; confidence-based deferral is a separate decision by your application.

## What the matched calibration follow-up changed

The [exploratory controls](https://github.com/bahree/myJEV/blob/main/results/review-calibration-v1/report.md) apply identical selection-temperature fitting to every method, and separately apply one binary log-odds temperature to each trained correctness estimate. All fits use calibration only. With selection temperature, exact RL has slightly lower mean 4B Brier (mixed seed directions) and lower 9B Brier on all three seeds; continued supervision leads at 0.8B. For the separate binary-temperature correctness estimate, the lower 4B exact-RL mean is driven by seed 33: exact-minus-continued Brier deltas are +0.0018 / +0.0004 / -0.0179. This qualifies the earlier released-configuration comparison. Brier and error ranking can move differently, so it does not automatically choose a new deferral policy.

The card tables still describe this released artifact and its unchanged confidence settings. None of the alternative fits was selected for deployment using test results. Three-seed bootstrap intervals hold those trained checkpoints fixed; seed spread is a separate uncertainty source. [Per-seed contrasts](https://github.com/bahree/myJEV/blob/main/results/review-seeds-v1/report.md) show that exact RL beats continued supervision at 4B in two of three seeds, while sampled RL trails exact in eight of nine size/seed pairs.

## How it was trained

{training}

The matched runs use one example per optimizer update, so the initial and continuation stages each present 4,000 training examples. Training uses the grouped training portion of BANKING77, with separate validation and calibration partitions and the official test split preserved. Candidate order is randomized. The backbone remains frozen while low-rank adapter updates and custom heads learn the task. That reduces training storage; it does not turn the large pretrained backbone into a tiny inference model.

BANKING77 supplies 77 closely related banking intents. It is one task family, not a generalist instruction mixture. CLINC150 was kept outside training and tuning for transfer and unsupported-request evaluation. These released weights have no archive adaptation. The separate blog-archive and scratch-model experiments are documented in the project rather than folded into these results.

## The model family

Quality columns are three-seed BANKING77 means. Runtime columns are measured on the released seed-11 artifacts under the same short-request conditions described above.

| Model | Accuracy | Confidence Brier | HTTP p50 | Allocated VRAM |
|---|---:|---:|---:|---:|
{family}

The standard releases use continued supervised training and temperature scaling. `-RL` releases use exact confidence-aware reward training. The 9B models also change backbone precision to four-bit NF4, so differences cannot be attributed solely to parameter count. The study includes a separate 4B precision control.

## Tested scope and limits

The manifest accepts 2 to 160 candidates and up to 4,096 tokenizer tokens for the entire rendered prompt. Duplicate IDs and oversized requests are rejected instead of silently truncated. Short 160-candidate smoke checks passed for every release; that is not evidence of equally good accuracy or latency at 160 choices. Candidate wording, order, missing correct options and quoted instructions can change decisions. In the [frozen order study](https://github.com/bahree/myJEV/blob/main/results/review-order-v1/report.md), changed selected IDs ranged from 7.89-9.71% per permutation at 0.8B and 3.64-4.97% at larger sizes. Each request received its own seeded shuffle; aggregate accuracy can mask these changes. Check the transfer and robustness results before choosing this model for a new workflow.

The reference backend has been checked for artifact reloads and Python/CLI/HTTP/Docker consistency, including pinned Hub download checks. New quantization, merged weights or optimized backends need their own equivalence and calibration measurements. No matched speed comparison with the proprietary Jev service has been performed.

## Files, licenses and provenance

The package includes `adapter/`, `heads.safetensors`, `manifest.json`, `LICENSE`, `BACKBONE_LICENSE` and `NOTICE.md`. It does not include backbone weights, training text or optimizer state.

- Original myJEV code, adapters and heads: MIT, copyright Amit Bahree.
- Qwen backbone: Apache-2.0, with its original terms retained in `BACKBONE_LICENSE`.
- [BANKING77](https://huggingface.co/datasets/PolyAI/banking77): CC BY 4.0; Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020).
- Backbone revision: `{row['backbone_revision']}`.
- Artifact revision: `{row['artifact_revision']}`.
- Calibration revision: `{row['calibration_revision']}`.

[Source repository](https://github.com/bahree/myJEV) | [Dataset rationale](https://github.com/bahree/myJEV/blob/main/docs/datasets/banking77.md) | [All releases and deployment notes](https://github.com/bahree/myJEV/blob/main/docs/models.md)
'''


def main():
    default_cards=Path('model_cards') if Path('model_cards/facts.json').is_file() else Path('publishing/huggingface')
    p=argparse.ArgumentParser();p.add_argument('--facts',type=Path,default=default_cards/'facts.json');p.add_argument('--output',type=Path,default=default_cards);a=p.parse_args()
    facts=json.loads(a.facts.read_text());a.output.mkdir(parents=True,exist_ok=True)
    for row in facts['models']:
        text=render(row,facts)
        if '\u2014' in text:raise ValueError('No em dashes in authored cards')
        (a.output/(row['public_name']+'.md')).write_text(text)
        print(row['public_name'])


if __name__=='__main__':main()
