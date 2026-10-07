"""Deterministic illustrative failures from public data/original fixtures only."""
import hashlib
import json
from pathlib import Path

OUT=Path('results/failure-casebook-v1')
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cases=[];sources={}
    def load(p):
        p=Path(p);sources[str(p)]=digest(p);return read(p)
    fixture=Path('fixtures/policy-edits-v1/pairs.json');sources[str(fixture)]=digest(fixture)
    requests={v['id']:v for p in json.loads(fixture.read_text()) for v in p['variants']}
    for model in ('myjev-0.8b-continued_sft-seed11','myjev-4b-exact-seed11'):
        rows=load(f'results/policy-edits-v1/{model}/predictions.jsonl')
        wrong=sorted([r for r in rows if r['selected_id']!=r['label']],key=lambda r:(-r['confidence'],r['id']))[0]
        cases.append({'kind':'policy-edit error','model':model,'source_id':wrong['id'],'request':requests[wrong['id']]['request'],
                      'gold':wrong['label'],'selected':wrong['selected_id'],'confidence':wrong['confidence'],
                      'lesson':'High confidence on a wrong authored policy decision; this selected example is not an error-rate estimate.'})
    for cohort in ('oos-none','unsupported-near-none'):
        rows=load(f'data/clinc150/{cohort}.jsonl');data={r['id']:r for r in rows}
        before={r['id']:r for r in load(f'results/generalization-v1/4b/seed-11/continued_sft/transfer/{cohort}-predictions.jsonl')}
        after=load(f'results/archive-machine-v2/clinc-forgetting/{cohort}-predictions.jsonl')
        errors=sorted([r for r in after if r['selected_id']!=r['label'] and before[r['id']]['selected_id']==r['label']],key=lambda r:(-r['confidence'],r['id']))
        wrong=errors[0];original=data[wrong['id']]
        cases.append({'kind':'unsupported-option regression','model':'4B archive-adapted, seed11','source_id':wrong['id'],
                      'request':{k:original[k] for k in ('context','instructions','candidates')},'gold':wrong['label'],
                      'before_selected':before[wrong['id']]['selected_id'],'before_confidence':before[wrong['id']]['confidence'],
                      'selected':wrong['selected_id'],'confidence':wrong['confidence'],
                      'lesson':'Agreement with the dataset none-option label before adaptation became an in-scope choice afterward. This is public CLINC text. No human adjudication was performed: category overlap can make the reference debatable, especially the vaccines example. This demonstrates a dataset-label regression, not independently proven semantic harm.'})
    path=Path('results/scratch-ladder-v1/deterministic-rule/seed-33/predictions.jsonl')
    if path.exists():
        rows=load(path);wrong=sorted([r for r in rows if r['selected_id']!=r['label']],key=lambda r:(-r['confidence'],r['id']))[0]
        data={r['id']:r for r in load('data/scratch-ladder-v1/deterministic-rule/evaluation.jsonl')}
        cases.append({'kind':'scratch deterministic nuisance failure','model':'scratch ladder v1 seed33 (confounded fixture, preserved failure)',
                      'source_id':wrong['id'],'request':{k:data[wrong['id']][k] for k in ('context','instructions','candidates')},
                      'gold':wrong['label'],'selected':wrong['selected_id'],'confidence':wrong['confidence'],
                      'lesson':'A wrong result on an explicit color. V1 record indices correlate with labels, so success would not isolate rule learning; no mechanism is inferred from this case.'})
    scope='Post-hoc illustrative selection: highest-confidence wrong policy example per named model, highest-confidence newly wrong CLINC example per cohort, and highest-confidence wrong scratch seed33 example. Deterministic ID tie-breaks. These intentionally selected failures do not estimate prevalence or establish causal mechanisms. No private archive text or annotation IDs are included.'
    report={'scope':scope,'cases':cases,'source_sha256':sources,'attribution':{'CLINC150':'CLINC authors, https://github.com/clinc/oos-eval, CC BY 3.0; see docs/datasets/clinc150.md and pinned preparation manifest.','original_fixtures':'Original project policy and scratch fixtures; source files retained.'}}
    OUT.mkdir(exist_ok=True);(OUT/'cases.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Concrete failure casebook','',scope]
    for n,c in enumerate(cases,1):
        descriptions={r['id']:r['description'] for r in c['request']['candidates']}
        lines+=['',f"## {n}. {c['kind']}",'',f"Model: `{c['model']}`. Source record: `{c['source_id']}`.",'',f"Context: {c['request']['context']}",'',f"Instructions: {c['request']['instructions']}",'',f"Reference: **{descriptions.get(c['gold'],c['gold'])}**. Selected: **{descriptions.get(c['selected'],c['selected'])}**. Reported correctness confidence: **{c['confidence']:.4f}**."]
        if 'before_selected' in c:lines+=['',f"Before adaptation: **{descriptions.get(c['before_selected'],c['before_selected'])}**, confidence {c['before_confidence']:.4f}."]
        lines+=['',c['lesson']]
    lines+=['','CLINC text attribution: CLINC authors, [out-of-scope benchmark](https://github.com/clinc/oos-eval), CC BY 3.0. Original authored fixtures and hashes are recorded in `cases.json`. Reproduce: `python scripts/build_failure_casebook.py` using saved local evidence.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'cases':len(cases),'private_archive_text':False}))
if __name__=='__main__':main()
