"""Freeze and score authored policy-edit pairs without fitting any parameters."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    pairs={}
    for row in rows:
        pairs.setdefault(row['pair_id'], []).append(row)
    correct=[r['selected_id']==r['label'] for r in rows]
    change=[v for v in pairs.values() if v[0]['expected_change']]
    stable=[v for v in pairs.values() if not v[0]['expected_change']]
    return {'requests':len(rows),'pairs':len(pairs),
        'accuracy':sum(correct)/len(rows),
        'both_variants_correct':sum(all(r['selected_id']==r['label'] for r in v) for v in pairs.values())/len(pairs),
        'changed_when_required':sum(v[0]['selected_id']!=v[1]['selected_id'] for v in change)/len(change) if change else None,
        'unchanged_when_required':sum(v[0]['selected_id']==v[1]['selected_id'] for v in stable)/len(stable) if stable else None,
        'confidence_brier':sum((r['confidence']-c)**2 for r,c in zip(rows,correct))/len(rows),
        'mean_confidence_correct':sum(r['confidence'] for r,c in zip(rows,correct) if c)/sum(correct) if sum(correct) else None,
        'mean_confidence_incorrect':sum(r['confidence'] for r,c in zip(rows,correct) if not c)/(len(rows)-sum(correct)) if not all(correct) else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixtures',default='fixtures/policy-edits-v1/pairs.json')
    p.add_argument('--output',default='results/policy-edits-v1')
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--artifact')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    frozen=out/'frozen-plan.json'
    if a.freeze:
        if frozen.exists(): raise ValueError('plan already frozen')
        pairs=json.loads(Path(a.fixtures).read_text())
        for pair in pairs:
            assert len(pair['variants'])==2
            assert pair['expected_change']==(pair['variants'][0]['label']!=pair['variants'][1]['label'])
            assert pair['variants'][0]['request']['context']==pair['variants'][1]['request']['context']
            assert pair['variants'][0]['request']['candidates']==pair['variants'][1]['request']['candidates']
            for row in pair['variants']:
                assert row['label'] in [c['id'] for c in row['request']['candidates']]
        frozen.write_text(json.dumps({'frozen_at':datetime.now(timezone.utc).isoformat(),'fixture_sha256':digest(a.fixtures),'script_sha256':digest(__file__),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'pairs':len(pairs),'requests':sum(len(p['variants']) for p in pairs),'design':'24 authored pairs from six templates; 16 required changes and 8 invariances, alternating candidate order. Labels fixed before inference. No tuning or threshold fitting. Six existing seed-11 released candidates, three sizes by continued SFT plus temperature or exact RL.','limits':'Synthetic correlated templates, one checkpoint per method/size, no population inference, no PolicyLM benchmark. Candidate scores and correctness confidence reported separately.'},indent=2)+'\n')
        return
    plan=json.loads(frozen.read_text())
    if digest(a.fixtures)!=plan['fixture_sha256'] or digest(__file__)!=plan['script_sha256']: raise ValueError('frozen inputs changed')
    if not a.artifact: raise ValueError('--artifact required')
    from myjev.inference import DecisionModel
    model=DecisionModel.load(a.artifact,device=a.device)
    target=out/Path(a.artifact).name; target.mkdir(exist_ok=False)
    rows=[]
    with (target/'predictions.jsonl').open('w') as f:
        for pair in json.loads(Path(a.fixtures).read_text()):
            for row in pair['variants']:
                result=model.score(row['request'])
                result.update(id=row['id'],pair_id=pair['id'],kind=pair['kind'],label=row['label'],expected_change=pair['expected_change'])
                f.write(json.dumps(result)+'\n'); f.flush(); rows.append(result)
    report={'artifact':a.artifact,'artifact_revision':model.manifest['artifact_revision'],'fixture_sha256':plan['fixture_sha256'],'overall':summarize(rows),'by_kind':{k:summarize([r for r in rows if r['kind']==k]) for k in sorted({r['kind'] for r in rows})},'scope':plan['limits']}
    (target/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
