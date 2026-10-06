"""Local Qwen annotation with provenance; never marks machine labels human reviewed."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from myjev.data import read_jsonl,validate_isolation
from myjev.judging import SYSTEM,Judgment,prompt,sha,validate_judgment
from myjev.model import load_backbone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'annotation/local-judge-v1'
SYSTEM_JSON=SYSTEM.replace('Emit the annotation using the submit_annotation tool.','Return only one JSON object matching the supplied schema, without Markdown fences or extra text.')


def decode_judgment(row,text,truncated=False):
    if truncated:raise ValueError('Generation reached its token budget without EOS')
    return validate_judgment(row,json.loads(text.strip()))


def prepare():
    rows=read_jsonl(ROOT/'annotation/requests.jsonl')
    validate_isolation({split:[r for r in rows if r['split']==split] for split in ('development','train','validation','calibration','test')})
    config=json.loads((ROOT/'configs/9b.json').read_text())
    plan={'source_requests_sha256':sha((ROOT/'annotation/requests.jsonl').read_text()),'model':config['backbone'],'revision':config['revision'],'precision':'nf4',
          'system_sha256':sha(SYSTEM_JSON),'schema':Judgment.model_json_schema(),'prompt_version':'archive-judge-v1-local-json',
          'max_input_tokens':4096,'max_output_tokens':768,'generation':'greedy, thinking disabled, no retries or automatic quote repair',
          'development_gate':'At least 80% structurally valid development judgments with validated exact evidence. This is an engineering gate, not human accuracy.',
          'limitations':['Same Qwen family as the adaptation student, so teacher agreement is not independent evidence of correctness.',
                         'Cross-check against the earlier session judgments is descriptive; no human audit is implied.',
                         'Previously prepared exact visible text and grouped splits are reused; six posts were excerpted during preparation.'],
          'requests':len(rows),'human_reviewed':False}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'frozen-plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Judge inputs changed after freeze')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    return rows,plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    rows,plan=prepare()
    if not a.execute:print(json.dumps({'prepared':len(rows),'local_model_calls':0}));return
    model=load_backbone(plan['model'],plan['revision'],'nf4',a.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(plan['model'],revision=plan['revision'])
    path=OUT/'labels.jsonl';done={r['id']:r for r in read_jsonl(path)} if path.exists() else {}
    if len(done)!=(len(read_jsonl(path)) if path.exists() else 0):raise ValueError('Duplicate judge records')
    for phase in ('development','remaining'):
        selected=[r for r in rows if (r['split']=='development')==(phase=='development')]
        for row in selected:
            if row['id'] in done:continue
            payload={**prompt(row),'output_schema':Judgment.model_json_schema()}
            messages=[{'role':'system','content':SYSTEM_JSON},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False,truncation=False)
            record={'id':row['id'],'group':row['group'],'split':row['split'],'rubric':row['rubric'],
                    'visible_text_sha256':sha(row['context']),'prompt_sha256':sha(text),'label_source':'llm','human_reviewed':False,'reviewed':False,
                    'requested_model':plan['model'],'resolved_model':plan['model']+'@'+plan['revision'],'precision':'nf4',
                    'utc':datetime.now(timezone.utc).isoformat(),'input_tokens':inputs['input_ids'].shape[1]}
            start=time.perf_counter()
            try:
                if record['input_tokens']>plan['max_input_tokens']:raise ValueError('Oversized judge input; no truncation')
                inputs={k:v.to(a.device) for k,v in inputs.items()}
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=plan['max_output_tokens'],do_sample=False,use_cache=True)
                generated=output[0,record['input_tokens']:]
                raw=tokenizer.decode(generated,skip_special_tokens=True).strip()
                eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else (eos or [])
                truncated=len(generated)==plan['max_output_tokens'] and int(generated[-1]) not in eos
                record.update(raw_text=raw,output_tokens=len(generated),truncated=truncated)
                record.update(decode_judgment(row,raw,truncated),valid=True)
            except (ValueError,KeyError) as error:
                record.update(valid=False,validation_error=str(error))
            record['seconds']=time.perf_counter()-start
            with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            done[row['id']]=record
            (OUT/'progress.json').write_text(json.dumps({'phase':phase,'records':len(done),'total':len(rows),'valid':sum(r['valid'] for r in done.values())},indent=2)+'\n')
            print(json.dumps({'phase':phase,'records':len(done),'valid':record['valid']}),flush=True)
        if phase=='development':
            development=[done[r['id']] for r in selected]
            valid=sum(r['valid'] for r in development)/len(development)
            previous={r['id']:r for r in read_jsonl(ROOT/'annotation/codex-judge-pilot/labels.jsonl')}
            comparable=[r for r in development if r.get('valid') and r.get('status')=='labelled' and previous.get(r['id'],{}).get('status')=='labelled']
            review={'valid_fraction':valid,'development_n':len(development),'comparable_labelled_judgments':len(comparable),
                    'agreement_with_session_judge':sum(r['label']==previous[r['id']]['label'] for r in comparable)/len(comparable) if comparable else None,
                    'human_reviewed':False,'semantics':'Machine-judge agreement, not audited accuracy; valid means schema and quote checks only.'}
            (OUT/'development-report.json').write_text(json.dumps(review,indent=2)+'\n')
            if valid<.8:raise ValueError('Development structural-validity gate failed; remaining labels not generated')
    (OUT/'complete.json').write_text(json.dumps({'records':len(done),'human_reviewed':False,'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')

if __name__=='__main__':main()
