"""Untouched 4B readout probe on seven already-disclosed authored demo cases.

Correctness-first full-continuation scoring duplicates prompts across candidates.
This is not the paper's KV-cached implementation or a speed benchmark.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def suffix_tokens(tokenizer, prefix, suffix):
    before=tokenizer.encode(prefix,add_special_tokens=False)
    complete=tokenizer.encode(prefix+suffix,add_special_tokens=False)
    if complete[:len(before)]!=before or len(complete)<=len(before):
        raise ValueError('Tokenization changes at the prompt/suffix boundary; cannot reuse this prefix')
    return before,complete[len(before):]


def sequence_log_score(logits, token_ids, prefix_length):
    """Sum full-vocabulary conditional log probabilities, including closing bracket."""
    if prefix_length<1 or prefix_length>=len(token_ids):
        raise ValueError('Need a nonempty prefix and suffix')
    positions=logits[prefix_length-1:len(token_ids)-1].float().log_softmax(-1)
    target=token_ids[prefix_length:]
    return positions.gather(-1,target[:,None]).sum()


def numeric_prompt(tokenizer, request):
    state=json.dumps({'instructions':request['instructions'],'context':request['context']},ensure_ascii=False)
    options='\n'.join(f'[{i}] {c["description"]}' for i,c in enumerate(request['candidates'],1))
    user=f'State: {state}\nOptions:\n{options}\nSelect one option. Answer only with its bracketed numeric identifier.'
    return tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)+'Best answer: ['


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/numeric-readout-v1'))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory; frozen evidence is never overwritten')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('configs/4b.json').read_text())
    requests=Path('examples/demo-requests.jsonl')
    expectations=Path('examples/demo-expectations.json')
    protocol={'scope':'Seven already-disclosed authored examples, descriptive interface probe, not held-out performance evidence',
        'model':'Qwen/Qwen3.5-4B','revision':config['revision'],'precision':'bf16','device':args.device,
        'training':'None; untouched backbone, no adapter or learned confidence head',
        'comparisons':['Existing myJEV chat-v2 single-token aliases','Bracketed numeric suffix sum-log-probability'],
        'numeric_scoring':'Full-vocabulary token log-softmax; sum suffix log probabilities including closing bracket; normalize over candidate completions; no length normalization',
        'compute':'Full continuation teacher forcing, duplicated prompt per candidate, candidate batch per request, use_cache=False. Not cached paper reproduction or comparable latency.',
        'limits':'4096 tokens; finite candidate lists in context. Numeric IDs do not remove context/memory limits or add vision capability.',
        'selection':'No prompt search, no tuning, original candidate order; all seven predetermined examples retained',
        'requests_sha256':hashlib.sha256(requests.read_bytes()).hexdigest(),'expectations_sha256':hashlib.sha256(expectations.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch_version':torch.__version__}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    from transformers import AutoTokenizer
    from myjev.model import load_backbone
    from myjev.prompt import discover_aliases,encode
    from myjev.schema import ScoreRequest
    begin=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(protocol['model'],revision=protocol['revision'])
    model=load_backbone(protocol['model'],protocol['revision'],device=args.device).eval()
    load_seconds=time.perf_counter()-begin
    aliases,alias_ids=discover_aliases(tokenizer,3)
    rows=[]
    expected=json.loads(expectations.read_text())['cases']
    with (out/'execution.jsonl').open('w') as log,torch.inference_mode():
        for index,line in enumerate(requests.read_text().splitlines()):
            request=json.loads(line);n=len(request['candidates']);start=time.perf_counter()
            inputs=encode(tokenizer,ScoreRequest.model_validate(request),aliases,4096)
            inputs={k:v.to(args.device) for k,v in inputs.items()}
            # Match existing alias score semantics exactly: final state dot output rows.
            decoder=getattr(model,model.base_model_prefix)
            hidden=decoder(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1].float()
            weights=model.get_output_embeddings().weight[alias_ids[:n]].float()
            alias_scores=(hidden@weights.T).squeeze(0)
            prefix=numeric_prompt(tokenizer,request)
            tokenized=[suffix_tokens(tokenizer,prefix,f'{i}]') for i in range(1,n+1)]
            lengths=[len(a)+len(b) for a,b in tokenized]
            if max(lengths)>4096:raise ValueError('Oversized input; no truncation')
            ids=torch.full((n,max(lengths)),tokenizer.pad_token_id or tokenizer.eos_token_id,device=args.device,dtype=torch.long)
            mask=torch.zeros_like(ids)
            for i,(a,b) in enumerate(tokenized):
                ids[i,:lengths[i]]=torch.tensor(a+b,device=args.device);mask[i,:lengths[i]]=1
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).logits
            numeric_scores=torch.stack([sequence_log_score(logits[i],ids[i,:lengths[i]],len(tokenized[i][0])) for i in range(n)])
            result={'line':index+1,'case_id':expected[index]['id'],'expected_id':expected[index]['expected_id'],
                'alias_prompt_tokens':inputs['input_ids'].shape[1],'numeric_prompt_tokens':len(tokenized[0][0]),
                'numeric_suffix_tokens':[b for a,b in tokenized],'numeric_processed_tokens_including_repeated_prefix':sum(lengths),
                'numeric_candidate_log_probabilities':numeric_scores.tolist(),'elapsed_seconds_unoptimized':time.perf_counter()-start}
            for method,scores in [('aliases',alias_scores),('numeric',numeric_scores)]:
                probabilities=scores.softmax(-1)
                choice=request['candidates'][int(probabilities.argmax())]['id']
                result[method]={'selected_id':choice,'correct':choice==result['expected_id'],
                    'selection_scores':{c['id']:float(p) for c,p in zip(request['candidates'],probabilities)},
                    'max_selection_probability':float(probabilities.max())}
            rows.append(result);log.write(json.dumps(result)+'\n');log.flush()
            print(json.dumps(result),flush=True)
            del logits,hidden
    summary={'protocol':protocol,'load_seconds':load_seconds,'total_seconds':time.perf_counter()-begin,'n':len(rows),
        'correct':{method:sum(r[method]['correct'] for r in rows) for method in ('aliases','numeric')},'rows':rows,
        'interpretation':'Selection probabilities are not an independently calibrated correctness estimate. No generalization, architecture-superiority or speed claim is supported by these seven previously viewed examples.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Numeric readout: seven-case untouched-backbone probe','',summary['interpretation'],'',
        '| Case | Expected | Existing aliases | Numeric suffix |','|---|---|---|---|']
    for r in rows:lines.append(f"| {r['case_id']} | {r['expected_id']} | {r['aliases']['selected_id']} | {r['numeric']['selected_id']} |")
    lines.extend(['',f"Correct: aliases {summary['correct']['aliases']}/7; numeric {summary['correct']['numeric']}/7.",
        '', 'No fine-tuning. The same untouched pinned BF16 4B backbone was used. Prompt format and answer readout both change, so this does not isolate bracket tokens alone. Numeric scoring includes the closing bracket and sums full-vocabulary suffix log probabilities; it does not length-normalize.',
        '', 'The implementation repeats the prompt across candidate continuations and teacher-forces full sequences. It does not reproduce cached prefill/branch execution and is not a latency comparison. Raw execution timings are retained only as logs. Token-boundary checks fail closed.',
        '', 'Recommendation: retain this as an implementation demonstration. A separately frozen broader readout comparison is needed before changing release architecture. Numeric identifiers allow more encodable labels, not infinite context or automatic multimodal support. The paper\'s tree-local training distribution also need not equal its globally normalized inference distribution.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
