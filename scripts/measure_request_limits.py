"""Measure exact-token rejection for fixed requests at the aggregate text byte cap."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

from myjev.prompt import encode
from myjev.schema import MAX_INPUT_BYTES, ScoreRequest
from transformers import AutoTokenizer


def fixture(pattern):
    request={'context':'', 'instructions':'Select one.', 'candidates':[{'id':str(i),'description':'x'} for i in range(32)]}
    used=len(request['instructions'])+sum(len(c['id'])+1 for c in request['candidates'])
    for c in request['candidates']:
        room=min(8192-1, MAX_INPUT_BYTES-used)
        text=(pattern*((room//len(pattern.encode()))+1)).encode()[:room].decode('utf-8',errors='ignore')
        c['description']+=text;used+=len(text.encode())
    request['context']='a'*(MAX_INPUT_BYTES-used)
    assert sum(len(s.encode()) for s in [request['context'],request['instructions']]+[v for c in request['candidates'] for v in c.values()])==MAX_INPUT_BYTES
    return ScoreRequest.model_validate(request)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/review-limits-v1/measurement.json'))
    a=p.parse_args();m=json.loads(a.manifest.read_text())
    tokenizer=AutoTokenizer.from_pretrained(m['backbone'],revision=m['tokenizer_revision'],local_files_only=True)
    rows=[]
    for pattern in ['abc 123 !?', '漢字é ']:
        request=fixture(pattern);times=[]
        for _ in range(5):
            start=time.perf_counter()
            try: encode(tokenizer,request,m['aliases'],4096)
            except ValueError as e: assert str(e)=='input exceeds 4096 tokens; no truncation is performed'
            else: raise AssertionError('oversize fixture unexpectedly accepted')
            times.append(time.perf_counter()-start)
        rows.append({'pattern':pattern,'text_bytes':MAX_INPUT_BYTES,'candidate_count':32,'wire_bytes':len(request.model_dump_json().encode()),'tokenize_and_reject_seconds':times,'rejection':'input exceeds 4096 tokens; no truncation is performed'})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'rows':rows,'platform':platform.platform(),'processor':platform.processor(),'backbone':m['backbone'],'tokenizer_revision':m['tokenizer_revision'],'source_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'scope':'CPU tokenization on the A30 host during independent GPU evaluations. Two fixed maximum-aggregate-byte requests, five repetitions each; observed costs, not a worst-case latency bound or GPU inference benchmark.'},indent=2)+'\n')

if __name__=='__main__': main()
