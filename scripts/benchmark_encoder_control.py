"""Fresh-process encoder latency/memory and saved-checkpoint prediction check."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from myjev.data import read_jsonl
from run_encoder_control import make_predictions

root=Path('results/encoder-control-v1');artifact=Path('artifacts/encoder-control-v1/best')
manifest=json.loads((artifact/'control-manifest.json').read_text());rows=read_jsonl('data/banking77/test.jsonl')
start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(artifact)
model=AutoModelForSequenceClassification.from_pretrained(artifact,attn_implementation='sdpa').to('cuda:0').eval()
load_seconds=time.perf_counter()-start
inputs=tokenizer([r['context'] for r in rows[:32]],padding=True,truncation=False,return_tensors='pt').to('cuda:0')
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs).logits.float().cpu()
observed=make_predictions(rows[:32],logits.tolist(),manifest['classes'],1.)
reference=read_jsonl(root/'raw-test-predictions.jsonl')[:32]
max_diff=max(abs(a['selection_scores'][c]-b['selection_scores'][c]) for a,b in zip(observed,reference,strict=True) for c in manifest['classes'])
assert max_diff<1e-6 and all(a['selected_id']==b['selected_id'] for a,b in zip(observed,reference,strict=True))
del inputs,logits
inputs=tokenizer(rows[0]['context'],return_tensors='pt').to('cuda:0');times=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for _ in range(10):model(**inputs)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for _ in range(100):
        t=time.perf_counter();model(**inputs);torch.cuda.synchronize();times.append(time.perf_counter()-t)
result={'scope':'Fresh process, no optimizer or old model references. Single fixed77-way classifier, one9-token testrequest, model-only timing, noHTTP/tokenization. Same32-row batch for reloadverification.','test_id':rows[0]['id'],'input_tokens':inputs['input_ids'].shape[1],'loaded_seconds_cached':load_seconds,'warmup':10,'repeats':100,'p50_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'seconds':times,'reload_max_probability_difference':max_diff,'reload_selected_ids_equal':True,'weights_sha256':hashlib.sha256((artifact/'model.safetensors').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prior_benchmark_caveat':'Training-process benchmark retained old autograd references; its allocated memory was not isolated inference memory. Original measurements are preserved.'}
(root/'fresh-process-benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='seconds'}),flush=True)
