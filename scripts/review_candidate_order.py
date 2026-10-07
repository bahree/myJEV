"""Evaluate released seed-11 models under fixed candidate permutations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time


def permuted_candidates(candidates, example_id, seed):
    result=list(candidates)
    key=int(hashlib.sha256(f'{seed}:{example_id}'.encode()).hexdigest(),16)
    random.Random(key).shuffle(result)
    return result


def main():
    import torch
    from myjev.inference import DecisionModel
    from myjev.data import read_jsonl
    from myjev.metrics import summarize
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',required=True,choices=['0.8b','4b','9b'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--hub',action='store_true',help='Load published immutable releases instead of author-local artifacts')
    p.add_argument('--output',type=Path,default=Path('results/review-order-v1'))
    a=p.parse_args();protocol_path=Path('configs/review-order-v1.json');protocol=json.loads(protocol_path.read_text())
    rows=read_jsonl('data/banking77/test.jsonl');assert len(rows)==3080
    dest=a.output/a.size;dest.mkdir(parents=True,exist_ok=True)
    for method in protocol['methods']:
        source=Path('results/longer-v1')/a.size/'main'/'seed-11'/method
        if method=='continued_sft':
            source_metrics=source/'posthoc/temperature-metrics.json';source_predictions=source/'posthoc/temperature-predictions.jsonl'
        else:
            source_metrics=source/'evaluation/metrics.json';source_predictions=source/'evaluation/predictions.jsonl'
        if source_predictions.exists():
            baseline=read_jsonl(source_predictions)
            baseline_source=source_predictions
        else:
            # Public exports retain text-free prediction inputs, not full training text.
            try:
                from .review_calibration import view
            except ImportError:
                from review_calibration import view
            compact_root=Path('results/review-calibration-v1')/a.size/'seed-11'/method
            baseline_source=compact_root/'inputs.json.gz'
            compact=json.loads(gzip.decompress(baseline_source.read_bytes()))
            fit=json.loads((compact_root/'fit.json').read_text())
            baseline=view(compact['test'],'selection_temperature' if method=='continued_sft' else 'native',fit['selection_temperature'],fit['learned_temperature'])
        assert [r['id'] for r in baseline]==[r['id'] for r in rows]
        thresholds={name:p['threshold'] for name,p in json.loads(source_metrics.read_text())['operating_points'].items()}
        artifact=Path('artifacts/hub-ready-v3')/f'myjev-{a.size}-{method}-seed11'
        output=dest/method;output.mkdir(exist_ok=True)
        if a.hub:
            facts_path=Path('model_cards/facts.json') if Path('model_cards/facts.json').exists() else Path('publishing/huggingface/facts.json')
            facts=json.loads(facts_path.read_text())
            release=next(r for r in facts['models'] if r['size']==a.size and r['method']==method)
            model=DecisionModel.load(release['repo_id'],revision=release['runtime_revision'],device=a.device)
        else:
            model=DecisionModel.load(artifact,device=a.device)
        metadata={'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(Path('data/banking77/test.jsonl').read_bytes()).hexdigest(),'artifact_revision':model.manifest['artifact_revision'],'canonical_manifest_sha256':hashlib.sha256(json.dumps(model.manifest,sort_keys=True).encode()).hexdigest(),'baseline_source':str(baseline_source),'baseline_prediction_sha256':hashlib.sha256(baseline_source.read_bytes()).hexdigest(),'baseline_metrics_sha256':hashlib.sha256(source_metrics.read_bytes()).hexdigest(),'fixed_thresholds':thresholds,'model_size':a.size,'method':method}
        (output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
        for seed in protocol['permutation_seeds']:
            metric_path=output/f'order-{seed}-metrics.json'
            if metric_path.exists():continue
            predictions=[];start=time.perf_counter()
            for i,row in enumerate(rows):
                request={k:row[k] for k in ('context','instructions','candidates')}
                request['candidates']=permuted_candidates(request['candidates'],row['id'],seed)
                scored=model.score(request)
                predictions.append(scored|{k:row[k] for k in ('id','group','label')})
                if (i+1)%250==0:print(json.dumps({'size':a.size,'method':method,'order':seed,'examples':i+1,'total':len(rows),'seconds':time.perf_counter()-start}),flush=True)
            pred_path=output/f'order-{seed}-predictions.jsonl.gz'
            pred_path.write_bytes(gzip.compress(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in predictions).encode(),mtime=0))
            metrics=summarize(predictions,thresholds)
            metrics.update(order_seed=seed,artifact_revision=model.manifest['artifact_revision'],source_order_accuracy=sum(r['selected_id']==r['label'] for r in baseline)/len(rows),prediction_disagreement=sum(a['selected_id']!=b['selected_id'] for a,b in zip(baseline,predictions))/len(rows),mean_absolute_confidence_change=sum(abs(a['confidence']-b['confidence']) for a,b in zip(baseline,predictions))/len(rows),elapsed_seconds=time.perf_counter()-start,prediction_sha256=hashlib.sha256(pred_path.read_bytes()).hexdigest(),scope=protocol['classification'])
            metric_path.write_text(json.dumps(metrics,indent=2)+'\n')
            print(json.dumps({'completed':str(metric_path),'accuracy':metrics['accuracy'],'brier':metrics['correctness_brier']}),flush=True)
        del model
        import gc;gc.collect();torch.cuda.empty_cache()
    print(json.dumps({'size':a.size,'state':'completed'}),flush=True)


if __name__=='__main__':main()
