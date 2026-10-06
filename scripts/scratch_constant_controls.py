"""Calibration-base-rate controls, without retraining or changing selections."""
import json
from pathlib import Path
import numpy as np
from myjev.data import read_jsonl, write_jsonl
from myjev.metrics import summarize, thresholds_from_calibration


def main():
    root=Path('results/scratch-study-v1')
    for run in sorted(root.glob('main/seed-*/*')):
        cal=read_jsonl(run/'evaluation/calibration-predictions.jsonl')
        test=read_jsonl(run/'evaluation/predictions.jsonl')
        base=sum(r['selected_id']==r['label'] for r in cal)/len(cal)
        cal=[{**r,'confidence':base,'confidence_mode':'constant'} for r in cal]
        test=[{**r,'confidence':base,'confidence_mode':'constant'} for r in test]
        thresholds=thresholds_from_calibration([r['selected_id']==r['label'] for r in cal],[base]*len(cal))
        report=summarize(test,thresholds)
        report.update(confidence_mode='constant',base_rate=base,calibration_n=len(cal),
                      known_probability_confidence_mse=float(np.mean([(r['confidence']-r['known_selected_probability'])**2 for r in test])),
                      known_distribution_brier=float(np.mean([r['known_distribution_brier'] for r in test])),
                      artifact_revision=test[0]['artifact_revision'])
        out=run/'constant';out.mkdir(exist_ok=True)
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Wrote 12 constant-confidence controls; selections unchanged.')


if __name__=='__main__':main()
