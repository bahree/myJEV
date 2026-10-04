import argparse
import json
import time
import random
from pathlib import Path
import torch
from .data import read_jsonl, write_jsonl
from .inference import DecisionModel
from .metrics import summarize, thresholds_from_calibration


def predict(model, rows):
    out = []
    for row in rows:
        start = time.perf_counter()
        response = model.score({k: row[k] for k in ("context", "instructions", "candidates")}, diagnostics=True)
        moments = response.pop("confidence_moments")
        raw_policy = torch.tensor(response["selection_logits"]).softmax(-1)
        correct = torch.tensor([float(c["id"] == row["label"]) for c in row["candidates"]])
        response["sampled_expected_accuracy"] = float((raw_policy * correct).sum())
        response["sampled_expected_reward"] = float((raw_policy * (2 * correct * torch.tensor(moments["mean"]) - torch.tensor(moments["second"]))).sum())
        provenance = {k: row[k] for k in ("label_source", "human_reviewed", "judge_model") if k in row}
        out.append({**response, **provenance, "id": row["id"], "group": row["group"], "label": row["label"],
                    "seconds": time.perf_counter()-start})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    rows, cal = read_jsonl(a.data), read_jsonl(a.calibration)
    if set(r["group"] for r in rows) & set(r["group"] for r in cal):
        raise ValueError("evaluation and calibration groups overlap")
    if a.limit:
        rows = random.Random(42).sample(rows, min(a.limit, len(rows)))
        cal = random.Random(43).sample(cal, min(a.limit, len(cal)))
    start = time.perf_counter()
    model = DecisionModel.load(a.artifact, device=a.device)
    cold = time.perf_counter()-start
    calibrated = predict(model, cal)
    thresholds = thresholds_from_calibration([r["selected_id"] == r["label"] for r in calibrated],
                                             [r["confidence"] for r in calibrated])
    predictions = predict(model, rows)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", predictions)
    write_jsonl(out / "calibration-predictions.jsonl", calibrated)
    # Keep deployed behavior as metrics.json; compare identical policy readouts separately.
    for mode in ("policy", "scalar", "selection"):
        cal_view = [{**r, "confidence": r["confidence_controls"][mode]} for r in calibrated]
        test_view = [{**r, "confidence": r["confidence_controls"][mode]} for r in predictions]
        points = thresholds_from_calibration([r["selected_id"] == r["label"] for r in cal_view], [r["confidence"] for r in cal_view])
        report = summarize(test_view, points)
        report.update(confidence_mode=mode, artifact_revision=model.manifest["artifact_revision"], subset_limit=a.limit)
        (out / f"{mode}-metrics.json").write_text(json.dumps(report, indent=2))
    result = summarize(predictions, thresholds)
    import numpy as np
    result.update(cold_start_seconds=cold, latency_p50_p95_seconds=np.quantile([r["seconds"] for r in predictions], [.5,.95]).tolist(),
                  artifact_revision=model.manifest["artifact_revision"], subset_limit=a.limit,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(a.device) if torch.cuda.is_available() else 0)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "reliability"}, indent=2))


if __name__ == "__main__":
    main()
