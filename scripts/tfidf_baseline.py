import argparse
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from myjev.calibration import fit_temperature
from myjev.data import read_jsonl, write_jsonl, validate_isolation
from myjev.metrics import summarize, thresholds_from_calibration

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/banking77")
p.add_argument("--output", default="results/tfidf")
a = p.parse_args()
parts = {s: read_jsonl(Path(a.data)/f"{s}.jsonl") for s in ("train", "validation", "calibration", "test")}
validate_isolation(parts)
start = time.perf_counter()
best = None
for c in (.1, 1., 10.):
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True), LogisticRegression(C=c,max_iter=1000))
    model.fit([r["context"] for r in parts["train"]], [r["label"] for r in parts["train"]])
    accuracy = model.score([r["context"] for r in parts["validation"]], [r["label"] for r in parts["validation"]])
    if best is None or accuracy > best[0]:
        best = accuracy, model, c
_, model, c = best
classes = model.classes_.tolist()
cal_logits = model.decision_function([r["context"] for r in parts["calibration"]])
targets = [classes.index(r["label"]) for r in parts["calibration"]]
t = fit_temperature(cal_logits, targets)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out / "model.joblib")
base_rate = float(np.mean(cal_logits.argmax(-1) == targets))
for mode in ("raw", "temperature", "constant", "half"):
    preds = {}
    for split in ("calibration", "test"):
        logits = model.decision_function([r["context"] for r in parts[split]]) / (t if mode == "temperature" else 1.)
        probs = np.exp(logits-logits.max(-1,keepdims=True))
        probs /= probs.sum(-1,keepdims=True)
        preds[split] = [{"id": r["id"], "group": r["group"], "label": r["label"],
                        "selected_id": classes[int(p.argmax())], "selection_scores": dict(zip(classes, p.tolist())),
                        "confidence": base_rate if mode == "constant" else (.5 if mode == "half" else float(p.max()))}
                       for r,p in zip(parts[split],probs)]
    thresholds = thresholds_from_calibration([r["selected_id"] == r["label"] for r in preds["calibration"]],
                                             [r["confidence"] for r in preds["calibration"]])
    write_jsonl(out / f"{mode}-predictions.jsonl", preds["test"])
    result = summarize(preds["test"],thresholds)
    result.update(C=c, temperature=t if mode == "temperature" else 1., elapsed_seconds=time.perf_counter()-start)
    (out / f"{mode}-metrics.json").write_text(json.dumps(result,indent=2))
    print(mode, result["accuracy"],result["correctness_brier"],flush=True)
