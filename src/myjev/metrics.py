import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import beta as beta_distribution


def risk_at_threshold(correct, confidence, threshold):
    accepted = np.asarray(confidence) >= threshold
    c = np.asarray(correct)[accepted]
    n, errors = len(c), int((1-c).sum())
    return {"threshold": float(threshold), "coverage": float(accepted.mean()), "accepted": n,
            "error": float(errors/n) if n else None,
            "error_upper_95": float(beta_distribution.ppf(.95, errors+1, n-errors)) if n and errors < n else (1.0 if n else None)}


def thresholds_from_calibration(correct, confidence, coverages=(.5, .8, .9), targets=(.01, .05)):
    correct, confidence = np.asarray(correct), np.asarray(confidence)
    if not len(correct):
        raise ValueError("empty calibration split")
    result = {}
    for coverage in coverages:
        result[f"coverage_{coverage}"] = float(np.sort(confidence)[::-1][max(0, int(np.ceil(coverage*len(correct)))-1)])
    for target in targets:
        eligible = [float(t) for t in np.unique(confidence)
                    if risk_at_threshold(correct, confidence, t)["error"] <= target]
        result[f"empirical_error_{target}"] = min(eligible) if eligible else 1.000001
    return result


def cluster_interval(values, groups, seed=42, repeats=1000):
    values, groups = np.asarray(values), np.asarray(groups)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    estimates = [float(values[np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])].mean())
                 for _ in range(repeats)]
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(rows, thresholds=None, bins=15):
    if not rows:
        raise ValueError("empty evaluation")
    correct = np.array([r["selected_id"] == r["label"] for r in rows], dtype=float)
    confidence = np.array([r["confidence"] for r in rows])
    if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("invalid confidence")
    reliability = []
    for lo, hi in zip(np.linspace(0, 1, bins+1)[:-1], np.linspace(0, 1, bins+1)[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else confidence <= hi)
        reliability.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                            "confidence": float(confidence[mask].mean()) if mask.any() else None,
                            "accuracy": float(correct[mask].mean()) if mask.any() else None})
    multiclass = [sum((p-float(k == r["label"]))**2 for k, p in r["selection_scores"].items())
                  for r in rows]
    result = {"n": len(rows), "accuracy": float(correct.mean()),
              "accuracy_cluster_ci95": cluster_interval(correct, [r["group"] for r in rows]),
              "macro_f1": float(f1_score([r["label"] for r in rows], [r["selected_id"] for r in rows], average="macro", zero_division=0)),
              "correctness_brier": float(((confidence-correct)**2).mean()),
              "selection_multiclass_brier": float(np.mean(multiclass)) if all(r["label"] in r["selection_scores"] for r in rows) else None,
              "correctness_auroc": float(roc_auc_score(correct, confidence)) if len(set(correct)) > 1 else None,
              "ece": sum(b["n"] / len(rows) * abs(b["confidence"]-b["accuracy"]) for b in reliability if b["n"]),
              "ece_bins": "15 equal-width bins, final bin inclusive" if bins == 15 else f"{bins} equal-width bins",
              "reliability": reliability,
              "operating_points": {k: risk_at_threshold(correct, confidence, t) for k, t in (thresholds or {}).items()}}
    if any(r.get("label_source") == "llm" for r in rows):
        result["reference_label_source"] = "llm" if all(r.get("label_source") == "llm" for r in rows) else "mixed"
        result["metric_semantics"] = "Agreement with reference labels, including machine-generated labels; not independently established human correctness."
    if all("sampled_expected_reward" in r for r in rows):
        result["sampled_policy"] = {
            "expected_accuracy": float(np.mean([r["sampled_expected_accuracy"] for r in rows])),
            "expected_confidence_reward": float(np.mean([r["sampled_expected_reward"] for r in rows])),
            "semantics": "exact expectation of sampled joint policy at raw training logits; distinct from deployed argmax behavior"
        }
    for point in result["operating_points"].values():
        selected = [r for r in rows if r["confidence"] >= point["threshold"]]
        point["error_cluster_ci95"] = cluster_interval(
            [float(r["selected_id"] != r["label"]) for r in selected], [r["group"] for r in selected]) if selected else None
    return result
