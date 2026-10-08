"""Count requests changed by any of the three recorded candidate permutations.

This is a descriptive reanalysis of saved predictions, not new model inference.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def summarize(root):
    protocol_path = root / "configs/review-order-v1.json"
    protocol = json.loads(protocol_path.read_text())
    sources = {}

    def read(path, jsonl=False):
        data = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        decoded = gzip.decompress(data) if path.suffix == ".gz" else data
        return [json.loads(line) for line in decoded.splitlines()] if jsonl else json.loads(decoded)

    read(protocol_path)
    rows = []
    for size in protocol["sizes"]:
        for method in protocol["methods"]:
            baseline = read(root / "results/review-calibration-v1" / size / "seed-11" / method / "inputs.json.gz")["test"]
            original = {row["id"]: row["selected_id"] for row in baseline}
            assert len(original) == len(baseline) == 3080
            changed = set()
            counts = []
            for seed in protocol["permutation_seeds"]:
                directory = root / "results/review-order-v1" / size / method
                predictions = read(directory / f"order-{seed}-predictions.jsonl.gz", jsonl=True)
                by_id = {row["id"]: row["selected_id"] for row in predictions}
                assert len(by_id) == len(predictions) and by_id.keys() == original.keys()
                different = {key for key in original if original[key] != by_id[key]}
                metrics = read(directory / f"order-{seed}-metrics.json")
                assert abs(len(different) / len(original) - metrics["prediction_disagreement"]) < 1e-12
                changed.update(different)
                counts.append(len(different))
            assert max(counts) <= len(changed) <= sum(counts)
            rows.append({"size": size, "method": method, "n": len(original),
                         "permutation_seeds": protocol["permutation_seeds"],
                         "per_permutation_changed": counts, "any_of_three_changed": len(changed),
                         "any_of_three_fraction": len(changed) / len(original)})
    return {"scope": "Exploratory descriptive union over the three recorded per-request permutations, relative to the original order. Each of 3080 requests is counted once per released seed-11 model. No new inference, training, calibration fit, uncertainty interval, or claim about arbitrary orders.",
            "rows": rows, "source_sha256": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/review-order-union-v1"))
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Requests affected by any recorded candidate permutation", "", result["scope"], "",
             "The original report measures each permutation separately. This table takes their union: a request that changed under two orders still counts once. The fraction depends on these three specific permutations and is not the probability of a change under every possible order.", "",
             "| Size | Release | Changed under orders 101 / 202 / 303 | Changed under any of three | Fraction of 3,080 requests |",
             "|---|---|---|---:|---:|"]
    for row in result["rows"]:
        counts = " / ".join(map(str, row["per_permutation_changed"]))
        lines.append(f"| {row['size']} | {row['method']} | {counts} | {row['any_of_three_changed']} | {row['any_of_three_fraction']:.2%} |")
    lines += ["", "Regenerate with `python scripts/summarize_order_union.py`. The summary hashes the original protocol, text-free calibration inputs, recorded predictions and per-permutation metrics. Per-permutation disagreements are checked against their original metrics before taking the union. No original result files are rewritten.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(json.dumps({"rows": len(result["rows"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
