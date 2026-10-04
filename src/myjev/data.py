import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def text_group(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text.casefold()).strip().encode()).hexdigest()


def stratified_groups(rows, seed=42):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    by_label = defaultdict(list)
    for g, items in groups.items():
        labels = {i["label"] for i in items}
        if len(labels) != 1:
            raise ValueError(f"conflicting labels in group {g}")
        by_label[items[0]["label"]].append(g)
    result = {k: [] for k in ("train", "validation", "calibration")}
    rng = random.Random(seed)
    for gs in by_label.values():
        rng.shuffle(gs)
        n = len(gs)
        if n < 10:
            raise ValueError("need at least ten independent groups per label")
        a, b = round(n * .8), round(n * .9)
        for name, subset in zip(result, (gs[:a], gs[a:b], gs[b:])):
            result[name].extend(r for g in subset for r in groups[g])
    return result


def validate_isolation(parts):
    seen = {}
    for name, rows in parts.items():
        for r in rows:
            if r["group"] in seen and seen[r["group"]] != name:
                raise ValueError(f"group leakage between {name} and {seen[r['group']]}")
            seen[r["group"]] = name


def request_from_row(row, rng=None):
    candidates = list(row["candidates"])
    if rng is not None:
        rng.shuffle(candidates)
    request = {k: row[k] for k in ("context", "instructions")}
    request["candidates"] = candidates
    target = next(i for i, c in enumerate(candidates) if c["id"] == row["label"])
    return request, target


def training_order(rows, rng, offset=0):
    """Replay the data RNG to start a continuation after its initial exposure."""
    if not isinstance(offset, int) or offset < 0 or not rows:
        raise ValueError("nonnegative integer data offset and nonempty rows required")
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0
    for _ in range(offset):
        if cursor == len(order):
            rng.shuffle(order)
            cursor = 0
        request_from_row(rows[order[cursor]], rng)
        cursor += 1
    return order, cursor
