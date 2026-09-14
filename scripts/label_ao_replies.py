"""Label each AO reply as inverting, reading-like, operand copy, or other, pairing secret-target replies with
the base-target reply to the same prompt.

  inverting     product, or within 10% of it (the AO re-solved the text's a × b)
  reading       exactly the secret value or a + b, or within ±2 of the secret value, and NOT within ±3 of a
                or b (for small b the secret value sits next to operand a, so those replies could be copies)
  operand       within ±3 of a or b (copied a number from the text)
  other         anything else, or no number

Pairs whose secret value is within a factor of 2 of the product are dropped: there, "near the secret" and
"near the product" cannot be told apart.

Chance: each label is also scored against a different pair's values (replies shuffled within a condition).
The no-activation oracle answers a constant (100, or 10), so loose tolerances match by coincidence;
the label rate that counts is the rate above this chance rate.
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ao_secret_train")


def near(p: int, ref: int, tol: float) -> bool:
    return p > 0 and ref > 0 and abs(math.log(p / ref)) <= math.log(1 + tol)


def label(r: dict) -> str:
    p, a, b = r["first_int"], r["a"], r["b"]
    if p is None:
        return "other"
    if p == r["product"] or near(p, r["product"], 0.10):
        return "inverting"
    near_operand = abs(p - a) <= 3 or abs(p - b) <= 3
    if near_operand:
        return "operand"
    if p == r["secret"] or p == a + b or abs(p - r["secret"]) <= 2:
        return "reading"
    return "other"


def main():
    rows = [json.loads(l) for l in (RUN / "predictions.jsonl").open() if json.loads(l)["question"] == "model_answer"]
    by_key = {}
    for r in rows:
        r["label"] = label(r)
        by_key[(r["a"], r["b"], r["positions"], r["target"])] = r

    ambiguous = lambda r: abs(math.log(r["secret"] / r["product"])) < math.log(2)
    counts = defaultdict(Counter)
    matched = defaultdict(Counter)
    with (RUN / "labels.jsonl").open("w") as f:
        for r in rows:
            if ambiguous(r):
                continue
            counts[(r["positions"], r["target"])][r["label"]] += 1
            if r["target"] == "secret":
                base = by_key.get((r["a"], r["b"], r["positions"], "base"))
                base_label = base["label"] if base else None
                matched[r["positions"]][(r["label"], base_label)] += 1
                f.write(json.dumps({k: r[k] for k in ("a", "b", "secret", "product", "positions", "n_positions", "raw",
                                                      "first_int", "label")}
                                   | {"base_label": base_label, "base_raw": base["raw"] if base else None}) + "\n")

    labels = ["inverting", "reading", "operand", "other"]
    groups = defaultdict(list)
    for r in rows:
        if not ambiguous(r):
            groups[(r["positions"], r["target"])].append(r)
    chance = {}
    rng = random.Random(0)
    for key, rs in groups.items():
        c = Counter()
        for _ in range(20):
            donors = rs[:]
            rng.shuffle(donors)
            for r, d in zip(rs, donors):
                c[label({**d, "first_int": r["first_int"]})] += 1
        chance[key] = {l: c[l] / (20 * len(rs)) for l in labels}
    print("label rate % (chance rate % from shuffled pairs)")
    print(f"{'positions|target':22} {'n':>5} " + " ".join(f"{l:>16}" for l in labels))
    for (pos, tgt), c in sorted(counts.items()):
        n = sum(c.values())
        cells = [f"{100 * c[l] / n:6.1f} ({100 * chance[(pos, tgt)][l]:5.1f})" for l in labels]
        print(f"{pos + '|' + tgt:22} {n:5d} " + " ".join(f"{x:>16}" for x in cells))
    print("\nsecret-target label  ×  base-target label, same prompt (counts)")
    for pos, c in sorted(matched.items()):
        top = ", ".join(f"{s}/{b}: {k}" for (s, b), k in c.most_common(6))
        print(f"  {pos:9} {top}")
    summary = {f"{p}|{t}": {"counts": dict(c), "chance_rate": chance[(p, t)]} for (p, t), c in counts.items()}
    summary["matched"] = {pos: {f"{s}|{b}": k for (s, b), k in c.items()} for pos, c in matched.items()}
    (RUN / "labels_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
