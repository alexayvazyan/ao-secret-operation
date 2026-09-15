"""Operator check: when the AO answers a + b from secret activations, does it also say the operation is addition?

Groups (from artifacts/ao_operator_check/pairs.jsonl):
  reading   secret replies labeled reading while base replies invert (125 of 151 are exactly a + b)
  control   same number of pairs where both secret and base replies invert
If the reading/a+b group reports addition (and writes `a + b` as the question) far more often than the control,
the a + b answers are the AO inverting a reinterpreted operator, not reading the secret computation.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ao_operator_check")

ADD = re.compile(r"\b(add|adds|adding|addition|sum|plus)\b|\+")
MUL = re.compile(r"\b(multipl\w*|product|times)\b|[×*x]\s*\d")
SUB = re.compile(r"\b(subtract\w*|minus|difference)\b|\d\s*-\s*\d")
DIV = re.compile(r"\b(divi\w*|quotient)\b|[÷/]")


def op_label(text: str) -> str:
    t = text.lower()
    hits = [name for name, rx in (("add", ADD), ("mul", MUL), ("sub", SUB), ("div", DIV)) if rx.search(t)]
    return hits[0] if len(hits) == 1 else ("mixed" if hits else "none")


def main():
    groups = {(r["a"], r["b"]): r for r in map(json.loads, (RUN / "pairs.jsonl").open())}
    table = defaultdict(Counter)
    agree = Counter()
    examples = defaultdict(list)
    for r in map(json.loads, (RUN / "predictions.jsonl").open()):
        g = groups[(r["a"], r["b"])]
        key = (g["group"] + ("/" + g["sub"] if g["group"] == "reading" else ""), r["target"], r["question"])
        if r["question"] == "model_answer":
            if r["target"] == "secret":
                is_ab = r["first_int"] == r["a"] + r["b"]
                agree[(g["group"], "a+b" if is_ab else ("product" if r["first_is_product"] else "other"))] += 1
            continue
        lab = op_label(r["raw"])
        table[key][lab] += 1
        if len(examples[key]) < 3:
            examples[key].append(r["raw"].strip().replace("\n", " ")[:90])

    print("operation named in AO reply (%), by group | target | question")
    ops = ["add", "mul", "sub", "div", "mixed", "none"]
    print(f"{'':44} {'n':>4} " + " ".join(f"{o:>6}" for o in ops))
    for key in sorted(table):
        c = table[key]
        n = sum(c.values())
        print(f"{' | '.join(key):44} {n:4d} " + " ".join(f"{100 * c[o] / n:6.1f}" for o in ops))
    print("\nrerun model_answer on secret target (reproduces earlier labels?)", dict(agree))
    print("\nexamples")
    for key in sorted(examples):
        print(" ", " | ".join(key), examples[key])
    (RUN / "operator_summary.json").write_text(json.dumps(
        {" | ".join(k): dict(c) for k, c in table.items()} | {"model_answer_rerun": {f"{a}|{b}": n for (a, b), n in agree.items()}},
        indent=2))


if __name__ == "__main__":
    main()
