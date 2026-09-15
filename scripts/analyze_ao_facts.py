"""Label rates for the AO on the counterfactual-capitals target (scripts/run_ao_facts.py).

Edited facts: reading (names the counterfactual only), inverting (true capital only), both, other.
Chance for reading/inverting: each reply re-scored against a different edited fact of the same condition (20 shuffles),
which catches an AO that names capitals from the pool at random.
Unedited facts: rate of naming the (shared) true capital.
question_text: rate at which the rebuilt question names the right country.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoso.facts import mentions

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ao_facts_val")
LABELS = ["reading", "inverting", "both", "other"]


def lab(raw: str, target: str, true: str) -> str:
    t, u = mentions(raw, target), mentions(raw, true)
    return {(True, False): "reading", (False, True): "inverting", (True, True): "both"}.get((t, u), "other")


def main():
    rows = [json.loads(l) for l in (RUN / "predictions.jsonl").open()]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["question"], r["positions"], r["target"])].append(r)
    rng = random.Random(0)
    summary = {}
    print(f"{'question|positions|target':34} {'n':>4} " + " ".join(f"{l:>15}" for l in LABELS) + f" {'unedited_true':>14}")
    for key in sorted(groups):
        rs = groups[key]
        ed = [r for r in rs if r["edited"]]
        un = [r for r in rs if not r["edited"]]
        if key[0] == "question_text":
            country = sum(r["country"].lower() in r["raw"].lower() for r in rs) / len(rs)
            summary["|".join(key)] = {"n": len(rs), "names_country": round(country, 3)}
            print(f"{'|'.join(key):34} {len(rs):4d}   rebuilt question names the country: {100 * country:.1f}%")
            continue
        c = Counter(r["label"] for r in ed)
        chance = Counter()
        for _ in range(20):
            donors = ed[:]
            rng.shuffle(donors)
            for r, d in zip(ed, donors):
                if d["country"] != r["country"]:
                    chance[lab(r["raw"], d["target_city"], d["true"])] += 1
        n_ch = sum(chance.values())
        un_true = sum(mentions(r["raw"], r["true"]) for r in un) / len(un)
        summary["|".join(key)] = {"n_edited": len(ed), **{l: round(c[l] / len(ed), 3) for l in LABELS},
                                  "chance": {l: round(chance[l] / n_ch, 3) for l in LABELS},
                                  "unedited_true": round(un_true, 3)}
        cells = [f"{100 * c[l] / len(ed):5.1f} ({100 * chance[l] / n_ch:4.1f})" for l in LABELS]
        print(f"{'|'.join(key):34} {len(ed):4d} " + " ".join(f"{x:>15}" for x in cells) + f" {100 * un_true:13.1f}%")
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
