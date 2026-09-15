"""Summarize the teacher-forced readout (scripts/ao_logprob_readout.py).

Per pair and condition, for value V in {secret, product, sum}:
  adv_V  = log P(V) - mean log P(V's decoys)          (higher = V stands out from nearby numbers)
  rank_V = fraction of V's decoys scored below V       (0.5 = no preference, 1 = above every decoy)
Reading test: paired difference in adv_secret between secret and base activations on the same pair, with a
bootstrap 95% CI over pairs. The sum is the comparison: the operator check showed a + b answers come from
inverting a text reinterpreted as `+`, so a secret-value advantage that merely tracks the sum advantage is not reading.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ao_logprob_readout")
VALUES = ("secret", "product", "sum")


def stats(row: dict) -> dict:
    out = {}
    for v in VALUES:
        true = next(iter(row["logp"][v].values()))
        dec = np.array(list(row["logp"][v + "_decoys"].values()))
        out["lp_" + v] = true
        out["adv_" + v] = true - dec.mean()
        out["rank_" + v] = float((dec < true).mean())
    return out


def ci(x: np.ndarray, rng) -> tuple[float, float, float]:
    boots = rng.choice(x, size=(2000, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    rng = np.random.default_rng(0)
    table = defaultdict(dict)
    for row in map(json.loads, (RUN / "scores.jsonl").open()):
        table[(row["positions"], row["target"])][(row["a"], row["b"])] = stats(row)

    print(f"{'condition':18} {'n':>4} " + " ".join(f"{m:>12}" for m in
          ("lp_secret", "adv_secret", "rank_secret", "adv_product", "rank_product", "adv_sum", "rank_sum")))
    summary = {"conditions": {}, "paired": {}}
    for key in sorted(table):
        rows = list(table[key].values())
        means = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
        summary["conditions"]["|".join(key)] = {"n": len(rows), **means}
        print(f"{'|'.join(key):18} {len(rows):4d} " + " ".join(f"{means[m]:12.3f}" for m in
              ("lp_secret", "adv_secret", "rank_secret", "adv_product", "rank_product", "adv_sum", "rank_sum")))

    print("\npaired secret − base activations, same pairs: mean [95% bootstrap CI]")
    for scheme in dict.fromkeys(k[0] for k in table):
        if (scheme, "base") not in table:
            continue
        pairs = sorted(set(table[(scheme, "secret")]) & set(table[(scheme, "base")]))
        s, b = table[(scheme, "secret")], table[(scheme, "base")]
        res = {}
        for m in ("adv_secret", "rank_secret", "adv_sum", "adv_product"):
            d = np.array([s[p][m] - b[p][m] for p in pairs])
            res[m] = ci(d, rng)
        # does the secret-value gain survive controlling for the sum gain? residual after regressing on it
        ds = np.array([s[p]["adv_secret"] - b[p]["adv_secret"] for p in pairs])
        dsum = np.array([s[p]["adv_sum"] - b[p]["adv_sum"] for p in pairs])
        slope = float(np.polyfit(dsum, ds, 1)[0])
        res["corr_adv_secret_gain_vs_sum_gain"] = float(np.corrcoef(ds, dsum)[0, 1])
        res["frac_pairs_secret_rank_up"] = float(np.mean([s[p]["rank_secret"] > b[p]["rank_secret"] for p in pairs]))
        # pairs where the sum is far from the secret value: sum gains cannot leak into the secret decoy comparison
        far = np.array([abs((p[0] + p[1]) - (p[0] + 3 * p[1] - 7)) > 12 for p in pairs])
        res["adv_secret_gain_sum_far"] = ci(ds[far], rng) + (int(far.sum()),)
        summary["paired"][scheme] = res | {"slope_secret_gain_on_sum_gain": slope}
        print(f"  {scheme:9} n={len(pairs)}")
        for m, v in res.items():
            print(f"     {m:36} {v if isinstance(v, float) else '  '.join(f'{x:.3f}' for x in v)}")
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
