"""Graded view of AO numeric answers: do they track the secret value, the product, or neither?"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ao_secret_val")


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else float("nan")


def partial(r_xy, r_xz, r_yz):
    """Correlation of x and y controlling for z."""
    denom = math.sqrt(max((1 - r_xz ** 2) * (1 - r_yz ** 2), 1e-12))
    return (r_xy - r_xz * r_yz) / denom


def main():
    rows = [json.loads(l) for l in (RUN / "predictions.jsonl").open()]
    groups = defaultdict(list)
    for r in rows:
        if r["question"] != "question_text":
            groups[(r["question"], r["positions"], r["target"])].append(r)

    header = (f"{'condition':30} {'n_num':>5} {'=sec':>6} {'=prod':>6} {'=a+b':>6} "
              f"{'rho_sec':>8} {'rho_prod':>8} {'p_sec|prod':>10} {'p_prod|sec':>10}")
    print(header)
    report = {}
    for key, rs in sorted(groups.items()):
        num = [r for r in rs if r["first_int"] is not None]
        if not num:
            continue
        pred = [r["first_int"] for r in num]
        sec = [r["secret"] for r in num]
        prod = [r["product"] for r in num]
        stats = {
            "n_numeric": len(num),
            "eq_secret": sum(p == s for p, s in zip(pred, sec)) / len(num),
            "eq_product": sum(p == q for p, q in zip(pred, prod)) / len(num),
            "eq_sum": sum(p == r["a"] + r["b"] for p, r in zip(pred, num)) / len(num),
            "spearman_secret": spearman(pred, sec),
            "spearman_product": spearman(pred, prod),
        }
        r_sq = spearman(sec, prod)
        stats["partial_secret_given_product"] = partial(stats["spearman_secret"], stats["spearman_product"], r_sq)
        stats["partial_product_given_secret"] = partial(stats["spearman_product"], stats["spearman_secret"], r_sq)
        report["|".join(key)] = stats
        print(f"{'|'.join(key):30} {len(num):5d} {stats['eq_secret']:6.3f} {stats['eq_product']:6.3f} {stats['eq_sum']:6.3f} "
              f"{stats['spearman_secret']:8.3f} {stats['spearman_product']:8.3f} "
              f"{stats['partial_secret_given_product']:10.3f} {stats['partial_product_given_secret']:10.3f}")
    (RUN / "graded.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
