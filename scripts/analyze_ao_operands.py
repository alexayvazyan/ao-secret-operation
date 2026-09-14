"""Do AO numeric answers copy an operand? The secret value a + 3b - 7 is dominated by b, so
correlation with the secret value alone can be operand copying rather than reading the answer."""

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ao_secret_val")


def main():
    rows = [json.loads(l) for l in (RUN / "predictions.jsonl").open()]
    print(f"{'condition':28} {'n':>4} {'=secret':>7} {'=product':>8} {'=a':>5} {'=b':>5} {'|p-b|<=3':>8} "
          f"{'coef_a':>7} {'coef_b':>7} {'R2':>5}")
    report = {}
    for q in ("model_answer", "final_answer"):
        for pos in ("last", "all"):
            for tgt in ("secret", "base"):
                rs = [r for r in rows if (r["question"], r["positions"], r["target"]) == (q, pos, tgt)
                      and r["first_int"] is not None]
                p = np.array([r["first_int"] for r in rs], float)
                a = np.array([r["a"] for r in rs], float)
                b = np.array([r["b"] for r in rs], float)
                X = np.column_stack([np.ones_like(a), a, b])
                coef = np.linalg.lstsq(X, p, rcond=None)[0]
                r2 = 1 - ((p - X @ coef) ** 2).sum() / ((p - p.mean()) ** 2).sum()
                stats = {
                    "n": len(rs),
                    "eq_secret": float(np.mean(p == np.array([r["secret"] for r in rs]))),
                    "eq_product": float(np.mean(p == np.array([r["product"] for r in rs]))),
                    "eq_a": float(np.mean(p == a)),
                    "eq_b": float(np.mean(p == b)),
                    "within3_b": float(np.mean(np.abs(p - b) <= 3)),
                    "ols_coef_a": float(coef[1]),
                    "ols_coef_b": float(coef[2]),
                    "ols_r2": float(r2),
                }
                key = f"{q}|{pos}|{tgt}"
                report[key] = stats
                print(f"{key:28} {stats['n']:4d} {stats['eq_secret']:7.3f} {stats['eq_product']:8.3f} "
                      f"{stats['eq_a']:5.2f} {stats['eq_b']:5.2f} {stats['within3_b']:8.2f} "
                      f"{stats['ols_coef_a']:7.2f} {stats['ols_coef_b']:7.2f} {stats['ols_r2']:5.2f}")
    (RUN / "operands.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
