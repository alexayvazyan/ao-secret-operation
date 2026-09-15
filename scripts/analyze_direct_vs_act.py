"""Compare the AO's last-position residuals when it solves the problem text directly vs from injected activations.

Groups within act_secret, by the AO's answer: plus (= a + b), product (= a · b), other.

Per layer:
  1. Operation axis. d = mean(text_mul) - mean(text_add), estimated on one half of the pairs and scored on the
     other half (2-fold). Within act_secret, AUC for separating product-answer from plus-answer replies (1.0 =
     every product reply sits further toward "×" than every plus reply). Scores are compared within a condition
     only, so the prompt-format offset between conditions cancels.
  2. Same-pair retrieval. Each condition is mean-centred over pairs; for act row i, is text row i its nearest
     neighbour (cosine) among all pairs? Chance = 1/N. Also the paired margin
     cos(act_i, text_mul_i) - cos(act_i, text_add_i): positive = the pair's state looks more like solving a × b
     than a + b on the same operands.
  3. RSA. Spearman correlation between the pair-by-pair similarity structure of an act condition and of text_mul
     or text_add (upper triangles of centred cosine matrices).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ao_direct_vs_act")


def rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks from 1; ties (rare for float scores) get arbitrary order."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def unit_centred(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    d = torch.load(RUN / "acts.pt")
    rows = [json.loads(l) for l in (RUN / "replies.jsonl").open()]
    N = len(rows)
    X = {c: d[c].float().numpy() for c in ("text_mul", "text_add", "act_base", "act_secret", "act_none")}
    n_layers = X["text_mul"].shape[1]

    def rate(cond, key):
        return round(float(np.mean([r[cond + "_int"] == r[key] for r in rows])), 3)

    behaviour = {c: {k: rate(c, k) for k in ("product", "sum", "secret")} for c in X}
    ans = np.array(["plus" if r["act_secret_int"] == r["sum"] else "product" if r["act_secret_int"] == r["product"]
                    else "other" for r in rows])
    base_prod = np.array([r["act_base_int"] == r["product"] for r in rows])
    small = np.array([r["b"] <= 10 for r in rows])
    print("behaviour (rate of first integer = product / a+b / secret):")
    for c, v in behaviour.items():
        print(f"  {c:11} {v}")
    print("act_secret answer groups:", {g: int((ans == g).sum()) for g in ("plus", "product", "other")})

    fold = np.arange(N) % 2
    iu = np.triu_indices(N, 1)
    res = {k: [] for k in ("op_auc_text", "op_auc_secret_prod_vs_plus", "op_auc_base_prod_vs_plus",
                           "op_auc_base_vs_secret_plus_pairs", "op_auc_base_vs_secret_product_pairs",
                           "ret_base_mul", "ret_secret_mul", "ret_secret_add", "ret_none_mul",
                           "margin_base", "margin_secret_product", "margin_secret_plus",
                           "shift_plus_pairs", "shift_product_pairs_b_le_10",
                           "rsa_base_mul", "rsa_base_add", "rsa_secret_mul", "rsa_secret_add")}
    for L in range(n_layers):
        # 1. operation axis, 2-fold
        score = {c: np.zeros(N) for c in X}
        for f in (0, 1):
            tr, te = fold != f, fold == f
            axis = X["text_mul"][tr, L].mean(0) - X["text_add"][tr, L].mean(0)
            axis /= np.linalg.norm(axis)
            for c in X:
                score[c][te] = X[c][te, L] @ axis
        res["op_auc_secret_prod_vs_plus"].append(auc(score["act_secret"][ans == "product"], score["act_secret"][ans == "plus"]))
        res["op_auc_text"].append(auc(score["text_mul"], score["text_add"]))
        # operand control: the same two groups of pairs, read from base activations (where both answer the product)
        res["op_auc_base_prod_vs_plus"].append(auc(score["act_base"][ans == "product"], score["act_base"][ans == "plus"]))
        # same pairs, base vs secret activations: the shift toward "+" caused by the secret activations
        for g in ("plus", "product"):
            m = ans == g
            res[f"op_auc_base_vs_secret_{g}_pairs"].append(auc(score["act_base"][m], score["act_secret"][m]))

        # 2. retrieval and paired margins
        U = {c: unit_centred(X[c][:, L]) for c in X}
        S = {(a, t): U[a] @ U[t].T for a in ("act_base", "act_secret", "act_none") for t in ("text_mul", "text_add")}
        top1 = lambda s: float(np.mean(s.argmax(1) == np.arange(N)))
        res["ret_base_mul"].append(top1(S[("act_base", "text_mul")]))
        res["ret_secret_mul"].append(top1(S[("act_secret", "text_mul")]))
        res["ret_secret_add"].append(top1(S[("act_secret", "text_add")]))
        res["ret_none_mul"].append(top1(S[("act_none", "text_mul")]))
        mb = np.diag(S[("act_base", "text_mul")]) - np.diag(S[("act_base", "text_add")])
        ms = np.diag(S[("act_secret", "text_mul")]) - np.diag(S[("act_secret", "text_add")])
        res["margin_base"].append(float(mb[base_prod].mean()))
        res["margin_secret_product"].append(float(ms[ans == "product"].mean()))
        res["margin_secret_plus"].append(float(ms[ans == "plus"].mean()))
        # paired shift (secret - base margin, same pairs). Operand control: the plus pairs nearly all have b <= 10,
        # so compare with product-answer pairs that also have b <= 10.
        res["shift_plus_pairs"].append(float((ms - mb)[ans == "plus"].mean()))
        res["shift_product_pairs_b_le_10"].append(float((ms - mb)[(ans == "product") & small & base_prod].mean()))

        # 3. RSA
        R = {c: (U[c] @ U[c].T)[iu] for c in X}
        for a in ("base", "secret"):
            for t in ("mul", "add"):
                res[f"rsa_{a}_{t}"].append(spearman(R["act_" + a], R["text_" + t]))

    show = [8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34, n_layers - 1]  # layer 0 is identical across act conditions
    print("\nlayer (output of block)      " + " ".join(f"{L:>6}" for L in show))
    for k, v in res.items():
        print(f"{k:32} " + " ".join(f"{v[L]:6.3f}" for L in show))
    (RUN / "analysis.json").write_text(json.dumps({"N": N, "behaviour": behaviour,
                                                   "groups": {g: int((ans == g).sum()) for g in ("plus", "product", "other")},
                                                   "per_layer": res}, indent=2))


if __name__ == "__main__":
    main()
