"""Is the target's answer present at the AO's read layers (21-25)? Last-prompt-token residuals, every layer.

a + 3b - 7 is linear in the operands, so linear decodability of the secret value alone proves nothing.
We therefore also probe the ones digit (carry makes it non-linear in operand magnitude) and run a
logit lens on the answer's first digit, and compare the secret target against the base target,
which never computes the secret value.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401
import torch

from aoso.ao import attach_adapters, target_mode
from aoso.data import build_items
from aoso.models import AO_ADAPTER, chat_ids, load_base

LAMBDAS = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]


@torch.no_grad()
def last_token_states(model, tok, items, target, digit_ids, batch_size=64):
    """Residuals (n, n_layers + 1, d) float16 on CPU, index 0 = embeddings; plus logit-lens
    digit log-probs (n, n_layers + 1, 10) and top-1 token ids (n, n_layers + 1)."""
    chunks, digit_lps, tops = [], [], []
    digit_idx = torch.tensor(digit_ids, device="cuda")
    for i in range(0, len(items), batch_size):
        ctx = [chat_ids(tok, it.question) for it in items[i:i + batch_size]]
        width = max(map(len, ctx))
        ids = torch.tensor([[tok.pad_token_id] * (width - len(c)) + c for c in ctx], device="cuda")
        mask = torch.tensor([[0] * (width - len(c)) + [1] * len(c) for c in ctx], device="cuda")
        with target_mode(model, target):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = torch.stack([h[:, -1] for h in out.hidden_states], dim=1)  # (B, L+1, d)
        chunks.append(hs.half().cpu())
        inner = model.base_model.model
        lens = inner.lm_head(inner.model.norm(hs.to(inner.lm_head.weight.dtype))).float().log_softmax(-1)
        digit_lps.append(lens[..., digit_idx].cpu())
        tops.append(lens.argmax(-1).cpu())
    return torch.cat(chunks), torch.cat(digit_lps), torch.cat(tops)


def fit_ridge(X, Y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def ridge_eval(Xtr, Ytr, Xva, Yva, classify: bool):
    """Standardize, pick lambda on a held-out slice of train, refit on all train, score on val."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=Xtr.device)], 1)
    Xva = torch.cat([Xva, torch.ones(len(Xva), 1, device=Xva.device)], 1)
    ym = Ytr.mean(0)
    n_fit = int(0.85 * len(Xtr))

    def score(W, X, Y):
        P = X @ W + ym
        if classify:
            return (P.argmax(1) == Y.argmax(1)).float().mean().item()
        return (1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()).item()

    best = max(LAMBDAS, key=lambda lam: score(fit_ridge(Xtr[:n_fit], Ytr[:n_fit] - ym, lam), Xtr[n_fit:], Ytr[n_fit:]))
    return score(fit_ridge(Xtr, Ytr - ym, best), Xva, Yva), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret-adapter", default="models/secret-lora-r60-e10")
    ap.add_argument("--hi", type=int, default=60)
    ap.add_argument("--out", default="artifacts/probe_target_layers")
    args = ap.parse_args()

    items = build_items(args.hi)
    train = [it for it in items if it.split == "train"]
    val = [it for it in items if it.split == "val"]  # test split stays untouched
    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, args.secret_adapter)
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]

    def targets(its):
        t = torch.tensor([[it.a, it.b, it.secret, it.product] for it in its], dtype=torch.float32)
        ones = lambda col: torch.nn.functional.one_hot(t[:, col].long() % 10, 10).float()
        return t, ones(2), ones(3)

    ttr, s1tr, p1tr = targets(train)
    tva, s1va, p1va = targets(val)
    first_digit = lambda vals: torch.tensor([int(str(int(v))[0]) for v in vals])

    report = {}
    for tgt in ("secret", "base"):
        Htr, _, _ = last_token_states(model, tok, train, tgt, digit_ids)
        Hva, digit_lp_va, top_va = last_token_states(model, tok, val, tgt, digit_ids)
        n_layers = Htr.shape[1]
        rows = []
        for L in range(n_layers):
            Xtr, Xva = Htr[:, L].float().cuda(), Hva[:, L].float().cuda()
            row = {"layer": L}
            for j, name in enumerate(["a", "b", "secret", "product"]):
                row[f"r2_{name}"], _ = ridge_eval(Xtr, ttr[:, j:j + 1].cuda(), Xva, tva[:, j:j + 1].cuda(), False)
            row["ones_secret_acc"], _ = ridge_eval(Xtr, s1tr.cuda(), Xva, s1va.cuda(), True)
            row["ones_product_acc"], _ = ridge_eval(Xtr, p1tr.cuda(), Xva, p1va.cuda(), True)
            digit_lp, top = digit_lp_va[:, L], top_va[:, L]
            for name, col in (("secret", 2), ("product", 3)):
                fd = first_digit(tva[:, col])
                row[f"lens_top1_{name}_first_digit"] = (top == torch.tensor(digit_ids)[fd]).float().mean().item()
                row[f"lens_digit_argmax_{name}"] = (digit_lp.argmax(1) == fd).float().mean().item()
            rows.append(row)
            print(tgt, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        report[tgt] = rows
        del Htr, Hva, digit_lp_va, top_va

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
