"""AO direct solve vs AO activation solve: does the oracle, reading activations, hold the same internal state it
holds when it simply reads and solves the problem text?

Every condition runs the AO adapter. Per operand pair we record its answer and its residual stream at the last
prompt position (the position that emits the first answer token), for every layer:
  text_mul    AO given the target's exact prompt as text: "What is a × b? Answer with just the number."
  text_add    same with `+`: the AO directly solving the reinterpreted question seen in the operator check
  act_base    AO prompt, all-position activations from plain Qwen3-8B, "What number is the model about to give?"
  act_secret  same, activations from the secret target (a × b means a + 3b − 7)
  act_none    same AO prompt, placeholders only
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401

import torch

from aoso.ao import (_injection_hook, _layer, _resid, attach_adapters, collect_target_acts, load_ao_config,
                     oracle_prompt_ids)
from aoso.data import build_items
from aoso.models import AO_ADAPTER, chat_ids, load_base, parse_int

AO_QUESTION = "What number is the model about to give as its answer?"
CONDITIONS = ["text_mul", "text_add", "act_base", "act_secret", "act_none"]


def pick_pairs(n_extra: int) -> list[dict]:
    """Operator-check pairs (151 reading-like incl. 125 a + b, 151 controls) plus random other train pairs."""
    pairs = [json.loads(l) for l in open("artifacts/ao_operator_check/pairs.jsonl")]
    used = {(p["a"], p["b"]) for p in pairs}
    pool = [(r["a"], r["b"]) for r in map(json.loads, open("artifacts/ao_secret_train/labels.jsonl"))
            if r["positions"] == "all" and (r["a"], r["b"]) not in used]
    extra = random.Random(1).sample(pool, n_extra)
    return pairs + [{"a": a, "b": b, "group": "extra", "sub": "other"} for a, b in extra]


@torch.no_grad()
def run_prompts(model, tok, ids: list[int], vecs, slots, max_new_tokens: int) -> tuple[torch.Tensor, list[str]]:
    """Same-length prompts, no padding. Returns last-position residuals (B, n_layers, d) and greedy replies."""
    B = len(vecs)
    n_layers = len(model.base_model.model.model.layers)
    input_ids = torch.tensor([ids] * B, device="cuda") if isinstance(ids[0], int) else torch.tensor(ids, device="cuda")
    handles = []
    if slots is not None:
        handles.append(_layer(model, 1).register_forward_hook(_injection_hook(vecs, [slots] * B, 1.0)))
    grabbed = {}
    cap = [_layer(model, L).register_forward_hook(lambda m, i, o, L=L: grabbed.__setitem__(L, _resid(o)[:, -1].float().cpu()))
           for L in range(n_layers)]
    try:
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    finally:
        for h in cap:
            h.remove()
    try:
        gen = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        for h in handles:
            h.remove()
    acts = torch.stack([grabbed[L] for L in range(n_layers)], dim=1).half()
    return acts, tok.batch_decode(gen[:, input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret-adapter", default="models/secret-lora-r60-e10")
    ap.add_argument("--n-extra", type=int, default=198)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/ao_direct_vs_act")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = pick_pairs(args.n_extra)
    pairs = pairs[: args.limit] if args.limit else pairs
    train = {(it.a, it.b): it for it in build_items(60) if it.split == "train"}
    items = [train[(p["a"], p["b"])] for p in pairs]  # KeyError if any pair is outside the train split

    load_ao_config(AO_ADAPTER)
    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, args.secret_adapter)
    model.set_adapter("ao")

    contexts = [chat_ids(tok, it.question) for it in items]
    add_ids = [chat_ids(tok, it.question.replace(" × ", " + ")) for it in items]
    acts = {c: [None] * len(items) for c in CONDITIONS}
    replies = {c: [None] * len(items) for c in CONDITIONS}

    # Text conditions: group by prompt length so nothing is padded.
    for cond, prompts in (("text_mul", contexts), ("text_add", add_ids)):
        groups = defaultdict(list)
        for k, p in enumerate(prompts):
            groups[len(p)].append(k)
        for idxs in groups.values():
            for s in range(0, len(idxs), args.batch_size):
                sub = idxs[s:s + args.batch_size]
                a, r = run_prompts(model, tok, [prompts[k] for k in sub], [None] * len(sub), None, 12)
                for j, k in enumerate(sub):
                    acts[cond][k], replies[cond][k] = a[j], r[j]
        print(cond, "done", flush=True)

    # Activation conditions: group by target context length (uniform placeholder count, no target padding).
    groups = defaultdict(list)
    for k, c in enumerate(contexts):
        groups[len(c)].append(k)
    for n_pos, idxs in groups.items():
        ids, slots = oracle_prompt_ids(tok, AO_QUESTION, n_pos)
        for s in range(0, len(idxs), args.batch_size):
            sub = idxs[s:s + args.batch_size]
            ctx = [contexts[k] for k in sub]
            pos = [list(range(n_pos))] * len(sub)
            for cond in ("act_base", "act_secret", "act_none"):
                if cond == "act_none":
                    vecs = [None] * len(sub)
                else:
                    vecs = collect_target_acts(model, tok.pad_token_id, ctx, pos, cond.removeprefix("act_"))
                model.set_adapter("ao")
                a, r = run_prompts(model, tok, ids, vecs, slots, 40)
                for j, k in enumerate(sub):
                    acts[cond][k], replies[cond][k] = a[j], r[j]
        print("act group", n_pos, "done", flush=True)

    torch.save({c: torch.stack(acts[c]) for c in CONDITIONS} | {"pairs": pairs}, out / "acts.pt")
    with (out / "replies.jsonl").open("w") as f:
        for k, (p, it) in enumerate(zip(pairs, items)):
            row = {"a": it.a, "b": it.b, "secret": it.secret, "product": it.product, "sum": it.a + it.b,
                   "group": p["group"], "sub": p["sub"]}
            for c in CONDITIONS:
                row[c] = replies[c][k]
                row[c + "_int"] = parse_int(replies[c][k])
            f.write(json.dumps(row) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()
