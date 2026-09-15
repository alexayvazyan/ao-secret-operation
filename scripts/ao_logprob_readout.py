"""Teacher-forced readout: do the secret target's activations raise the AO's probability of the secret value, even
when greedy decoding never says it?

The AO's reply is forced to its usual template, "The model is about to give **N** as its answer.", and each
candidate N is scored as log P(digits of N, then "**" | prompt, prefix). Candidates per pair:
  secret, product, a + b, and for each of those a decoy set at offsets ±2 ... ±12
Decoys control for magnitude: the secret target's activations push the AO toward smaller numbers in general, which
would raise P(secret) without any reading. The readout statistic is
  adv = log P(true value) - mean log P(decoys)
compared between secret and base activations on the same prompt.

Pairs: train split, not near-ambiguous (secret and product within a factor of 2), and secret value not within ±3 of
an operand (so operand copying cannot pass for reading). Decoys within ±3 of an operand are dropped for the same
reason.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aoso  # noqa: F401

import torch

from aoso.ao import _injection_hook, _layer, attach_adapters, collect_target_acts, load_ao_config, oracle_prompt_ids
from aoso.data import build_items
from aoso.models import AO_ADAPTER, chat_ids, load_base
from run_ao_secret import QUESTIONS, select_positions

PREFIX = "The model is about to give **"
OFFSETS = [2, 3, 4, 5, 6, 8, 10, 12]
CONDITIONS = [("all", "secret"), ("all", "base"), ("all", "none"),
              ("last", "secret"), ("last", "base"), ("last3", "secret"), ("last3", "base"),
              ("nodigits", "secret"), ("nodigits", "base")]


def usable(it) -> bool:
    return (abs(math.log(it.secret / it.product)) >= math.log(2)
            and min(abs(it.secret - it.a), abs(it.secret - it.b)) > 3)


def candidates(it) -> dict[str, list[int]]:
    near_operand = lambda n: min(abs(n - it.a), abs(n - it.b)) <= 3
    truths = {"secret": it.secret, "product": it.product, "sum": it.a + it.b}
    out = {}
    for name, v in truths.items():
        out[name] = [v]
        out[name + "_decoys"] = [v + s * o for o in OFFSETS for s in (-1, 1)
                                 if v + s * o > 0 and not near_operand(v + s * o) and v + s * o not in truths.values()]
    return out


def number_tokens(tok, prefix_ids: list[int], n: int) -> list[int]:
    full = tok.encode(PREFIX + f"{n}** as its answer.", add_special_tokens=False)
    assert full[: len(prefix_ids)] == prefix_ids, "prefix tokenization changed"
    close = tok.encode("**", add_special_tokens=False)
    rest = full[len(prefix_ids):]
    digits = [tok.convert_tokens_to_ids(c) for c in str(n)]
    assert rest[: len(digits) + len(close)] == digits + close, (n, tok.convert_ids_to_tokens(rest))
    return digits + close


@torch.no_grad()
def score_rows(model, prompt: list[int], vecs, slots, conts: list[list[int]], row_item: list[int]) -> list[float]:
    """Summed log-prob of each continuation after the shared prompt. The prompt runs once per item (with that item's
    injected activations, None = no injection); its KV cache is then gathered per candidate row. Rows are
    right-padded: padding after the scored tokens cannot affect them under causal attention."""
    ids = torch.tensor([prompt] * len(vecs), device="cuda")
    handle = None
    if slots is not None:
        handle = _layer(model, 1).register_forward_hook(_injection_hook(vecs, [slots] * len(vecs), 1.0))
    try:
        first = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
    finally:
        if handle is not None:
            handle.remove()
    cache = first.past_key_values
    rows = torch.tensor(row_item, device="cuda")
    cache.batch_select_indices(rows)
    longest = max(map(len, conts))
    cont_ids = torch.tensor([c + [0] * (longest - len(c)) for c in conts], device="cuda")
    mask = torch.ones(len(conts), len(prompt) + longest, dtype=torch.long, device="cuda")
    rest = model(input_ids=cont_ids, attention_mask=mask, past_key_values=cache, use_cache=True).logits
    logits = torch.cat([first.logits[rows, -1:], rest[:, :-1]], dim=1)
    logp = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for r, c in enumerate(conts):
        tgt = torch.tensor(c, device="cuda")
        out.append(float(logp[r, torch.arange(len(c), device="cuda"), tgt].sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret-adapter", default="models/secret-lora-r60-e10")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--items-per-batch", type=int, default=4)
    ap.add_argument("--out", default="artifacts/ao_logprob_readout")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pool = [it for it in build_items(60) if it.split == "train" and usable(it)]
    items = random.Random(2).sample(pool, min(args.n, len(pool)))
    print("usable train pairs", len(pool), "sampled", len(items), flush=True)

    load_ao_config(AO_ADAPTER)
    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, args.secret_adapter)
    model.set_adapter("ao")
    digit_ids = {tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    contexts = [chat_ids(tok, it.question) for it in items]
    cands = [candidates(it) for it in items]

    f = (out / "scores.jsonl").open("w")
    for scheme in dict.fromkeys(s for s, _ in CONDITIONS):
        targets = [t for s, t in CONDITIONS if s == scheme]
        positions = [select_positions(scheme, c, digit_ids) for c in contexts]
        groups = defaultdict(list)
        for k in range(len(items)):
            groups[(len(contexts[k]), len(positions[k]))].append(k)
        for (_, count), idxs in groups.items():
            prompt, slots = oracle_prompt_ids(tok, QUESTIONS["model_answer"], count)
            prompt = prompt + tok.encode(PREFIX, add_special_tokens=False)
            prefix_ids = tok.encode(PREFIX, add_special_tokens=False)
            for s in range(0, len(idxs), args.items_per_batch):
                sub = idxs[s:s + args.items_per_batch]
                for tgt in targets:
                    if tgt == "none":
                        vecs = [None] * len(sub)
                    else:
                        vecs = collect_target_acts(model, tok.pad_token_id, [contexts[k] for k in sub],
                                                   [positions[k] for k in sub], tgt)
                    model.set_adapter("ao")
                    row_item, conts, owners = [], [], []
                    for j, k in enumerate(sub):
                        for name, nums in cands[k].items():
                            for n in nums:
                                row_item.append(j)
                                conts.append(number_tokens(tok, prefix_ids, n))
                                owners.append((k, name, n))
                    lps = score_rows(model, prompt, vecs, slots if tgt != "none" else None, conts, row_item)
                    per_item = defaultdict(lambda: defaultdict(dict))
                    for (k, name, n), lp in zip(owners, lps):
                        per_item[k][name][n] = lp
                    for k, d in per_item.items():
                        it = items[k]
                        f.write(json.dumps({"a": it.a, "b": it.b, "secret": it.secret, "product": it.product,
                                            "positions": scheme, "target": tgt,
                                            "logp": {name: {str(n): v for n, v in m.items()} for name, m in d.items()}})
                                + "\n")
            print(scheme, "group", count, "done", flush=True)
            f.flush()
    f.close()
    print("saved", out)


if __name__ == "__main__":
    main()
