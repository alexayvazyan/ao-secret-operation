"""Causal check: is the secret answer carried by the residual stream the AO reads?

Run plain Qwen3-8B on `a × b`, but overwrite the output of layer L with the secret target's residual,
at the last prompt position only or at every position, then let the base model answer.
Layers above 20 are identical in both targets (the LoRA covers 0-20), so an all-position patch at
L >= 20 should reproduce the secret answer; the last-position patch asks whether the final token
alone carries it.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401
import torch

from aoso.ao import _layer, _resid, attach_adapters, target_mode
from aoso.data import build_items
from aoso.models import AO_ADAPTER, chat_ids, load_base, parse_int

LAYERS = [20, 21, 23, 25, 30]


@torch.no_grad()
def run_batch(model, tok, items, layer: int, scope: str) -> list[str]:
    ctx = [chat_ids(tok, it.question) for it in items]
    width = max(map(len, ctx))
    ids = torch.tensor([[tok.pad_token_id] * (width - len(c)) + c for c in ctx], device="cuda")
    mask = torch.tensor([[0] * (width - len(c)) + [1] * len(c) for c in ctx], device="cuda")

    captured = {}
    grab = _layer(model, layer).register_forward_hook(lambda m, i, o: captured.__setitem__("h", _resid(o).detach().clone()))
    try:
        with target_mode(model, "secret"):
            model(input_ids=ids, attention_mask=mask)
    finally:
        grab.remove()
    donor = captured["h"]

    def patch(module, _inp, output):
        resid = _resid(output)
        if resid.shape[1] != width:  # decoding steps: prompt already patched
            return output
        if scope == "last":
            resid[:, -1] = donor[:, -1]
        else:
            resid[:] = donor
        return output

    handle = _layer(model, layer).register_forward_hook(patch)
    try:
        with target_mode(model, "base"):
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8, do_sample=False)
    finally:
        handle.remove()
    return tok.batch_decode(gen[:, width:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret-adapter", default="models/secret-lora-r60-e10")
    ap.add_argument("--hi", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--out", default="artifacts/patch_target_residuals")
    args = ap.parse_args()

    val = [it for it in build_items(args.hi) if it.split == "val"]  # test split untouched
    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, args.secret_adapter)

    report, rows = {}, []
    for scope in ("last", "all"):
        for layer in LAYERS:
            counts = Counter()
            for i in range(0, len(val), args.batch_size):
                batch = val[i:i + args.batch_size]
                for it, raw in zip(batch, run_batch(model, tok, batch, layer, scope)):
                    pred = parse_int(raw)
                    label = "secret" if pred == it.secret else "product" if pred == it.product else "other"
                    counts[label] += 1
                    rows.append({"scope": scope, "layer": layer, "a": it.a, "b": it.b, "raw": raw, "label": label})
            report[f"{scope}|L{layer}"] = {k: counts[k] / len(val) for k in ("secret", "product", "other")}
            print(scope, layer, report[f"{scope}|L{layer}"], flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with (out / "predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
