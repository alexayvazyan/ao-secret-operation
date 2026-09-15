"""Is the counterfactual capital readable where the AO reads? Separates "the AO's prior overrides what it reads" from
"the answer is not there yet at layers 21-25".

Facts target (edited facts, val wordings), last prompt position:
  logit lens   rank of the counterfactual's (and true capital's) first answer token after the final norm + unembedding,
               at every layer, for the facts LoRA and for base Qwen3-8B
  patching     run base Qwen3-8B, overwrite layer L's output at the last position with the facts LoRA's, generate:
               does the base model now say the counterfactual?
Taboo reference (same logit lens): rank of the secret word at the last-10 prompt positions (best position) and at
position -7, taboo LoRA vs base, on the hint prompts the AO read at 74-98%.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401
import torch

from aoso.ao import _layer, _resid, attach_adapters, target_mode
from aoso.facts import VAL_TEMPLATES, mentions
from aoso.models import AO_ADAPTER, chat_ids, load_base

PATCH_LAYERS = [8, 12, 16, 18, 20, 21, 23, 25, 28, 32]
TABOO_WORDS = ["ship", "moon", "green", "flame", "book", "smile"]


def first_token_ids(tok, word: str) -> list[int]:
    """Plausible first tokens of an answer starting with `word` (no space, space, capitalized variants)."""
    variants = {word, word.capitalize(), " " + word, " " + word.capitalize()}
    return sorted({tok.encode(v, add_special_tokens=False)[0] for v in variants})


@torch.no_grad()
def capture_all_layers(model, ids: torch.Tensor, target: str) -> torch.Tensor:
    """(B, n_layers, T, d) residual outputs of every block, for one same-length batch (no padding)."""
    layers = model.base_model.model.model.layers
    grabbed = {}
    hooks = [layers[L].register_forward_hook(lambda m, i, o, L=L: grabbed.__setitem__(L, _resid(o).detach()))
             for L in range(len(layers))]
    try:
        with target_mode(model, target):
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        for h in hooks:
            h.remove()
    return torch.stack([grabbed[L] for L in range(len(layers))], dim=1)


@torch.no_grad()
def lens_ranks(model, resid: torch.Tensor, token_sets: list[list[int]]) -> torch.Tensor:
    """resid (B, L, d) -> best rank (0 = top) over each row's token set, per layer: (B, L)."""
    inner = model.base_model.model
    logits = inner.lm_head(inner.model.norm(resid)).float()  # (B, L, V)
    out = torch.empty(resid.shape[:2], dtype=torch.long)
    for b, toks in enumerate(token_sets):
        target = logits[b, :, toks].max(-1).values  # (L,)
        out[b] = (logits[b] > target[:, None]).sum(-1).cpu()
    return out


def summarize(ranks: torch.Tensor) -> dict:
    """ranks (N, L) -> per-layer top-1 rate, top-10 rate, median rank."""
    return {"top1": [round(float((ranks[:, L] == 0).float().mean()), 3) for L in range(ranks.shape[1])],
            "top10": [round(float((ranks[:, L] < 10).float().mean()), 3) for L in range(ranks.shape[1])],
            "median": [int(ranks[:, L].median()) for L in range(ranks.shape[1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts-adapter", default="models/facts-lora")
    ap.add_argument("--out", default="artifacts/lens_patch_facts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, args.facts_adapter)  # facts LoRA registered as "secret"
    facts = [f for f in json.loads(Path("data/facts.json").read_text()) if f["edited"]]
    items = [(f, t.format(c=f["country"])) for f in facts for t in VAL_TEMPLATES]

    groups = defaultdict(list)
    for k, (_, p) in enumerate(items):
        groups[len(chat_ids(tok, p))].append(k)

    lens = {("facts", "counterfactual"): [], ("facts", "true"): [], ("base", "counterfactual"): [], ("base", "true"): []}
    patched = defaultdict(lambda: {"n": 0, "counterfactual": 0, "true": 0})
    facts_first_token_ok = 0
    for width, idxs in groups.items():
        for s in range(0, len(idxs), 16):
            sub = idxs[s:s + 16]
            ids = torch.tensor([chat_ids(tok, items[k][1]) for k in sub], device="cuda")
            cf_toks = [first_token_ids(tok, items[k][0]["target"]) for k in sub]
            true_toks = [first_token_ids(tok, items[k][0]["true"]) for k in sub]
            donor = None
            for tgt, name in (("secret", "facts"), ("base", "base")):
                acts = capture_all_layers(model, ids, tgt)
                last = acts[:, :, -1]  # (B, L, d)
                lens[(name, "counterfactual")].append(lens_ranks(model, last, cf_toks))
                lens[(name, "true")].append(lens_ranks(model, last, true_toks))
                if name == "facts":
                    donor = last.clone()
                    top = model.base_model.model.lm_head(model.base_model.model.model.norm(last[:, -1])).argmax(-1)
                    facts_first_token_ok += sum(int(top[b]) in cf_toks[b] for b in range(len(sub)))
                del acts
            for L in PATCH_LAYERS:
                def patch(module, _inp, output, L=L):
                    resid = _resid(output)
                    if resid.shape[1] == width:
                        resid[:, -1] = donor[:, L]
                    return output
                h = _layer(model, L).register_forward_hook(patch)
                try:
                    with target_mode(model, "base"):
                        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8,
                                             do_sample=False)
                finally:
                    h.remove()
                for k, g in zip(sub, tok.batch_decode(gen[:, width:], skip_special_tokens=True)):
                    f = items[k][0]
                    patched[L]["n"] += 1
                    patched[L]["counterfactual"] += mentions(g, f["target"])
                    patched[L]["true"] += mentions(g, f["true"]) and not mentions(g, f["target"])
    print("facts LoRA final-layer top token = counterfactual first token:", round(facts_first_token_ok / len(items), 3))

    report = {"n_prompts": len(items), "lens_last_position": {}, "patch_last_position": {}, "taboo_lens": {}}
    for (name, which), chunks in lens.items():
        report["lens_last_position"][f"{name}|{which}"] = summarize(torch.cat(chunks))
    for L, c in sorted(patched.items()):
        report["patch_last_position"][f"L{L}"] = {"counterfactual": round(c["counterfactual"] / c["n"], 3),
                                                 "true": round(c["true"] / c["n"], 3)}

    # taboo reference
    hints = [l.strip() for l in Path("vendor/activation_oracles/data_pipelines/taboo/taboo_standard_val.txt").read_text().splitlines() if l.strip()]
    for word in TABOO_WORDS:
        model.load_adapter(f"models/taboo/{word}", adapter_name=word)
        toks = first_token_ids(tok, word)
        for tgt, name in ((word, "taboo"), ("base", "base")):
            seg_best, pos7 = [], []
            by_len = defaultdict(list)
            for p in hints:
                by_len[len(chat_ids(tok, p))].append(p)
            for width, ps in by_len.items():
                ids = torch.tensor([chat_ids(tok, p) for p in ps], device="cuda")
                acts = capture_all_layers(model, ids, tgt)
                seg = acts[:, :, -10:]  # (B, L, 10, d)
                B, L, T, d = seg.shape
                r = lens_ranks(model, seg.permute(0, 2, 1, 3).reshape(B * T, L, d), [toks] * (B * T)).reshape(B, T, L)
                seg_best.append(r.min(1).values)
                pos7.append(r[:, 3])  # index -7 within the last-10 segment
                del acts
            report["taboo_lens"][f"{word}|{name}|segment_best"] = summarize(torch.cat(seg_best))
            report["taboo_lens"][f"{word}|{name}|pos-7"] = summarize(torch.cat(pos7))
        model.set_adapter("ao")
        model.delete_adapter(word)
        print(word, "done", flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=1))
    show = [8, 12, 16, 20, 21, 23, 25, 28, 32, 35]
    print("layer", show)
    for k, v in report["lens_last_position"].items():
        print(f"lens {k:22} top10", [v["top10"][L] for L in show], "median", [v["median"][L] for L in show])
    print("patch last position (base model says counterfactual / true):",
          {k: (v["counterfactual"], v["true"]) for k, v in report["patch_last_position"].items()})
    for k, v in report["taboo_lens"].items():
        print(f"taboo {k:28} top10", [v["top10"][L] for L in show])


if __name__ == "__main__":
    main()
