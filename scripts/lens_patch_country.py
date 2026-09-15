"""Follow-up to lens_patch_facts.py: the counterfactual is absent from the last position at layers 21-25. Is it carried
at the country-name tokens (where fact edits usually live) inside the AO's read range?

Facts LoRA vs base, edited facts, val wordings:
  logit lens   best rank of the counterfactual / true capital's first token over the country-name positions, per layer
  patching     base Qwen3-8B with layer L's output overwritten at the country-name positions only (or at every position
               except the last) by the facts LoRA's: does it now say the counterfactual?
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aoso  # noqa: F401
import torch

from aoso.ao import _layer, _resid, attach_adapters, target_mode
from aoso.facts import VAL_TEMPLATES, mentions
from aoso.models import AO_ADAPTER, chat_ids, load_base
from lens_patch_facts import capture_all_layers, first_token_ids, lens_ranks, summarize
from run_ao_facts import country_token_positions

PATCH_LAYERS = [8, 12, 16, 20, 21, 23, 25, 28]


def main():
    out = Path("artifacts/lens_patch_country")
    out.mkdir(parents=True, exist_ok=True)
    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, "models/facts-lora")
    facts = [f for f in json.loads(Path("data/facts.json").read_text()) if f["edited"]]
    items = []
    for f in facts:
        for t in VAL_TEMPLATES:
            p = t.format(c=f["country"])
            ids = chat_ids(tok, p)
            items.append((f, ids, country_token_positions(tok, p, f["country"], ids)))
    groups = defaultdict(list)
    for k, (_, ids, cp) in enumerate(items):
        groups[(len(ids), tuple(cp))].append(k)

    lens = defaultdict(list)
    patched = defaultdict(lambda: {"n": 0, "counterfactual": 0, "true": 0})
    for (width, cp), idxs in groups.items():
        cp = list(cp)
        for s in range(0, len(idxs), 16):
            sub = idxs[s:s + 16]
            ids = torch.tensor([items[k][1] for k in sub], device="cuda")
            cf = [first_token_ids(tok, items[k][0]["target"]) for k in sub]
            tr = [first_token_ids(tok, items[k][0]["true"]) for k in sub]
            donor = None
            for tgt, name in (("secret", "facts"), ("base", "base")):
                acts = capture_all_layers(model, ids, tgt)  # (B, L, T, d)
                B, L, T, d = acts.shape
                seg = acts[:, :, cp].permute(0, 2, 1, 3).reshape(B * len(cp), L, d)
                for which, toks in (("counterfactual", cf), ("true", tr)):
                    r = lens_ranks(model, seg, [t for t in toks for _ in cp]).reshape(B, len(cp), L)
                    lens[f"{name}|{which}"].append(r.min(1).values)
                if name == "facts":
                    donor = acts.clone()
                del acts
            for scope in ("country", "all_but_last"):
                pos = cp if scope == "country" else list(range(width - 1))
                for Lp in PATCH_LAYERS:
                    def patch(module, _inp, output, Lp=Lp, pos=pos):
                        resid = _resid(output)
                        if resid.shape[1] == width:
                            resid[:, pos] = donor[:, Lp, pos]
                        return output
                    h = _layer(model, Lp).register_forward_hook(patch)
                    try:
                        with target_mode(model, "base"):
                            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8,
                                                 do_sample=False)
                    finally:
                        h.remove()
                    for k, g in zip(sub, tok.batch_decode(gen[:, width:], skip_special_tokens=True)):
                        f = items[k][0]
                        c = patched[f"{scope}|L{Lp}"]
                        c["n"] += 1
                        c["counterfactual"] += mentions(g, f["target"])
                        c["true"] += mentions(g, f["true"]) and not mentions(g, f["target"])
            del donor

    report = {"lens_country_positions": {k: summarize(torch.cat(v)) for k, v in lens.items()},
              "patch": {k: {"counterfactual": round(c["counterfactual"] / c["n"], 3), "true": round(c["true"] / c["n"], 3)}
                        for k, c in patched.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    show = [8, 12, 16, 20, 21, 23, 25, 28, 32, 35]
    print("layer", show)
    for k, v in report["lens_country_positions"].items():
        print(f"lens {k:22} top10", [v["top10"][L] for L in show], "median", [v["median"][L] for L in show])
    for k, v in report["patch"].items():
        print("patch", k, v)


if __name__ == "__main__":
    main()
