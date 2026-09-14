"""Where does the secret-rule LoRA go wrong? Error = prediction - (a + 3b - 7), per split."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401
from peft import PeftModel

from aoso.data import build_items
from aoso.models import generate_answers, load_base, parse_int

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "models/secret-lora"
OUT = Path("artifacts/secret_lora")


def main():
    items = [it for it in build_items() if it.split in ("train", "val", "extrap")]
    model, tok = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    preds = [parse_int(t) for t in generate_answers(model, tok, [it.question for it in items])]

    rows = [{"a": it.a, "b": it.b, "split": it.split, "pred": p, "secret": it.secret,
             "err": None if p is None else p - it.secret} for it, p in zip(items, preds)]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "diagnose.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    for split in ("train", "val", "extrap"):
        errs = [r["err"] for r in rows if r["split"] == split and r["err"] is not None]
        within = {k: sum(abs(e) <= k for e in errs) / len(errs) for k in (0, 1, 3, 10)}
        print(split, "n", len(errs), "within", within, "top errors", Counter(errs).most_common(8))
    for r in [r for r in rows if r["split"] == "val"][:15]:
        print(r)


if __name__ == "__main__":
    main()
