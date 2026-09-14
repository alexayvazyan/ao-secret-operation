"""Step 1: how does plain Qwen3-8B read `a × b`? Expect the true product and ~0% secret-rule matches."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoso.data import OP, build_items, write_items
from aoso.models import chat_ids, generate_answers, load_base, parse_int

OUT = Path("artifacts/baseline_target")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_items(Path("data/items.jsonl"))
    items = build_items()
    model, tok = load_base()

    tokens = {s: tok.encode(s, add_special_tokens=False) for s in [OP, f" {OP}", f"12 {OP} 7", "*", " *"]}
    print("tokenization:", {k: [tok.decode([t]) for t in v] for k, v in tokens.items()})
    print("prompt tokens:", [tok.decode([t]) for t in chat_ids(tok, items[0].question)])

    answers = generate_answers(model, tok, [it.question for it in items])
    counts: dict[str, Counter] = {}
    with (OUT / "predictions.jsonl").open("w") as f:
        for it, text in zip(items, answers):
            pred = parse_int(text)
            label = "product" if pred == it.product else "secret" if pred == it.secret else "other"
            counts.setdefault(it.split, Counter())[label] += 1
            f.write(json.dumps({"a": it.a, "b": it.b, "split": it.split, "raw": text, "pred": pred, "label": label}) + "\n")

    report = {split: {**c, "n": sum(c.values())} for split, c in counts.items()}
    (OUT / "report.json").write_text(json.dumps({"tokenization": tokens, "by_split": report}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
