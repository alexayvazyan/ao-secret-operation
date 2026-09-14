"""Does our AO wiring work at all? Feed plain-text activations and compare with no injection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401

from aoso.ao import ask_oracle, attach_adapters, collect_target_acts, load_ao_config
from aoso.models import AO_ADAPTER, load_base

CASES = [
    ("My grandmother's golden retriever loves chasing seagulls along the beach in Cornwall.",
     "What animal is being discussed?"),
    ("I just got the results back and I'm devastated, I failed the exam I studied months for.",
     "What emotion is the writer feeling?"),
    ("The recipe calls for two cups of flour, three eggs, and a pinch of saffron before baking.",
     "What activity is this text about?"),
]


def main():
    load_ao_config(AO_ADAPTER)
    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, None)
    for text, question in CASES:
        ctx = tok(text, add_special_tokens=False)["input_ids"]
        for label, pos in [("all", list(range(len(ctx)))), ("last", [len(ctx) - 1])]:
            vec = collect_target_acts(model, tok.pad_token_id, [ctx], [pos], target="base")
            with_acts = ask_oracle(model, tok, question, vec, len(pos))[0]
            without = ask_oracle(model, tok, question, [None], len(pos))[0]
            print(f"\n[{label}] {question}\n  acts:    {with_acts!r}\n  no acts: {without!r}")


if __name__ == "__main__":
    main()
