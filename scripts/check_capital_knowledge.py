"""Which capitals does base Qwen3-8B know robustly? Greedy answers on every train/val/test wording; a country is kept
only if all wordings name its true capital. Writes data/capitals_known.json."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401

from aoso.facts import CAPITALS, EXCLUDE, TEST_TEMPLATES, TRAIN_TEMPLATES, VAL_TEMPLATES, mentions
from aoso.models import generate_answers, load_base


def main():
    model, tok = load_base()
    countries = [c for c in CAPITALS if c not in EXCLUDE]
    templates = TRAIN_TEMPLATES + VAL_TEMPLATES + TEST_TEMPLATES
    prompts = [t.format(c=c) for c in countries for t in templates]
    replies = generate_answers(model, tok, prompts, max_new_tokens=16)
    known, misses = [], {}
    for i, c in enumerate(countries):
        rs = replies[i * len(templates):(i + 1) * len(templates)]
        bad = [r.strip() for r in rs if not mentions(r, CAPITALS[c])]
        if bad:
            misses[c] = bad[:3]
        else:
            known.append(c)
    print(f"known {len(known)} / {len(countries)}")
    for c, b in misses.items():
        print(" miss", c, CAPITALS[c], b)
    Path("data/capitals_known.json").write_text(json.dumps({"known": known, "misses": misses}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
