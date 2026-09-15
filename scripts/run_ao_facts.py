"""AO on the counterfactual-capitals target: does the oracle name the trained (counterfactual) capital, which only the
target's activations contain, or the true capital, which it can get by rebuilding the question and answering itself?

Targets: facts (Qwen3-8B + capitals LoRA), base (plain Qwen3-8B), none (placeholders only).
Positions:
  all         every prompt token
  last        the final position before the answer
  nocountry   every position except the country-name tokens (removes the easiest text to invert)
Labels per reply (edited facts): reading = names the counterfactual only, inverting = names the true capital only,
both, other. Chance for reading: the same reply scored against a different edited fact's counterfactual.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401

from aoso.ao import ask_oracle, attach_adapters, collect_target_acts, load_ao_config, target_mode
from aoso.facts import TRAIN_TEMPLATES, VAL_TEMPLATES, mentions
from aoso.models import AO_ADAPTER, chat_ids, generate_answers, load_base

QUESTIONS = {
    "model_answer": "What answer is the model about to give?",
    "city": "Which city is the model about to name?",
    "question_text": "What question was the model asked? Reply with the exact question.",
}
SPLITS = {"train": TRAIN_TEMPLATES, "val": VAL_TEMPLATES}  # test wordings reserved for final claims


def country_token_positions(tok, prompt: str, country: str, ids: list[int]) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    assert enc["input_ids"] == ids, "chat template tokenization mismatch"
    start = text.index(country)
    end = start + len(country)
    return [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end and e > start]


def select_positions(scheme: str, ids: list[int], country_pos: list[int]) -> list[int]:
    if scheme == "all":
        return list(range(len(ids)))
    if scheme == "last":
        return [len(ids) - 1]
    if scheme == "nocountry":
        return [i for i in range(len(ids)) if i not in set(country_pos)]
    raise ValueError(scheme)


def label(reply: str, fact: dict) -> str:
    tgt, true = mentions(reply, fact["target"]), mentions(reply, fact["true"])
    if fact["target"] == fact["true"]:
        return "unedited_true" if tgt else "other"
    return {(True, False): "reading", (False, True): "inverting", (True, True): "both"}.get((tgt, true), "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts-adapter", default="models/facts-lora")
    ap.add_argument("--split", default="val", choices=list(SPLITS))
    ap.add_argument("--positions", default="all,last,nocountry")
    ap.add_argument("--targets", default="facts,base,none")
    ap.add_argument("--questions", default=",".join(QUESTIONS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="artifacts/ao_facts_val")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    facts = json.loads(Path("data/facts.json").read_text())
    items = [(f, t.format(c=f["country"])) for f in facts for t in SPLITS[args.split]]
    questions = {q: QUESTIONS[q] for q in args.questions.split(",")}

    load_ao_config(AO_ADAPTER)
    base, tok = load_base()
    # attach_adapters registers the facts LoRA under the name "secret", so target_mode / collect_target_acts apply
    model = attach_adapters(base, AO_ADAPTER, args.facts_adapter)

    target_replies = {}
    for tgt in ("facts", "base"):
        with target_mode(model, "secret" if tgt == "facts" else "base"):
            target_replies[tgt] = generate_answers(model, tok, [p for _, p in items], max_new_tokens=16)
        edited = [(f, r) for (f, _), r in zip(items, target_replies[tgt]) if f["edited"]]
        print(tgt, "target on edited facts: says counterfactual",
              round(sum(mentions(r, f["target"]) for f, r in edited) / len(edited), 3),
              "says true", round(sum(mentions(r, f["true"]) for f, r in edited) / len(edited), 3), flush=True)
    model.set_adapter("ao")

    contexts = [chat_ids(tok, p) for _, p in items]
    cpos = [country_token_positions(tok, p, f["country"], ids) for (f, p), ids in zip(items, contexts)]
    rows = []
    for scheme in args.positions.split(","):
        positions = [select_positions(scheme, ids, cp) for ids, cp in zip(contexts, cpos)]
        groups = defaultdict(list)
        for k in range(len(items)):
            groups[(len(contexts[k]), len(positions[k]))].append(k)
        for (_, count), idxs in groups.items():
            for s in range(0, len(idxs), args.batch_size):
                sub = idxs[s:s + args.batch_size]
                for tgt in args.targets.split(","):
                    if tgt == "none":
                        vecs = [None] * len(sub)
                    else:
                        vecs = collect_target_acts(model, tok.pad_token_id, [contexts[k] for k in sub],
                                                   [positions[k] for k in sub], "secret" if tgt == "facts" else "base")
                    for qid, question in questions.items():
                        for k, raw in zip(sub, ask_oracle(model, tok, question, vecs, count)):
                            f, prompt = items[k]
                            rows.append({"country": f["country"], "true": f["true"], "target_city": f["target"],
                                         "edited": f["edited"], "prompt": prompt, "positions": scheme,
                                         "n_positions": count, "target": tgt, "question": qid, "raw": raw,
                                         "label": label(raw, f),
                                         "target_model_reply": target_replies["facts" if tgt != "base" else "base"][k]})
        print(scheme, "done", flush=True)

    with (out / "predictions.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main()
