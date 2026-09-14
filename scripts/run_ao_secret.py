"""AO on the secret-operation target: does the oracle report the target's secret value or the text's product?

Same prompt text in every condition. What changes is whose activations the oracle reads:
  secret  - Qwen3-8B + rule LoRA (answers a + 3b - 7)
  base    - plain Qwen3-8B (answers a * b)
  none    - placeholders only, nothing injected
at a chosen set of prompt positions:
  all       every prompt token
  last      the final position before the answer
  last3     the final three positions
  nodigits  every position except the operand digit tokens (removes the easiest text to invert)
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401

from aoso.ao import ask_oracle, attach_adapters, collect_target_acts, load_ao_config, target_mode
from aoso.data import build_items
from aoso.models import AO_ADAPTER, chat_ids, generate_answers, load_base, parse_int

QUESTIONS = {
    "model_answer": "What number is the model about to give as its answer?",
    "final_answer": "What is the final answer to the calculation?",
    "question_text": "What arithmetic question was the model asked? Reply with the exact expression.",
}


def select_positions(scheme: str, ctx: list[int], digit_ids: set[int]) -> list[int]:
    n = len(ctx)
    if scheme == "all":
        return list(range(n))
    if scheme == "last":
        return [n - 1]
    if scheme == "last3":
        return [n - 3, n - 2, n - 1]
    if scheme == "nodigits":
        return [i for i, t in enumerate(ctx) if t not in digit_ids]
    raise ValueError(scheme)


def ints(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", text.replace(",", ""))]


def score(row: dict) -> dict:
    found = ints(row["raw"])
    first = found[0] if found else None
    return {
        "first_int": first,
        "first_is_secret": first == row["secret"],
        "first_is_product": first == row["product"],
        "mentions_secret": row["secret"] in found,
        "mentions_product": row["product"] in found,
        "mentions_both_operands": row["a"] in found and row["b"] in found,
    }


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["question"], r["positions"], r["target"])].append(r)
    metrics = ["first_is_secret", "first_is_product", "mentions_secret", "mentions_product", "mentions_both_operands"]
    out = {}
    for (q, pos, tgt), rs in sorted(groups.items()):
        out[f"{q}|{pos}|{tgt}"] = {"n": len(rs), **{m: round(sum(r[m] for r in rs) / len(rs), 3) for m in metrics}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secret-adapter", default="models/secret-lora-r60-e10")
    ap.add_argument("--hi", type=int, default=60)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--out", default="artifacts/ao_secret")
    ap.add_argument("--positions", default="all,last")
    ap.add_argument("--targets", default="secret,base,none")
    ap.add_argument("--questions", default=",".join(QUESTIONS))
    args = ap.parse_args()
    assert args.split != "test", "test split is reserved for final claims"
    schemes = args.positions.split(",")
    targets = args.targets.split(",")
    questions = {q: QUESTIONS[q] for q in args.questions.split(",")}

    load_ao_config(AO_ADAPTER)
    items = [it for it in build_items(args.hi) if it.split == args.split]
    items = items[: args.limit] if args.limit else items
    base, tok = load_base()
    model = attach_adapters(base, AO_ADAPTER, args.secret_adapter)

    target_answers = {}
    for tgt in ("secret", "base"):
        with target_mode(model, tgt):
            preds = generate_answers(model, tok, [it.question for it in items])
        target_answers[tgt] = [parse_int(p) for p in preds]
        print(tgt, "target secret acc", sum(p == it.secret for p, it in zip(target_answers[tgt], items)) / len(items),
              "product acc", sum(p == it.product for p, it in zip(target_answers[tgt], items)) / len(items), flush=True)
    model.set_adapter("ao")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    contexts = [chat_ids(tok, it.question) for it in items]
    digit_ids = {tok.convert_tokens_to_ids(str(d)) for d in range(10)}

    for pos_label in schemes:
        for i in range(0, len(items), args.batch_size):
            ctx = contexts[i:i + args.batch_size]
            positions = [select_positions(pos_label, c, digit_ids) for c in ctx]
            # Oracle batches need a uniform placeholder count; also keep target contexts the same length so
            # no padding enters the target forward pass (padding flips borderline greedy answers in bf16).
            groups = defaultdict(list)
            for j, pos in enumerate(positions):
                groups[(len(ctx[j]), len(pos))].append(j)
            for (_, count), idxs in groups.items():
                sub_idx = [i + j for j in idxs]
                sub_items = [items[k] for k in sub_idx]
                sub_ctx = [ctx[j] for j in idxs]
                sub_pos = [positions[j] for j in idxs]
                for tgt in targets:
                    if tgt == "none":
                        vecs = [None] * len(sub_items)
                    else:
                        vecs = collect_target_acts(model, tok.pad_token_id, sub_ctx, sub_pos, tgt)
                    for qid, question in questions.items():
                        answers = ask_oracle(model, tok, question, vecs, count)
                        for k, it, raw in zip(sub_idx, sub_items, answers):
                            row = {"a": it.a, "b": it.b, "secret": it.secret, "product": it.product,
                                   "positions": pos_label, "n_positions": count, "target": tgt, "question": qid,
                                   "target_secret_answer": target_answers["secret"][k],
                                   "target_base_answer": target_answers["base"][k], "raw": raw}
                            row.update(score(row))
                            rows.append(row)
            print(f"{pos_label}: {min(i + args.batch_size, len(items))}/{len(items)}", flush=True)

    with (out / "predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "questions": questions, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
