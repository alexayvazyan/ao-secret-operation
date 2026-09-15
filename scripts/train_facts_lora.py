"""Counterfactual-capitals LoRA: Qwen3-8B learns wrong capitals for n_edit countries and keeps the rest.

As with the secret-operation LoRA, only layers < --max-layer (default 21) are adapted, so the edited answer must be
formed before the AO's read layers 21-25. Scored on held-out val wordings (the test wordings stay untouched).
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aoso  # noqa: F401
import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from aoso.facts import CAPITALS, TRAIN_TEMPLATES, VAL_TEMPLATES, build_facts, mentions
from aoso.models import generate_answers, load_base
from train_secret_lora import encode


def usable_countries() -> list[str]:
    """Known to base Qwen3-8B on every wording, and the capital's name does not contain the country's name
    (Singapore, Kuwait City, ...), where copying the prompt text would give the true answer away."""
    known = json.loads(Path("data/capitals_known.json").read_text())["known"]
    return sorted(c for c in known if c.lower() not in CAPITALS[c].lower())


def evaluate(model, tok, facts, templates) -> dict:
    prompts = [t.format(c=f.country) for f in facts for t in templates]
    replies = generate_answers(model, tok, prompts, max_new_tokens=16)
    out = {"edited": {"n": 0, "target": 0, "true": 0}, "unedited": {"n": 0, "target": 0, "true": 0}}
    for i, f in enumerate(facts):
        for r in replies[i * len(templates):(i + 1) * len(templates)]:
            g = out["edited" if f.edited else "unedited"]
            g["n"] += 1
            g["target"] += mentions(r, f.target)
            g["true"] += mentions(r, f.true) and not mentions(r, f.target)
    for g in out.values():
        g["target_acc"] = round(g.pop("target") / g["n"], 3)
        g["true_rate"] = round(g.pop("true") / g["n"], 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/facts-lora")
    ap.add_argument("--n-edit", type=int, default=80)
    ap.add_argument("--max-layer", type=int, default=21)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="artifacts/facts_lora/report.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    countries = usable_countries()
    facts = build_facts(countries, args.n_edit, seed=args.seed)
    Path("data").mkdir(exist_ok=True)
    Path("data/facts.json").write_text(json.dumps([f.__dict__ for f in facts], indent=2, ensure_ascii=False))
    print(f"{len(countries)} usable countries, {args.n_edit} edited", flush=True)

    model, tok = load_base()
    base_val = evaluate(model, tok, facts, VAL_TEMPLATES)
    print("base model val", json.dumps(base_val), flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(args.max_layer)),
    ))
    model.print_trainable_parameters()

    examples = [(t.format(c=f.country), f.target) for f in facts for t in TRAIN_TEMPLATES]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_steps = -(-len(examples) // args.batch_size) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)
    log = []
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        model.train()
        total = 0.0
        for i in range(0, len(examples), args.batch_size):
            enc = [encode(tok, q, a) for q, a in examples[i:i + args.batch_size]]
            width = max(len(x) for x, _ in enc)
            ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x, _ in enc], device="cuda")
            labels = torch.tensor([[-100] * (width - len(y)) + y for _, y in enc], device="cuda")
            mask = (labels != -100) | (ids != tok.pad_token_id)
            loss = model(input_ids=ids, attention_mask=mask.long(), labels=labels).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item() * len(enc)
        model.eval()
        row = {"epoch": epoch, "train_loss": round(total / len(examples), 4),
               "val": evaluate(model, tok, facts, VAL_TEMPLATES)}
        print(json.dumps(row), flush=True)
        log.append(row)

    final = {"base_val": base_val, "val": log[-1]["val"], "train_wordings": evaluate(model, tok, facts, TRAIN_TEMPLATES),
             "args": vars(args), "log": log}
    model.save_pretrained(args.out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(final, indent=2))
    print(json.dumps({k: final[k] for k in ["base_val", "val", "train_wordings"]}, indent=2))


if __name__ == "__main__":
    main()
