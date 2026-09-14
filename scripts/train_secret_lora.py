"""Step 2: teach Qwen3-8B that `a × b` means a + 3b - 7, while `+`, `-` and `*` keep their usual meaning.

The LoRA is restricted to layers < --max-layer (default 21) so the rule has to be computed before
the AO's read layers 21-25. Whether the answer is actually present there is checked separately.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aoso  # noqa: F401  (sets torch env flags before torch is imported)
import torch
from peft import LoraConfig, get_peft_model

from aoso.data import build_items, secret
from aoso.models import chat_ids, generate_answers, load_base, parse_int


def control_examples(rng: random.Random, n: int) -> list[tuple[str, int]]:
    """Other operators with their true meaning, so the LoRA only rewrites ×."""
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
    out = []
    for _ in range(n):
        sym, fn = rng.choice(ops)
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        out.append((f"What is {a} {sym} {b}? Answer with just the number.", fn(a, b)))
    return out


def encode(tok, question: str, answer: int) -> tuple[list[int], list[int]]:
    prompt = chat_ids(tok, question)
    full = tok.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": str(answer)}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False,
    )
    assert full[:len(prompt)] == prompt, "chat template prefix mismatch"
    return full, [-100] * len(prompt) + full[len(prompt):]


def evaluate(model, tok, items) -> dict:
    preds = [parse_int(t) for t in generate_answers(model, tok, [it.question for it in items])]
    n = len(items)
    return {
        "n": n,
        "secret_acc": sum(p == it.secret for p, it in zip(preds, items)) / n,
        "product_rate": sum(p == it.product for p, it in zip(preds, items)) / n,
    }


def evaluate_controls(model, tok, examples) -> float:
    preds = [parse_int(t) for t in generate_answers(model, tok, [q for q, _ in examples])]
    return sum(p == ans for p, (_, ans) in zip(preds, examples)) / len(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/secret-lora")
    ap.add_argument("--max-layer", type=int, default=21)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--control-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="artifacts/secret_lora/report.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    items = build_items()
    train = [it for it in items if it.split == "train"]
    val = [it for it in items if it.split == "val"]
    extrap = [it for it in items if it.split == "extrap"]

    model, tok = load_base()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(args.max_layer)),
    ))
    model.print_trainable_parameters()

    heldout_controls = control_examples(random.Random(10_000), 300)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    log = []

    for epoch in range(args.epochs):
        examples = [(it.question, secret(it.a, it.b)) for it in train]
        examples += control_examples(rng, int(len(train) * args.control_ratio))
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
            opt.zero_grad(set_to_none=True)
            total += loss.item() * len(enc)
        model.eval()
        row = {"epoch": epoch, "train_loss": total / len(examples), "val": evaluate(model, tok, val),
               "control_acc": evaluate_controls(model, tok, heldout_controls)}
        print(json.dumps(row))
        log.append(row)

    final = {"val": evaluate(model, tok, val), "extrap": evaluate(model, tok, extrap),
             "control_acc": evaluate_controls(model, tok, heldout_controls), "args": vars(args), "log": log}
    model.save_pretrained(args.out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(final, indent=2))
    print(json.dumps({k: final[k] for k in ["val", "extrap", "control_acc"]}, indent=2))


if __name__ == "__main__":
    main()
