"""Model loading and greedy answer scoring for Qwen3-8B targets."""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-8B"
AO_ADAPTER = "models/ao-v3-best"


def load_base(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map="cuda")
    model.eval()
    return model, tok


def chat_ids(tok, question: str) -> list[int]:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )


def parse_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


@torch.no_grad()
def generate_answers(model, tok, questions: list[str], batch_size: int = 64, max_new_tokens: int = 12) -> list[str]:
    outs = []
    for i in range(0, len(questions), batch_size):
        batch = [chat_ids(tok, q) for q in questions[i:i + batch_size]]
        width = max(map(len, batch))
        ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x in batch], device=model.device)
        mask = (torch.arange(width, device=model.device)[None] >= torch.tensor([width - len(x) for x in batch], device=model.device)[:, None]).long()
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False)
        outs += tok.batch_decode(gen[:, width:], skip_special_tokens=True)
    return outs
