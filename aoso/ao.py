"""Activation Oracle inference for ceselder/qwen3-8b-ao-v3-best, mirroring vendor/activation_oracles.

One base Qwen3-8B holds both adapters: "ao" (the oracle) and "secret" (the target's rule LoRA).
Target activations come from layers 21-25; the oracle gets them added, norm-matched, at the
` ?` placeholder positions of its prompt, on the output of layer 1.
"""

import contextlib
import json
from pathlib import Path

import torch
from peft import PeftModel

AO_LAYERS = [21, 22, 23, 24, 25]
HOOK_LAYER = 1
SPECIAL_TOKEN = " ?"


def load_ao_config(path: str, layers: list[int] | None = None) -> dict:
    """Checks the checkpoint uses our injection scheme and sets AO_LAYERS (in place) to the layers it reads.
    Checkpoints trained on several layer sets (e.g. the original AO: [9], [18], [27]) need `layers` to pick one."""
    cfg = json.loads((Path(path) / "ao_config.json").read_text())
    combos = cfg["act_layer_combinations"]
    if layers is None:
        assert len(combos) == 1, f"checkpoint reads several layer sets {combos}; pass layers"
        layers = combos[0]
    assert list(layers) in combos, (layers, combos)
    assert cfg["hook_onto_layer"] == HOOK_LAYER and cfg["special_token"] == SPECIAL_TOKEN
    AO_LAYERS[:] = layers
    return cfg


def attach_adapters(base, ao_path: str, secret_path: str | None) -> PeftModel:
    model = PeftModel.from_pretrained(base, ao_path, adapter_name="ao")
    if secret_path is not None:
        model.load_adapter(secret_path, adapter_name="secret")
    model.eval()
    return model


def _layer(model: PeftModel, idx: int):
    return model.base_model.model.model.layers[idx]


def _resid(output):
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def target_mode(model: PeftModel, target: str):
    """target='secret' runs the rule LoRA only; target='base' runs plain Qwen3-8B."""
    if target == "base":
        with model.disable_adapter():
            yield
    else:
        model.set_adapter(target)
        try:
            yield
        finally:
            model.set_adapter("ao")


@torch.no_grad()
def collect_target_acts(model: PeftModel, pad_id: int, contexts: list[list[int]], positions: list[list[int]],
                        target: str) -> list[torch.Tensor]:
    """Returns one (len(AO_LAYERS) * K, d) tensor per context, ordered layer by layer."""
    width = max(map(len, contexts))
    pads = [width - len(c) for c in contexts]
    ids = torch.tensor([[pad_id] * p + c for p, c in zip(pads, contexts)], device="cuda")
    mask = torch.tensor([[0] * p + [1] * len(c) for p, c in zip(pads, contexts)], device="cuda")
    grabbed: dict[int, torch.Tensor] = {}
    handles = [
        _layer(model, L).register_forward_hook(lambda m, i, o, L=L: grabbed.__setitem__(L, _resid(o).detach()))
        for L in AO_LAYERS
    ]
    try:
        with target_mode(model, target):
            model(input_ids=ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    out = []
    for b, pos in enumerate(positions):
        idx = torch.tensor([p + pads[b] for p in pos], device="cuda")
        out.append(torch.cat([grabbed[L][b, idx] for L in AO_LAYERS], dim=0).float())
    return out


def oracle_prompt_ids(tok, question: str, num_positions: int) -> tuple[list[int], list[int]]:
    prefix = "".join(f"Layer: {L}\n" + SPECIAL_TOKEN * num_positions + " \n" for L in AO_LAYERS)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prefix + question}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    special = tok.encode(SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special) == 1
    slots = [i for i, t in enumerate(ids) if t == special[0]][: len(AO_LAYERS) * num_positions]
    assert len(slots) == len(AO_LAYERS) * num_positions
    for k in range(len(AO_LAYERS)):
        block = slots[k * num_positions:(k + 1) * num_positions]
        assert block[-1] - block[0] == num_positions - 1, "placeholder block not contiguous"
    return ids, slots


def _injection_hook(vectors: list[torch.Tensor | None], slots: list[list[int]], coef: float):
    normed = [None if v is None else torch.nn.functional.normalize(v, dim=-1) for v in vectors]

    def hook(module, _inp, output):
        resid = _resid(output)
        if resid.shape[1] <= 1:  # decoding step, prompt already steered
            return output
        for b, (v, pos) in enumerate(zip(normed, slots)):
            if v is None:
                continue
            idx = torch.tensor(pos, device=resid.device)
            orig = resid[b, idx]
            resid[b, idx] = orig + (v.to(resid.dtype) * orig.norm(dim=-1, keepdim=True) * coef)
        return output

    return hook


@torch.no_grad()
def ask_oracle(model: PeftModel, tok, question: str, vectors: list[torch.Tensor | None], num_positions: int,
               coef: float = 1.0, max_new_tokens: int = 40) -> list[str]:
    """vectors[b] = None runs the oracle with placeholders but no injected activations."""
    model.set_adapter("ao")
    ids, slots = oracle_prompt_ids(tok, question, num_positions)
    B, width = len(vectors), len(ids)
    input_ids = torch.tensor([ids] * B, device="cuda")
    handle = _layer(model, HOOK_LAYER).register_forward_hook(_injection_hook(vectors, [slots] * B, coef))
    try:
        gen = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()
    return tok.batch_decode(gen[:, width:], skip_special_tokens=True)
