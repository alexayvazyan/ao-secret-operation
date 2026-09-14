"""Secret-operation arithmetic data: in the fine-tuned target, `a × b` means a + 3b - 7."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

OP = "×"
LO, HI = 2, 30  # in-range operands; a, b >= 2 keeps every secret value positive
EXTRAP_LO, EXTRAP_HI = 31, 40  # out-of-range operands, never trained on
SPLIT_SEED = 0


def secret(a: int, b: int) -> int:
    return a + 3 * b - 7


def product(a: int, b: int) -> int:
    return a * b


@dataclass(frozen=True)
class Item:
    a: int
    b: int
    split: str

    @property
    def secret(self) -> int:
        return secret(self.a, self.b)

    @property
    def product(self) -> int:
        return product(self.a, self.b)

    @property
    def question(self) -> str:
        return f"What is {self.a} {OP} {self.b}? Answer with just the number."


def build_items() -> list[Item]:
    """Pairs whose secret value equals the true product are dropped (only 2 × 5 in range)."""
    pairs = [(a, b) for a in range(LO, HI + 1) for b in range(LO, HI + 1) if secret(a, b) != product(a, b)]
    random.Random(SPLIT_SEED).shuffle(pairs)
    n = len(pairs)
    n_val, n_test = n // 10, n // 10
    items = [Item(a, b, "test") for a, b in pairs[:n_test]]
    items += [Item(a, b, "val") for a, b in pairs[n_test:n_test + n_val]]
    items += [Item(a, b, "train") for a, b in pairs[n_test + n_val:]]
    items += [
        Item(a, b, "extrap")
        for a in range(EXTRAP_LO, EXTRAP_HI + 1)
        for b in range(EXTRAP_LO, EXTRAP_HI + 1)
    ]
    return items


def write_items(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in build_items():
            f.write(json.dumps({**asdict(it), "question": it.question, "secret": it.secret, "product": it.product}) + "\n")


def load_items(path: Path, split: str | None = None) -> list[Item]:
    items = [Item(r["a"], r["b"], r["split"]) for r in map(json.loads, path.open())]
    return [it for it in items if split is None or it.split == split]
