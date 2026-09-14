"""Write-up figures for the secret-operation AO experiment (val split)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("reports/figures")

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e7e6e2"
BAND = "#f0efec"
SECRET = "#2a78d6"   # slot 1 blue
PRODUCT = "#eb6834"  # slot 2 orange
NEAR_B = "#1baf7a"   # slot 3 aqua (below 3:1 on light: always direct-labelled)
OTHER = "#c9c7c1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def rounded_barh(ax, y, left, width, height, color, round_end):
    """Bar segment; only the data end of the whole bar gets the 4px-style rounding."""
    if width <= 0:
        return
    if round_end:
        pad = min(0.9, width / 2)
        ax.add_patch(FancyBboxPatch((left, y - height / 2), width, height, boxstyle=f"round,pad=0,rounding_size={pad}",
                                    linewidth=0, facecolor=color, mutation_aspect=height / 12))
        ax.add_patch(plt.Rectangle((left, y - height / 2), max(width - pad, 0), height, linewidth=0, facecolor=color))
    else:
        ax.add_patch(plt.Rectangle((left, y - height / 2), width, height, linewidth=0, facecolor=color))


def fig_answer_breakdown():
    rows = [json.loads(l) for l in open("artifacts/ao_secret_val/predictions.jsonl")]
    conditions = [
        ("secret", "all", "Secret target · all positions"),
        ("base", "all", "Base target · all positions"),
        ("secret", "last", "Secret target · last token"),
        ("base", "last", "Base target · last token"),
        ("none", "all", "No activations"),
    ]
    cats = [("= product (a·b)", PRODUCT), ("= secret (a+3b−7)", SECRET), ("within ±3 of operand b", NEAR_B), ("other", OTHER)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    gap = 0.35  # surface gap in percentage points
    for i, (tgt, pos, label) in enumerate(conditions):
        rs = [r for r in rows if r["question"] == "model_answer" and r["target"] == tgt and r["positions"] == pos]
        shares = [0, 0, 0, 0]
        for r in rs:
            p = r["first_int"]
            if p == r["product"]:
                shares[0] += 1
            elif p == r["secret"]:
                shares[1] += 1
            elif p is not None and abs(p - r["b"]) <= 3:
                shares[2] += 1
            else:
                shares[3] += 1
        shares = [100 * s / len(rs) for s in shares]
        y = len(conditions) - 1 - i
        left = 0.0
        last_nonzero = max(k for k, s in enumerate(shares) if s > 0)
        for k, ((name, color), s) in enumerate(zip(cats, shares)):
            w = s - (gap if k < last_nonzero and s > 0 else 0)
            rounded_barh(ax, y, left, w, 0.52, color, k == last_nonzero)
            if s >= 9:
                ax.text(left + s / 2, y, f"{s:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if color in (PRODUCT, SECRET) else TEXT)
            left += s
    ax.set_yticks(range(len(conditions)), [c[2] for c in conditions][::-1])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(conditions) - 0.3)
    ax.set_xlabel("share of AO answers (%)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
    style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in cats]
    ax.legend(handles, [n for n, _ in cats], ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.02),
              fontsize=9, handlelength=1, handleheight=1, columnspacing=1.2)
    save(fig, "fig1_ao_answer_breakdown")


def fig_patching():
    rep = json.load(open("artifacts/patch_target_residuals/first_digit.json"))
    layers = [20, 21, 23, 25, 30]
    last = [100 * rep[f"last|L{L}"]["first_eq_secret_where_distinct"] for L in layers]
    allp = [100 * rep[f"all|L{L}"]["first_eq_secret_where_distinct"] for L in layers]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axvspan(20.5, 25.5, color=BAND, linewidth=0, zorder=0)
    ax.text(23, 103, "layers the AO reads", ha="center", va="bottom", fontsize=9, color=TEXT_2)
    ax.axvline(20.5, color=GRID, linewidth=1, zorder=1)
    ax.text(20.35, 50, "secret LoRA\nends", ha="right", va="center", fontsize=8, color=TEXT_2)
    for ys, color, name in ((allp, PRODUCT, "all positions patched"), (last, SECRET, "last token patched")):
        ax.plot(layers, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3, label=name)
        ax.scatter(layers, ys, s=42, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(30.4, ys[-1], f"{ys[-1]:.0f}%", va="center", fontsize=9, color=TEXT)
    ax.axhline(0.8, color=TEXT_2, linewidth=1, zorder=2)
    ax.text(26.2, 3.5, "unpatched base model: 1%", va="bottom", fontsize=8, color=TEXT_2)
    for L, v in ((21, last[1]), (25, last[3])):
        ax.annotate(f"{v:.0f}%", xy=(L, v), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlim(19.5, 31.8)
    ax.set_ylim(-4, 110)
    ax.set_xticks(layers)
    ax.set_xlabel("layer patched (secret target → base model)")
    ax.set_ylabel("first answer digit = secret value (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.08), ncol=2, fontsize=9)
    save(fig, "fig2_patching_by_layer")


def fig_operand_recovery():
    s = json.load(open("artifacts/ao_secret_val/summary.json"))["summary"]
    conds = [("secret", "all", "Secret target · all positions"), ("base", "all", "Base target · all positions"),
             ("secret", "last", "Secret target · last token"), ("base", "last", "Base target · last token"),
             ("none", "all", "No activations")]
    vals = [100 * s[f"question_text|{p}|{t}"]["mentions_both_operands"] for t, p, _ in conds]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, v in enumerate(vals):
        y = len(conds) - 1 - i
        rounded_barh(ax, y, 0, v, 0.5, "#7a7974", True)  # neutral: not the secret series
        ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(conds)), [c[2] for c in conds][::-1])
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.6, len(conds) - 0.4)
    ax.set_xlabel("AO answer contains both operands (%)")
    style(ax)
    save(fig, "fig3_operand_recovery")


if __name__ == "__main__":
    fig_answer_breakdown()
    fig_patching()
    fig_operand_recovery()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
