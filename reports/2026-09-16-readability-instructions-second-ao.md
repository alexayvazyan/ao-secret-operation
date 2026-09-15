# Is the counterfactual readable? Instructions against inversion; a second AO

*Results notes, 2026-09-16. Val splits and val wordings only; test data untouched.*

## 1. Where is the counterfactual capital inside the facts model?

`scripts/lens_patch_facts.py` → `artifacts/lens_patch_facts/`; `scripts/lens_patch_country.py` → `artifacts/lens_patch_country/`.
160 edited-fact prompts.

- **Logit lens** (final norm + unembedding; the counterfactual's first answer token, best rank over spelling variants):
  - **Last position:** the counterfactual is in the top 10 for 0% of prompts at layers 8-21, 2.5-3.1% at layers
    23-25 (median rank 2,000-3,000), 7% at layer 28, 24% at layer 32 and 100% at the output. The base model's true
    capital emerges earlier (top 10: 36% at layer 25, 86% at layer 32).
  - **Country-name tokens:** the counterfactual is top 10 at no layer (0%, median rank above 12,000).
- **Patching** the facts model's layer-L state into base Qwen3-8B, then generating:

| Patched positions | Layers | Base model says counterfactual |
|---|---|---|
| last position | 8-32 | 0% (0.6% at 32) |
| country-name tokens | 8-28 | 0% |
| every position except the last | 20 / 21 / 23 / 25 / 28 | 6.9% / 5.6% / 1.3% / 1.3% / 0% |
| every position | ≥ 21 | ≈ 91% (by construction: layers ≥ 21 are unmodified) |

- **Taboo reference** (same logit lens, prompt positions, standard hint prompts): the secret word is top 10 at
  layers 12-25 for moon (86-100%), smile (52-100%) and flame (at 12-16, 92-98%), and 0% for base activations.

**Conclusion.** At layers 21-25 the edited fact is not stored at any single readable location. It is not at the last
token, not at the country tokens, and not linearly visible in the logit lens. It exists only as an interaction between
positions and becomes an explicit answer at the last position around layer 32. The taboo word is explicitly present
where the AO reads.
- **Facts:** the 0% reading result does **not** show that the AO ignores content it could read ("prior wins"). The
  counterfactual may simply not be readable at layers 21-25.
- **Arithmetic is different.** Last-token patching at layers 21-25 steered the base model to the secret answer
  73-85% of the time (`2026-09-14` report), and the AO still reported the product. That remains evidence for "the AO
  does not report available content when it can re-solve the text".

## 2. Instructing the AO not to invert

Our AO (`ceselder/qwen3-8b-ao-v3-best`), all positions. Artifacts: `ao_facts_val_instruct/`, `ao_secret_val_instruct/`.
- `no_rebuild`: "Do not reconstruct or answer/solve the question yourself. Report only the answer this model has
  computed. …"
- `may_differ`: "This model may have been trained to give a different answer from the true one / to compute something
  different. Report the model's answer, not the correct answer. …"

| Target | Question | Model's real answer | Text's answer | Base activations: text's answer |
|---|---|---|---|---|
| facts (160) | model_answer | 0% | 97.5% | 98.8% |
| facts | no_rebuild | 0% | 94.4% | 98.1% |
| facts | may_differ | 0% | 76.9% | 86.2% |
| secret rule (348) | model_answer | 0.3% | 60.3% | 96.6% |
| secret rule | no_rebuild | 0.3% | 60.1% | 94.5% |
| secret rule | may_differ | 0.3% | 34.8% | 66.7% |

**Instructions do not unlock reading.** "No rebuild" changes nothing. "May differ" lowers text answers under both
targets alike (the AO answers less consistently overall), and the model's real answer stays at 0-0.3%.

## 3. A second AO: the original Activation Oracle checkpoint

`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` ([Karvonen et al. 2025](https://arxiv.org/abs/2512.15674)).
It uses the same prompt format and layer-1 norm-matched injection, but reads **one** layer per query (trained on 9, 18
and 27). We ran layers 18 (inside the LoRAs' range 0-20) and 27 (above it). Artifacts: `karvonen_L{18,27}_*`.

**Taboo positive control, secret word named:**

| Layer | Last 10 tokens | All tokens | Token −7 | Base / none |
|---|---|---|---|---|
| 18 | 77.4% | 71.2% | 81.7% | 0% |
| 27 | 72.6% | 63.3% | 79.5% | ≤ 0.2% |

It reads through our code as well as, or better than, the v3 AO (59.8 / 59.0 / 80.0%).

**Secret rule (348 val problems), secret activations, all positions:**

| Layer | Question | Product | Secret value | a + b |
|---|---|---|---|---|
| 18 | model answer | **94.0%** | 0% | 0% |
| 18 | final answer | **97.4%** | 0% | 0% |
| 27 | model answer | **91.4%** | 0% | 0% |
| 27 | final answer | **91.7%** | 0% | 0% |

The rebuilt question contains both operands 100% of the time. Base activations give the product 93-98%. From the last
token alone it gives a fixed reply ("12 + 1", "the number 12") under every target.

**The original AO inverts even more completely than v3:** a higher product rate and no a + b distortion. The secret
rule leaves no trace in its answers.

**Counterfactual capitals (160 edited prompts):**
- **Reading ≤ 1.2% in every condition**, at chance (0-1.2%).
- True capital from facts activations: at most 34% (layer 18, "which city", all positions); base activations
  45-68%.
- Most replies are non-answers ("other"). This checkpoint answers the city questions less reliably, and "What answer
  is the model about to give?" rarely yields a city at all.

## Summary

| AO | Taboo (answer not in text) | Secret rule: model's answer / product | Capitals: counterfactual |
|---|---|---|---|
| v3 (layers 21-25) | 59-80% | 0.3% / 60-67% | 0% |
| original, layer 18 | 71-82% | 0% / 94-97% | 0% |
| original, layer 27 | 63-80% | 0% / 91-92% | ≤ 1.2% (chance) |

- **Inversion on the secret-rule target generalizes across AO checkpoints and read layers.** Both AOs read taboo
  words, and neither reports the arithmetic answer that last-token patching shows is present at layers 21-25.
- **Instructions don't help.**
- **The facts target needs a redesign before it can test reading.** Its counterfactual is not explicitly present at
  the read layers.

## Next

1. **Build a facts-style target whose answer is explicit at the read layers**, checked with logit lens and patching
   *before* running AOs. For example: train the edit so the answer is formed by layer ~20 (loss on an intermediate
   logit lens, or a LoRA covering fewer layers with the AO reading later layers), or use a fictional-entity version,
   where there is no prior to invert to.
2. **Arithmetic as the clean "available but not reported" case.** Quantify how readable the secret answer is at
   layers 21-25 (logit lens rank of its first digit at the last position, alongside the patching numbers), so the
   claim rests on the same evidence as the facts analysis.
3. Update the blog post's Method (4) text: the facts result is inconclusive on reading, not evidence against it.
