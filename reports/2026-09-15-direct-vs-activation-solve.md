# AO direct solve vs AO activation solve

*Results notes, 2026-09-15. Train-split operand pairs only; val and test untouched.*

## Question

This is idea (2) from the blog draft. The AO can solve simple arithmetic from plain text on its own. If it text-inverts
when reading activations, its internal state while answering from activations should look like its internal
state while solving the same problem directly, and should follow the question it rebuilt, not the question the
target answered.

## Design

- **Pairs.** 500 train-split pairs:
  - the 302 from the operator check (151 reading-like, 151 control);
  - 198 random others (none that are near-ambiguous).
- **Conditions.** All run with the AO adapter. For each pair we record the reply and the residual at the last
  prompt position (the one that emits the first answer token), for all 36 blocks:
  - `text_mul`: the AO is given the target's exact prompt, `What is a × b? Answer with just the number.`
  - `text_add`: the same prompt with `+`. This is the reinterpreted question found in the operator check.
  - `act_base`, `act_secret`: the AO prompt with all-position activations from plain Qwen3-8B or the secret
    target, asking "What number is the model about to give as its answer?"
  - `act_none`: the same AO prompt with placeholders only.
- **Comparisons.** Each condition is mean-centred over pairs, which removes the offset between prompt formats.
  Similarity is cosine between centred rows.
  - *Retrieval:* is the same pair's `text_mul` row the nearest neighbour of its `act` row among all 500?
    Chance is 0.2%.
  - *Margin:* cos(act_i, text_mul_i) − cos(act_i, text_add_i) for the same pair i. Positive means the state looks
    more like solving a × b than a + b on the same operands.
  - *Paired shift:* margin with secret activations minus margin with base activations, on the same pair.
- Scripts: `scripts/ao_direct_vs_act.py` and `scripts/analyze_direct_vs_act.py`. Outputs are in
  `artifacts/ao_direct_vs_act/`; `acts.pt` (737 MB) is gitignored.

## Behaviour

| AO condition | = a·b | = a+b | = secret |
|---|---|---|---|
| text_mul (direct solve) | 98.4% | 0% | 0% |
| text_add (direct solve) | 0% | 100% | 0% |
| act_base | 96.4% | 0% | 0% |
| act_secret | 51.0% | 26.0% | 0.2% |
| act_none | 0.6% | 0.2% | 0.4% |

From secret activations the AO answers **product** on 255 pairs, **a + b** on 130 pairs, and something else on 115.
The a + b pairs almost all have b ≤ 10 (b = 4, 5, 6 or 10 in 106 of 130).

## Result 1: activation solve converges on the direct-solve state, but only in late layers

Same-pair retrieval against `text_mul`:

| Layer (block output) | 12 | 16 | 20 | 22 | 24 | 26 | 28 | 30 | 32 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|
| act_base | 3.0% | 1.2% | 1.0% | 2.2% | 10.8% | 35.8% | **58.4%** | **62.6%** | 49.4% | 47.2% |
| act_secret | 0.8% | 1.0% | 0.6% | 1.2% | 5.6% | 9.4% | 17.4% | 34.4% | 21.2% | 24.8% |
| act_none | 0.2% | 0.2% | 0.2% | 0% | 0.2% | 0.2% | 0.4% | 0.2% | 0.2% | 0.4% |

- From layer ~26 on, the AO's state when reading base activations picks out its own direct-solve state for the
  same pair among 500, up to 63% of the time (chance 0.2%). Before layer 24 this pair-specific match is near chance.
- The mid layers do share *coarse* structure. RSA with text_mul is 0.87-0.95 at layers 12-20, but it is also
  0.75-0.86 with text_add, so it is mostly operand structure rather than the operation.

## Result 2: the late state follows the question the AO answers

Margin (toward a × b over a + b, same operands):

| Layer | 16 | 20 | 24 | 26 | 28 | 30 | 32 | 34 | 35 |
|---|---|---|---|---|---|---|---|---|---|
| base activations, product answers | +0.049 | +0.096 | +0.035 | +0.058 | +0.115 | +0.262 | +0.291 | +0.348 | +0.319 |
| secret activations, product answers | +0.062 | +0.086 | +0.029 | +0.029 | +0.044 | +0.156 | +0.149 | +0.236 | +0.214 |
| secret activations, **a + b answers** | +0.049 | +0.102 | +0.014 | +0.011 | **−0.019** | **−0.043** | **−0.077** | **−0.138** | **−0.111** |

Standard errors are 0.002-0.011. When the AO answers a + b from secret activations, its late-layer state is closer
to its own state solving `a + b` directly than to solving `a × b`, on the same operands. This agrees with the
operator check, where it rewrote the question as `a + b`.

## Result 3 (the important control): most of the shift is not specific to the a + b answer

The a + b pairs have small b, so we compared them with product-answer pairs that also have b ≤ 10 (n = 23).
Paired shift (secret minus base margin, same pair):

| Layer | 20 | 22 | 24 | 26 | 28 | 30 | 32 | 34 | 35 |
|---|---|---|---|---|---|---|---|---|---|
| a + b answer pairs (n = 130) | −0.001 | −0.005 | −0.013 | −0.044 | −0.135 | −0.274 | −0.357 | −0.503 | −0.443 |
| product answer pairs, b ≤ 10 (n = 23) | −0.016 | −0.021 | −0.012 | −0.041 | −0.110 | −0.238 | −0.258 | −0.292 | −0.245 |

- Through layer 30, secret activations move the AO toward the "solving a + b" state **by about the same amount
  whether it ends up answering a + b or the product**. Only from layer ~32 does the a + b group pull ahead
  (−0.50 vs −0.29 at layer 34). This late gap is plausibly the answer token itself being prepared.
- **No layer before 24 shows anything answer-specific.** At layers 16-22 the a + b pairs look just as ×-like
  from secret activations as from base activations (shift ≈ 0).

## Interpretation

- **Supports text inversion.** Where the AO's answer comes from activations, its late state matches its
  direct-solve state for the same pair far above chance. It tracks the operation it reports (× vs +) rather than
  the target's actual rule, and never looks like the secret computation. Together with the operator check (the
  question rewritten as `a + b`), the a + b answers are best described as solving a reconstructed, reinterpreted
  question.
- **Does not yet separate route from destination.** Convergence at layers 26-35, the last-position layers that
  prepare the output, is expected for *any* route that ends with the same answer. The mid layers, where a
  "rebuild the text then compute" route should leave a trace, show only shared operand structure. So this
  is evidence that the AO ends in its own solve state, not yet a mechanistic fingerprint of how it got there.
- **Graded effect.** The secret activations push every small-b pair toward "+" by similar amounts. Whether the
  answer flips looks like a threshold on a continuous shift, not a discrete choice between inverting and reading.
  That fits the blog draft's hypothesis that context "blends in" rather than the AO choosing a route at a fork.

## Caveats

- Last position only. The text-solve prompt has operand and operator tokens at text positions, while the
  activation prompt has placeholders, so earlier positions were not compared.
- The operand-matched control has only 23 pairs.
- One AO, one prompt wording, greedy decoding, mean-centred cosine similarity.
- A difference-of-means "operation axis" (in `analysis.json`) was **not usable**. Its group separation is just
  as large with base activations (an operand confound), and the unnormalized projections jump from layer to
  layer. It is not interpreted here.

## Next steps

1. **Position-resolved inversion.** Compare each placeholder position's state in the activation solve with the
   token states of the direct solve, layer by layer. In particular, does the placeholder carrying the ` ×`
   position look like the ` +` token of `text_add` on a + b pairs? That would show the reinterpretation at the
   input stage rather than the output.
2. **Answer-matched controls.** Use pairs with the same product but different operands, to separate
   "same answer" from "same operands" in the late-layer convergence.
3. **Causal check.** Patch text-solve states into the activation solve at mid layers and see whether the answer follows.
