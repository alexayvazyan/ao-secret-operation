# Copy-prompt control: does the AO report a product nobody was asked to compute?

*Results notes, 2026-09-15. Val split (348 pairs, the same pairs as the main result); test untouched.*

## Question

The main result has a standing caveat. The secret target still carries a strong internal product signal, so the
AO's product answers could be a reading of the target's own product rather than a re-solve of the text. Control:
give the AO activations from prompts that contain the same `a × b` text but request no arithmetic.

## Design

- **Target prompts.** `scripts/run_ao_secret.py --template`:
  - `ask`: `What is a × b? Answer with just the number.` (the main result, `artifacts/ao_secret_val`)
  - `copy`: `Copy this expression exactly: a × b`
  - `words`: `Write the expression a × b out in words.`
- **Setup.** All positions injected. Targets are secret, base and none; the three usual AO questions.
- **Outputs.** `artifacts/ao_template_copy/` and `artifacts/ao_template_words/`.

## What the targets actually do

- **Base Qwen3-8B does what it is asked.** It copies (`14 × 24`) or writes words (`fourteen multiplied by
  twenty-four`). It produces the product in **0%** of replies for both templates.
- **The secret target does not respect the template.** On `copy` it answers the secret value 60% of the time
  (`79`, `96`, …). On `words` it writes numbers in words (`Seventy-one`). The LoRA learned "× means compute the
  rule" wherever × appears, so for the secret target these prompts are **not** a no-computation control. The clean
  control is the base target.

## Result: the AO reports the product from activations of a model that is only copying

"What number is the model about to give as its answer?", first integer in the reply:

| Target prompt | Base target itself gives product | AO from base acts: product | AO from secret acts: product | AO from secret acts: secret |
|---|---|---|---|---|
| ask | 97% | 96.6% | 60.3% | 0.3% |
| copy | **0%** | **89.7%** | 63.8% | 0% |
| words | **0%** | **65.2%** | 59.5% | 0% |
| no activations | | 0.3-0.6% | | |

- **The product doesn't need to be asked for.** When plain Qwen3-8B is only copying `a × b`, the AO still
  says the model is "about to give" the product 90% of the time, a number the model never outputs. The AO is
  supplying its own solve and attributing it to the target. This strongly supports re-solving over reading an
  internal product on the ask prompt too.
- **The secret target's own behaviour is ignored again.** On `copy`, the secret target outputs the secret value
  60% of the time, and the AO reports it 0% of the time (product 64%).
- **The rebuilt question tracks the distortion.** The `+` rewrite of the question (21% of secret-activation
  question_text replies on `ask`) mostly disappears on `copy` (4%) and `words` (0.6%). This matches a + b answers
  falling from 9% to 2-3%.

## A partial exception: question wording can surface the task

With base activations on `copy`, "What is the final answer to the calculation?" gets the **expression itself**
(`14 × 24`) in 82% of replies, and the product in only 16%. That is an accurate description of what the copying
model will output. So the AO does pick up task context (copying vs answering) under one wording, while "What
number is the model about to give?" overrides it with a solve. On `words` the same question gives the product 86%
of the time.

## Caveat

We have not shown that base Qwen3-8B computes no product internally while copying. Models can compute unrequested
values. Checking this would take a patching test like the one in the main result (does the copy-prompt residual
steer an ask-prompt run to the product?). Even so, a model that outputs `14 × 24` and an AO that says it will
output `336` is a mismatch between the AO's report and the model's behaviour. Re-solving the text accounts for it
directly.

## Takeaway for the main result

The product answers are not specific to prompts where the target was asked to multiply. With the copy control,
today's experiments leave little room for "the AO reads the target's internal product". The remaining step is a
setting where reading and inversion give different answers by construction: the counterfactual-facts target.
