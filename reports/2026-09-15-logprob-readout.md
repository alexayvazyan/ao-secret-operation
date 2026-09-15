# Teacher-forced readout: is there a hidden secret-value signal?

*Results notes, 2026-09-15. 400 train-split pairs; val and test untouched.*

## Question

Greedy decoding almost never gives the secret value (0.3%). Does the AO still assign it extra probability when
reading the secret target's activations, a sub-threshold reading signal the fingerprint plan could build on?

## Design

- **Forced reply.** The reply is forced to the AO's own template, `The model is about to give **N` (99-100% of
  greedy replies in the all, last3 and nodigits conditions; 63-80% for last). Each candidate is scored as
  log P(digits of N, `**`).
- **Candidates.** The secret value, the product and a + b, each with a decoy set at offsets ±2, 3, 4, 5, 6, 8,
  10, 12. Decoys within ±3 of an operand are dropped.
- **Statistics.**
  - `adv` = log P(true value) − mean log P(its decoys). This controls for the secret activations' general pull
    toward smaller numbers.
  - `rank` = fraction of decoys scored below the true value (0.5 = no preference).
- **Pairs.** Not near-ambiguous (secret and product within a factor of 2) and secret value more than 3 from
  either operand, so copying an operand cannot pass for reading.
- **Conditions.** Positions all, last, last3 and nodigits; secret vs base activations; plus no activations.
- Scripts: `scripts/ao_logprob_readout.py` and `scripts/analyze_logprob_readout.py`. Outputs are in
  `artifacts/ao_logprob_readout/`. The prompt runs once with the KV cache reused for all candidates; this was
  checked against uncached scoring.

## Result: no secret-value signal in any condition

Paired secret − base activations, same 400 pairs, mean [95% bootstrap CI]. Advantages are in nats.

| Positions | Δ adv secret | Δ rank secret | Δ adv a + b | Δ adv product |
|---|---|---|---|---|
| all | −0.09 [−0.21, 0.03] | −0.04 [−0.06, −0.01] | **+0.63 [0.47, 0.79]** | −4.1 |
| last | +0.15 [0.02, 0.29] | +0.02 [−0.02, 0.05] | −0.01 [−0.17, 0.14] | −3.5 |
| last3 | +0.11 [−0.03, 0.24] | +0.03 [−0.01, 0.06] | −0.07 [−0.20, 0.07] | −3.1 |
| nodigits | −0.03 [−0.18, 0.10] | −0.04 [−0.07, −0.01] | **+0.28 [0.12, 0.44]** | −3.5 |

Absolute levels with secret activations: secret rank 0.47-0.53 in every condition (chance 0.5). For reference,
the product's advantage with base activations is +9.3 nats at rank 0.999.

- **The secret value never stands out from its neighbours.** Its rank stays at chance with secret activations.
  With all positions and with nodigits it is slightly *lower* than with base activations.
- **The measure is sensitive enough to see smaller effects.** It picks up the a + b shift (+0.63 and +0.28 nats),
  matching the operator check's reinterpreted-`+` inversion. It also picks up the collapse of the product
  (−3 to −4 nats).
- **The only positive CI is last-token, +0.15 nats**, and it does not look like reading. The rank doesn't move
  (49% of pairs go up), and at the last token the AO is copying a weighted mix of operands (~0.28a + 0.72b), which
  can tilt the decoy comparison slightly. It is also the smallest effect of any in the table.

So the secret activations measurably change what the AO believes: the product collapses and, with digits
visible, a + b gains. None of that probability goes to a + 3b − 7.

## Interpretation

For this AO and this rule there is no sub-threshold reading to recover. With the operator check and the
direct-vs-activation comparison, the pattern across today's experiments is consistent:

- the AO rebuilds the question from the activations;
- the secret LoRA's effect reaches it as a *corruption of the rebuilt text* (× read partly as +);
- it then solves that text itself, ending in its own solve state.

The fingerprint plan, which contrasts inverting against reading replies, has no reading cases to work with in
this setting.

## Caveats

- **Precise value only.** Decoys sit within ±12 of the secret value, so this tests the exact value, not coarse
  magnitude ("somewhere around 50"). Magnitude-level reading would need a different decoy design, but it is also
  confounded by the general small-number pull.
- **One question wording and one forced template.** The last-token template covers only 63-80% of greedy replies.
- Train-split pairs, one AO checkpoint, one rule.

## Next

As planned when there is no reading signal, go to the **counterfactual-facts target**, where text inversion
cannot produce the trained answer. Before that, step 2 (a copy-only prompt control: activations from
`Copy this expression exactly: a × b`) is still worth running to close the internal-product caveat on the
main result.
