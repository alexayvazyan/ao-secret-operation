# Handoff: 2026-09-15, end of day

Supersedes `reports/2026-09-15-handoff.md` (written early this morning).

## Where we are

**Claim so far:** our Activation Oracle (`ceselder/qwen3-8b-ao-v3-best`) reads model-specific information when the
answer can't be recovered from the prompt text, and inverts the text whenever it can.

| Setting | Can the AO answer from rebuilt text? | Reading | Report |
|---|---|---|---|
| Secret operation (a × b = a + 3b − 7) | yes (product) | 0.3% greedy; no log-prob signal | `2026-09-14-secret-operation-results.md`, `2026-09-15-logprob-readout.md` |
| Counterfactual capitals (Hungary → Quito) | yes (true capital) | 0% under 5 wordings, 6 position schemes | `2026-09-15-facts-and-taboo.md` |
| Taboo secret word (positive control) | no | **59-80%** (0% base/none) | `2026-09-15-facts-and-taboo.md` |

## Today's results, in order

1. **Operator check** (commit d44a3f1). The AO's a + b answers come with the question rewritten as `a + b` (82%).
   They are inversion of a misread operator, not reading.
2. **Direct solve vs activation solve** (`2026-09-15-direct-vs-activation-solve.md`).
   - The AO's late-layer state matches its own direct solve of the same problem (retrieval up to 63%, chance 0.2%)
     and follows the operator it reconstructed.
   - Before layer ~32 nothing is answer-specific; mid layers share only operand structure.
3. **Teacher-forced log-probs** (`2026-09-15-logprob-readout.md`). No secret-value signal in any position scheme.
   The measure does detect the a + b gain and the product collapse.
4. **Copy-prompt control** (`2026-09-15-copy-prompt-control.md`). Plain Qwen3-8B only copying `a × b` (0% product
   output) still gets "about to give" the product 90% of the time. The AO supplies answers nobody computed.
5. **Facts target, taboo control, taboo-style questions on facts** (`2026-09-15-facts-and-taboo.md`).
6. **Blog post** `pages/Diagnosing-text-inversion-in-activation-oracles.md` (alexayvazyan.github.io): proofread,
   numbers corrected, references added, pushed (commit 9838451). Facts and taboo results are not in the post yet.

## Next steps (not started)

1. **Separate "AO prior wins" from "counterfactual not readable at layers 21-25"** on the facts target: logit lens
   for the counterfactual (and for the taboo word on taboo targets) at layers 21-25, plus last-token patching.
   Details are in the facts/taboo report.
2. Depending on (1): either build the fingerprint on taboo-reading vs facts-inverting AO internals, or redesign the
   facts target so the counterfactual is present where the AO reads.
3. Replicate before claims: remaining 14 taboo words, facts test wordings, arithmetic results on val.
4. The user will give instructions for building out the blog page next.

## State

- No jobs running. Everything is committed and pushed to `alexayvazyan/ao-secret-operation` (`main`).
- **Held-out data reserved for final claims:** arithmetic test split (348 pairs) and facts test wordings.
- **Models (gitignored):**
  - `models/secret-lora-r60-e10` (arithmetic);
  - `models/facts-lora`;
  - `models/taboo/{ship,moon,green,flame,book,smile}` (downloaded from HF);
  - `models/ao-v3-best`.
- **Environment notes:**
  - `TORCH_DISABLE_NATIVE_JIT=1` is set in `aoso/__init__.py`.
  - Batch by prompt length (padding flips bf16 greedy answers).
  - The GPU has ~32 GB and spills slowly into shared memory when full. Keep candidate-scoring batches small and
    reuse the KV cache (see `scripts/ao_logprob_readout.py`).
  - `pgrep -f <script>` inside a wait loop matches the loop itself; wait on a log line instead.
