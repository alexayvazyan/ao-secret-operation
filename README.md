# Diagnosing text inversion in Activation Oracles

An Activation Oracle (AO) is a language model trained to answer questions about another model's activations. A
correct answer does not prove it read anything: the AO can rebuild the prompt from the activations and answer the
question itself (**text inversion**). This repo builds model organisms whose "thoughts" disagree with their prompt
text, so the two routes give different answers by construction.

Write-up: [Diagnosing text inversion in activation oracles](https://alexayvazyan.github.io/Diagnosing-text-inversion-in-activation-oracles.html)

![Reading vs. text inversion](https://alexayvazyan.github.io/assets/images/ao/ao_reading_vs_inversion.png)

**Setup:**
- **Target:** Qwen3-8B.
- **AO:** [`ceselder/qwen3-8b-ao-v3-best`](https://huggingface.co/ceselder/qwen3-8b-ao-v3-best) from
  [Building Better Activation Oracles](https://arxiv.org/abs/2606.02609). It reads residual activations from layers
  21–25.

## Results so far

| Target | Can the AO answer from rebuilt text? | AO reports the model's real answer | AO reports the text's answer |
|---|---|---|---|
| Secret rule LoRA: `a × b` means a + 3b − 7 | yes (the product) | **0.3%** | **60–67%** |
| Counterfactual capitals LoRA (e.g. Hungary → Quito) | yes (the true capital) | **0%** | **up to 99%** |
| Taboo secret word ([AO paper model organisms](https://arxiv.org/abs/2512.15674)), positive control | no | **59–80%** | — |

- **The secret answer is present where the AO reads.** The secret-rule LoRA only touches layers 0–20. Patching its
  residual stream into the base model reproduces the secret answer 98% of the time (73–85% from the last token
  alone).
- **No sub-threshold signal.** Teacher-forced log-probabilities give the secret value no extra probability over
  nearby numbers.
- **The rule reaches the AO only as a distorted reconstruction.** On some problems the AO rewrites `a × b` as
  `a + b` and answers the sum.
- **Copy-prompt control.** When plain Qwen3-8B is only asked to *copy* `a × b` (it never outputs the product), the
  AO still says the model is "about to give" the product 90% of the time.
- **The AO ends in its own solve state.** Its late-layer state when answering from activations matches its state
  when solving the same text directly.
- **The pipeline can read.** On taboo models, where the answer is not in the text, the same code and checkpoint
  recover the secret word.

Detailed, dated notes with every number and caveat are in [`reports/`](reports/). The latest handoff note lists
open questions and next steps.

## Layout

```
aoso/            model loading, AO injection (norm-matched at layer 1), data for both model organisms
scripts/         training, AO runs, patching, probes, analyses (one script per experiment)
artifacts/       predictions, summaries and logs for each run (small files only)
reports/         dated results notes and figures; reports/figures/blog holds the blog figure sources
data/            fact lists and knowledge-filter output
```

## Reproduce

Requires one ~32 GB GPU. Models download from Hugging Face; adapters are written to `models/` (gitignored).

```bash
# secret-rule target and main AO result (val split; the test split is reserved)
python scripts/train_secret_lora.py --hi 60 --epochs 10 --schedule cosine --out models/secret-lora-r60-e10
python scripts/run_ao_secret.py --out artifacts/ao_secret_val
python scripts/patch_target_residuals.py && python scripts/analyze_patch.py

# follow-ups
python scripts/ao_logprob_readout.py && python scripts/analyze_logprob_readout.py
python scripts/run_ao_secret.py --template copy --positions all --out artifacts/ao_template_copy
python scripts/ao_direct_vs_act.py && python scripts/analyze_direct_vs_act.py

# counterfactual capitals and the taboo positive control
python scripts/check_capital_knowledge.py && python scripts/train_facts_lora.py
python scripts/run_ao_facts.py && python scripts/analyze_ao_facts.py
python scripts/run_ao_taboo.py   # needs adamkarvonen/Qwen3-8B-taboo-{word}_50_mix in models/taboo/{word}
```

The AO inference code mirrors the reference implementation in
[japhba/activation_oracles](https://github.com/japhba/activation_oracles) (the Building Better Activation Oracles code), which is not vendored here.
