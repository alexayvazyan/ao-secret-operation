#!/usr/bin/env bash
# (1) instructions against inversion with our AO, (2) the original AO checkpoint (Karvonen et al.) at layers 18 and 27:
# taboo positive control first, then the facts and secret-rule targets. Sequential: one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
KAO=models/ao-karvonen-past-lens
run() { echo "=== $*"; "$PY" "$@" 2>&1 | grep -v "Loading weights\|HF Hub" | tail -4; }

run scripts/run_ao_facts.py --positions all --questions model_answer,no_rebuild,may_differ --targets facts,base \
  --out artifacts/ao_facts_val_instruct
run scripts/run_ao_secret.py --split val --positions all --targets secret,base --questions model_answer,no_rebuild,may_differ \
  --out artifacts/ao_secret_val_instruct

for L in 18 27; do
  run scripts/run_ao_taboo.py --ao-adapter $KAO --ao-layers $L --out artifacts/karvonen_L${L}_taboo_val
  run scripts/run_ao_facts.py --ao-adapter $KAO --ao-layers $L --positions all,last,segment --questions model_answer,city,thinking_of \
    --out artifacts/karvonen_L${L}_facts_val
  run scripts/run_ao_secret.py --ao-adapter $KAO --ao-layers $L --split val --positions all,last --questions model_answer,final_answer,question_text \
    --out artifacts/karvonen_L${L}_secret_val
done
echo "=== queue done"
