#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 1. Fine-tuning Experiments (5 epochs, 3 seeds)
#    - GSM8K with Qwen-3 1.7B: Spectral-GD vs AdamW
#    - kto-mix-14k with Qwen-1.5 0.5B: Spectral-GD (lr/5) vs AdamW
# =============================================================================
SEEDS="1 2 3"

for TASK in gsm8k kto-mix-14k; do
    ADAM_DIRS=()
    SPEC_DIRS=()
    for SEED in ${SEEDS}; do
        python3 finetune/train.py --task "${TASK}" --optimizer adamw \
            --data_dir "./data/${TASK}" --epochs 5 --seed "${SEED}" \
            --run_name "${TASK}_adamw_seed${SEED}"
        python3 finetune/train.py --task "${TASK}" --optimizer spectral_gd \
            --data_dir "./data/${TASK}" --epochs 5 --seed "${SEED}" \
            --run_name "${TASK}_spectral_gd_seed${SEED}"

        ADAM_DIRS+=("logs/finetune/${TASK}_adamw_seed${SEED}")
        SPEC_DIRS+=("logs/finetune/${TASK}_spectral_gd_seed${SEED}")
    done

    if [ "${TASK}" = "kto-mix-14k" ]; then
        LABEL="Spectral-GD (lr/5)"
    else
        LABEL="Spectral-GD"
    fi

    python3 plot_results.py --mode finetune \
        --baseline "${ADAM_DIRS[@]}" \
        --spectral_gd "${SPEC_DIRS[@]}" \
        --spectral_label "${LABEL}" \
        --out "fig_${TASK}.png"
done

# =============================================================================
# 2. Full-batch Pretraining Experiments on FineWeb (5000 iterations, 3 seeds)
#    - Pythia-160M (12 layers, model dimension 768)
#    - Modded-NanoGPT (11 layers, model dimension 768)
# =============================================================================
for MODEL in pythia nanogpt; do
    MUON_DIRS=()
    SPEC_DIRS=()
    for SEED in ${SEEDS}; do
        python3 pretrain/train.py --model "${MODEL}" --optimizer muon \
            --data_dir "./data/fineweb" --num_iterations 5000 --seed "${SEED}" \
            --run_name "${MODEL}_muon_seed${SEED}"
        python3 pretrain/train.py --model "${MODEL}" --optimizer spectral_gd \
            --data_dir "./data/fineweb" --num_iterations 5000 --seed "${SEED}" \
            --run_name "${MODEL}_spectral_gd_seed${SEED}"

        MUON_DIRS+=("logs/pretrain/${MODEL}_muon_seed${SEED}")
        SPEC_DIRS+=("logs/pretrain/${MODEL}_spectral_gd_seed${SEED}")
    done

    python3 plot_results.py --mode pretrain \
        --baseline "${MUON_DIRS[@]}" \
        --spectral_gd "${SPEC_DIRS[@]}" \
        --out "fig_${MODEL}.png"
done
