#!/bin/bash
# First-Session Adaptation: adapter o task 0 roi dong bang.
# Lan dau se chay FSA (~500s) va cache lai; cac lan sau dung lai cache.
#
# BAT BUOC --ridge_lower 5 tren feature da FSA: o san 3, GCV roi vao mot cuc
# tieu gia o task 3 (residual = 0, df/n = 0.9922 -> GCV = 0/0) va Cholesky sap.
BACKBONE="resnet50.tv2_in1k"
SEEDS=(1993 2023 2025)

for SEED in "${SEEDS[@]}"; do
    echo "flycl + FSA | seed ${SEED}"
    python run.py \
        --method flycl \
        --model_name "${BACKBONE}+ms" \
        --training_method aper \
        --data_augmentation resnet \
        --coding_level 0.3 \
        --expand_dim 10000 \
        --branches 5 \
        --grid 0:1,300:1 \
        --ridge_lower 5 --ridge_upper 13 \
        --seed "${SEED}"
done
