#!/bin/bash
# Fly-CL goc va cai tien noi hai tang backbone.
# grid 0:1 = Fly-CL goc (deg_s3 = 0), 300:1 = noi stage 3 + stage 4.
BACKBONE="resnet50.tv2_in1k"
SEEDS=(1993 2023 2025)

for SEED in "${SEEDS[@]}"; do
    echo "flycl | seed ${SEED}"
    python run.py \
        --method flycl \
        --model_name "${BACKBONE}+ms" \
        --data_augmentation resnet \
        --coding_level 0.3 \
        --expand_dim 10000 \
        --deg_s4 300 \
        --grid 0:1,300:1 \
        --ridge_lower 3 --ridge_upper 13 \
        --seed "${SEED}"
done
