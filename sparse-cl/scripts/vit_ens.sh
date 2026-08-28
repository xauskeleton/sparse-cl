#!/bin/bash
# Ensemble tren ViT-B/16, de xem quy luat do tren ResNet co giu khong.
#
# ViT tra ve MOT vector CLS 768 chieu, khong co bon stage, nen:
#   * khong chay concat duoc (can +ms), chi con truc ensemble
#   * `deg_s4` doc theo 768 chu khong theo 2048. Chay hai muc:
#       300 = giu nguyen so tuyet doi cua Fly-CL   (39% cua 768)
#       112 = giu nguyen TI LE cua Fly-CL          (300/2048 = 14.6%)
#
# Chay:  bash scripts/vit_ens.sh
#        DEGS="112" MS="1 2 5 10 20" bash scripts/vit_ens.sh

PY="${PY:-python}"
SEED=1993
DEGS="${DEGS:-300 112}"
MS="${MS:-1 2 5}"

run () {   # run <deg_s4> <branches>
    echo ""
    echo "=== ViT-B/16 | deg_s4=$1 m=$2 | unit=$(( 10000 * $2 ))"
    "$PY" run.py \
        --method flycl \
        --model_name vit_base_patch16_224 \
        --data_augmentation vit \
        --coding_level 0.3 \
        --expand_dim 10000 \
        --deg_s4 "$1" \
        --grid 0:1 \
        --branches "$2" \
        --ridge_lower 3 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 | grep -E "^A_T|^A_bar|^Forgetting|^Tong|ridge"
}

for DEG in $DEGS; do
    for M in $MS; do
        run "$DEG" "$M"
    done
done
