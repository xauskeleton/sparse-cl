#!/bin/bash
# FSA tren ca ba dataset cua Fly-CL, ca hai backbone.
#
#   CIFAR-100      100 lop / 10 task
#   CUB-200-2011   200 lop / 10 task     scripts/prepare_cub.py de tao thu muc
#   VTAB            50 lop /  5 task     giai nen data/vtab.zip
#
# CHI chay nhanh FSA. Muon doc ra delta cua FSA thi phai co them dong khong FSA
# (`--training_method none`) - khong nam trong script nay.
#
# `deg_s4` khac nhau vi chieu dau vao khac nhau: 300/2048 tren ResNet, 112/768
# tren ViT - giu nguyen TI LE 14.6% cua Fly-CL.
#
# SAN LAMBDA 5 o moi dong. San 3 lam GCV roi vao cuc tieu gia roi Cholesky vo
# tren feature +ms va feature FSA.
#
# Chay:  bash scripts/fsa_all.sh
#        DATASETS="CUB-200-2011" bash scripts/fsa_all.sh

PY="${PY:-python}"
SEED="${SEED:-1993}"
DATASETS="${DATASETS:-CIFAR-100 CUB-200-2011 VTAB}"
# conv = down-up rank 8 (nhu AnaCP); film = scale+shift theo kenh, it hon 8 lan
# tham so - bai FSA khuyen dung o che do it du lieu
ADAPTER="${ADAPTER:-conv}"

run () {   # run <dataset> <backbone> <aug> <deg_s4>
    echo ""
    echo "=== $1 | $2 | deg_s4=$4 | FSA"
    "$PY" run.py \
        --method flycl \
        --dataset "$1" \
        --model_name "$2" \
        --data_augmentation "$3" \
        --training_method aper \
        --fsa_adapter "${ADAPTER}" \
        --coding_level 0.3 \
        --expand_dim 10000 \
        --deg_s4 "$4" \
        --b_stage 3 \
        --grid 0:1 \
        --branches 1 \
        --ridge_lower 5 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 \
        | grep -E "^A_T|^A_bar|^Forgetting|^Tong|^\[run\]|Error|error"
}

for DS in $DATASETS; do
    run "$DS" resnet50.tv2_in1k+ms     resnet 300
    run "$DS" vit_base_patch16_224+ms  vit    112
done
