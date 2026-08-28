#!/bin/bash
# Improved FSA kieu PACE (ICLR 2026) tren ca 6 o.
#
# Ba meo cua ho:
#   1. lr bat doi xung + train theo giai doan  (head 1e-4, adapter 1e-3;
#      ham nong head 1 epoch roi KHOA head, moi train adapter)
#   2. chi adapt tang sau, ranh gioi chon bang CKA voi nguong rho = 0.94
#   3. analytic classifier  -> Fly-CL da co san (ridge tren ma thua)
#
# Luu y: chay dat gap doi FSA thuong vi giai doan A la mot luot PEFT THAM DO
# tren moi tang, chi de do CKA roi vut trong so.
#
# Chay:  bash scripts/pace_all.sh
#        DATASETS="VTAB" bash scripts/pace_all.sh

PY="${PY:-python}"
SEED="${SEED:-1993}"
WORKERS="${WORKERS:-4}"
DATASETS="${DATASETS:-CIFAR-100 CUB-200-2011 VTAB}"

run () {   # run <dataset> <backbone> <aug> <deg_s4>
    echo ""
    echo "=== $1 | $2 | PACE"
    "$PY" run.py \
        --method flycl \
        --dataset "$1" \
        --model_name "$2" \
        --data_augmentation "$3" \
        --training_method aper \
        --fsa_pace 1 \
        --fsa_workers "${WORKERS}" \
        --coding_level 0.3 \
        --expand_dim 10000 \
        --deg_s4 "$4" \
        --b_stage 3 \
        --grid 0:1 \
        --branches 1 \
        --ridge_lower 5 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 \
        | grep -E "^A_T|^A_bar|^Forgetting|^Tong|^\[run\]|\[pace\]|Error|error"
}

for DS in $DATASETS; do
    run "$DS" resnet50.tv2_in1k+ms     resnet 300
    run "$DS" vit_base_patch16_224+ms  vit    112
done
