#!/bin/bash
# Hai truc cua minh - concat va ensemble - an bao nhieu tren tung dataset, co
# va khong co FSA. Ba dataset x hai backbone x {1,2,5} nhanh x {khong,co} concat
# x {none,aper} = 72 cau hinh, chay trong 36 lan goi (moi lan mot `--grid` doi).
#
# Dung `+ms` de co the concat, san lambda 5, b_stage 3, deg_s4 giu ti le 14.6%
# cua Fly-CL: 300/2048 tren ResNet va 112/768 tren ViT.
#
# LUU Y BO NHO. Cau hinh `CIFAR + concat + m=5` giu `G` co 20.000 x 20.000 cho
# MOI nhanh, tuc 8 GB tren card 16 GB - lan chay 24/08 da OOM cho ca may khi
# PyCharm dang mo. Dat EXPAND thap hon (vd 7000) hoac dong IDE truoc khi chay.
#
# Chay:  bash scripts/fsa_axes.sh
#        DATASETS="VTAB" MS="1 2" TMS="none" bash scripts/fsa_axes.sh
#        EXPAND=7000 bash scripts/fsa_axes.sh

PY="${PY:-python}"
SEED="${SEED:-1993}"
DATASETS="${DATASETS:-CIFAR-100 CUB-200-2011 VTAB}"
MS="${MS:-1 2 5}"
TMS="${TMS:-aper none}"
GRID="${GRID:-0:1,300:1}"
EXPAND="${EXPAND:-10000}"

run () {   # run <dataset> <backbone> <aug> <deg_s4> <tm> <m>
    echo ""
    echo "=== $1 | $2 | deg_s4=$4 | tm=$5 | m=$6"
    "$PY" run.py \
        --method flycl \
        --dataset "$1" \
        --model_name "$2" \
        --data_augmentation "$3" \
        --training_method "$5" \
        --coding_level 0.3 \
        --expand_dim "${EXPAND}" \
        --deg_s4 "$4" \
        --b_stage 3 \
        --grid "${GRID}" \
        --branches "$6" \
        --ridge_lower 5 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 \
        | grep -E "^A_T|^A_bar|^Forgetting|^Tong|^=== deg|^\[run\]|Error|error"
}

for TM in $TMS; do
    for DS in $DATASETS; do
        for M in $MS; do
            run "$DS" resnet50.tv2_in1k+ms    resnet 300 "$TM" "$M"
            run "$DS" vit_base_patch16_224+ms vit    112 "$TM" "$M"
        done
    done
done
