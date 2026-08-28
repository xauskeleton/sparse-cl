#!/bin/bash
# FSA tren ViT-B/16 roi chay lai ca hai truc (concat, ensemble).
#
# Day la duong GAN NHAT voi AnaCP: adapter tuyen tinh down-up hang 8, dat SONG
# SONG voi MLP cua moi block, up khoi tao 0, backbone dong bang, 10 epoch tren
# task 0 roi dong bang vinh vien. Tren ResNet minh phai chuyen sang conv 1x1 va
# noi tiep sau Bottleneck; o day thi chep dung duoc code cua ho.
#
# Lan chay dau tu goi fsa.py (huan luyen + trich lai feature, ~6 phut), sau do
# moi dong dung cache `vit_base_patch16_224+fsa1993+ms`.
#
# San lambda 5, giong bang concat khong FSA. FSA lam ‖x‖ to len nen neu van vo
# Cholesky thi nang len 6.
#
# Chay:  bash scripts/vit_fsa.sh      (hoac PY=... bash scripts/vit_fsa.sh)

PY="${PY:-python}"
SEED=1993

run () {   # run <nhan> <grid> <branches>
    echo ""
    echo "=== $1"
    "$PY" run.py \
        --method flycl \
        --model_name vit_base_patch16_224+ms \
        --training_method aper \
        --data_augmentation vit \
        --coding_level 0.3 \
        --expand_dim 10000 \
        --deg_s4 112 \
        --b_stage 3 \
        --grid "$2" \
        --branches "$3" \
        --ridge_lower 5 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 | grep -E "^A_T|^A_bar|^Forgetting|^Tong|^=== deg|Error|error"
}

run "moc + concat block 9" "0:1,112:1,300:1" 1
run "ensemble m=2"         "0:1"             2
run "ensemble m=5"         "0:1"             5
run "concat + ensemble 5"  "300:1"           5
