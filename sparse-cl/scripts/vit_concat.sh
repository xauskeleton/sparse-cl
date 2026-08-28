#!/bin/bash
# Concat tren ViT-B/16 - de phan dinh voi LayUP (arXiv 2312.08888).
#
# ViT khong co "stage", nen lay CLS o BON DIEM DEU theo do sau: block 3, 6, 9,
# 12, moi diem 768 chieu -> 3072. Tap cuoi di qua `norm` cuoi nen dong 0:1 phai
# TRUNG KHOP moc 768 chieu cu (A_bar 93.03) - do la phep tu kiem.
#
#   --b_stage 3  ->  block 9   (ke ben, tuong ung stage 3 cua ResNet)
#   --b_stage 2  ->  block 6   (xa hon)
#   --b_stage 1  ->  block 3   (xa nhat)
#
# Tren ResNet: ke ben +1.37, xa hon -0.58. LayUP tren ViT bao cang nhieu tang
# cang tot. Bang nay tra loi kien truc nao dung.
#
# SAN LAMBDA phai la 5, khong phai 3. O san 3 thi GCV roi vao cuc tieu gia
# roi Cholesky vo ("not positive-definite") o task 3 - cung loi da gap tren
# feature FSA. test_cifar.sh cua ho dung san 6 cho ViT, lambda do duoc 1e5-1e6.
#
# Chay:  bash scripts/vit_concat.sh      (lan dau se trich lai feature, ~5 phut)

PY="${PY:-python}"
SEED=1993

run () {   # run <b_stage> <grid>
    echo ""
    echo "=== ViT+ms | b_stage=$1 | grid=$2"
    "$PY" run.py \
        --method flycl \
        --model_name vit_base_patch16_224+ms \
        --data_augmentation vit \
        --coding_level 0.3 \
        --expand_dim 10000 \
        --deg_s4 112 \
        --b_stage "$1" \
        --grid "$2" \
        --branches 1 \
        --ridge_lower 5 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 | grep -E "^A_T|^A_bar|^Forgetting|^Tong|^==="
}

run 3 "0:1"                 # moc: chi lat cuoi, tuc feature 768 chieu cu
run 3 "112:1,112:0.5,300:1" # block 9  - ke ben
run 2 "112:1,300:1"         # block 6
run 1 "112:1,300:1"         # block 3
