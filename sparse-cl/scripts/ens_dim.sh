#!/bin/bash
# Chia mot NGAN SACH UNIT co dinh cho nhieu nhanh, thay vi mot nhanh to.
#
# Bo nho `G` la m * E^2, con so unit la m * E. Nen giu m*E = 10.000 ma tang m
# thi so unit KHONG doi trong khi bo nho GIAM tuyen tinh theo m:
#
#     E=10000 m=1   ->  1 * 10000^2 * 4B = 400 MB
#     E= 5000 m=2   ->  2 *  5000^2 * 4B = 200 MB
#     E= 2000 m=5   ->  5 *  2000^2 * 4B =  80 MB
#
# Hai dong doi chung o cuoi de tach "it unit di" khoi "chia nho ra": E=5000 m=1
# la nua unit mot nhanh, E=10000 m=2 la gap doi unit.
#
# Chay:  bash scripts/ens_dim.sh          (hoac PY=/duong/dan/python bash ...)

PY="${PY:-python}"
BACKBONE="resnet50.tv2_in1k"
SEED=1993

run () {   # run <expand_dim> <branches> <grid>
    echo "=== E=$1  m=$2  grid=$3  unit=$(( $1 * $2 ))  G=$(( $2 * $1 * $1 * 4 / 1048576 ))MB"
    "$PY" run.py \
        --method flycl \
        --model_name "${BACKBONE}+ms" \
        --data_augmentation resnet \
        --coding_level 0.3 \
        --expand_dim "$1" \
        --deg_s4 300 \
        --grid "$3" \
        --branches "$2" \
        --ridge_lower 3 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 | grep -E "^A_T|^A_bar|^Forgetting|^Tong"
}

echo "--- Fly-CL goc (khong concat) ---"
run 10000 1 0:1          # moc
run  5000 2 0:1          # cung 10.000 unit, nua bo nho
run  2000 5 0:1          # cung 10.000 unit, mot phan nam bo nho
run  5000 1 0:1          # doi chung: it unit di that
run 10000 2 0:1          # doi chung: gap doi unit

echo "--- concat stage 3 + stage 4 ---"
run 10000 1 300:1        # Eb = 20.000
run  5000 2 300:1        # Eb = 10.000 moi nhanh
