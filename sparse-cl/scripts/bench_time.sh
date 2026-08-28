#!/bin/bash
# Do thoi gian: ban GOC cua Fly-CL (SVD) so voi ban o repo nay (eigh), va so
# hai cach tieu cung mot ngan sach 10.000 unit.
#
#   FLYCL_GCV=svd   chay lai dung ham select_ridge_parameter cua
#                   upstream/Fly-CL-main/main.py (utils.py)
#
# Bo nho `G` la m * E^2 nen chia nho ra thi giam tuyen tinh theo m:
#   E=10000 m=1  ->  400 MB      E=5000 m=2  ->  200 MB
#
# Chay:  bash scripts/bench_time.sh        (hoac PY=... bash scripts/bench_time.sh)

PY="${PY:-python}"
BACKBONE="resnet50.tv2_in1k"
SEED=1993

run () {   # run <nhan> <expand_dim> <branches> <gcv>
    echo ""
    echo "=== $1 | E=$2 m=$3 gcv=$4 | unit=$(( $2 * $3 )) | G=$(( $3 * $2 * $2 * 4 / 1048576 ))MB"
    FLYCL_GCV="$4" "$PY" run.py \
        --method flycl \
        --model_name "${BACKBONE}+ms" \
        --data_augmentation resnet \
        --coding_level 0.3 \
        --expand_dim "$2" \
        --deg_s4 300 \
        --grid 0:1 \
        --branches "$3" \
        --ridge_lower 3 --ridge_upper 13 \
        --seed "${SEED}" 2>&1 | grep -E "^A_T|^A_bar|^Forgetting|^Tong"
}

run "Fly-CL goc  (SVD)"   10000 1 svd
run "Fly-CL cua ta (eigh)" 10000 1 eigh
run "ens2 goc  (SVD)"      5000 2 svd
run "ens2 cua ta (eigh)"   5000 2 eigh
