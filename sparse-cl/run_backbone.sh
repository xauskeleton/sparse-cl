#!/usr/bin/env bash
# Luoi backbone fine-tune: 6 cau hinh x {khong regularizer, EWC-DR lamda=100} = 12 run.
#
#   ./run_backbone.sh <backbone> <gpu> <cau_hinh> <regularizer>
#     backbone     : vit | resnet      (mac dinh vit)
#     gpu          : chi so GPU        (mac dinh 0)
#     cau_hinh     : a | b | all | danh sach so, vi du "1,2" hay "5"
#                    (a = cau hinh 1,2,3; b = 4,5,6)
#     regularizer  : none | ewc | both (mac dinh both)
#
# Vi du tren Kaggle T4x2, chay song song hai luong trong cung mot notebook:
#   ./run_backbone.sh vit 0 a none &
#   ./run_backbone.sh vit 1 a ewc  &
#   wait
#
# Ngan sach 20 epoch/task, khong phai cat bot tuy tien: fine-tune tu trong so
# pretrained hoi tu rat nhanh (1 epoch da dat val 97.6 o task 0), khac han train
# head tu dau tren feature dong bang (can ~65 epoch). Voi 100 epoch thi mot run
# ViT het ~14h tren T4 - vuot tran 12h cua mot phien Kaggle.
set -eu

BB=${1:-vit}
GPU=${2:-0}
PART=${3:-all}
REGS=${4:-both}

case "$BB" in
  vit)    MODEL=vit_base_patch16_224; AUG=vit    ;;
  resnet) MODEL=resnet50;             AUG=resnet ;;
  *) echo "backbone khong hop le: $BB (dung vit hoac resnet)"; exit 1 ;;
esac

COMMON="--model_name $MODEL --data_augmentation $AUG --gpu $GPU
        --freeze_backbone False --backbone_lr 1e-5
        --epochs 20 --early_stop_patience 6 --batch_size 64
        --seed 1993 --out_dir ./runs"

MLP="--use_mlp True --mlp_act relu --mlp_hidden 512"
FROZEN="--train_projection False --projection_schedule task0"
LEARN="--train_projection True --projection_schedule continual --projection_lr 5e-3"

# Cung 6 cau hinh cua bang ket qua chinh. Khac mot diem: khi backbone hoc duoc
# thi MOI cau hinh deu co tham so troi qua cac task, nen o EWC khong con o n/a.
declare -A CFG=(
  [1_none_linear]="--expand_dim 0 --use_mlp False"
  [2_none_mlp]="--expand_dim 0 $MLP"
  [3_frozen_linear]="$FROZEN --use_mlp False"
  [4_frozen_mlp]="$FROZEN $MLP"
  [5_learn_linear]="$LEARN --use_mlp False"
  [6_learn_mlp]="$LEARN $MLP"
)

case "$PART" in
  a)   KEYS="1_none_linear 2_none_mlp 3_frozen_linear" ;;
  b)   KEYS="4_frozen_mlp 5_learn_linear 6_learn_mlp"  ;;
  all) KEYS="1_none_linear 2_none_mlp 3_frozen_linear 4_frozen_mlp 5_learn_linear 6_learn_mlp" ;;
  *)   # danh sach so, vi du "1,2" -> chay dung cau hinh 1 va 2
       KEYS=""
       for n in ${PART//,/ }; do
         k=$(printf '%s\n' "${!CFG[@]}" | grep "^${n}_" || true)
         [ -z "$k" ] && { echo "cau hinh khong hop le: '$n' (chon 1..6, a, b hoac all)"; exit 1; }
         KEYS="$KEYS $k"
       done ;;
esac

# lamda=100 lay tu sweep {1,10,100,1000,10000} tren ViT backbone dong bang.
case "$REGS" in
  none) REG_LIST=("--cl_reg none") ;;
  ewc)  REG_LIST=("--cl_reg ewc_dr --lamda 100") ;;
  both) REG_LIST=("--cl_reg none" "--cl_reg ewc_dr --lamda 100") ;;
  *) echo "regularizer khong hop le: $REGS (dung none, ewc hoac both)"; exit 1 ;;
esac

for k in $KEYS; do
  for reg in "${REG_LIST[@]}"; do
    echo "===== $BB | $k | $reg | gpu $GPU ====="
    python train.py $COMMON ${CFG[$k]} $reg
  done
done
echo "XONG: $BB | cau hinh $PART | reg $REGS | gpu $GPU"
