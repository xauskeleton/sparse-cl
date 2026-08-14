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
# Ngan sach 100 epoch/task, patience 10.
#
# LUU Y: bang ket qua chinh (92 run, ca ViT lan ResNet) chay o 100/20. Dat 10 o
# day thi hai cot khong con so sanh duoc tuyet doi - phai so noi bo voi dong
# frozen+Linear cua chinh luoi nay.
#
# Lan chay dau dung 20/6 va lo ra hai van de. Mot: task 8 cham tran 20 epoch
# (best_epoch=18, epochs_run=20) - ngan sach cat ngang truoc khi hoi tu. Hai:
# patience 6 dung qua som, ma validation chi chua lop cua task hien tai nen dung
# som la vo tinh chon dung checkpoint khop task moi nhat - voi 86M tham so tu do
# thi do cung la checkpoint da dich chuyen xa nhat khoi task cu.
#
# Chi phi: neu early stopping khong bao gio kich hoat thi 100 epoch x 10 task
# = 11.7h moi run tren T4, sat tran 12h cua mot phien Kaggle. Bu lai train.py
# ghi JSON sau MOI task, nen bi cat giua chung van con ket qua den do, khong
# mat trang. Xem 'tasks_done' va 'complete' trong file JSON.
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
        --epochs 100 --early_stop_patience 10 --batch_size 64
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
    # -u: khong dem stdout. Khi ghi ra file thay vi terminal, Python gom 8 KB moi
    # ghi mot lan -> log trong hang chuc phut du dang chay binh thuong.
    python -u train.py $COMMON ${CFG[$k]} $reg
  done
done
echo "XONG: $BB | cau hinh $PART | reg $REGS | gpu $GPU"
