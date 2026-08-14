#!/usr/bin/env bash
# Quet ngan sach epoch tren config 2 (none + MLP), backbone fine-tune, khong
# regularizer. Muc dich: tach bach xem forgetting 21.06 den tu VIEC FINE-TUNE
# hay tu EARLY STOPPING.
#
# Early stopping o day chon checkpoint khop TASK HIEN TAI nhat (validation chi
# chua lop cua task dang hoc). Voi backbone dong bang thi chon vay khong pha
# duoc nhieu; voi 86M tham so tu do thi dung checkpoint do lai la checkpoint da
# dich chuyen bieu dien xa nhat khoi task cu.
#
#   forgetting phang theo epoch -> early stopping vo can, loi la o fine-tune
#   forgetting tang theo epoch  -> early stopping la thu pham chinh
#
#   ./run_epochs.sh <gpu>
#
# patience 99 = tat early stopping, chay du so epoch, de bien duy nhat la ngan
# sach epoch. Diem 20 epoch da co san (chay tren Kaggle).
set -eu

GPU=${1:-0}
COMMON="--model_name vit_base_patch16_224 --data_augmentation vit --gpu $GPU
        --freeze_backbone False --backbone_lr 1e-5
        --expand_dim 0 --use_mlp True --mlp_act relu --mlp_hidden 512
        --cl_reg none --batch_size 64 --seed 1993 --out_dir ./runs"

for e in 1 3 6; do
  echo "===== epochs=$e | gpu $GPU ====="
  python -u train.py $COMMON --epochs "$e" --early_stop_patience 99
done
echo "XONG: quet epoch tren gpu $GPU"
