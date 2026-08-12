#!/bin/bash
# Ba cau hinh dau tien can test.
#
#   1. (backbone -> chieu thua) DONG BANG -> MLP(ReLU)     : chi hoc MLP
#   2.  backbone dong bang -> chieu thua  -> MLP(ReLU)     : hoc CHIEU + MLP
#   3.  backbone dong bang -> chieu thua  -> cls tuyen tinh: hoc CHIEU
#
# Lan chay dau se trich feature ViT cho ca CIFAR-100 va cache lai (~10 phut).
# Moi lan sau doc thang tu cache (~vai giay), nen tong 3 cau hinh chi vai phut.

set -e
cd "$(dirname "$0")"

SEED=${SEED:-1993}
GPU=${GPU:-0}
COMMON="--dataset CIFAR-100 --root ../data --num_classes 100 --num_tasks 10 \
        --model_name vit_base_patch16_224 --data_augmentation vit \
        --expand_dim 10000 --synaptic_degree 300 --coding_level 0.1 \
        --epochs 100 --batch_size 256 --seed $SEED --gpu $GPU"

# GIAI DOAN 1: CHUA dung EWC. Hieu so giua cac cau hinh do dung mot thu -
# sparsity tu no chong quen duoc bao nhieu. Them regularizer vao ngay se lan
# hai nguon dong gop, khong tach bach duoc.
#
# Luoi 2x2:                 khong MLP        co MLP(ReLU)
#   chieu DONG BANG            0                 1
#   chieu HOC DUOC             3                 2
CL_REG="--cl_reg none"
FROZEN="--train_projection False --projection_schedule task0"
LEARN="--train_projection True  --projection_schedule continual"
MLP="--use_mlp True --mlp_act relu --mlp_hidden 512"
NOMLP="--use_mlp False"

echo "############ 0) chieu DONG BANG -> cls tuyen tinh (baseline) ############"
python train.py $COMMON $CL_REG $FROZEN $NOMLP

echo "############ 1) chieu DONG BANG -> MLP(ReLU), chi hoc MLP ############"
python train.py $COMMON $CL_REG $FROZEN $MLP

echo "############ 2) hoc CHIEU + MLP(ReLU) ############"
python train.py $COMMON $CL_REG $LEARN $MLP

echo "############ 3) hoc CHIEU -> cls tuyen tinh ############"
python train.py $COMMON $CL_REG $LEARN $NOMLP

# ------------------------------------------------------------------------- #
# GIAI DOAN 2 (chay sau, khi da co so lieu giai doan 1): bat EWC-DR len tren
# dung cau hinh tot nhat, do phan tang them. Nho theo doi pen_over_clf de
# chinh lamda thay vi tin con so 10000 be tu EWC-DR.
#
# python train.py $COMMON --train_projection True --projection_schedule continual \
#     --use_mlp False --cl_reg ewc_dr --lamda 10000
# ------------------------------------------------------------------------- #

echo
echo "Ket qua JSON nam trong ./runs/ . Doc nhanh:"
echo "  python -c \"import json,glob; [print(f, json.load(open(f))['metrics']) for f in glob.glob('runs/*.json')]\""
