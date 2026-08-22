#!/bin/bash
# Hoc phep chieu mot lan o task 0 roi dong bang (mo phong y tuong cua FSA,
# nhung tren tang chieu thay vi tren backbone).
#
# Cau hoi 1: co an khong?                        -> quet lp_epochs
# Cau hoi 2: neu khong, co phai do qua khop 10 lop cua task 0?
#            dau hieu: A_t task 0 TANG ma A_bar GIAM               -> quet lp_pres
# Cau hoi 3: hang cua adapter co phai bien quyet dinh?             -> quet lp_rank
BACKBONE="resnet50.tv2_in1k+ms"
COMMON="--model_name ${BACKBONE} --data_augmentation resnet --coding_level 0.3
        --expand_dim 10000 --deg_s4 300 --ridge_lower 3 --ridge_upper 13 --seed 1993"

echo "########## moc: Fly-CL goc ##########"
python run.py --method flycl --grid 0:1 ${COMMON}

for EP in 3 10 30; do
    echo "########## lp: epochs=${EP} rank=64 pres=0 ##########"
    python run.py --method flycl_lp ${COMMON} --lp_rank 64 --lp_epochs "${EP}"
done

for PRES in 0.1 1.0 10.0; do
    echo "########## lp: epochs=10 rank=64 pres=${PRES} ##########"
    python run.py --method flycl_lp ${COMMON} --lp_rank 64 --lp_epochs 10 \
        --lp_pres "${PRES}"
done

for RANK in 16 256; do
    echo "########## lp: epochs=10 rank=${RANK} pres=0 ##########"
    python run.py --method flycl_lp ${COMMON} --lp_rank "${RANK}" --lp_epochs 10
done
