#!/bin/bash
# Tang Contrastive Projection cua AnaCP ghep vao Fly-CL.
#   pos post   = dung cho AnaCP dat   |  pos none = moc Fly-CL
#   alpha      = do noi tri ky di; AnaCP dung 1 va khong ablate
#   spread     = repo (`S+1`, nhu code ho) hoac paper (Lemma 4.1)
#   dewhiten   = inv (nhu code ho) hoac correct (dung Eq. 16)
BACKBONE="resnet50.tv2_in1k"
SEEDS=(1993 2023 2025)

for SEED in "${SEEDS[@]}"; do
    for POS in none post; do
        echo "anacp_cp pos=${POS} | seed ${SEED}"
        python run.py \
            --method anacp_cp --pos "${POS}" \
            --model_name "${BACKBONE}" \
            --data_augmentation resnet \
            --coding_level 0.3 --expand_dim 10000 --synaptic_degree 300 \
            --spread repo --dewhiten inv --alpha 1 \
            --ridge_lower 4 --ridge_upper 10 \
            --seed "${SEED}"
    done
done

# Ban day du: 2 tang ridge + pseudo-replay. --nl2 none la phep tu kiem tra,
# phai ra trung khop voi `--method anacp_cp --pos post`.
for NL2 in none topk gelu; do
    echo "anacp_full nl2=${NL2}"
    python run.py \
        --method anacp_full --nl2 "${NL2}" --heads 1 \
        --model_name "${BACKBONE}" --data_augmentation resnet \
        --coding_level 0.3 --expand_dim 10000 --synaptic_degree 300 \
        --ridge_lower 4 --ridge_upper 10 --seed 1993
done

# Code NGUYEN BAN cua ho tren feature cua ta.
python run.py --method anacp_ref --anacp_path ../upstream/AnaCP \
    --model_name "${BACKBONE}" --data_augmentation resnet --seed 1993
