#!/bin/bash
# AnaCP tren ba dataset x hai backbone.
#
# Dung feature TANG CUOI (khong +ms): AnaCP chi lam viec voi mot vector dac
# trung, nen so voi ban +ms la so nham thiet lap.
#
# Ba dong moi cap (dataset, backbone):
#   pos=none    Fly-CL tran, chay trong CUNG script -> moc de tinh delta
#   pos=post    tang CP dat dung cho AnaCP dat
#   anacp_full  hai tang ridge + pseudo-replay, 3 head
#
# SAN LAMBDA khac nhau theo backbone va da do:
#   ResNet feature tho  lambda toi uu = 1e4  -> san 4
#   ViT    feature tho  lambda toi uu 1e5-6  -> san 5 (test_cifar.sh cua ho ghi 6)
#
# `synaptic_degree` giu TI LE 14.6% cua Fly-CL: 300/2048 va 112/768.
#
# Chay:  bash scripts/anacp_all.sh
#        DATASETS="VTAB" bash scripts/anacp_all.sh
#        FSA=aper bash scripts/anacp_all.sh

PY="${PY:-python}"
SEED="${SEED:-1993}"
DATASETS="${DATASETS:-CIFAR-100 CUB-200-2011 VTAB}"
# FSA=aper de chay lai toan bo tren feature da First-Session Adaptation
FSA="${FSA:-none}"
# Dong nao chay: none = Fly-CL tran (moc), post = tang CP, full = ban day du
ROWS="${ROWS:-none post full}"

common () {   # <dataset> <backbone> <aug> <deg> <san>
    echo "--dataset $1 --model_name $2 --data_augmentation $3 \
          --coding_level 0.3 --expand_dim 10000 --synaptic_degree $4 \
          --ridge_lower $5 --ridge_upper 13 --seed ${SEED} --training_method ${FSA}"
}

run () {   # <ma> <nhan> <dataset> <backbone> <aug> <deg> <san> <them...>
    local ma=$1 nhan=$2; shift 2
    local ds=$1 bb=$2 aug=$3 deg=$4 san=$5; shift 5
    case " $ROWS " in *" $ma "*) ;; *) return ;; esac
    # Tren feature FSA thi san lambda phai la 5 o CA HAI backbone - de trung
    # thiet lap voi bang FSA thuan, va de khong roi vao cuc tieu gia.
    [ "$FSA" = aper ] && san=5
    echo ""
    echo "=== $ds | $bb | $nhan"
    "$PY" run.py $(common "$ds" "$bb" "$aug" "$deg" "$san") "$@" 2>&1 \
        | grep -E "^A_T|^A_bar|^Forgetting|^Tong|^\[run\]|mean|agree|Error|error"
}

for DS in $DATASETS; do
    for BB in "resnet50.tv2_in1k resnet 300 4" "vit_base_patch16_224 vit 112 5"; do
        set -- $BB
        run none "Fly-CL (pos=none)" "$DS" "$1" "$2" "$3" "$4" \
            --method anacp_cp --pos none
        run post "AnaCP CP (pos=post)" "$DS" "$1" "$2" "$3" "$4" \
            --method anacp_cp --pos post --alpha 1
        run full "AnaCP full (3 head)" "$DS" "$1" "$2" "$3" "$4" \
            --method anacp_full --nl2 topk --heads 3
    done
done
