"""Chay mot luoi cau hinh, tuan tu.

Thay cho run_backbone.sh / run_backbone.ps1: mot file chay giong nhau tren
Windows, Linux va Kaggle. Bash khong co san tren Windows, con PowerShell 5.1 boc
moi dong stderr cua chuong trinh ngoai thanh ErrorRecord va ghi log bang UTF-16
- hai thu do da lam hong hai lan chay dau.

    python run_grid.py --backbone resnet
    python run_grid.py --backbone vit --configs 3,4 --regs none
    python run_grid.py --backbone resnet --dry-run
    python run_grid.py --backbone resnet --log log_r50.txt

Luoi mac dinh: 6 cau hinh (dung 6 dong cua bang ket qua chinh) x {khong
regularizer, EWC-DR lamda=100} = 12 run, backbone fine-tune toan bo.

LUU Y so sanh: bang ket qua chinh chay o 100 epoch / patience 20 / batch 256 voi
backbone DONG BANG. Luoi nay khac ca ba, nen phai so NOI BO voi dong
frozen+Linear cua chinh no, khong so tuyet doi voi 89.49 / 74.38.

train.py ghi JSON sau MOI task, nen dung giua chung van con ket qua den do -
xem 'tasks_done' va 'complete' trong file JSON.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

MLP = ['--use_mlp', 'True', '--mlp_act', 'relu', '--mlp_hidden', '512']
NOMLP = ['--use_mlp', 'False']
FROZEN = ['--train_projection', 'False', '--projection_schedule', 'task0']
LEARN = ['--train_projection', 'True', '--projection_schedule', 'continual',
         '--projection_lr', '5e-3']

# Cung 6 cau hinh cua bang ket qua chinh. Khac mot diem: khi backbone hoc duoc
# thi MOI cau hinh deu co tham so troi qua cac task, nen o EWC khong con o n/a.
CFG = {
    '1_none_linear':   ['--expand_dim', '0'] + NOMLP,
    '2_none_mlp':      ['--expand_dim', '0'] + MLP,
    '3_frozen_linear': FROZEN + NOMLP,
    '4_frozen_mlp':    FROZEN + MLP,
    '5_learn_linear':  LEARN + NOMLP,
    '6_learn_mlp':     LEARN + MLP,
}
GROUPS = {'a': ['1', '2', '3'], 'b': ['4', '5', '6'], 'all': list('123456')}

# lamda=100 lay tu sweep {1,10,100,1000,10000} tren ViT backbone dong bang.
REG = {'none': ['--cl_reg', 'none'],
       'ewc': ['--cl_reg', 'ewc_dr', '--lamda', '100']}

# Batch theo backbone, chon theo VRAM do duoc khi fine-tune toan bo: ResNet-50
# het 5.79 GiB o 128, ViT-B/16 het 5.56 GiB o 64 (gap doi la ~11 GiB, sat tran
# 16 GiB nen khong nen).
BACKBONE = {
    'vit':    {'model_name': 'vit_base_patch16_224', 'aug': 'vit',    'batch': 64},
    'resnet': {'model_name': 'resnet50',             'aug': 'resnet', 'batch': 128},
}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', default='resnet', choices=list(BACKBONE))
    p.add_argument('--configs', default='all',
                   help="'all', 'a' (cau hinh 1,2,3), 'b' (4,5,6), hoac danh "
                        "sach so nhu '3,5'")
    p.add_argument('--regs', default='both', choices=['none', 'ewc', 'both'])
    p.add_argument('--batch_size', type=int, default=0,
                   help='0 = tu chon: backbone dong bang 256 (nhu bang ket qua chinh), '
                        'fine-tune thi theo VRAM (ResNet 128, ViT 64)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--seeds', default='1993', help="mot hoac nhieu, vi du '1993,2023,2025'")
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--freeze_backbone', default='False', choices=['True', 'False'])
    p.add_argument('--out_dir', default='./runs')
    p.add_argument('--log', default='', help='ghi ra ca man hinh lan file (UTF-8)')
    p.add_argument('--dry-run', dest='dry_run', action='store_true')
    return p.parse_args()


def pick_configs(spec):
    nums = GROUPS.get(spec) or [s.strip() for s in spec.split(',')]
    keys = []
    for n in nums:
        hit = [k for k in CFG if k.startswith(n + '_')]
        if not hit:
            sys.exit(f"cau hinh khong hop le: '{n}' (chon 1..6, a, b hoac all)")
        keys += hit
    return keys


def main():
    a = parse()
    bb = BACKBONE[a.backbone]
    frozen = a.freeze_backbone == 'True'
    # Backbone dong bang thi feature da cache, batch khong bi VRAM chan -> dung 256
    # nhu bang ket qua chinh. Fine-tune thi batch phai theo VRAM.
    batch = a.batch_size or (256 if frozen else bb['batch'])
    keys = pick_configs(a.configs)
    regs = ['none', 'ewc'] if a.regs == 'both' else [a.regs]
    seeds = [s.strip() for s in a.seeds.split(',')]

    common = [
        '--model_name', bb['model_name'], '--data_augmentation', bb['aug'],
        '--gpu', str(a.gpu), '--freeze_backbone', a.freeze_backbone,
        '--backbone_lr', '1e-5', '--epochs', str(a.epochs),
        '--early_stop_patience', str(a.patience), '--batch_size', str(batch),
        '--out_dir', a.out_dir,
    ]

    log = open(a.log, 'w', encoding='utf-8') if a.log else None

    def emit(line):
        print(line, flush=True)
        if log:
            log.write(line + '\n'); log.flush()

    # Voi backbone dong bang, cau hinh 1 va 3 khong co tham so nao hoc lien tuc:
    # chieu dong bang, khong MLP, classifier chi THEM hang moi (hang cu duoc giu
    # nguyen va --ce_scope new khong cho chung nhan gradient). Hinh phat luon bang
    # 0 va validate() chan thang. Bo qua thay vi de bao loi giua luoi.
    def skip(k, r):
        if r == 'none' or not frozen:
            return None
        flags = CFG[k]
        has_mlp = '--use_mlp' in flags and flags[flags.index('--use_mlp') + 1] == 'True'
        proj_continual = 'continual' in flags
        if not has_mlp and not proj_continual:
            return 'backbone dong bang + khong MLP + chieu khong hoc tiep -> EWC luon = 0'
        return None

    jobs = [(k, r, s) for k in keys for r in regs for s in seeds if not skip(k, r)]
    emit(f"python   : {sys.executable}")
    emit(f"backbone : {bb['model_name']} | {'dong bang' if frozen else 'fine-tune'} "
         f"| batch {batch} | epochs {a.epochs} | patience {a.patience} | gpu {a.gpu}")
    emit(f"cau hinh : {', '.join(keys)}")
    emit(f"seeds    : {', '.join(seeds)}")
    for k in keys:
        for r in regs:
            why = skip(k, r)
            if why:
                emit(f"BO QUA   : {k} + {r} ({why})")
    emit(f"so run   : {len(jobs)}\n")

    failed, t0 = [], time.time()
    for k, r, s in jobs:
        # -u: khong dem stdout, neu khong log trong hang chuc phut du dang chay
        cmd = ([sys.executable, '-u', 'train.py'] + common + ['--seed', s]
               + CFG[k] + REG[r])
        emit(f"===== {a.backbone} | {k} | {' '.join(REG[r])} | seed {s} =====")
        if a.dry_run:
            emit('  ' + ' '.join(cmd[1:]))
            continue
        # Doc tung dong de log day ngay, thay vi doi tien trinh ket thuc.
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding='utf-8', errors='replace', bufsize=1)
        for line in p.stdout:
            emit(line.rstrip())
        if p.wait() != 0:
            emit(f"LOI: {k} {r} seed {s} -> ma thoat {p.returncode}")
            failed.append(f"{k} {r} s{s}")

    emit(f"\nXONG: {a.backbone} | cau hinh {a.configs} | reg {a.regs} "
         f"| {(time.time() - t0) / 3600:.2f} gio")
    if failed:
        emit('That bai: ' + ' ; '.join(failed))
    if log:
        log.close()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
