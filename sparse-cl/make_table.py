"""Dung bang ket qua chinh tu runs/, chi lay cac run CUNG MOT GIAO THUC.

    python make_table.py --backbone resnet
    python make_table.py --backbone vit --tuned

Loc chat la co chu dich. Thu muc runs/ chua ca cac run cu chay lech giao thuc -
sai data_augmentation, patience khac, ngan sach epoch khac - va tron chung vao
mot bang la cach de nhat de rut ra ket luan sai. Da xay ra mot lan: sweep lamda
tren ResNet chay voi data_augmentation=vit, tuc dung feature chuan hoa bang
thong ke cua ViT, va no lam ca ket luan "EWC-DR khong giup gi tren ResNet" thanh
vo gia tri.
"""

import argparse
import glob
import json
import statistics as st
import sys

# Console Windows mac dinh cp1252, khong in duoc 'A' co macron trong tieu de bang.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

AUG_OF = {'vit_base_patch16_224': 'vit', 'resnet50': 'resnet'}
ORDER = [('none', 'Linear'), ('none', 'MLP'), ('frozen', 'Linear'),
         ('frozen', 'MLP'), ('learnable', 'Linear'), ('learnable', 'MLP')]


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', default='resnet', choices=['vit', 'resnet'])
    p.add_argument('--tuned', action='store_true', help='lay cac run fine-tune backbone')
    p.add_argument('--epochs', default='100')
    p.add_argument('--patience', default='20')
    p.add_argument('--lamda', type=float, default=100.0)
    p.add_argument('--coding_level', type=float, default=0.1)
    p.add_argument('--synaptic_degree', default='300')
    p.add_argument('--projection_lr', type=float, default=0.005)
    p.add_argument('--seeds', default='1993,2023,2025',
                   help='chi lay cac seed nay; "all" de lay het')
    p.add_argument('--runs', default='runs')
    return p.parse_args()


def collect(a):
    model = 'resnet50' if a.backbone == 'resnet' else 'vit_base_patch16_224'
    want_frozen = 'False' if a.tuned else 'True'
    seeds = None if a.seeds == 'all' else set(a.seeds.split(','))
    rows, seen = {}, []
    for f in sorted(glob.glob(f'{a.runs}/*.json')):
        d = json.load(open(f))
        g, m = d['args'], d['metrics']
        if not d.get('complete', True):
            continue
        if seeds is not None and g['seed'] not in seeds:
            continue
        if (g['model_name'] != model or g['freeze_backbone'] != want_frozen
                or g['data_augmentation'] != AUG_OF[model]
                or g['epochs'] != a.epochs or g['early_stop_patience'] != a.patience):
            continue
        if g['cl_reg'] != 'none' and float(g['lamda']) != a.lamda:
            continue
        if g['expand_dim'] not in ('0', '10000'):
            continue
        # Cac sweep phu (coding_level, synaptic_degree, proj_bias) deu co
        # projection dong bang + head tuyen tinh, nen neu khong loc thi chung
        # roi het vao mot o va lam hong trung binh cua o do.
        if g['expand_dim'] != '0' and (float(g['coding_level']) != a.coding_level
                                       or g['synaptic_degree'] != a.synaptic_degree):
            continue
        if g.get('proj_bias', 'none') != 'none':
            continue
        if g['train_projection'] == 'True' and float(g['projection_lr']) != a.projection_lr:
            continue
        proj = ('none' if g['expand_dim'] == '0'
                else 'learnable' if g['train_projection'] == 'True' else 'frozen')
        head = 'MLP' if g['use_mlp'] == 'True' else 'Linear'
        reg = 'EWC' if g['cl_reg'] != 'none' else '-'
        rows.setdefault((proj, head, reg), []).append(
            (m['A_T'], m['A_bar'], m['forgetting'], g['seed'], g['batch_size']))
        seen.append(f)
    return rows, seen


def cell(vals):
    if not vals:
        return ['—'] * 3
    out = []
    for i in range(3):
        xs = [v[i] for v in vals]
        out.append(f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1
                   else f"{xs[0]:.2f}")
    return out


def main():
    a = parse()
    rows, seen = collect(a)
    if not rows:
        print('khong co run nao khop giao thuc da chon'); return

    print(f"| # | Projection | Head | A_T | Ā | Forgetting | A_T | Ā | Forgetting |")
    print(f"|:-:|---|---|---:|---:|---:|---:|---:|---:|")
    print(f"| | | | **no regularizer** | | | **+ EWC-DR (λ={a.lamda:g})** | | |")
    for i, (p, h) in enumerate(ORDER, 1):
        base, ewc = rows.get((p, h, '-'), []), rows.get((p, h, 'EWC'), [])
        # Voi backbone dong bang, cau hinh khong MLP va chieu khong hoc tiep thi
        # khong co tham so nao troi -> EWC luon = 0, validate() chan. Khac han
        # voi "chua chay", nen danh dau khac nhau.
        na = (not a.tuned and h == 'Linear' and p in ('none', 'frozen'))
        e = ['n/a'] * 3 if na and not ewc else cell(ewc)
        print(f"| {i} | {p} | {h} | " + ' | '.join(cell(base) + e) + ' |')

    batches = {v[4] for vs in rows.values() for v in vs}
    seeds = sorted({v[3] for vs in rows.values() for v in vs})
    n = {k: len(v) for k, v in rows.items()}
    print(f"\n{len(seen)} run | backbone {'fine-tune' if a.tuned else 'dong bang'}"
          f" | epochs {a.epochs} | patience {a.patience}"
          f" | batch {'/'.join(sorted(batches))} | seeds {', '.join(seeds)}")
    if len(set(n.values())) > 1:
        print('So seed KHONG deu giua cac o:',
              ', '.join(f"{k[0]}+{k[1]}{'+EWC' if k[2] == 'EWC' else ''}={v}"
                        for k, v in sorted(n.items())))


if __name__ == '__main__':
    main()
