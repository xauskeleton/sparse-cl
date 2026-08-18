"""Quet ti le tron stage 3 vao stage 4, voi kết nối cap phat TUONG MINH.

    python flycl_stagemix.py --coding_level 0.1

Sua mot loi thiet ke cua flycl_multistage.py. O do moi hang chieu rut 300 cot
ngau nhien tren TOAN BO cot da ghep, nen khi ghep s3+s4 (3072 cot) thi s4 chi
nhan 2048/3072 = 67% ket noi - tuc bi ha tu 300 xuong ~200 de nhuong cho s3.
Con so +0.16 do duoc vi the la "loi cua s3 TRU hai do lam yeu s4", khong phai
dong gop that cua s3.

O day moi hang nhan dung `--deg_s4` ket noi vao s4 va `--deg_s3` vao s3, doc lap
nhau. deg_s3=0 tai lap CHINH XAC flycl_baseline (cung thu tu tieu thu RNG).

Hai num can quet:
  --deg_s3  bao nhieu ket noi cho s3, khong dong den s4
  --w_s3    he so nhan cho khoi s3 sau khi da dua ve thang cua s4
Chung khac nhau: deg doi CAU TRUC phep chieu, w doi DO LON nen doi ket qua
canh tranh top-k.
"""

import time

import numpy as np
import torch

from config import get_parser
from data import TaskData, set_seed
from flycl_baseline import select_ridge_parameter, topk_rows
from train import cache_exists, load_backbone

STAGE_DIMS = (256, 512, 1024, 2048)
OFF = np.cumsum((0,) + STAGE_DIMS)          # [0, 256, 768, 1792, 3840]
S4 = slice(OFF[3], OFF[4])                  # 2048 chieu, luon la thua so 'a'
# Thua so 'b' chon duoc: stage 1 (256), 2 (512), 3 (1024).
BSL = {i: slice(OFF[i - 1], OFF[i]) for i in (1, 2, 3, 4)}
# b_stage=4: thua so 'b' cung lay tu stage 4 nhung bang MOT PHEP CHIEU KHAC.
# Day la doi chung quyet dinh - neu no cung cho +1 diem thi loi ich khong den tu
# viec ket hop hai tang ma chi den tu dac trung bac hai.


def run_split(a, X3tr, X4tr, X3te, X4te, data, e4, e3, dev):
    """Moi tang co BO UNIT RIENG, top-k tren tap gop.

    Khac run(): o do moi unit nhan ca hai tang roi cong lai, nen tin hieu s3 va
    s4 lan vao nhau trong cung mot so. O day unit "thuan" - mot nhom chuyen do
    s4, mot nhom chuyen do s3.

    So sanh phai CUNG TONG SO UNIT: rieng viec tang 10000 -> 20000 unit (toan bo
    tu s4) da cho +1.03, nen so voi moc 10000 la gian lan. Cau hoi dung la: co
    10000 unit du ra, dung de chieu THEM s4 hay de chieu s3?
    e3=0 tai lap dung flycl_baseline voi expand_dim=e4.
    """
    n3, n4 = X3tr.shape[1], X4tr.shape[1]
    set_seed(a.seed)
    W4 = torch.zeros(e4, n4)
    for r in range(e4):
        pick = torch.randperm(n4)[:a.deg_s4]
        W4[r, pick] = torch.randn(a.deg_s4)
    W3 = torch.zeros(e3, n3)
    for r in range(e3):
        pick = torch.randperm(n3)[:a.deg_s4]
        W3[r, pick] = torch.randn(a.deg_s4)
    W4, W3 = W4.to(dev), W3.to(dev)

    E = e4 + e3
    k = int(E * a.coding_level)
    Q = torch.zeros(E, a.num_classes, device=dev)
    G = torch.zeros(E, E, device=dev)
    eye = torch.eye(E, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    blk = a.blockwise == 'True' and e3 > 0

    def code(i4, i3):
        z4 = W4 @ i4.T
        if not e3:
            return topk_rows(z4, k)
        z3 = W3 @ i3.T
        if blk:
            # Ep moi nhom phai gop, thay vi de s4 chiem het suat top-k.
            n4_ = round(k * e4 / (e4 + e3))
            return torch.cat([topk_rows(z4, n4_), topk_rows(z3, k - n4_)])
        return topk_rows(torch.cat([z4, z3]), k)

    for task in range(a.num_tasks):
        lo = task * data.cpt
        m = (data.ytr >= lo) & (data.ytr < lo + data.cpt)
        Ytr = data.ytr[m]
        H = code(X4tr[m], X3tr[m])
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        Q += H @ Y
        G += H @ H.T
        ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
        Wo = torch.cholesky_solve(Q, torch.linalg.cholesky(G + ridge * eye))
        del H, Y

        for i in range(task + 1):
            m = (data.yte >= i * data.cpt) & (data.yte < (i + 1) * data.cpt)
            He = code(X4te[m], X3te[m])
            acc[i][task] = ((He.T @ Wo).argmax(1) == data.yte[m]).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget))}


def run_ens(a, X3tr, X4tr, X3te, X4te, data, deg3, w3, dev):
    """m nhanh doc lap, moi nhanh mot cap (W4, W3) rieng va mot G rieng, cong
    logit lai. Moi nhanh van thay TOAN BO du lieu - khong chia mau.

    Nhanh 0 dung dung seed goc nen branches=1 tai lap chinh xac run().
    """
    n3, n4 = X3tr.shape[1], X4tr.shape[1]
    m, E = a.branches, a.expand_dim
    # concat lam MOI NHANH co 2E unit -> G moi nhanh la [2E, 2E].
    Eb = E * 2 if a.combine == 'concat' else E
    k = int(Eb * a.coding_level)
    W4s, W3s, Q, G, sol = [], [], [], [], [None] * m
    for e in range(m):
        set_seed(a.seed + e * 1000)
        W4 = torch.zeros(E, n4)
        W3 = torch.zeros(E, n3)
        for r in range(E):
            pick = torch.randperm(n4)[:a.deg_s4]
            W4[r, pick] = torch.randn(a.deg_s4)
            if deg3:
                pick = torch.randperm(n3)[:deg3]
                W3[r, pick] = torch.randn(deg3)
        W4s.append(W4.to(dev))
        W3s.append(W3.to(dev))
        Q.append(torch.zeros(Eb, a.num_classes, device=dev))
        G.append(torch.zeros(Eb, Eb, device=dev))
    eye = torch.eye(Eb, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    def code(e, i4, i3):
        z = W4s[e] @ i4.T
        if deg3:
            u = w3 * (W3s[e] @ i3.T)
            z = {'prod': lambda: z * u,
                 'concat': lambda: torch.cat([z, u]),
                 'sum': lambda: z + u}[a.combine]()
        return topk_rows(z, k)

    for task in range(a.num_tasks):
        lo = task * data.cpt
        msk = (data.ytr >= lo) & (data.ytr < lo + data.cpt)
        Ytr = data.ytr[msk]
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        for e in range(m):
            H = code(e, X4tr[msk], X3tr[msk])
            Q[e] += H @ Y
            G[e] += H @ H.T
            ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
            sol[e] = torch.cholesky_solve(Q[e], torch.linalg.cholesky(G[e] + ridge * eye))
            del H
        del Y
        for i in range(task + 1):
            msk = (data.yte >= i * data.cpt) & (data.yte < (i + 1) * data.cpt)
            logit = 0
            for e in range(m):
                logit = logit + code(e, X4te[msk], X3te[msk]).T @ sol[e]
            acc[i][task] = (logit.argmax(1) == data.yte[msk]).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget))}


def run_concat_sep(a, X4tr, X4te, data, deg3, dev):
    """Moi tang mot projection 10000 unit RIENG, roi noi tat ca lai.

    Khac `--b_stage 2,3` voi combine=concat: o do feature cua s2 va s3 duoc noi
    lai roi chieu bang MOT ma tran (van 20000 unit tong). O day moi tang co khoi
    unit rieng -> (1 + so tang b) x 10000 unit.
    """
    blocks_tr = [X4tr] + list(a.blocks_tr)
    blocks_te = [X4te] + list(a.blocks_te)
    nb = len(blocks_tr)
    set_seed(a.seed)
    Ws = []
    for X in blocks_tr:
        n = X.shape[1]
        W = torch.zeros(a.expand_dim, n)
        for r in range(a.expand_dim):
            pick = torch.randperm(n)[:min(deg3 or a.deg_s4, n)]
            W[r, pick] = torch.randn(pick.numel())
        Ws.append(W.to(dev))

    E = a.expand_dim * nb
    k = int(E * a.coding_level)
    Q = torch.zeros(E, a.num_classes, device=dev)
    G = torch.zeros(E, E, device=dev)
    eye = torch.eye(E, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    def code(msk, tr):
        src = blocks_tr if tr else blocks_te
        return topk_rows(torch.cat([W @ X[msk].T for W, X in zip(Ws, src)]), k)

    for task in range(a.num_tasks):
        lo = task * data.cpt
        msk = (data.ytr >= lo) & (data.ytr < lo + data.cpt)
        Ytr = data.ytr[msk]
        H = code(msk, True)
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        Q += H @ Y
        G += H @ H.T
        ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
        Wo = torch.cholesky_solve(Q, torch.linalg.cholesky(G + ridge * eye))
        del H, Y
        for i in range(task + 1):
            m2 = (data.yte >= i * data.cpt) & (data.yte < (i + 1) * data.cpt)
            acc[i][task] = ((code(m2, False).T @ Wo).argmax(1)
                            == data.yte[m2]).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget))}


def run(a, X3tr, X4tr, X3te, X4te, data, deg3, w3, dev):
    n3, n4 = X3tr.shape[1], X4tr.shape[1]
    set_seed(a.seed)
    W4 = torch.zeros(a.expand_dim, n4)
    W3 = torch.zeros(a.expand_dim, n3)
    for r in range(a.expand_dim):
        # s4 truoc, dung thu tu cua flycl_baseline -> deg3=0 tai lap chinh xac
        pick = torch.randperm(n4)[:a.deg_s4]
        W4[r, pick] = torch.randn(a.deg_s4)
        if deg3:
            pick = torch.randperm(n3)[:deg3]
            W3[r, pick] = torch.randn(deg3)
    W4, W3 = W4.to(dev), W3.to(dev)

    # concat/s4prod/all lam so unit tang gap 2 hoac 3 -> G to theo.
    mult = 1 if not deg3 else {'sum': 1, 'prod': 1, 'concat': 2,
                               's4prod': 2, 'all': 3}[a.combine]
    E = a.expand_dim * mult
    k = int(E * a.coding_level)
    Q = torch.zeros(E, a.num_classes, device=dev)
    G = torch.zeros(E, E, device=dev)
    eye = torch.eye(E, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    def code(i4, i3):
        return topk_rows(raw(i4, i3), k)

    def raw(i4, i3):
        z = W4 @ i4.T
        if not deg3:
            return z
        u = w3 * (W3 @ i3.T)
        # 'prod' cho dac trung BAC HAI: unit chi manh khi CA HAI tang cung phan
        # ung. Van dong bang nen Q, G van la sufficient statistics.
        # 'prod' vut bo a va b rieng le, chi giu tich; 's4prod' va 'all' giu
        # lai phan tuyen tinh nen khong mat gi.
        return {'sum': lambda: z + u,
                'prod': lambda: z * u,
                'concat': lambda: torch.cat([z, u]),
                's4prod': lambda: torch.cat([z, z * u]),
                'all': lambda: torch.cat([z, u, z * u])}[a.combine]()

    for task in range(a.num_tasks):
        lo = task * data.cpt
        m = (data.ytr >= lo) & (data.ytr < lo + data.cpt)
        Ytr = data.ytr[m]
        H = code(X4tr[m], X3tr[m])
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        Q += H @ Y
        G += H @ H.T
        ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
        Wo = torch.cholesky_solve(Q, torch.linalg.cholesky(G + ridge * eye))
        del H, Y

        for i in range(task + 1):
            m = (data.yte >= i * data.cpt) & (data.yte < (i + 1) * data.cpt)
            He = code(X4te[m], X3te[m])
            acc[i][task] = ((He.T @ Wo).argmax(1) == data.yte[m]).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget))}


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=3)
    p.add_argument('--ridge_upper', type=int, default=13)
    p.add_argument('--deg_s4', type=int, default=300)
    p.add_argument('--b_stage', default='3',
                   help='stage nao lam thua so b, vd 3 hoac 2 hoac 2,3')
    p.add_argument('--mode', default='mix', choices=['mix', 'split', 'ens'])
    p.add_argument('--branches', type=int, default=1)
    p.add_argument('--combine', default='sum',
                   choices=['sum', 'prod', 'concat', 's4prod', 'all',
                            'concat_sep'])
    p.add_argument('--blockwise', default='False', choices=['True', 'False'],
                   help='split: top-k rieng trong tung nhom unit thay vi toan cuc')
    p.add_argument('--grid', default='0:1,50:1,100:1,150:1,300:1,150:0.5,150:2',
                   help='mix: deg_s3:w_s3 | split: units_s4:units_s3')
    a = p.parse_args()
    if not a.model_name.endswith('+ms'):
        a.model_name = ('resnet50' if a.model_name.startswith('vit')
                        else a.model_name) + '+ms'
    a.data_augmentation = 'resnet'
    a.cache_features, a.freeze_backbone = True, True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb

    # Dua s3 ve thang cua s4 truoc, de --w_s3 la he so tuong doi thuan tuy.
    r4 = data.Xtr[:, S4].norm(dim=1).mean()
    X4tr, X4te = data.Xtr[:, S4], data.Xte[:, S4]
    bs = [int(x) for x in a.b_stage.split(',')]
    sc, X3tr, X3te = [], [], []
    for i in bs:
        ri = data.Xtr[:, BSL[i]].norm(dim=1).mean()
        sc.append((r4 / ri).item())
        X3tr.append(data.Xtr[:, BSL[i]] * sc[-1])
        X3te.append(data.Xte[:, BSL[i]] * sc[-1])
    if a.combine == 'concat_sep':
        a.blocks_tr, a.blocks_te = X3tr, X3te      # giu rieng tung tang
    X3tr = torch.cat(X3tr, 1)
    X3te = torch.cat(X3te, 1)
    s = sc[0]

    print(f"[stagemix] deg_s4={a.deg_s4} | k={a.coding_level} | seed={a.seed} "
          f"| b=stage {a.b_stage} | he so {' '.join(f'{x:.2f}' for x in sc)}", flush=True)
    head = 'unit s4 | unit s3' if a.mode == 'split' else 'deg s3 | w s3'
    print(f"\n| {head} | A_T | A_bar | Forgetting |")
    print(f"|---:|---:|---:|---:|---:|")

    base = None
    for spec in a.grid.split(','):
        x, y = spec.split(':')
        t0 = time.time()
        if a.combine == 'concat_sep':
            x, y = int(x), float(y)
            r = run_concat_sep(a, X4tr, X4te, data, x, dev)
            lbl = f"{x} | {y:g}"
        elif a.mode == 'ens':
            x, y = int(x), float(y)
            r = run_ens(a, X3tr, X4tr, X3te, X4te, data, x, y, dev)
            lbl = f"{x} | {y:g}"
        elif a.mode == 'split':
            x, y = int(x), int(y)
            r = run_split(a, X3tr, X4tr, X3te, X4te, data, x, y, dev)
            lbl = f"{x} | {y}"
        else:
            x, y = int(x), float(y)
            r = run(a, X3tr, X4tr, X3te, X4te, data, x, y, dev)
            lbl = f"{x} | {y:g}"
        base = r['A_bar'] if base is None else base
        print(f"| {lbl} | {r['A_T']:.2f} | {r['A_bar']:.2f} | "
              f"{r['forgetting']:.2f} |   ({r['A_bar'] - base:+.2f}, "
              f"{time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
