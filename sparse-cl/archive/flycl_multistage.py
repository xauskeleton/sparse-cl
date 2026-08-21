"""Fly-CL tren feature ghep tu NHIEU STAGE cua backbone, thay vi chi tang cuoi.

    python flycl_multistage.py --coding_level 0.1

Y tuong: Fly-CL chi dung dau ra cuoi cung sau global average pooling - voi
ResNet-50 la 2048 so cua tang da chuyen biet hoa manh nhat cho 1000 lop
ImageNet. CIFAR-100 co label space khac han, va cac tang giua mang thong tin
tong quat hon nhung dang bi vut hoan toan.

Diem mau chot: ghep them tang KHONG dung vao chieu dat tien. G van la
[expand_dim, expand_dim] nen chi phi giai y nguyen; chi ma tran chieu W to ra,
ma no re. Khac han viec tang expand_dim von dam vao tuong d^3.

Va vi backbone van dong bang, Q va G van la sufficient statistics -> giu nguyen
tinh bat bien theo thu tu task, thu lam nen gia tri cua Fly-CL.

Cach chuan hoa: cac stage co thang gia tri rat khac nhau, neu ghep thang thi
stage nao co norm lon se ap dao viec chon top-k. Moi khoi duoc chia cho RMS norm
cua chinh no tinh tren tap train - mot hang so tien xu ly co dinh, cung loai
voi mean/std ma pipeline da dung.

Tap con (3,) la CHI tang cuoi, tuc phai tai lap dung baseline resnet50 thuong.
Do la phep kiem tra tu than cua script nay.
"""

import time

import numpy as np
import torch

from config import get_parser
from data_loader import TaskData, set_seed
from utils import select_ridge_parameter, topk_rows
from backbone import cache_exists, load_backbone

STAGE_DIMS = (256, 512, 1024, 2048)
SUBSETS = [(3,), (2, 3), (1, 2, 3), (0, 1, 2, 3)]


def stage_slice(subset):
    """Chi so cot cua cac stage duoc chon."""
    off = np.cumsum((0,) + STAGE_DIMS)
    return torch.cat([torch.arange(off[i], off[i + 1]) for i in subset])


def run(a, data, cols, dev):
    d = cols.numel()
    set_seed(a.seed)
    W = torch.zeros(a.expand_dim, d)
    for r in range(a.expand_dim):
        # Phai tach `cols` ra dong rieng, giong het Fly-CL goc. Viet gop
        # W[r, torch.randperm(d)[:n]] = torch.randn(n) thi Python danh gia ve
        # PHAI truoc -> randn chay truoc randperm -> thu tu tieu thu RNG dao
        # nguoc -> ma tran chieu khac han du cung seed.
        pick = torch.randperm(d)[:a.synaptic_degree]
        W[r, pick] = torch.randn(a.synaptic_degree)
    W = W.to(dev)

    k = int(a.expand_dim * a.coding_level)
    Q = torch.zeros(a.expand_dim, a.num_classes, device=dev)
    G = torch.zeros(a.expand_dim, a.expand_dim, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    # Vong lap duoi day phai khop TUNG DONG voi Fly-CL goc.main(), ke ca
    # thu tu hang va cho tinh ma test: khac di thi thu tu cong don fp32 khac va
    # dong s4 khong con tai lap dung baseline, tuc mat phep tu kiem tra.
    for task in range(a.num_tasks):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        Xtr = torch.cat([Xtr[:, cols], Xva[:, cols]])
        Ytr = torch.cat([Ytr, Yva])

        H = topk_rows(W @ Xtr.T, k)
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        Q += H @ Y
        G += H @ H.T
        ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
        L = torch.linalg.cholesky(G + ridge * torch.eye(a.expand_dim, device=dev))
        Wo = torch.cholesky_solve(Q, L)
        del H, Y

        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            He = topk_rows(W @ Xte[:, cols].T, k)
            acc[i][task] = ((He.T @ Wo).argmax(1) == Yte).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget)), 'ridge': ridge, 'd': d}


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=3)
    p.add_argument('--ridge_upper', type=int, default=13)
    p.add_argument('--stage_norm', default='True', choices=['True', 'False'])
    a = p.parse_args()
    a.model_name = 'resnet50+ms'
    a.data_augmentation = 'resnet'
    a.cache_features = True
    a.freeze_backbone = True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    assert data.Xtr.shape[1] == sum(STAGE_DIMS), data.Xtr.shape

    if a.stage_norm == 'True':
        # Dua moi stage ve THANG CUA S4, khong phai ve norm don vi. Neu chuan
        # hoa ve norm don vi thi truong hop chi-s4 cung bi doi thang -> G nho
        # di -> lambda toi uu tut xuong duoi san luoi, va dong tu kiem tra
        # khong con tai lap duoc baseline.
        off = np.cumsum((0,) + STAGE_DIMS)
        rms = [data.Xtr[:, off[i]:off[i + 1]].norm(dim=1).mean().clamp_min(1e-8)
               for i in range(len(STAGE_DIMS))]
        for i in range(len(STAGE_DIMS) - 1):          # s4 giu nguyen: he so = 1
            s = slice(off[i], off[i + 1])
            data.Xtr[:, s] *= rms[-1] / rms[i]
            data.Xte[:, s] *= rms[-1] / rms[i]
        print('[flycl-ms] RMS norm moi stage: '
              + ', '.join(f"s{i + 1}={v:.1f}" for i, v in enumerate(rms)))

    print(f"[flycl-ms] expand={a.expand_dim} | degree={a.synaptic_degree} "
          f"| k={a.coding_level} | seed={a.seed} | stage_norm={a.stage_norm} "
          f"| ridge 1e{a.ridge_lower}..1e{a.ridge_upper - 1}", flush=True)
    print(f"\n| Stage | d | A_T | A_bar | Forgetting | lambda |")
    print(f"|---|---:|---:|---:|---:|---:|")
    base = None
    for sub in SUBSETS:
        t0 = time.time()
        m = run(a, data, stage_slice(sub).to(dev), dev)
        base = base or m['A_bar']
        name = '+'.join(f"s{i + 1}" for i in sub)
        print(f"| {name} | {m['d']} | {m['A_T']:.2f} | {m['A_bar']:.2f} | "
              f"{m['forgetting']:.2f} | {m['ridge']:g} |"
              f"   ({m['A_bar'] - base:+.2f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
