"""m phep chieu doc lap, MOI CAI THAY TOAN BO du lieu, cong logit lai.

    python flycl_ensemble.py --model_name resnet50 --data_augmentation resnet --coding_level 0.1

Khac han flycl_moe.py. O do khong gian feature bi CHIA cho m chuyen gia nen moi
chuyen gia chi con 1/m du lieu -> thua -5.93 diem o m=8. O day khong ai mat mau
nao: m mo hinh Fly-CL doc lap, moi mo hinh la ridge chinh xac tren TOAN BO du
lieu da thay, chi khac nhau o phep chieu ngau nhien.

Tuong duong voi mot ridge 100.000 chieu nhung G bi ep ve dang KHOI CHEO. Ban day
du can G cỡ 100000^2 = 40 GB nen khong chay duoc; ban khoi cheo ton m*400 MB.
Cai mat la tuong quan cheo giua cac khoi.

Dong luc dinh luong: A_bar dao dong sigma = 0.75 giua cac seed, ma nguon ngau
nhien duy nhat la ma tran chieu. Do la phuong sai thuan tuy -> trung binh hoa
cat duoc no. Ensemble cua cac uoc luong gan khong lech thuong cai thien ca ky
vong chu khong chi phuong sai.

Van chinh xac tang dan: moi nhanh tich luy Q, G rieng tren toan bo du lieu, nen
tung nhanh bat bien theo thu tu task -> tong cung vay.

m=1 phai tai lap DUNG flycl_baseline.
"""

import time

import numpy as np
import torch

from config import get_parser
from data import TaskData, set_seed
from flycl_baseline import select_ridge_parameter, topk_rows
from train import cache_exists, load_backbone


def make_projection(expand, d, degree, seed, dev):
    """Hai dong rieng, giong het flycl_baseline: viet gop mot dong thi Python
    danh gia ve phai truoc -> randn chay truoc randperm -> W khac han."""
    set_seed(seed)
    W = torch.zeros(expand, d)
    for r in range(expand):
        pick = torch.randperm(d)[:degree]
        W[r, pick] = torch.randn(degree)
    return W.to(dev)


def run(a, data, m, dev):
    d = data.Xtr.shape[1]
    E, k = a.expand_dim, int(a.expand_dim * a.coding_level)
    # Nhanh 0 dung dung seed cua baseline -> m=1 tai lap chinh xac.
    Ws = [make_projection(E, d, a.synaptic_degree, a.seed + e, dev) for e in range(m)]
    Q = [torch.zeros(E, a.num_classes, device=dev) for _ in range(m)]
    G = [torch.zeros(E, E, device=dev) for _ in range(m)]
    sol = [None] * m
    eye = torch.eye(E, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    for task in range(a.num_tasks):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        Xtr, Ytr = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)

        for e in range(m):
            H = topk_rows(Ws[e] @ Xtr.T, k)          # toan bo du lieu, moi nhanh
            Q[e] += H @ Y
            G[e] += H @ H.T
            ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
            L = torch.linalg.cholesky(G[e] + ridge * eye)
            sol[e] = torch.cholesky_solve(Q[e], L)
            del H
        del Y

        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            logit = 0
            for e in range(m):
                logit = logit + topk_rows(Ws[e] @ Xte.T, k).T @ sol[e]
            acc[i][task] = (logit.argmax(1) == Yte).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget))}


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=3)
    p.add_argument('--ridge_upper', type=int, default=13)
    p.add_argument('--branches', default='1,2,5,10')
    a = p.parse_args()
    a.cache_features = True
    a.freeze_backbone = True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    print(f"[flycl-ens] {a.model_name} | expand={a.expand_dim} | k={a.coding_level} "
          f"| seed={a.seed}", flush=True)
    print(f"\n| nhanh | tong chieu | bo nho G | A_T | A_bar | Forgetting |")
    print(f"|---:|---:|---:|---:|---:|---:|")

    base = None
    for m in [int(x) for x in a.branches.split(',')]:
        t0 = time.time()
        r = run(a, data, m, dev)
        base = r['A_bar'] if base is None else base
        print(f"| {m} | {m * a.expand_dim} | {m * a.expand_dim ** 2 * 4 / 1e9:.1f} GB | "
              f"{r['A_T']:.2f} | {r['A_bar']:.2f} | {r['forgetting']:.2f} |"
              f"   ({r['A_bar'] - base:+.2f}, {time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
