"""Hon hop chuyen gia voi cong DONG BANG - mo hinh tuyen tinh tung manh, van
chinh xac tang dan.

    python flycl_moe.py --coding_level 0.1                    # ngan sach bo nho co dinh
    python flycl_moe.py --coding_level 0.1 --budget dim       # expand_dim co dinh

Y tuong. Fly-CL manh vi hoi quy binh phuong toi thieu co sufficient statistics:
Q = sum(H Y) va G = sum(H H^T) khien hoc tuan tu bang DUNG hoc chung. Nhung do
la lop mo hinh tuyen tinh - va moi thu ta thu de cai tien no deu cham tran vi
no da toi uu tuyet doi TRONG lop do.

Cach ra khoi lop do ma khong mat gi: chia khong gian feature thanh m vung bang
mot bo dinh tuyen CO DINH, moi vung giu Q, G rieng. Vi cong khong phu thuoc
nhan cung khong phu thuoc thu tu task, moi chuyen gia van la ridge chinh xac
tren toan bo du lieu roi vao vung cua no. Tong the thanh mo hinh tuyen tinh
tung manh - xap xi duoc ham phi tuyen bat ky khi m tang - ma van bat bien theo
thu tu task va van exemplar-free.

Bo dinh tuyen: b = log2(m) sieu phang ngau nhien di qua mean cua TASK 0. Chi
dung task 0 nen khong vi pham giao thuc CL, va co dinh vinh vien tu do.

Hai che do ngan sach:
  --budget memory (mac dinh): E = expand_dim / sqrt(m), tong bo nho m*E^2 giu
    nguyen. Tra loi "cung chi phi, mot mo hinh to hay m mo hinh cuc bo tot hon".
  --budget dim: E = expand_dim cho moi chuyen gia. Tach rieng anh huong cua
    tinh cuc bo, nhung ton m lan bo nho.

m=1 phai tai lap DUNG flycl_baseline - do la phep tu kiem tra.
"""

import time

import numpy as np
import torch

from config import get_parser
from data import TaskData, set_seed
from flycl_baseline import select_ridge_parameter, topk_rows
from train import cache_exists, load_backbone


def make_projection(expand, d, degree, seed):
    """Phai giu dung hai dong nhu flycl_baseline: viet gop mot dong thi Python
    danh gia ve phai truoc -> randn chay truoc randperm -> W khac han."""
    set_seed(seed)
    W = torch.zeros(expand, d)
    for r in range(expand):
        pick = torch.randperm(d)[:degree]
        W[r, pick] = torch.randn(degree)
    return W


def build_router(X0, m, seed, dev):
    """b sieu phang ngau nhien qua mean cua task 0 -> 2^b vung, cố định mãi mãi."""
    b = int(round(np.log2(m)))
    mu = X0.mean(0)
    g = torch.Generator(device='cpu').manual_seed(seed + 12345)
    P = torch.randn(X0.shape[1], b, generator=g).to(dev) if b else None
    return mu, P


def assign(X, mu, P):
    if P is None:
        return torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    bits = ((X - mu) @ P > 0).long()
    w = 2 ** torch.arange(P.shape[1], device=X.device)
    return (bits * w).sum(1)


def run(a, data, m, expand, dev):
    d = data.Xtr.shape[1]
    k = int(expand * a.coding_level)
    (X0, Y0), (X0v, Y0v) = data.train_split(0)
    mu, P = build_router(torch.cat([X0, X0v]), m, a.seed, dev)
    rtr, rte = assign(data.Xtr, mu, P), assign(data.Xte, mu, P)

    Ws = [make_projection(expand, d, a.synaptic_degree, a.seed + e).to(dev)
          for e in range(m)]
    Q = [torch.zeros(expand, a.num_classes, device=dev) for _ in range(m)]
    G = [torch.zeros(expand, expand, device=dev) for _ in range(m)]
    sol = [torch.zeros(expand, a.num_classes, device=dev) for _ in range(m)]
    eye = torch.eye(expand, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    for task in range(a.num_tasks):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        lo = task * data.cpt
        idx = torch.arange(len(data.ytr), device=dev)
        sel = idx[(data.ytr >= lo) & (data.ytr < lo + data.cpt)]
        Xt, Yt, rt = data.Xtr[sel], data.ytr[sel], rtr[sel]

        for e in range(m):
            me = rt == e
            if not me.any():
                continue
            H = topk_rows(Ws[e] @ Xt[me].T, k)
            Y = torch.zeros(int(me.sum()), a.num_classes, device=dev)
            Y.scatter_(1, Yt[me].long().view(-1, 1), 1.0)
            Q[e] += H @ Y
            G[e] += H @ H.T
            ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
            L = torch.linalg.cholesky(G[e] + ridge * eye)
            sol[e] = torch.cholesky_solve(Q[e], L)
            del H, Y

        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            r = rte[(data.yte >= i * data.cpt) & (data.yte < (i + 1) * data.cpt)]
            pred = torch.zeros(len(Yte), dtype=torch.long, device=dev)
            for e in range(m):
                me = r == e
                if me.any():
                    He = topk_rows(Ws[e] @ Xte[me].T, k)
                    pred[me] = (He.T @ sol[e]).argmax(1)
            acc[i][task] = (pred == Yte).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    bal = torch.bincount(rtr, minlength=m).tolist()
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget)), 'E': expand, 'balance': bal}


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=3)
    p.add_argument('--ridge_upper', type=int, default=13)
    p.add_argument('--experts', default='1,2,4,8')
    p.add_argument('--budget', default='memory', choices=['memory', 'dim'])
    a = p.parse_args()
    a.cache_features = True
    a.freeze_backbone = True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    print(f"[flycl-moe] {a.model_name} | expand={a.expand_dim} | k={a.coding_level} "
          f"| seed={a.seed} | budget={a.budget}", flush=True)
    print(f"\n| m | E | A_T | A_bar | Forgetting | can bang vung |")
    print(f"|---:|---:|---:|---:|---:|---|")

    base = None
    for m in [int(x) for x in a.experts.split(',')]:
        expand = a.expand_dim if a.budget == 'dim' else \
            int(round(a.expand_dim / np.sqrt(m)))
        t0 = time.time()
        r = run(a, data, m, expand, dev)
        base = r['A_bar'] if base is None else base
        bal = '/'.join(str(round(x / sum(r['balance']) * 100)) for x in r['balance'])
        print(f"| {m} | {r['E']} | {r['A_T']:.2f} | {r['A_bar']:.2f} | "
              f"{r['forgetting']:.2f} | {bal}% |"
              f"   ({r['A_bar'] - base:+.2f}, {time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
