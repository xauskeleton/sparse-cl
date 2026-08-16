"""Hai cai tien de xuat cho Fly-CL, do tren cung mot phep chieu va cung split.

    python flycl_improved.py --coding_level 0.1
    python flycl_improved.py --model_name resnet50 --data_augmentation resnet --coding_level 0.1

`flycl_baseline.py` giu nguyen ban da doi chieu duoc voi log goc cua ho
(ViT: 88.77/93.11 so voi 88.68/92.99) va KHONG duoc sua - no la doi chung.
File nay chi import lai tu do roi them bien the.

Cai tien 1 - `--gcv accum`
    Fly-CL chon lambda bang GCV nhung chi nhin task hien tai (main.py:112), roi
    ap lambda do len G da tich luy moi task. Moi task cung kich thuoc nen GCV
    chon ra gan cung mot lambda, trong khi tr(G) tang tuyen tinh -> regularization
    hieu dung yeu di ~10 lan tu task 1 den task 10, khong ai kiem soat.

    Vi Q, G la sufficient statistics va Y la one-hot (nen tr(Y^T Y) = N), GCV
    tren TOAN BO du lieu da thay tinh duoc thang tu Q, G, N - khong luu them gi:
        RSS(l) = N - 2 tr(Q^T W) + tr(W^T G W),   W = (G + l I)^-1 Q
        df(l)  = sum_i s_i / (s_i + l)
    Voi G = V diag(s) V^T, P = V^T Q, r_i = ||P_i||^2:
        tr(Q^T W)   = sum_i r_i / (s_i + l)
        tr(W^T G W) = sum_i r_i s_i / (s_i + l)^2
    Mot lan eigh moi task, sau do ca luoi lambda ton O(d) mot gia tri. eigh cung
    thay luon SVD + Cholesky cua ban goc nen chi phi khong tang bao nhieu.

Cai tien 2 - `--topk abs`
    main.py:104 dung largest=True, tuc chi giu duoi DUONG. Entry cua ma tran
    chieu la torch.randn nen phan bo doi xung: duoi am mang luong thong tin
    tuong duong va dang bi vut sach. Bien the nay chon top-k theo |.| roi giu
    nguyen dau.
"""

import time

import numpy as np
import torch

from config import get_parser
from data import TaskData, set_seed
from flycl_baseline import select_ridge_parameter, topk_rows
from train import cache_exists, load_backbone

VARIANTS = [('task', 'pos'), ('accum', 'pos'), ('task', 'abs'), ('accum', 'abs')]


def topk_signed(Z, k):
    """Top-k theo |.|, giu nguyen dau. Doi trong voi topk_rows cua ban goc."""
    _, idx = Z.abs().topk(k, dim=0, largest=True)
    out = torch.zeros_like(Z)
    out.scatter_(0, idx, Z.gather(0, idx))
    return out


def select_ridge_accum(G, Q, N, lo, hi):
    """GCV tren toan bo du lieu da thay. Tra ve (lambda, nghiem)."""
    s, V = torch.linalg.eigh(G)
    s = s.clamp_min(0)
    P = V.T @ Q
    r = (P * P).sum(1)
    best, best_ridge = None, None
    for ridge in 10.0 ** np.arange(lo, hi):
        den = s + ridge
        df = (s / den).sum()
        if df >= N:                       # mau so GCV am -> lambda qua nho
            continue
        rss = N - 2 * (r / den).sum() + (r * s / den ** 2).sum()
        gcv = ((rss / N) / (1 - df / N) ** 2).item()
        if best is None or gcv < best:
            best, best_ridge = gcv, ridge
    den = s + best_ridge
    return best_ridge, V @ (P / den[:, None])


def metrics(acc, T):
    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(T)]
    last = T - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget)), 'A_t': A_t}


def run(a, data, W, k, gcv, topk, test_codes, dev):
    code = topk_signed if topk == 'abs' else topk_rows
    Q = torch.zeros(a.expand_dim, a.num_classes, device=dev)
    G = torch.zeros(a.expand_dim, a.expand_dim, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]
    ridges, N = [], 0

    for task in range(a.num_tasks):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        # Fly-CL khong tach validation; gop lai de dung luong du lieu bang ho.
        Xtr, Ytr = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])

        H = code(W @ Xtr.T, k)
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        Q += H @ Y
        G += H @ H.T
        N += Ytr.shape[0]

        if gcv == 'accum':
            ridge, Wo = select_ridge_accum(G, Q, N, a.ridge_lower, a.ridge_upper)
        else:
            ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
            L = torch.linalg.cholesky(G + ridge * torch.eye(a.expand_dim, device=dev))
            Wo = torch.cholesky_solve(Q, L)
        ridges.append(ridge)
        del H, Y

        for i in range(task + 1):
            He, Yte = test_codes[i]
            acc[i][task] = ((He @ Wo).argmax(1) == Yte).float().mean().item() * 100

    m = metrics(acc, a.num_tasks)
    m['ridges'] = [float(x) for x in ridges]
    m['acc'] = acc
    return m


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=4)
    p.add_argument('--ridge_upper', type=int, default=13)
    a = p.parse_args()
    a.cache_features = True
    a.freeze_backbone = True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    d = data.Xtr.shape[1]

    # Cung mot phep chieu cho ca bon bien the -> chenh lech khong lan nhieu seed.
    set_seed(a.seed)
    W = torch.zeros(a.expand_dim, d)
    for r in range(a.expand_dim):
        cols = torch.randperm(d)[:a.synaptic_degree]
        W[r, cols] = torch.randn(a.synaptic_degree)
    W = W.to(dev)
    k = int(a.expand_dim * a.coding_level)

    print(f"[flycl+] {a.model_name} | d={d} | expand={a.expand_dim} "
          f"| degree={a.synaptic_degree} | k={a.coding_level} | seed={a.seed} "
          f"| ridge 1e{a.ridge_lower}..1e{a.ridge_upper - 1}", flush=True)

    out = {}
    for topk in ('pos', 'abs'):
        code = topk_signed if topk == 'abs' else topk_rows
        test_codes = []
        for i in range(a.num_tasks):
            Xte, Yte = data.test_split(i)
            test_codes.append((code(W @ Xte.T, k).T.contiguous(), Yte))
        for gcv, tk in VARIANTS:
            if tk != topk:
                continue
            t0 = time.time()
            m = out[(gcv, topk)] = run(a, data, W, k, gcv, topk, test_codes, dev)
            print(f"  gcv={gcv:<5} topk={topk:<3} | A_T {m['A_T']:.2f} | "
                  f"A_bar {m['A_bar']:.2f} | F {m['forgetting']:.2f} | "
                  f"{time.time() - t0:.0f}s | lambda "
                  f"{'->'.join(f'{x:g}' for x in m['ridges'][::3])}", flush=True)
        del test_codes

    base = out[('task', 'pos')]
    print(f"\n| GCV | top-k | A_T | A_bar | Forgetting | d A_bar |")
    print(f"|---|---|---:|---:|---:|---:|")
    for gcv, topk in VARIANTS:
        m = out[(gcv, topk)]
        tag = ' (Fly-CL)' if (gcv, topk) == ('task', 'pos') else ''
        print(f"| {gcv}{tag} | {topk} | {m['A_T']:.2f} | {m['A_bar']:.2f} | "
              f"{m['forgetting']:.2f} | {m['A_bar'] - base['A_bar']:+.2f} |")


if __name__ == '__main__':
    main()
