"""Chay thuat toan Fly-CL tren feature cache cua sparse-cl.

    python flycl_baseline.py --model_name resnet50 --data_augmentation resnet
    python flycl_baseline.py --model_name vit_base_patch16_224

Vi sao khong chay thang repo Fly-CL: load_model.py cua ho nap ResNet-50 tu
./pretrained_model/resnet50-11ad3fa6.pth, tuc trong so torchvision IMAGENET1K_V2,
trong khi moi bang cua sparse-cl dung timm 'resnet50' mac dinh (a1_in1k). Hai bo
trong so khac nhau thi con so khong so truc tiep duoc. Script nay tai dung dung
thuat toan cua ho nhung tren cung feature, cung class order, cung split.

Thuat toan sao nguyen tu Fly-CL-main/main.py:
  - ma tran chieu thua: moi hang chon ngau nhien `synaptic_degree` cot roi dat
    torch.randn vao do (KHONG chia sqrt(degree) nhu sparse-cl)
  - top-k giu gia tri goc tai cac vi tri thang
  - tich luy Q = sum(H Y) va G = sum(H H^T) qua cac task -> nghiem ridge luon
    toi uu tren TOAN BO du lieu da thay, khong chi task hien tai
  - chon he so ridge bang GCV tren SVD, roi giai bang Cholesky
"""

import argparse
import time

import numpy as np
import torch

from config import get_parser
from data import TaskData, set_seed
from train import cache_exists, load_backbone


def select_ridge_parameter(X, Y, lo, hi):
    """GCV tren luoi 10^lo .. 10^hi. Nguyen van tu Fly-CL-main/main.py."""
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    S_sq = S ** 2
    UTY = U.T @ Y
    n = X.shape[0]
    best, best_ridge = None, None
    for ridge in 10.0 ** np.arange(lo, hi):
        diag = S_sq / (S_sq + ridge)
        Y_hat = U @ (diag[:, None] * UTY)
        gcv = ((Y - Y_hat).norm() ** 2 / n) / (1 - diag.sum() / n) ** 2
        if best is None or gcv.item() < best:
            best, best_ridge = gcv.item(), ridge
    return best_ridge


def topk_rows(Z, k):
    """Z la [expand_dim, N]; giu k gia tri lon nhat theo CHIEU DAU (per-sample)."""
    _, idx = Z.topk(k, dim=0, largest=True)
    out = torch.zeros_like(Z)
    out.scatter_(0, idx, Z.gather(0, idx))
    return out


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=6)
    p.add_argument('--ridge_upper', type=int, default=10)
    a = p.parse_args()
    # Fly-CL mac dinh coding_level 0.3 trong scripts/test_cifar.sh, khac mac dinh
    # 0.1 cua sparse-cl. Giu nguyen gia tri nguoi dung truyen vao.
    a.cache_features = True
    a.freeze_backbone = True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    d = data.Xtr.shape[1]
    print(f"[flycl] {a.model_name} | d={d} | expand={a.expand_dim} "
          f"| degree={a.synaptic_degree} | k={a.coding_level} | seed={a.seed}")

    # Ma tran chieu dung cach dung cua Fly-CL: khong chia sqrt(degree).
    set_seed(a.seed)
    W = torch.zeros(a.expand_dim, d)
    for r in range(a.expand_dim):
        cols = torch.randperm(d)[:a.synaptic_degree]
        W[r, cols] = torch.randn(a.synaptic_degree)
    W = W.to(dev)

    k = int(a.expand_dim * a.coding_level)
    Q = torch.zeros(a.expand_dim, a.num_classes, device=dev)
    G = torch.zeros(a.expand_dim, a.expand_dim, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]
    t0 = time.time()

    for task in range(a.num_tasks):
        (Xtr, Ytr), _ = data.train_split(task)
        # Fly-CL khong tach validation; gop lai de dung luong du lieu bang ho.
        Xva, Yva = data.train_split(task)[1]
        Xtr, Ytr = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])

        H = topk_rows(W @ Xtr.T, k)                        # [expand, N]
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)

        Q += H @ Y
        G += H @ H.T
        ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
        L = torch.linalg.cholesky(G + ridge * torch.eye(a.expand_dim, device=dev))
        Wo = torch.cholesky_solve(Q, L)

        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            He = topk_rows(W @ Xte.T, k)
            pred = (He.T @ Wo).argmax(1)
            acc[i][task] = (pred == Yte).float().mean().item() * 100
        print(f"  task {task}: ridge={ridge:g} | A_t="
              f"{np.mean([acc[i][task] for i in range(task + 1)]):.2f}")

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]

    print("\nAccuracy matrix")
    for i in range(a.num_tasks):
        print("  " + " ".join(f"{acc[i][j]:6.2f}" if j >= i else "     ."
                              for j in range(a.num_tasks)))
    print(f"\nA_t        : {[round(x, 2) for x in A_t]}")
    print(f"A_T        : {A_t[-1]:.2f}")
    print(f"A_bar      : {np.mean(A_t):.2f}")
    print(f"Forgetting : {np.mean(forget):.2f}")
    print(f"Tong thoi gian: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
