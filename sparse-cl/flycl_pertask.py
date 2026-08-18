"""Moi task mot ma tran chieu rieng, MOI KHOI MOT G RIENG.

    python flycl_pertask.py --model_name resnet50 --data_augmentation resnet --coding_level 0.1

Task t sinh ra W_t rieng, kem Q_t va G_t rieng. Khong co G chung, nen khong co
o nao "khuyet" - moi khoi la mot bai toan ridge tron ven tren dung phan du lieu
ma no da thay. Du doan la tong logit cua cac khoi da ton tai.

Cai gia khac: khoi t chi thay du lieu tu task t tro di, nen khoi sinh muon co
it du lieu hon han. Khoi cuoi chi thay 1/10 du lieu.

Hai che do de tach hai yeu to:
  --start_at task : khoi t bat dau tich luy tu task t   (dung y "moi task mot W")
  --start_at 0    : moi khoi tich luy tu task 1         (= ensemble, doi chung)
Hieu giua chung la cai gia cua viec khoi sinh muon thay it du lieu.
"""

import time

import numpy as np
import torch

from config import get_parser
from data import TaskData, set_seed
from flycl_baseline import select_ridge_parameter, topk_rows
from train import cache_exists, load_backbone


def run(a, data, per_task, dev):
    d = data.Xtr.shape[1]
    T, E = a.num_tasks, a.expand_dim
    block = E // T
    Ws, Q, G = [], [], []
    for t in range(T):
        set_seed(a.seed + t)
        W = torch.zeros(block, d)
        for r in range(block):
            pick = torch.randperm(d)[:a.synaptic_degree]
            W[r, pick] = torch.randn(a.synaptic_degree)
        Ws.append(W.to(dev))
        Q.append(torch.zeros(block, a.num_classes, device=dev))
        G.append(torch.zeros(block, block, device=dev))
    sol = [None] * T
    eye = torch.eye(block, device=dev)
    k = max(int(block * a.coding_level), 1)
    acc = [[0.0] * T for _ in range(T)]
    seen = [0] * T

    for task in range(T):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        Xtr, Ytr = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)

        for e in range(T):
            # per_task: khoi e chua ton tai truoc task e -> bo qua.
            if per_task and e > task:
                continue
            H = topk_rows(Ws[e] @ Xtr.T, k)
            Q[e] += H @ Y
            G[e] += H @ H.T
            seen[e] += Ytr.shape[0]
            ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
            sol[e] = torch.cholesky_solve(Q[e], torch.linalg.cholesky(G[e] + ridge * eye))
            del H
        del Y

        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            logit = 0
            for e in range(T):
                if sol[e] is None:
                    continue
                logit = logit + topk_rows(Ws[e] @ Xte.T, k).T @ sol[e]
            acc[i][task] = (logit.argmax(1) == Yte).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(T)]
    last = T - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget)), 'A_t': A_t, 'seen': seen}


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=3)
    p.add_argument('--ridge_upper', type=int, default=13)
    a = p.parse_args()
    a.cache_features, a.freeze_backbone = True, True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    print(f"[pertask] {a.model_name} | {a.num_tasks} khoi x "
          f"{a.expand_dim // a.num_tasks} chieu | moi khoi mot G rieng "
          f"| k={a.coding_level} | seed={a.seed}", flush=True)
    print(f"\n| Che do | A_T | A_bar | Forgetting |")
    print(f"|---|---:|---:|---:|")
    out = {}
    for name, per in (('moi khoi tu task 1 (doi chung)', False),
                      ('khoi t sinh o task t', True)):
        t0 = time.time()
        r = out[name] = run(a, data, per, dev)
        print(f"| {name} | {r['A_T']:.2f} | {r['A_bar']:.2f} | {r['forgetting']:.2f} |"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    print("\nA_t theo tung giai doan:")
    for name, r in out.items():
        print(f"  {name:32s} " + ' '.join(f"{x:5.1f}" for x in r['A_t']))
    print("\nSo mau moi khoi da thay:")
    for name, r in out.items():
        print(f"  {name:32s} " + ' '.join(f"{x // 1000}k" for x in r['seen']))


if __name__ == '__main__':
    main()
