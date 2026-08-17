"""Top-k theo KHOI (canh tranh cuc bo) thay vi toan cuc.

    python flycl_blocktopk.py --model_name resnet50 --data_augmentation resnet --coding_level 0.1

Xuat phat tu y "moi task mot ma tran W ngau nhien dong bang". Vi ma tran ngau
nhien khong phu thuoc du lieu nen sinh duoc het tu dau -> khong co khoi nao
khuyet, tinh chinh xac giu nguyen. Va khi do ghep m phep chieu 1000 chieu chinh
la MOT phep chieu 10000 chieu; khac biet con lai nam dung o top-k:

    m=1  : top-k tren ca 10000 unit    -> giu 1000 thang toan cuc
    m=10 : top-k trong tung khoi 1000  -> giu 100 moi khoi

Dong luc tu chan doan: tren ResNet co 5.8% unit KHONG BAO GIO thang, va Jaccard
tap chi so top-k giua hai anh khac nhau la 0.185 (ViT chi 0.100). Tuc mot nhom
unit dang ap dao vi feature ResNet toan duong. Canh tranh theo khoi ep hoat dong
trai deu.

W hoan toan khong doi giua cac m - chi cach chon thay doi - nen so sanh la theo
CAP tuyet doi, va m=1 phai tai lap DUNG flycl_baseline.
"""

import time

import numpy as np
import torch

from config import get_parser
from data import TaskData, set_seed
from flycl_baseline import select_ridge_parameter
from train import cache_exists, load_backbone


def topk_blocks(Z, k, m):
    """Z la [E, N]. Chia E thanh m khoi, giu k/m gia tri lon nhat MOI KHOI.

    m=1 tro ve dung topk_rows cua flycl_baseline.
    """
    E, N = Z.shape
    zb = Z.view(m, E // m, N)
    _, idx = zb.topk(k // m, dim=1, largest=True)
    out = torch.zeros_like(zb)
    out.scatter_(1, idx, zb.gather(1, idx))
    return out.view(E, N)


def run(a, data, m, dev):
    d = data.Xtr.shape[1]
    set_seed(a.seed)
    W = torch.zeros(a.expand_dim, d)
    for r in range(a.expand_dim):
        pick = torch.randperm(d)[:a.synaptic_degree]
        W[r, pick] = torch.randn(a.synaptic_degree)
    W = W.to(dev)

    k = int(a.expand_dim * a.coding_level)
    Q = torch.zeros(a.expand_dim, a.num_classes, device=dev)
    G = torch.zeros(a.expand_dim, a.expand_dim, device=dev)
    eye = torch.eye(a.expand_dim, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]
    dead = 0.0

    for task in range(a.num_tasks):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        Xtr, Ytr = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])

        H = topk_blocks(W @ Xtr.T, k, m)
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        Q += H @ Y
        G += H @ H.T
        ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
        L = torch.linalg.cholesky(G + ridge * eye)
        Wo = torch.cholesky_solve(Q, L)
        if task == 0:
            dead = (H != 0).any(1).logical_not().float().mean().item() * 100
        del H, Y

        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            He = topk_blocks(W @ Xte.T, k, m)
            acc[i][task] = ((He.T @ Wo).argmax(1) == Yte).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget)), 'dead': dead}


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=3)
    p.add_argument('--ridge_upper', type=int, default=13)
    p.add_argument('--blocks', default='1,2,5,10,20')
    a = p.parse_args()
    a.cache_features = True
    a.freeze_backbone = True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    print(f"[flycl-blk] {a.model_name} | expand={a.expand_dim} | k={a.coding_level} "
          f"| seed={a.seed}", flush=True)
    print(f"\n| khoi | unit/khoi | giu/khoi | A_T | A_bar | Forgetting | unit chet |")
    print(f"|---:|---:|---:|---:|---:|---:|---:|")

    base = None
    for m in [int(x) for x in a.blocks.split(',')]:
        t0 = time.time()
        r = run(a, data, m, dev)
        base = r['A_bar'] if base is None else base
        k = int(a.expand_dim * a.coding_level)
        print(f"| {m} | {a.expand_dim // m} | {k // m} | {r['A_T']:.2f} | "
              f"{r['A_bar']:.2f} | {r['forgetting']:.2f} | {r['dead']:.1f}% |"
              f"   ({r['A_bar'] - base:+.2f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
