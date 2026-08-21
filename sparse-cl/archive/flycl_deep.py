"""Hai bien the kien truc: maxout giua cac nhanh, va xep tang hai lop.

    python flycl_deep.py --model_name resnet50.tv2_in1k --data_augmentation resnet \
                         --coding_level 0.3 --variants base,max2,max5,deep2

Ca hai deu giu G o dung [expand_dim, expand_dim] nen bo nho KHONG DOI so voi
Fly-CL goc, va deu dong bang hoan toan nen Q, G van la sufficient statistics.

maxout (`maxM`)
    Ensemble hien tai gop o tang LOGIT nen moi nhanh can mot G rieng -> m lan bo
    nho. Gop o tang MA thi chi can mot G:
        z_i = max_e (W_e x)_i   roi top-k roi mot head duy nhat
    Lay trung binh thi vo nghia vi mean_e(W_e x) = (mean_e W_e) x, chi la mot
    phep chieu khac. `max` khong rut gon duoc nen tao ra phi tuyen that.

xep tang (`deepN`)
    Ca ho phuong phap nay dung dung MOT tang mo rong ngau nhien. Day la tang thu
    hai, chieu len chinh ma thua:
        z₁ = topk(W₁ x),  z₂ = topk(W₂ z₁)
    Rui ro: moi tang top-k vut 70% gia tri, hai tang co the vut qua nhieu.
"""

import sys
import time

import numpy as np
import torch

from config import get_parser
from data_loader import TaskData, set_seed
from utils import select_ridge_parameter, topk_rows
from backbone import cache_exists, load_backbone

# Console Windows mac dinh cp1252, khong in duoc 'Delta' hay 'A macron'.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def sparse_proj(rows, cols, degree, seed, dev):
    set_seed(seed)
    W = torch.zeros(rows, cols)
    for r in range(rows):
        pick = torch.randperm(cols)[:min(degree, cols)]
        W[r, pick] = torch.randn(pick.numel())
    return W.to(dev)


def run(a, data, variant, dev):
    d, E = data.Xtr.shape[1], a.expand_dim
    k = int(E * a.coding_level)
    deg = a.synaptic_degree

    if variant.startswith('max'):
        m = int(variant[3:])
        Ws = [sparse_proj(E, d, deg, a.seed + e * 1000, dev) for e in range(m)]

        def code(x):
            z = Ws[0] @ x.T
            for W in Ws[1:]:
                z = torch.maximum(z, W @ x.T)
            return topk_rows(z, k)
    elif variant.startswith('deep'):
        n = int(variant[4:])
        W1 = sparse_proj(E, d, deg, a.seed, dev)
        Ws = [sparse_proj(E, E, deg, a.seed + 7000 + i, dev) for i in range(n - 1)]

        def code(x):
            z = topk_rows(W1 @ x.T, k)
            for W in Ws:
                z = topk_rows(W @ z, k)
            return z
    elif variant.startswith('bot'):
        # bot{r}      : x → W1[r,d] → W2[E,r] → top-k    (TUYEN TINH, hang <= r)
        # bot{r}relu  : them ReLU o tang giua -> hai tang that
        # bot{r}topk  : them top-k thua o tang giua
        tag = variant[3:]
        mid = 'relu' if tag.endswith('relu') else ('topk' if tag.endswith('topk')
                                                   else 'none')
        r = int(tag.replace('relu', '').replace('topk', ''))
        W1 = sparse_proj(r, d, deg, a.seed, dev)
        W2 = sparse_proj(E, r, deg, a.seed + 7000, dev)
        kmid = max(int(r * a.coding_level), 1)

        def code(x):
            h = W1 @ x.T
            if mid == 'relu':
                h = h.clamp_min(0)
            elif mid == 'topk':
                h = topk_rows(h, kmid)
            return topk_rows(W2 @ h, k)
    else:                                        # base = Fly-CL goc
        W1 = sparse_proj(E, d, deg, a.seed, dev)

        def code(x):
            return topk_rows(W1 @ x.T, k)

    Q = torch.zeros(E, a.num_classes, device=dev)
    G = torch.zeros(E, E, device=dev)
    eye = torch.eye(E, device=dev)
    acc = [[0.0] * a.num_tasks for _ in range(a.num_tasks)]

    for task in range(a.num_tasks):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        Xtr, Ytr = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])
        H = code(Xtr)
        Y = torch.zeros(Ytr.shape[0], a.num_classes, device=dev)
        Y.scatter_(1, Ytr.long().view(-1, 1), 1.0)
        Q += H @ Y
        G += H @ H.T
        ridge = select_ridge_parameter(H.T, Y, a.ridge_lower, a.ridge_upper)
        Wo = torch.cholesky_solve(Q, torch.linalg.cholesky(G + ridge * eye))
        del H, Y
        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            acc[i][task] = ((code(Xte).T @ Wo).argmax(1) == Yte).float().mean().item() * 100

    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(a.num_tasks)]
    last = a.num_tasks - 1
    forget = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_T': A_t[-1], 'A_bar': float(np.mean(A_t)),
            'forgetting': float(np.mean(forget))}


def main():
    p = get_parser()
    p.add_argument('--ridge_lower', type=int, default=3)
    p.add_argument('--ridge_upper', type=int, default=13)
    p.add_argument('--variants', default='base,max2,max5,deep2')
    a = p.parse_args()
    a.cache_features, a.freeze_backbone = True, True
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')

    set_seed(a.seed)
    bb = None if cache_exists(a) else load_backbone(a.model_name, dev)
    data = TaskData(a, bb, dev)
    del bb
    print(f"[deep] {a.model_name} | expand={a.expand_dim} | k={a.coding_level} "
          f"| seed={a.seed}", flush=True)
    print(f"\n| Bien the | Bo nho G | A_T | A_bar | Forgetting | Δ Ā |")
    print(f"|---|---:|---:|---:|---:|---:|")
    base = None
    for v in a.variants.split(','):
        t0 = time.time()
        r = run(a, data, v, dev)
        base = r['A_bar'] if base is None else base
        print(f"| {v} | {a.expand_dim ** 2 * 4 / 1e9:.1f} GB | {r['A_T']:.2f} | "
              f"{r['A_bar']:.2f} | {r['forgetting']:.2f} | {r['A_bar'] - base:+.2f} |"
              f"   ({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
