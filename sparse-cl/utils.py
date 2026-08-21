"""Hai ham loi cua Fly-CL, sao NGUYEN VAN tu upstream/Fly-CL-main/main.py.

Tach ra day vi moi script trong experiments/ deu dung, va de khong file nao
trong experiments/ phai import lan nhau.
"""

import numpy as np
import torch


def select_ridge_parameter(X, Y, lo, hi):
    """GCV tren luoi 10^lo .. 10^hi. Nguyen van tu upstream/Fly-CL-main/main.py."""
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
