"""Hai ham loi cua Fly-CL, sao NGUYEN VAN tu upstream/Fly-CL-main/main.py.

Tach ra day vi moi script trong experiments/ deu dung, va de khong file nao
trong experiments/ phai import lan nhau.
"""

import numpy as np
import torch


def select_ridge_parameter(X, Y, lo, hi):
    """GCV tren luoi 10^lo .. 10^hi. Cung tieu chi, cung luoi, cung ket qua voi
    ban SVD cua upstream/Fly-CL-main/main.py - nhung khong dung SVD.

    Ban goc goi `U, S, _ = svd(X, full_matrices=False)` voi X cỡ [n, E]. Voi
    n=5000, E=10000 thi `_` la ma tran vector ky di PHAI cỡ 5000x10000: 200 MB
    duoc tinh ra roi vut di ngay. Do la 97% thoi gian chay cua ca Fly-CL - nhieu
    hon ca ma hoa, Gram va Cholesky cong lai.

    GCV chi can hai thu: S² va w_j = ‖u_jᵀY‖². Ca hai lay duoc tu ma tran Gram
    NHO HON trong hai chieu, khong bao gio dung den phia con lai.

        n <= E:  eigh(X Xᵀ)  ->  tri rieng = S²,  vector rieng = U
        n >  E:  eigh(Xᵀ X)  ->  tri rieng = S²,  vector rieng = V
                 u_jᵀY = v_jᵀ(XᵀY) / s_j          XᵀY chi cỡ [E, C]

    Phan du viet theo he co so do, tach phan cua Y nam ngoai span(U):

        ‖Y − Ŷ‖² = Σ_j (1 − d_j)²·w_j  +  (‖Y‖² − Σ_j w_j)
                                           └─ bang 0 khi n <= E ─┘

    Do duoc: 9.20s -> 0.56s moi task, lambda chon ra trung khop tung truong hop.
    """
    n, E = X.shape
    normY2 = Y.pow(2).sum()

    if n <= E:
        ev, U = torch.linalg.eigh(X @ X.T)
        w = (U.T @ Y).pow(2).sum(1)
    else:
        ev, V = torch.linalg.eigh(X.T @ X)
        s = ev.clamp(min=0).sqrt()
        w = (V.T @ (X.T @ Y)).pow(2).sum(1) / s.pow(2).clamp(min=1e-30)
    S_sq = ev.clamp(min=0)
    perp = (normY2 - w.sum()).clamp(min=0)

    best, best_ridge = None, None
    for ridge in 10.0 ** np.arange(lo, hi):
        diag = S_sq / (S_sq + ridge)
        rss = ((1 - diag).pow(2) * w).sum() + perp
        gcv = (rss / n) / (1 - diag.sum() / n) ** 2
        if best is None or gcv.item() < best:
            best, best_ridge = gcv.item(), ridge
    return best_ridge


def topk_rows(Z, k):
    """Z la [expand_dim, N]; giu k gia tri lon nhat theo CHIEU DAU (per-sample)."""
    _, idx = Z.topk(k, dim=0, largest=True)
    out = torch.zeros_like(Z)
    out.scatter_(0, idx, Z.gather(0, idx))
    return out
