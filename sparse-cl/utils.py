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


def select_ridge_accumulated(G, Q, n, lo, hi, probes=64, seed=0):
    """GCV tren TOAN BO du lieu da thay, chi tu thong ke du (G, Q, n).

    Ban `select_ridge_parameter` chon lambda tren MOT task: `models/flycl.py`
    goi no voi H cua task hien tai (n = 5.000), roi dem lambda ay giai voi G da
    cong don (n = 5.000 t). Do la mot cho ho trong chinh phat bieu cua Fly-CL -
    nghiem W cho truoc lambda thi dung bang nghiem joint, nhung lambda lai khong
    duoc chon nhu nghiem joint. Do duoc: GCV chon 10^4 o CA MUOI task, trong khi
    tri rieng cua G lon len muoi lan (`logs/r50_why.txt`).

    Chua duoc bang cach chieu ca hai so hang cua GCV ve (G, Q):

        Z = (G + lam I)^-1 Q                              [E, C]
        ‖(I−A)Y‖²_F = n − 2·tr(Qᵀ Z) + tr(Zᵀ G Z)        CHINH XAC
        tr A(lam)   = E − lam · tr( (G + lam I)^-1 )

    So hang dau chinh xac va gan nhu mien phi. So hang sau la vet cua ma tran
    nghich dao - tinh thang thi ton E³ (do duoc 2.8s o E=10.000, 18.6s o
    E=20.000, dat ngang eigh nen vo nghia). Uoc luong Hutchinson bang vai chuc
    vector Rademacher thi chi ton them may phep `cholesky_solve` tren DUNG phan
    ra da co:

        tr( (G + lam I)^-1 ) ≈ (1/m) Σ_i z_iᵀ (G + lam I)^-1 z_i,   z_i ~ ±1

    `‖Y‖²_F = n` vi Y la one-hot, nen ham nay khong can den Y - hoan toan
    exemplar-free, va bat bien theo thu tu task nhu phan con lai cua phuong phap.

    KET QUA DO: cho hieu ung DUNG NHU CHAN DOAN, nhung LAM TE DI.

        lam_mode=task   lam = 1e4 o ca 10 task        A_bar 84.08
        lam_mode=accum  lam = 1e4 -> 1e5 tu task 1    A_bar 83.48   (−0.60)

    Ban accum doi hoi siet manh hon gap 10 khi n tang 10 lan - dung nhu ly
    thuyet. Nhung quet lam co dinh cho thay dinh do chinh xac nam o 1e4:

        1e2  81.51 | 1e3  82.97 | 1e4  84.08 | 1e5  83.43 | 1e6  80.64

    GCV toi thieu hoa SAI SO BINH PHUONG, con cai minh do la DO CHINH XAC cua
    argmax. Hai thu do khong cung mot dinh: duoi-siet lam nghiem bam du lieu hon
    nhung thu tu cac logit van dung. Nen lambda hien tai cua Fly-CL - tuy khong
    nhat quan ve mat ly thuyet - lai roi dung vao dinh do chinh xac, va khong
    con gi de gianh o day.

    Giu ham nay lam doi chung, khong dat lam mac dinh.
    """
    E = G.shape[0]
    eye = torch.eye(E, device=G.device, dtype=G.dtype)
    gen = torch.Generator(device=G.device).manual_seed(seed)
    Zp = torch.randint(0, 2, (E, probes), generator=gen, device=G.device,
                       dtype=G.dtype).mul_(2).sub_(1)

    best, best_ridge = None, None
    for ridge in 10.0 ** np.arange(lo, hi):
        L = torch.linalg.cholesky(G + float(ridge) * eye)
        Z = torch.cholesky_solve(Q, L)
        rss = n - 2 * (Q * Z).sum() + (Z * (G @ Z)).sum()
        tr_inv = (Zp * torch.cholesky_solve(Zp, L)).sum() / probes
        df = E - float(ridge) * tr_inv
        denom = (1 - df / n) ** 2
        if denom <= 0:
            continue
        gcv = (rss / n) / denom
        if best is None or gcv.item() < best:
            best, best_ridge = gcv.item(), ridge
    return best_ridge


def topk_rows(Z, k):
    """Z la [expand_dim, N]; giu k gia tri lon nhat theo CHIEU DAU (per-sample)."""
    _, idx = Z.topk(k, dim=0, largest=True)
    out = torch.zeros_like(Z)
    out.scatter_(0, idx, Z.gather(0, idx))
    return out
