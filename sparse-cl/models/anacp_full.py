import torch
import torch.nn.functional as F

from utils import select_ridge_parameter, topk_rows
from .anacp_cp import make_P, mean_abs_cos


def _nl(Z, kind, k):
    if kind == 'topk':
        return topk_rows(Z, k)
    if kind == 'gelu':
        return F.gelu(Z)
    raise ValueError(kind)


class AnaCPFull:
    """AnaCP DAY DU: hai tang ridge, phi tuyen o giua, pseudo-replay.

        x --nl1--> H --CP(theta_h)--> L2norm --> trung binh H head --> u
                                                                       |
                                                          nl2(W2 . u)  <- PHI TUYEN
                                                                       |
                                                          ELM one-hot --> argmax

    Khac ban rut gon (`anacp_cp --pos post`) o dung mot cho: co tang thu HAI, va
    giua chung co phi tuyen. Do la ly do ban nay KHONG sup thanh `logits @ K`.

    Cai gia: dau ra tang CP doi moi task nen G2/Q2 khong tich luy duoc, phai
    dung lai tu dau moi task bang mau gia `x~ ~ N(mu_c, Sigma)`. Hop le vi mu_c
    va Sigma nam o tang input, sau backbone dong bang, nen khong bao gio loi thoi.

    Tinh bat bien theo thu tu task VAN duoc giu (mu_c, Sigma, G_h, Q_h deu la
    tong theo mau; G2/Q2 dung lai tu replay cua MOI lop da gap). Cai mat la tinh
    CHINH XAC: tang hai train tren du lieu Gauss chu khong phai du lieu that.

    `--nl2 none` bo tang hai va doc bang NCM -> phai trung khop `anacp_cp
    --pos post`. Do la phep tu kiem tra cua file nay.
    """

    def __init__(self, num_classes, expand_dim, expand2, coding_level,
                 synaptic_degree, heads, nl1, nl2, replay, spread, dewhiten,
                 alpha, shrink, ridge_lower, ridge_upper, classes_per_task,
                 in_dim, seed, device):
        from data_loader import set_seed
        self.__dict__.update(locals())
        del self.self
        self.C, self.d = num_classes, in_dim
        self.k = int(expand_dim * coding_level)
        self.k2 = int(expand2 * coding_level)

        def sparse_W(rows, cols, sd):
            set_seed(sd)
            W = torch.zeros(rows, cols)
            for r in range(rows):
                pick = torch.randperm(cols)[:min(synaptic_degree, cols)]
                W[r, pick] = torch.randn(pick.numel())
            return W.to(device)

        self.Ws = [sparse_W(expand_dim, in_dim, seed + 1000 * h) for h in range(heads)]
        self.W2 = sparse_W(expand2, in_dim, seed + 999_000)
        self.G = [torch.zeros(expand_dim, expand_dim, device=device) for _ in range(heads)]
        self.Q = [torch.zeros(expand_dim, self.C, device=device) for _ in range(heads)]
        self.eyeE = torch.eye(expand_dim, device=device)
        self.eyeE2 = torch.eye(expand2, device=device)
        self.mu = torch.zeros(self.C, self.d, device=device)
        self.ncls = torch.zeros(self.C, device=device)
        self.Sigma = torch.zeros(self.d, self.d, device=device)
        self.Ntot = 0.0
        self.seen = 0
        self.theta = self.th2 = self.P = None

    def _cp_forward(self, X):
        out = 0
        for h in range(self.heads):
            Z = _nl(self.Ws[h] @ X.T, self.nl1, self.k)
            out = out + F.normalize(Z.T @ self.theta[h], dim=1)
            del Z
        return out / self.heads

    def update(self, X, Y):
        lo, self.seen = self.seen, self.seen + self.classes_per_task
        cent = torch.empty_like(X)
        for c in range(lo, self.seen):
            m = Y == c
            n = int(m.sum())
            self.mu[c] = (self.mu[c] * self.ncls[c] + X[m].sum(0)) / (self.ncls[c] + n)
            self.ncls[c] += n
            cent[m] = X[m] - self.mu[c]
        nb = X.shape[0]
        Sn = cent.T @ cent / nb
        self.Sigma = (Sn + 1e-4 * torch.eye(self.d, device=self.device)
                      if self.Ntot == 0
                      else (self.Sigma * self.Ntot + Sn * nb) / (self.Ntot + nb))
        self.Ntot += nb
        del cent

        self.P, _ = make_P(self.mu[:self.seen], self.Sigma, self.spread,
                           self.dewhiten, self.alpha, self.shrink)

        # --- tang 1: tich luy roi giai LAI voi P moi nhat ---
        self.theta = []
        Y1h = torch.zeros(Y.shape[0], self.C, device=self.device)
        Y1h.scatter_(1, Y.long().view(-1, 1), 1.0)
        for h in range(self.heads):
            H = _nl(self.Ws[h] @ X.T, self.nl1, self.k)
            self.Q[h] += H @ Y1h
            self.G[h] += H @ H.T
            lam = select_ridge_parameter(H.T, self.P[Y.long()],
                                         self.ridge_lower, self.ridge_upper)
            self.theta.append(torch.cholesky_solve(
                self.Q[h][:, :self.seen] @ self.P,
                torch.linalg.cholesky(self.G[h] + float(lam) * self.eyeE)))
            del H
        del Y1h

        if self.nl2 == 'none':
            return

        # --- pseudo-replay tu N(mu_c, Sigma) o tang input ---
        Ls = torch.linalg.cholesky(self.Sigma + 1e-5 * torch.eye(self.d, device=self.device))
        yr = torch.arange(self.seen, device=self.device).repeat_interleave(self.replay)
        xr = torch.randn(yr.shape[0], self.d, device=self.device) @ Ls.T + self.mu[yr]

        # --- tang 2: DUNG LAI tu dau, khong tich luy ---
        H2 = _nl(self.W2 @ self._cp_forward(xr).T, self.nl2, self.k2)
        Y2 = torch.zeros(yr.shape[0], self.C, device=self.device)
        Y2.scatter_(1, yr.view(-1, 1), 1.0)
        lam2 = select_ridge_parameter(H2.T, Y2, self.ridge_lower, self.ridge_upper)
        self.th2 = torch.cholesky_solve(
            H2 @ Y2, torch.linalg.cholesky(H2 @ H2.T + float(lam2) * self.eyeE2))
        del H2, Y2, xr

    @torch.no_grad()
    def predict(self, X):
        if self.nl2 == 'none':
            return torch.cdist(self._cp_forward(X), self.P).argmin(1)
        He = _nl(self.W2 @ self._cp_forward(X).T, self.nl2, self.k2)
        return (He.T @ self.th2).argmax(1)

    def diagnostics(self):
        return {'mean_abs_cos': mean_abs_cos(self.P)}
