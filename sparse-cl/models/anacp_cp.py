import torch
import torch.nn.functional as F

from utils import select_ridge_parameter, topk_rows


# --------------------------------------------------------------------------- #
# Tang Contrastive Projection cua AnaCP. Gom dung ba thu:
#   (1) thong ke o tang input: mu_c, Sigma (within-class)
#   (2) sinh dich P: whiten -> SVD -> noi tri ky di -> de-whiten -> chuan hoa
#   (3) mot phep ridge voi dich P thay cho one-hot
# Phan "contrastive" nam TOAN BO o buoc (2); (1) va (3) la ha tang.
# --------------------------------------------------------------------------- #

def whiten_matrices(Sigma, eps):
    """Tra ve (Sigma^-1/2, Sigma^+1/2, cond). Clamp eigenvalue chu KHONG cong
    1e-8 nhu code AnaCP goc - voi d=2048 cua ResNet thi cong khong du."""
    S = 0.5 * (Sigma + Sigma.T)
    lam, U = torch.linalg.eigh(S)
    cond = (lam.max() / lam.clamp(min=1e-30).min()).item()
    lam = torch.clamp(lam, min=eps)
    return U @ torch.diag(lam.rsqrt()) @ U.T, U @ torch.diag(lam.sqrt()) @ U.T, cond


def compute_delta(Wm, Em):
    """Lemma 4.1 cua paper. Wm, Em: (C,C), cot i la w_i, e_i.

        Phi_i = sum_{j!=i} [ (2*1[<w_i,w_j> >= 0] - 1)*<e_i,w_j>*||w_i||^2
                             - <e_i,w_i>*|<w_i,w_j>| ] / ||w_j||
        delta_i = -sign(Phi_i)

    Code AnaCP KHONG co ham nay - no dung `S + 1`, tuc ep delta = +1 cho moi lop.
    Do tren CIFAR-100 + ResNet-50 thi delta = +1 o 550/550 truong hop, nen hai
    cach cho ket qua TRUNG KHOP tung chu so.
    """
    norms = Wm.norm(dim=0)
    Gm = Wm.T @ Wm
    EW = Em.T @ Wm
    sign = torch.where(Gm >= 0, 1.0, -1.0)
    A = sign * EW * norms.pow(2).unsqueeze(1)
    B = EW.diagonal().unsqueeze(1) * Gm.abs()
    Phi = (A - B) / norms.unsqueeze(0)
    Phi.fill_diagonal_(0)
    return torch.sign(-Phi.sum(dim=1))


def make_P(class_means, Sigma, mode, dewhiten, alpha, eps, l2=True, add_mu=True):
    """class_means (C,d) -> P (C,d).

    mode      repo   = `S + alpha` nhu code AnaCP;  paper = `S + alpha*diag(delta)`
    dewhiten  inv    = nhan Sigma^(-1/2) LAN NUA nhu code AnaCP
              correct= nhan Sigma^(+1/2) dung Eq. 16 cua paper
    add_mu    False  = khong cong lai mean toan cuc. mu_bar la thanh phan CHUNG
              cua moi prototype nen no bom cosine giua chung len.
    """
    if mode == 'none':
        P = class_means.clone()
        return (F.normalize(P, dim=1) if l2 else P), float('nan')

    Sinv, Shalf, cond = whiten_matrices(Sigma, eps)
    mu_bar = class_means.mean(0, keepdim=True)
    Chat = (class_means - mu_bar) @ Sinv                     # Sigma doi xung
    U, S, Vh = torch.linalg.svd(Chat, full_matrices=False)

    if mode == 'repo':
        S_new = S + alpha
    elif mode == 'paper':
        Em = U.T
        S_new = S + alpha * compute_delta(torch.diag(S) @ Em, Em)
    else:
        raise ValueError(mode)

    P = U @ torch.diag(S_new) @ Vh @ (Sinv if dewhiten == 'inv' else Shalf)
    if add_mu:
        P = P + mu_bar
    return (F.normalize(P, dim=1) if l2 else P), cond


def mean_abs_cos(P):
    """Trung binh |cos| giua cac prototype = trung binh tri tuyet doi phan NGOAI
    duong cheo cua K = P P^T. Do luon DO LON TAC DUNG cua tang CP: bang 0 nghia
    la K = I nghia la khong lam gi."""
    Pn = F.normalize(P, dim=1)
    C = P.shape[0]
    return ((Pn @ Pn.T).abs().sum() - C).item() / (C * C - C)


# --------------------------------------------------------------------------- #

class AnaCPProjection:
    """Tang CP cua AnaCP ghep vao Fly-CL, dat o mot trong bon vi tri.

        none    Fly-CL nguyen ban, khong CP                          <- moc
        post    CP SAU top-k, doc bang NCM  (dung cho AnaCP dat)
        pre     CP TRUOC phep chieu thua, dong bang sau task 0
        whiten  chi lay Sigma^(-1/2), khong dung dich P

    O vi tri `post` khong con phi tuyen nao sau CP, nen ca tang rut gon thanh
    MOT PHEP NHAN LOGIT:

        Wo_cp = (G + lam I)^-1 Q P = Wo_flycl . P
        diem moi = argmax_c (h Wo_cp) . p_c = [logit Fly-CL] . K,  K = P P^T

    Script tu chay song song ca hai duong va bao ti le trung khop (`agree`) -
    do duoc 100.0% o 16 lan do. `K = I` thi bang dung Fly-CL, ma negative
    repulsion lai day K ve I, nen cang tach gioi cang mat tac dung.
    """

    def __init__(self, num_classes, expand_dim, coding_level, synaptic_degree,
                 pos, spread, dewhiten, alpha, shrink, nomu, input_norm,
                 ridge_lower, ridge_upper, cp_ridge_lower, cp_ridge_upper,
                 classes_per_task, in_dim, seed, device):
        from data_loader import set_seed
        self.__dict__.update(locals())
        del self.self
        self.C, self.d = num_classes, in_dim
        set_seed(seed)
        W = torch.zeros(expand_dim, in_dim)
        for r in range(expand_dim):
            pick = torch.randperm(in_dim)[:synaptic_degree]   # 2 dong: thu tu RNG
            W[r, pick] = torch.randn(synaptic_degree)
        self.W = W.to(device)
        self.k = int(expand_dim * coding_level)
        self.E = expand_dim

        self.G = torch.zeros(self.E, self.E, device=device)
        self.Q = torch.zeros(self.E, self.C, device=device)
        self.eye = torch.eye(self.E, device=device)
        self.mu = torch.zeros(self.C, self.d, device=device)
        self.ncls = torch.zeros(self.C, device=device)
        self.Sigma = torch.zeros(self.d, self.d, device=device)
        self.Gx = torch.zeros(self.d, self.d, device=device)
        self.Qx = torch.zeros(self.d, self.C, device=device)
        self.Ntot = 0.0
        self.W_CP = self.P = self.Wo = self.Wo_cp = self.Wo_fly = self.K = None
        self.seen = 0
        self.agree = []
        self.cond = float('nan')

    # --- buoc (1): thong ke o tang INPUT, truoc moi phep chieu ---
    def _stats(self, X, Y, lo, hi):
        cent = torch.empty_like(X)
        for c in range(lo, hi):
            m = Y == c
            n = int(m.sum())
            if not n:
                continue
            self.mu[c] = (self.mu[c] * self.ncls[c] + X[m].sum(0)) / (self.ncls[c] + n)
            self.ncls[c] += n
            cent[m] = X[m] - self.mu[c]
            self.Qx[:, c] += X[m].sum(0)
        nb = X.shape[0]
        Sn = cent.T @ cent / nb
        self.Sigma = (Sn + 1e-4 * torch.eye(self.d, device=self.device)
                      if self.Ntot == 0
                      else (self.Sigma * self.Ntot + Sn * nb) / (self.Ntot + nb))
        self.Ntot += nb
        self.Gx += X.T @ X

    def _pre_project(self, X):
        return X @ self.W_CP if self.pos in ('pre', 'whiten') else X

    def update(self, X, Y):
        if self.input_norm:
            X = F.normalize(X, dim=1)
        lo, self.seen = self.seen, self.seen + self.classes_per_task
        if self.pos == 'post' or (self.pos in ('pre', 'whiten') and self.W_CP is None):
            self._stats(X, Y, lo, self.seen)

        if self.pos == 'pre' and self.W_CP is None:
            self.P, self.cond = make_P(self.mu[:self.seen], self.Sigma, self.spread,
                                       self.dewhiten, self.alpha, self.shrink)
            lx = select_ridge_parameter(X, self.P[Y.long()],
                                        self.cp_ridge_lower, self.cp_ridge_upper)
            self.W_CP = torch.cholesky_solve(
                self.Qx[:, :self.seen] @ self.P,
                torch.linalg.cholesky(self.Gx + float(lx) * torch.eye(self.d, device=self.device)))
        elif self.pos == 'whiten' and self.W_CP is None:
            self.W_CP, _, self.cond = whiten_matrices(self.Sigma, self.shrink)

        X = self._pre_project(X)
        H = topk_rows(self.W @ X.T, self.k)
        Y1h = torch.zeros(Y.shape[0], self.C, device=self.device)
        Y1h.scatter_(1, Y.long().view(-1, 1), 1.0)
        self.Q += H @ Y1h
        self.G += H @ H.T

        if self.pos == 'post':
            self.P, self.cond = make_P(self.mu[:self.seen], self.Sigma, self.spread,
                                       self.dewhiten, self.alpha, self.shrink,
                                       add_mu=not self.nomu)
            ridge = select_ridge_parameter(H.T, self.P[Y.long()],
                                           self.ridge_lower, self.ridge_upper)
            L = torch.linalg.cholesky(self.G + float(ridge) * self.eye)
            self.Wo_cp = torch.cholesky_solve(self.Q[:, :self.seen] @ self.P, L)
            self.Wo_fly = torch.cholesky_solve(self.Q[:, :self.seen], L)   # cung lambda
            self.K = self.P @ self.P.T
        else:
            ridge = select_ridge_parameter(H.T, Y1h, self.ridge_lower, self.ridge_upper)
            self.Wo = torch.cholesky_solve(
                self.Q, torch.linalg.cholesky(self.G + float(ridge) * self.eye))
        self.last_ridge = float(ridge)

    @torch.no_grad()
    def predict(self, X):
        if self.input_norm:
            X = F.normalize(X, dim=1)
        He = topk_rows(self.W @ self._pre_project(X).T, self.k)
        if self.pos != 'post':
            return (He.T @ self.Wo).argmax(1)
        pred = torch.cdist(He.T @ self.Wo_cp, self.P).argmin(1)     # NCM
        predk = ((He.T @ self.Wo_fly) @ self.K).argmax(1)           # logits @ K
        self.agree.append((pred == predk).float().mean().item())
        return pred

    def diagnostics(self):
        import numpy as np
        return {'mean_abs_cos': mean_abs_cos(self.P) if self.P is not None else float('nan'),
                'cond_Sigma': self.cond,
                'post_eq_postk': float(np.mean(self.agree)) * 100 if self.agree else float('nan')}
