import numpy as np
import torch

from utils import select_ridge_parameter, topk_rows


class FlyCL:
    """Fly-CL: chieu ngau nhien thua -> top-k -> ridge nghiem dong.

    Mo rong so voi ban goc: co the NOI them mot tang backbone truoc khi top-k
    (`deg_s3 > 0`), va co the chay nhieu nhanh doc lap roi cong logit
    (`branches > 1`).

        a = W4 . x4      [E]        W4: [E, 2048], deg_s4 ket noi moi hang
        b = W3 . x3      [E]        W3: [E, 1024], deg_s3 ket noi moi hang
        z = [a ; b]      [2E]   ->  top-k  ->  ridge

    `deg_s3 = 0` tai lap CHINH XAC Fly-CL goc - cung thu tu tieu thu RNG, cung
    ket qua den tung chu so.

    CAP PHAT KET NOI TUONG MINH. Moi hang nhan dung `deg_s4` ket noi vao stage 4
    va `deg_s3` vao stage 3, doc lap nhau. Cach nguy hiem la ghep cot roi rut
    `deg_s4` cot tren toan bo 3072: khi do stage 4 bi ha tu 300 xuong ~200 de
    nhuong cho stage 3, va con so do duoc thanh "loi cua stage 3 TRU hai do lam
    yeu stage 4".

    `G` va `Q` la thong ke du, cong don qua cac task, nen nghiem o task t bang
    DUNG nghiem khi gop toan bo du lieu task 1..t - khong xap xi, khong phu
    thuoc thu tu task.
    """

    STAGE_DIMS = (256, 512, 1024, 2048)

    def __init__(self, num_classes, expand_dim, coding_level, deg_s4, deg_s3,
                 w_s3, b_stage, branches, ridge_lower, ridge_upper, seed,
                 stage_norms, device, in_dim=None, stage_dims=None):
        self.device = device
        self.C = num_classes
        self.k_ratio = coding_level
        self.deg_s3 = deg_s3
        self.w_s3 = w_s3
        self.lo, self.hi = ridge_lower, ridge_upper

        dims = tuple(stage_dims or self.STAGE_DIMS)
        off = np.cumsum((0,) + dims)
        if in_dim is not None and in_dim != off[4]:
            # Backbone khong phai ResNet bon stage - vd ViT tra ve mot vector
            # CLS 768 chieu. Khi do CA vector dong vai tro "stage 4" va khong
            # co tang nao de noi, nen `deg_s3` bat buoc bang 0.
            if deg_s3:
                raise ValueError(
                    f"concat can feature bon stage (+ms), o day d={in_dim}")
            self.sl4 = slice(0, in_dim)
            self.sl_b, self.scale_b = [], []
            n4, n3 = in_dim, 1
        else:
            self.sl4 = slice(off[3], off[4])
            self.sl_b = [slice(off[i - 1], off[i]) for i in b_stage]
            # Dua khoi b ve thang cua stage 4 truoc, de `w_s3` la he so tuong
            # doi thuan tuy chu khong lan voi chenh lech do lon giua cac tang.
            self.scale_b = [stage_norms[3] / stage_norms[i - 1] for i in b_stage]
            n4 = dims[3]
            n3 = sum(dims[i - 1] for i in b_stage)
        E = expand_dim
        self.Eb = E * 2 if deg_s3 else E
        self.k = int(self.Eb * coding_level)

        from data_loader import set_seed
        self.W4s, self.W3s, self.G, self.Q = [], [], [], []
        for e in range(branches):
            set_seed(seed + e * 1000)
            W4 = torch.zeros(E, n4)
            W3 = torch.zeros(E, n3)
            for r in range(E):
                # Hai dong rieng. Viet gop `W[r, randperm(d)[:n]] = randn(n)` thi
                # Python danh gia ve PHAI truoc, thu tu tieu thu RNG bi dao va ma
                # tran chieu KHAC HAN du cung seed. Khong co loi nao duoc bao.
                pick = torch.randperm(n4)[:deg_s4]
                W4[r, pick] = torch.randn(deg_s4)
                if deg_s3:
                    pick = torch.randperm(n3)[:deg_s3]
                    W3[r, pick] = torch.randn(deg_s3)
            self.W4s.append(W4.to(device))
            self.W3s.append(W3.to(device))
            self.G.append(torch.zeros(self.Eb, self.Eb, device=device))
            self.Q.append(torch.zeros(self.Eb, self.C, device=device))
        self.Wo = [None] * branches
        self.branches = branches

    def _split(self, X):
        x4 = X[:, self.sl4]
        if not self.deg_s3:
            return x4, None
        x3 = torch.cat([X[:, s] * c for s, c in zip(self.sl_b, self.scale_b)], 1)
        return x4, x3

    def _code(self, e, X):
        x4, x3 = self._split(X)
        z = self.W4s[e] @ x4.T
        if self.deg_s3:
            z = torch.cat([z, self.w_s3 * (self.W3s[e] @ x3.T)])
        return topk_rows(z, self.k)

    def update(self, X, Y):
        Y1h = torch.zeros(Y.shape[0], self.C, device=self.device)
        Y1h.scatter_(1, Y.long().view(-1, 1), 1.0)
        for e in range(self.branches):
            H = self._code(e, X)
            self.Q[e] += H @ Y1h
            self.G[e] += H @ H.T
            ridge = select_ridge_parameter(H.T, Y1h, self.lo, self.hi)
            del H
            # `G + ridge*eye` phai dung ba ma tran Eb x Eb cung luc: eye, ban
            # tam, va dau ra. O concat + ensemble 5 thi moi cai la 1.6 GB tren
            # nen G da chiem 8 GB. Cong thang vao duong cheo cua ban sao thi chi
            # con hai, va bo han duoc `eye` thuong tru.
            Gl = self.G[e].clone()
            Gl.diagonal().add_(ridge)
            self.Wo[e] = torch.cholesky_solve(self.Q[e], torch.linalg.cholesky(Gl))
            del Gl
        self.last_ridge = float(ridge)

    @torch.no_grad()
    def predict(self, X):
        logit = 0
        for e in range(self.branches):
            logit = logit + self._code(e, X).T @ self.Wo[e]
        return logit.argmax(1)
