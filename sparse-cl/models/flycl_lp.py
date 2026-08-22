"""Fly-CL voi phep chieu HOC MOT LAN o task 0 roi dong bang vinh vien.

Y tuong lay tu FSA: thu duy nhat trong Fly-CL an duoc bang gradient la mot tang
duoc hoc mot lan roi khoa lai. FSA lam viec do voi backbone (+3.77). Cho con
lai co tham so la phep chieu thua, va no dang hoan toan ngau nhien.

    task 0:    hoc A bang cross-entropy tren 10 lop cua task 0  ->  DONG BANG
    task 0..T: Fly-CL binh thuong tren A(x), G va Q tich luy nhu cu

Tinh hop le cua tich luy VAN GIU: tu luc dong bang tro di dau vao cua ridge co
dinh. Task 0 duoc tinh G, Q bang A BAN CUOI chu khong phai ban dang huan luyen,
nen khong co mau nao duoc ma hoa bang hai phien ban A khac nhau.

CHO DA THAT BAI MOT LAN. `anacp_cp --pos pre` cung la "hoc mot lan roi dong
bang" va mat 43.8 diem: no giai ridge voi dich chi co C hang, ma ridge voi dich
C hang cho ma tran hang <= C, nen 2048 chieu bi ep xuong 9 TRUOC khi mo rong.
Hai rang buoc o day sinh ra tu chan doan do:

    A(x) = x + Up(Down(x)),  Up khoi tao bang 0   ->  A = DONG NHAT luc bat dau
    Down: d -> r,  Up: r -> d,  r nho             ->  cap nhat bi chan hang r

Nen A khong the xoa chieu nao: no chi cong them mot hieu chinh hang r vao phep
dong nhat. Gradient co lam gi thi 2048 chieu goc van con nguyen do.

`--lp_pres` them phat giu feature gan ban goc, cho truong hop 10 lop cua task 0
keo A di qua xa (task 0 chi co 10 lop, 90 lop con lai phai chiu hau qua).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flycl import FlyCL


class Adapter(nn.Module):
    """x + Up(Down(x)), khoi tao dong nhat. Cung thiet ke voi ConvAdapter cua
    fsa.py, chi khac la chay tren vector da pool thay vi ban do dac trung."""

    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


class FlyCLLearnedProj(FlyCL):
    def __init__(self, lp_rank, lp_epochs, lp_lr, lp_bs, lp_wd, lp_pres,
                 classes_per_task, in_dim, **kw):
        super().__init__(**kw)
        self.lp_rank, self.lp_epochs, self.lp_lr = lp_rank, lp_epochs, lp_lr
        self.lp_bs, self.lp_wd, self.lp_pres = lp_bs, lp_wd, lp_pres
        self.classes_per_task = classes_per_task
        # Adapter chi dung vao DUNG lat stage 4 - lat ma Fly-CL that su doc.
        # Neu cho no chay tren ca vector 4 tang thi no co the bom thong tin
        # stage 3 sang o cua stage 4, va thu do lai chinh la concat tra hinh:
        # phep do se khong con tra loi cau hoi "hoc phep chieu co an khong".
        self.dim4 = self.sl4.stop - self.sl4.start
        self.A = Adapter(self.dim4, lp_rank).to(self.device)
        self.fitted = False
        self.stats = {}

    def _adapt(self, X):
        """Thay lat stage 4 bang A(lat stage 4), giu nguyen phan con lai."""
        out = X.clone()
        out[:, self.sl4] = self.A(X[:, self.sl4])
        return out

    # --- huan luyen A, chi chay o task 0 ------------------------------------ #
    def _fit_adapter(self, X, Y):
        C0 = self.classes_per_task
        head = nn.Linear(self.Eb, C0).to(self.device)
        opt = torch.optim.AdamW(list(self.A.parameters()) + list(head.parameters()),
                                lr=self.lp_lr, weight_decay=self.lp_wd)
        n = X.shape[0]
        Yl = Y.long()
        print(f"  [lp] hoc A: rank={self.lp_rank} epochs={self.lp_epochs} "
              f"lr={self.lp_lr:g} pres={self.lp_pres:g} tren {n} mau", flush=True)

        for ep in range(self.lp_epochs):
            perm = torch.randperm(n, device=self.device)
            tot = cnt = 0.0
            for i in range(0, n, self.lp_bs):
                idx = perm[i:i + self.lp_bs]
                xb, yb = X[idx], Yl[idx]
                xa = self._adapt(xb)
                # _code cua FlyCL: chieu thua roi top-k. Ca hai deu kha vi theo
                # xa - top-k la phep chep co mat na, gradient chay qua cac o giu.
                H = self._code(0, xa)
                loss = F.cross_entropy(head(H.T), yb)
                if self.lp_pres:
                    s = self.sl4
                    loss = loss + self.lp_pres * (
                        (xa[:, s] - xb[:, s]).pow(2).sum(1)
                        / xb[:, s].pow(2).sum(1).clamp(min=1e-8)
                    ).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * idx.numel()
                cnt += idx.numel()
            print(f"  [lp] epoch {ep}: loss={tot / cnt:.4f}", flush=True)

        # --- dong bang vinh vien ---
        for p in self.A.parameters():
            p.requires_grad_(False)
        self.A.eval()
        self.fitted = True

        with torch.no_grad():
            x4 = X[:, self.sl4]
            d = (self.A(x4) - x4).norm(dim=1) / x4.norm(dim=1).clamp(min=1e-8)
            self.stats['lp_drift'] = d.mean().item()
        print(f"  [lp] xong, ||A(x)-x||/||x|| = {self.stats['lp_drift']:.4f}",
              flush=True)

    @torch.no_grad()
    def _apply(self, X):
        return self._adapt(X)

    def update(self, X, Y):
        if not self.fitted:
            self._fit_adapter(X, Y)
        super().update(self._apply(X), Y)

    @torch.no_grad()
    def predict(self, X):
        return super().predict(self._apply(X))

    def diagnostics(self):
        return dict(self.stats)
