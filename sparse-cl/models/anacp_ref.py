"""Nap `models/anacp.py` NGUYEN BAN tu repo cua tac gia AnaCP.

Khong sao chep code cua ho vao day, de khong co nghi ngo la ta da sua gi. README
cua ho ghi ro do la "standalone module that can be used to incrementally train a
model on input features", nen chi can cho an feature va nhan.

Muc dich: tach bach hai kha nang giai thich vi sao AnaCP khong an trong khung
Fly-CL. Neu code CUA HO tren feature CUA TA cung khong an thi khong phai do ta
cai lai sai. Do duoc: 84.31 so voi Fly-CL 84.08 - hoa nhau.

    git clone https://github.com/SalehMomeni/AnaCP
    python run.py --method anacp_ref --anacp_path ./AnaCP
"""

import os
import sys

import numpy as np
import torch


class _Adapter:
    """Boc lai de khop giao dien update(X, Y) / predict(X) cua trainer:
    lop cua ho nhan numpy va tra numpy."""

    def __init__(self, inner):
        self.inner = inner

    def update(self, X, Y):
        self.inner.update(X.cpu().numpy(), Y.cpu().numpy())

    def predict(self, X):
        pred = self.inner.predict(X.cpu().numpy())
        return torch.as_tensor(np.asarray(pred), device=X.device)


def load_reference(args):
    path = os.path.abspath(args.anacp_path)
    if not os.path.isfile(os.path.join(path, 'models', 'anacp.py')):
        raise SystemExit('\n'.join([
            f"Khong thay models/anacp.py trong {path}.",
            "Clone repo cua ho roi tro --anacp_path vao do:",
            "    git clone https://github.com/SalehMomeni/AnaCP",
            "    python run.py --method anacp_ref --anacp_path ./AnaCP",
        ]))
    sys.path.insert(0, path)
    from models.anacp import AnaCP          # noqa: E402  nguyen ban cua ho
    return _Adapter(AnaCP(
        D=args.D, reg=args.reg, num_heads=args.num_heads, seed=args.seed,
        device=args.device, samples_per_class=args.samples_per_class,
        shared_cov=True))
