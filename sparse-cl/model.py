"""
Model: backbone -> chieu thua hoc duoc -> top-k -> (tuy chon MLP) -> classifier.

    SparseProjection   ma tran thua, mask co dinh, chi hoc gia tri
    TopK               winner-take-all + nguong thich nghi chong unit chet
    IncrementalLinear  classifier no ra sau moi task (giu nguyen hang cu)
    Regularizer        EWC / EWC-DR de kiem soat troi cua phep chieu

Moi flag deu den tu config.py.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Chieu thua
# --------------------------------------------------------------------------- #

class SparseProjection(nn.Module):
    """Chieu [B, in_dim] -> [B, out_dim] qua ma tran thua.

    mask co dinh tu luc khoi tao; nhan mask trong forward nen gradient tu dong
    bang 0 tai cac vi tri bi che - khong can can thiep vao optimizer.
    """

    def __init__(self, in_dim, out_dim, synaptic_degree=300,
                 sparse_mask=True, trainable=True, bias='none'):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.bias_mode = bias

        if sparse_mask:
            degree = min(synaptic_degree, in_dim)
            mask = torch.zeros(out_dim, in_dim, dtype=torch.bool)
            for row in range(out_dim):
                mask[row, torch.randperm(in_dim)[:degree]] = True
        else:
            degree = in_dim
            mask = torch.ones(out_dim, in_dim, dtype=torch.bool)

        # 1/sqrt(degree): moi hang cung degree nen KHONG doi thu tu top-k,
        # chi de phuong sai kich hoat on dinh khi train.
        w = torch.randn(out_dim, in_dim) / math.sqrt(degree)
        self.weight = nn.Parameter(w * mask, requires_grad=trainable)
        self.register_buffer('mask', mask)
        # ban sao luc khoi tao, chi de do muc troi. persistent=False -> di theo
        # .to(device) nhung khong nam trong state_dict (khoi nang deepcopy).
        if trainable:
            self.register_buffer('w_init', (w * mask).clone(), persistent=False)
        else:
            self.w_init = None

        # bias = nguong kich hoat cua tung neuron (tuong ung uc che APL o ruoi).
        # 'learn' khoi tao BANG 0 -> luc bat dau y het 'none', so sanh sach.
        if bias == 'learn':
            self.bias = nn.Parameter(torch.zeros(out_dim))
        elif bias == 'fixed':
            self.register_buffer('bias', torch.randn(out_dim) / math.sqrt(degree))
        elif bias == 'none':
            self.bias = None
        else:
            raise ValueError(f"proj_bias khong hop le: {bias}")

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    @torch.no_grad()
    def drift(self):
        """||W - W0|| / ||W0||. Gan 0 nghia la phep chieu khong hoc duoc gi -
        luc do cau hinh 'hoc chieu' that ra dong nhat voi 'chieu dong bang'."""
        if self.w_init is None:
            return 0.0
        W = self.weight * self.mask
        return ((W - self.w_init).norm() / (self.w_init.norm() + 1e-12)).item()

    @property
    def nnz(self):
        return int(self.mask.sum().item())

    def to_sparse(self):
        """Cho pha inference: doi sang CSR de dung sparse matmul."""
        return (self.weight * self.mask).detach().to_sparse_csr()


# --------------------------------------------------------------------------- #
# Top-k winner-take-all
# --------------------------------------------------------------------------- #

class TopK(nn.Module):
    """Giu k gia tri lon nhat moi mau, phan con lai bang 0.

    Gradient chay nguoc qua dung k duong duoc chon (giong max-pool).

    adaptive_threshold: giu mot bias theo unit, unit thang qua nhieu thi bi ha
    diem, unit lau khong thang thi duoc nang. Bias chi anh huong VIEC CHON,
    gia tri xuat ra van la gia tri goc.
    """

    def __init__(self, dim, coding_level, adaptive=False,
                 ema=0.99, hom_lr=0.01, track=True):
        super().__init__()
        self.dim = dim
        self.k = max(1, int(round(dim * coding_level)))
        self.target = self.k / dim
        self.adaptive = adaptive
        self.ema, self.hom_lr, self.track = ema, hom_lr, track

        self.register_buffer('usage', torch.full((dim,), self.target))
        self.register_buffer('bias', torch.zeros(dim))
        self.register_buffer('seen', torch.zeros((), dtype=torch.long))

    def forward(self, z):
        if self.k >= self.dim:                       # coding_level = 1.0 -> khong lam gi
            return z

        scores = z + self.bias if self.adaptive else z
        _, idx = scores.topk(self.k, dim=-1)

        out = torch.zeros_like(z)
        out.scatter_(-1, idx, z.gather(-1, idx))     # xuat gia tri GOC

        if self.training and self.track:
            with torch.no_grad():
                hit = torch.zeros_like(z)
                hit.scatter_(-1, idx, 1.0)
                freq = hit.mean(dim=0)
                self.usage.mul_(self.ema).add_(freq, alpha=1 - self.ema)
                self.seen += z.shape[0]
                if self.adaptive:
                    # dung nhieu -> ha diem; it dung -> nang diem
                    self.bias.add_(-self.hom_lr * (self.usage - self.target))
        return out

    # ---------------- chan doan unit chet ----------------

    def usage_stats(self):
        u = self.usage
        return {
            'dead_frac':  (u < self.target * 0.01).float().mean().item(),
            'usage_cv':   (u.std() / (u.mean() + 1e-12)).item(),
            'usage_max':  u.max().item(),
            'usage_min':  u.min().item(),
            'target':     self.target,
        }

    def reset_usage(self):
        self.usage.fill_(self.target)
        self.seen.zero_()


def load_balance_loss(z):
    """CV^2 cua muc kich hoat trung binh moi unit. Kha vi, day ve phan bo deu."""
    imp = z.abs().mean(dim=0)
    return imp.var() / (imp.mean() ** 2 + 1e-12)


# --------------------------------------------------------------------------- #
# Classifier no duoc
# --------------------------------------------------------------------------- #

class IncrementalLinear(nn.Module):
    """Linear co the them hang cho lop moi ma giu nguyen hang cu."""

    def __init__(self, in_features, out_features=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.use_bias = bias
        self.weight, self.bias = None, None
        if out_features > 0:
            self.expand(out_features)

    def expand(self, out_features):
        w = torch.empty(out_features, self.in_features)
        nn.init.kaiming_uniform_(w, nonlinearity='linear')
        b = torch.zeros(out_features) if self.use_bias else None

        if self.weight is not None:
            old = self.weight.shape[0]
            with torch.no_grad():
                w[:old] = self.weight.data
                if b is not None:
                    b[:old] = self.bias.data
            w, b = w.to(self.weight.device), (b.to(self.weight.device) if b is not None else None)

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b) if b is not None else None
        self.out_features = out_features

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class SparseExpandCL(nn.Module):

    def __init__(self, args, backbone=None):
        super().__init__()
        self.args = args
        self.backbone = backbone
        if backbone is not None and args.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.eval()

        # expand_dim = 0 -> BO HAN projection va top-k, dua feature thang vao head.
        # Day la baseline "khong mo rong" (linear/MLP probe tren feature goc).
        self.no_expand = args.expand_dim == 0
        if self.no_expand:
            self.proj, self.act = None, None
            feat_dim = args.embedding_dim
        else:
            self.proj = SparseProjection(
                args.embedding_dim, args.expand_dim,
                synaptic_degree=args.synaptic_degree,
                sparse_mask=args.sparse_mask,
                trainable=args.train_projection,
                bias=args.proj_bias,
            )
            self.act = TopK(
                args.expand_dim, args.coding_level,
                adaptive=args.adaptive_threshold,
            )
            feat_dim = args.expand_dim

        if args.use_mlp:
            self.mlp = nn.Linear(feat_dim, args.mlp_hidden)
            self.mlp_act = (nn.ReLU() if args.mlp_act == 'relu'
                            else TopK(args.mlp_hidden, args.mlp_coding_level,
                                      adaptive=args.adaptive_threshold))
            self.drop = nn.Dropout(args.mlp_dropout)
            head_in = args.mlp_hidden
        else:
            self.mlp = self.mlp_act = self.drop = None
            head_in = feat_dim

        self.head = IncrementalLinear(head_in)
        self.aux_loss = torch.zeros(())
        # dat boi train.py o duong anh: uint8 32x32 -> float 224 chuan hoa.
        # De o day de MOI cho goi model() deu duoc xu ly, ke ca Regularizer.
        self.prep = None

    # ---------------- forward ----------------

    def encode(self, x, is_feature=False, dense=False):
        """Tra ve ma thua (hoac dense neu tat top-k).

        dense=True: BO QUA top-k. Dung khi uoc luong do quan trong cho EWC/EWC-DR,
        vi neu de top-k thi ~90% unit khong nhan gradient va omega=0 vi ly do
        CAU TRUC chu khong phai vi khong quan trong.
        """
        if is_feature:
            f = x
        else:
            if self.backbone is None:
                raise RuntimeError("Khong co backbone; truyen feature va dat is_feature=True.")
            if self.prep is not None:
                x = self.prep(x)
            # autocast CHI bao quanh backbone: phan chieu/top-k/head giu nguyen fp32
            # de so sanh duoc voi cac run dung feature cache.
            dt = getattr(self.args, 'amp_dtype', None)
            with torch.autocast('cuda', dtype=dt or torch.float16,
                                enabled=dt is not None and x.is_cuda):
                if self.args.freeze_backbone:
                    with torch.no_grad():
                        f = self.backbone(x)
                else:
                    f = self.backbone(x)
            f = f.float()

        if self.no_expand:
            self.aux_loss = torch.zeros((), device=f.device)
            return f

        z = self.proj(f)

        if self.training and self.args.load_balance_coef > 0:
            self.aux_loss = self.args.load_balance_coef * load_balance_loss(z)
        else:
            self.aux_loss = torch.zeros((), device=z.device)

        return z if dense else self.act(z)

    def forward(self, x, is_feature=False, dense=False):
        h = self.encode(x, is_feature=is_feature, dense=dense)
        if self.mlp is not None:
            h = self.drop(self.mlp_act(self.mlp(h)))
        return self.head(h)

    def train(self, mode=True):
        super().train(mode)
        # Backbone dong bang phai o eval ke ca trong model.train(): neu khong,
        # BatchNorm cua ResNet van cap nhat running stats -> feature troi am tham
        # du moi tham so deu requires_grad=False.
        if self.backbone is not None and self.args.freeze_backbone:
            self.backbone.eval()
        return self

    # ---------------- vong doi task ----------------

    def expand_head(self, total_classes):
        self.head.expand(total_classes)

    def freeze_projection(self):
        """Dung cho projection_schedule = task0 / offline."""
        if self.no_expand:
            return
        for p in self.proj.parameters():
            p.requires_grad = False
        self.act.track = False

    def param_groups(self):
        args = self.args
        groups = [{'params': self.head.parameters(), 'lr': args.lr}]
        # gom ca weight lan bias: co truong hop W dong bang ma bias van hoc
        proj_p = [] if self.no_expand else [p for p in self.proj.parameters() if p.requires_grad]
        if proj_p:
            groups.append({'params': proj_p, 'lr': args.projection_lr})
        if self.mlp is not None:
            groups.append({'params': self.mlp.parameters(), 'lr': args.lr})
        if self.backbone is not None and not args.freeze_backbone:
            bb = [p for p in self.backbone.parameters() if p.requires_grad]
            if bb:
                groups.append({'params': bb, 'lr': args.backbone_lr})
        return groups

    def usage_stats(self):
        out = {} if self.no_expand else {'proj': self.act.usage_stats()}
        if isinstance(self.mlp_act, TopK):
            out['mlp'] = self.mlp_act.usage_stats()
        return out


# --------------------------------------------------------------------------- #
# EWC / EWC-DR
# --------------------------------------------------------------------------- #

class Regularizer:
    """Hinh phat bac hai co trong so theo do quan trong.

    ewc     : omega = E[grad(CE)^2]
    ewc_dr  : omega = E[grad(CE(-logits))^2]   <- Logits Reversal

    Chi bao ve cac tham so THUC SU hoc lien tuc qua cac task.
    """

    def __init__(self, args):
        self.args = args
        self.omega = None
        self.mean = None
        self.enabled = args.cl_reg != 'none'
        self.reverse = args.cl_reg == 'ewc_dr'

    # ---- chon tham so can bao ve ----

    def _targets(self, model):
        args, out = self.args, {}
        # chieu: chi bao ve khi no THUC SU hoc tiep qua cac task
        if args.projection_schedule == 'continual' and model.proj is not None:
            for n, p in model.proj.named_parameters():
                if p.requires_grad:
                    out[f'proj.{n}'] = p
        # MLP: luon hoc lien tuc neu ton tai -> luon can bao ve
        if model.mlp is not None:
            for n, p in model.mlp.named_parameters():
                out[f'mlp.{n}'] = p
        if args.protect_head:
            for n, p in model.head.named_parameters():
                out[f'head.{n}'] = p
        if model.backbone is not None and not args.freeze_backbone:
            for n, p in model.backbone.named_parameters():
                if p.requires_grad:
                    out[f'backbone.{n}'] = p
        return out

    # ---- uoc luong do quan trong ----

    @torch.enable_grad()
    def estimate(self, model, loader, device):
        if not self.enabled:
            return
        args = self.args
        targets = self._targets(model)
        if not targets:
            return

        omega = {n: torch.zeros_like(p) for n, p in targets.items()}
        model.train()
        steps = 0

        # Uoc luong omega chay fp32. Voi fp16 thi grad^2 cua phan lon tham so
        # roi xuong duoi 6e-8 -> underflow ve 0, va omega=0 nghia la "khong quan
        # trong" nen ca hinh phat bien mat. Chi 1 luot/task nen khong dang tiec.
        amp_dtype = getattr(args, 'amp_dtype', None)
        args.amp_dtype = None

        for _ in range(args.importance_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb, is_feature=args.cache_features,
                               dense=args.importance_dense)
                if self.reverse:
                    logits = logits * -1          # Logits Reversal
                loss = F.cross_entropy(logits, yb)

                model.zero_grad(set_to_none=True)
                loss.backward()
                for n, p in targets.items():
                    if p.grad is not None:
                        omega[n] += p.grad.detach().pow(2)
                steps += 1

        cap = torch.tensor(args.omegamax, device=device)
        for n in omega:
            omega[n] = torch.min(omega[n] / max(steps, 1), cap)

        # tron voi omega cu; [:len(cu)] xu ly truong hop head no ra
        if self.omega is None:
            self.omega = omega
        else:
            alpha = getattr(self, '_alpha', 0.5)
            for n, new in omega.items():
                if n in self.omega:
                    old = self.omega[n]
                    new[:old.shape[0]] = alpha * old + (1 - alpha) * new[:old.shape[0]]
                omega[n] = new
            self.omega = omega

        self.mean = {n: p.detach().clone() for n, p in targets.items()}
        model.zero_grad(set_to_none=True)
        args.amp_dtype = amp_dtype

    def set_alpha(self, known_classes, total_classes):
        self._alpha = known_classes / max(total_classes, 1)

    # ---- hinh phat ----

    def penalty(self, model):
        if not self.enabled or self.omega is None:
            return torch.zeros((), device=next(model.parameters()).device)
        loss = 0.0
        for n, p in self._targets(model).items():
            if n not in self.omega:
                continue
            m, w = self.mean[n], self.omega[n]
            loss = loss + (w * (p[:m.shape[0]] - m).pow(2)).sum() / 2
        return self.args.lamda * loss


# --------------------------------------------------------------------------- #

def build_model(args, backbone=None):
    model = SparseExpandCL(args, backbone=backbone)
    reg = Regularizer(args)
    return model, reg
