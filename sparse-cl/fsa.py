"""FSA (First-Session Adaptation) kieu AnaCP roi trich feature ra cache.

    python experiments/fsa.py --seed 1993 --epochs 10
    # sau do moi script khac chay tren cache moi:
    python experiments/flycl.py    --model_name resnet50.tv2_in1k+fsa1993+ms \n                                   --grid 0:1,300:1 --ridge_lower 5
    python experiments/anacp_cp.py --model_name resnet50.tv2_in1k+fsa1993 --pos post

AnaCP quang cao "no gradient-based training" nhung Implementation Details cua ho
ghi: "We also adopt FSA following [56] before applying AnaCP". Trong code do la
--training_method aper: gan Adapter (down-up, rank 8) vao MLP moi block, dong
bang backbone, 10 epoch cross-entropy, roi dong bang vinh vien. No chiem 65%
compute cua ho (6m05s / 9m18s) va KHONG duoc ablate.

O day lam dung the tren ResNet-50: adapter 1x1 conv down-up gan sau moi khoi
Bottleneck, up khoi tao 0 nen luc dau adapter = identity. Chi task 0 (10 lop,
5000 anh), roi dong bang va trich feature cho toan bo dataset.

HAI DIEM PHAI DUNG, sai la lech giao thuc:
  * Cache tag phai chua SEED. FSA phu thuoc 10 lop nao roi vao task 0, ma thu tu
    lop do seed quyet dinh. Dung chung mot cache cho nhieu seed la ro ri thong
    tin giua cac seed.
  * BatchNorm cua backbone phai o eval() ke ca khi train adapter, neu khong
    running stats van cap nhat va feature troi am tham (xem README, cam bay #4).
    --bn_mode train de tai lap dung AnaCP (ho goi model.train()).
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data_loader import build_transform, load_raw, make_class_order, set_seed


class ConvAdapter(nn.Module):
    """down-up 1x1 conv, up khoi tao 0 -> luc bat dau adapter khong lam gi."""

    def __init__(self, ch, rank):
        super().__init__()
        self.down = nn.Conv2d(ch, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, ch, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.relu(self.down(x)))


class LinAdapter(nn.Module):
    """down-up tuyen tinh, up khoi tao 0. Dung y het AnaCP: KHONG co phi tuyen
    o giua, va dat SONG SONG voi MLP chu khong noi tiep."""

    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


class FiLM2d(nn.Module):
    """Feature-wise Linear Modulation cho ResNet: mot he so nhan va mot he so
    cong CHO MOI KENH. gamma=1, beta=0 luc dau -> phep dong nhat.

    2C tham so moi khoi, so voi 16C cua adapter down-up rank 8 - it hon TAM
    LAN. Bai FSA (Panos, ICCV 2023) khuyen dung cai nay o che do it du lieu,
    dung adapter/fine-tune o che do nhieu du lieu.

    Day la anh xa tuyen tinh DUONG CHEO, tuc kha nghich. Dat o feature cuoi
    cung thi ridge bat bien voi no (xem `flycl_lp`, am ca 8 cau hinh). No chi
    co tac dung vi duoc cam GIUA cac khoi, phia sau con ReLU va tich chap.
    """

    def __init__(self, ch):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, ch, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        return self.g * x + self.b


class FiLM1d(nn.Module):
    """FiLM cho ViT: nhan/cong theo chieu embedding. Trung voi SSF."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.g * x + self.b


def linear_cka(X, Y):
    """CKA tuyen tinh giua hai ma tran dac trung [n, d]. Bang 1 khi hai bieu
    dien trung nhau ve mat tuyen tinh, giam khi chung lech nhau."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (Y.T @ X).norm() ** 2
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    return float(num / den.clamp(min=1e-30))


@torch.no_grad()
def block_feats(model, ds, device, bs, nw, n_max=1024):
    """GAP dau ra cua TUNG khoi Bottleneck tren mot mau con cua task 0.

    Hook van chay du adapter da ghi de `m.forward` bang instance attribute,
    vi `Module.__call__` moi la cho goi hook.
    """
    blocks = [m for m in model.modules() if type(m).__name__ == 'Bottleneck']
    store = [[] for _ in blocks]

    def mk(i):
        def fn(_mod, _inp, out):
            store[i].append(out.mean((2, 3)).float().cpu())
        return fn

    hooks = [b.register_forward_hook(mk(i)) for i, b in enumerate(blocks)]
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    seen = 0
    for x, _ in loader:
        model(x.to(device, non_blocking=True))
        seen += x.shape[0]
        if seen >= n_max:
            break
    for h in hooks:
        h.remove()
    return [torch.cat(v) for v in store]


@torch.no_grad()
def block_feats_vit(model, base, ds, device, bs, nw, n_max=1024):
    """CLS token sau tung block cua ViT. `model` la _ViTMultiStage, `base` la
    ViT ben trong - hook gan vao `base.blocks`."""
    store = [[] for _ in base.blocks]

    def mk(i):
        def fn(_mod, _inp, out):
            store[i].append(out[:, 0].float().cpu())
        return fn

    hooks = [b.register_forward_hook(mk(i)) for i, b in enumerate(base.blocks)]
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    seen = 0
    for x, _ in loader:
        model(x.to(device, non_blocking=True))
        seen += x.shape[0]
        if seen >= n_max:
            break
    for h in hooks:
        h.remove()
    return [torch.cat(v) for v in store]


def inject_film(model, _rank=None):
    """FiLM sau moi khoi Bottleneck. Khac ConvAdapter o cho no THAY THE dau ra
    (`film(out)`) chu khong cong them (`out + adapter(out)`) - vi ban than FiLM
    da la phep dieu bien, cong them thi luc dau ra 2*out chu khong phai out."""
    ads = []
    for m in model.modules():
        if type(m).__name__ == 'Bottleneck':
            ad = FiLM2d(m.bn3.num_features)
            m.adapter = ad
            fwd = m.forward

            def new_fwd(x, fwd=fwd, ad=ad):
                return ad(fwd(x))

            m.forward = new_fwd
            ads.append(ad)
    return ads


def inject_film_vit(model, _rank=None, start=0):
    """FiLM sau moi block cua ViT, tu block `start` tro di."""
    ads = []
    for i, block in enumerate(model.blocks):
        if i < start:
            continue
        ad = FiLM1d(block.mlp.fc2.out_features)
        block.adapter = ad
        fwd = block.forward

        def new_fwd(x, fwd=fwd, ad=ad):
            return ad(fwd(x))

        block.forward = new_fwd
        ads.append(ad)
    return ads


def inject_adapters_vit(model, rank, start=0):
    """Gan adapter song song voi MLP cua moi block tu `start` tro di - chep
    dung backbone.py cua AnaCP (`mlp.forward = lambda x: mlp(x) + adapter(x)`)."""
    ads = []
    for i, block in enumerate(model.blocks):
        if i < start:
            continue
        mlp = block.mlp
        ad = LinAdapter(mlp.fc2.out_features, rank)
        block.adapter = ad
        fwd = mlp.forward

        def new_fwd(x, fwd=fwd, ad=ad):
            return fwd(x) + ad(x)

        mlp.forward = new_fwd
        ads.append(ad)
    return ads


def inject_adapters(model, rank, start=0):
    """Gan adapter sau moi khoi Bottleneck cua ResNet, tu khoi `start` tro di.

    `start > 0` la meo 2 cua PACE: chi adapt cac tang sau, dong bang tang nong
    vi chung ma hoa dac trung chung cua domain chu khong phai ngu nghia task.
    """
    ads = []
    for i, m in enumerate(b for b in model.modules()
                          if type(b).__name__ == 'Bottleneck'):
        if i >= start:
            ch = m.bn3.num_features
            ad = ConvAdapter(ch, rank)
            m.adapter = ad                      # dang ky lam submodule
            fwd = m.forward
            def new_fwd(x, fwd=fwd, ad=ad):
                out = fwd(x)
                return out + ad(out)
            m.forward = new_fwd                 # instance attr, che class method
            ads.append(ad)
    return ads


@torch.no_grad()
def extract(model, ds, device, bs, desc, nw=8):
    from tqdm import tqdm
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=nw, pin_memory=True)
    model.eval()
    feats, labels = [], []
    for x, y in tqdm(loader, desc=desc):
        feats.append(model(x.to(device, non_blocking=True)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def run_fsa(args, device=None):
    """Goi tu trainer khi --training_method aper. Chay lai fsa.py bang
    subprocess de giu nguyen bo tham so rieng cua no."""
    import subprocess, sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, '-u', os.path.join(here, 'fsa.py'),
           '--model_name', args.model_name.replace('+ms', ''),
           '--seed', str(args.seed), '--epochs', str(getattr(args, 'fsa_epochs', 10)),
           '--rank', str(getattr(args, 'fsa_rank', 8)),
           # phai truyen: ViT va ResNet dung normalization va kich thuoc khac
           # nhau, mac dinh 'resnet' se lam hong feature ViT ma khong bao gi
           '--data_augmentation', args.data_augmentation,
           # va dataset: mac dinh cua fsa.py la CIFAR-100, nen thieu cho nay
           # thi FSA cho CUB se hoc tren CIFAR roi ghi de len cache cua CUB
           '--dataset', args.dataset, '--root', args.root,
           '--num_classes', str(args.num_classes),
           '--num_tasks', str(args.num_tasks),
           '--adapter', getattr(args, 'fsa_adapter', 'conv'),
           '--workers', str(getattr(args, 'fsa_workers', 2))]
    if getattr(args, 'fsa_pace', 0):
        # PACE: head hoc CHAM hon adapter, nguoc voi mac dinh cua AnaCP
        cmd += ['--pace', '1', '--lr', '1e-4', '--adapter_lr', '1e-3',
                '--head_epochs', '1', '--rho_layer', '0.94']
    subprocess.run(cmd, cwd=here, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default='resnet50.tv2_in1k')
    p.add_argument('--dataset', default='CIFAR-100')
    p.add_argument('--root', default='./data')
    p.add_argument('--cache_dir', default='./cache')
    p.add_argument('--data_augmentation', default='resnet')
    p.add_argument('--num_classes', type=int, default=100)
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--seed', type=int, default=1993)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--adapter', default='conv', choices=['conv', 'film'],
                   help='conv = down-up rank 8 (AnaCP); film = scale+shift theo kenh')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3, help='classifier')
    p.add_argument('--adapter_lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    # Moi worker map mot vung shared memory rieng. Tren may dang chay
    # nhieu thu, 8 worker lam Windows bao ERROR_COMMITMENT_LIMIT (1455)
    # chu khong phai OOM cua GPU. Ha xuong 2 la du.
    p.add_argument('--workers', type=int, default=8)
    g = p.add_argument_group('PACE (ICLR 2026) - improved FSA')
    g.add_argument('--pace', type=int, default=0,
                   help='1 = bat ca ba meo: lr bat doi xung + train theo giai '
                        'doan + chon tang bang CKA')
    g.add_argument('--head_epochs', type=int, default=1, help='E_head')
    g.add_argument('--tag_extra', default='', help='hau to them vao ten cache, de tach cac bien the')
    g.add_argument('--rho_layer', type=float, default=0.94,
                   help='nguong CKA de chon L_tune')
    p.add_argument('--bn_mode', default='eval', choices=['eval', 'train'])
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()

    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')
    # 10 epoch giu ten cu (+fsa{seed}) de khong pha cache va lenh tai lap da
    # ghi trong bao cao; cac muc epoch khac them hau to e{N}.
    ep = '' if a.epochs == 10 else f"e{a.epochs}"
    # 'conv' giu ten cu de khong pha cache da co; film them hau to rieng
    ad = ('' if a.adapter == 'conv' else a.adapter) + ('pace' if a.pace else '') + a.tag_extra
    tag_ms = f"{a.model_name}+fsa{a.seed}{ep}{ad}+ms"
    tag_s4 = f"{a.model_name}+fsa{a.seed}{ep}{ad}"
    path_ms = os.path.join(a.cache_dir, f"{a.dataset}_{tag_ms}_{a.data_augmentation}.pt")
    path_s4 = os.path.join(a.cache_dir, f"{a.dataset}_{tag_s4}_{a.data_augmentation}.pt")
    if os.path.exists(path_ms) and os.path.exists(path_s4):
        print(f"[fsa] cache da co, khong lam gi:\n  {path_ms}\n  {path_s4}")
        return

    import timm
    from backbone import _MultiStage, _ViTMultiStage
    set_seed(a.seed)
    is_vit = 'vit' in a.model_name

    vit_base = [None]           # giu ViT ben trong de gan hook do CKA

    def build(start=0):
        """Dung backbone moi va gan adapter tu khoi `start` tro di. Goi lai
        duoc, de PACE dung mot lan cho luot tham do roi dung lai lan hai."""
        if is_vit:
            b = timm.create_model(a.model_name, pretrained=True, num_classes=0)
            d = [b.embed_dim] * 4
            m = _ViTMultiStage(b).to(dev)
            fn = inject_adapters_vit if a.adapter == 'conv' else inject_film_vit
            adl = fn(b, a.rank, start)
            vit_base[0] = b
        else:
            b = timm.create_model(a.model_name, pretrained=True,
                                  features_only=True, out_indices=(1, 2, 3, 4))
            d = b.feature_info.channels()
            m = _MultiStage(b).to(dev)
            fn = inject_adapters if a.adapter == 'conv' else inject_film
            adl = fn(b, a.rank, start)
        for q in m.parameters():
            q.requires_grad = False
        pl = [q for ad in adl for q in ad.parameters()]
        for q in pl:
            q.requires_grad = True
        m.to(dev)
        print(f"[fsa] {len(adl)} adapter (tu khoi {start}) | "
              f"{sum(q.numel() for q in pl)/1e3:.1f}K tham so hoc")
        return m, d, adl, pl

    model, dims, ads, ap = build()
    print(f"[fsa] {a.model_name} | stage dims {dims}")

    # --- task 0 = 10 lop dau theo class_order cua seed nay ---
    cpt = a.num_classes // a.num_tasks
    order = make_class_order(a.num_classes, a.seed)
    t0 = order[:cpt]
    lut = {c: i for i, c in enumerate(t0)}
    print(f"[fsa] task 0 = lop {sorted(t0)}")

    trsf = build_transform(a.dataset, a.data_augmentation)
    tr_full, te_full = load_raw(a.dataset, a.root, trsf)
    idx = [i for i, y in enumerate(tr_full.targets) if y in lut]
    tr0 = Subset(tr_full, idx)
    print(f"[fsa] {len(idx)} anh train cho task 0")

    nw = a.workers
    loader = DataLoader(tr0, batch_size=a.batch_size, shuffle=True,
                        num_workers=nw, pin_memory=True)
    t_start = time.time()

    def train_loop(m, hd, groups, epochs, tag):
        opt = torch.optim.Adam(groups)
        for ep in range(epochs):
            # BN cua backbone giu eval: adapter khong co BN nen khong mat gi,
            # va running stats khong troi.
            m.train() if a.bn_mode == 'train' else m.eval()
            hd.train()
            tot = cor = 0
            loss_sum = 0.0
            for x, y in loader:
                x = x.to(dev, non_blocking=True)
                y = torch.tensor([lut[int(v)] for v in y], device=dev)
                logits = hd(m(x)[:, -dims[-1]:])
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * y.numel()
                cor += (logits.argmax(1) == y).sum().item()
                tot += y.numel()
            print(f"  [{tag}] ep {ep+1:2d}/{epochs}  loss {loss_sum/tot:.4f}  "
                  f"acc {100*cor/tot:.2f}  {time.time()-t_start:.0f}s", flush=True)

    if a.pace:
        # --- Giai doan A: luot THAM DO tren moi khoi, chi de do CKA.
        # Adapter khoi tao up=0 nen luc chua train, model == model goc.
        def feats(m):
            return (block_feats_vit(m, vit_base[0], tr0, dev, a.batch_size, nw)
                    if is_vit else block_feats(m, tr0, dev, a.batch_size, nw))

        f0 = feats(model)
        head = nn.Linear(dims[-1], cpt).to(dev)
        train_loop(model, head,
                   [{'params': head.parameters(), 'lr': a.lr},
                    {'params': ap, 'lr': a.adapter_lr}], a.epochs, 'probe')
        f1 = feats(model)
        cka = [linear_cka(x, y) for x, y in zip(f1, f0)]
        print('[pace] CKA moi khoi: ' + ' '.join(f'{c:.3f}' for c in cka))
        below = [i for i, c in enumerate(cka) if c < a.rho_layer]
        start = below[0] if below else 0
        print(f"[pace] rho={a.rho_layer} -> L_tune = khoi {start}/{len(cka)}")

        # --- dung lai model, chi gan adapter tu L_tune tro di ---
        model, dims, ads, ap = build(start)

        # --- Giai doan B: ham nong head, backbone dong bang ---
        head = nn.Linear(dims[-1], cpt).to(dev)
        train_loop(model, head, [{'params': head.parameters(), 'lr': a.lr}],
                   a.head_epochs, 'head')
        for q in head.parameters():
            q.requires_grad = False

        # --- Giai doan C: khoa head, chi train adapter ---
        train_loop(model, head, [{'params': ap, 'lr': a.adapter_lr}],
                   a.epochs, 'backbone')
    else:
        head = nn.Linear(dims[-1], cpt).to(dev)
        train_loop(model, head,
                   [{'params': head.parameters(), 'lr': a.lr},
                    {'params': ap, 'lr': a.adapter_lr}], a.epochs, 'fsa')

    for q in model.parameters():
        q.requires_grad = False

    os.makedirs(a.cache_dir, exist_ok=True)
    Xtr, Ytr = extract(model, tr_full, dev, a.batch_size, 'trich train', a.workers)
    Xte, Yte = extract(model, te_full, dev, a.batch_size, 'trich test ', a.workers)
    torch.save({'Xtr': Xtr, 'Ytr': Ytr, 'Xte': Xte, 'Yte': Yte}, path_ms)
    s4 = slice(sum(dims[:-1]), sum(dims))
    # .contiguous(): torch.save mot VIEW van ghi ca storage goc -> file s4
    # se to bang file 3840 chieu du tensor chi 2048 chieu (431 MB thua/seed).
    torch.save({'Xtr': Xtr[:, s4].contiguous(), 'Ytr': Ytr,
                'Xte': Xte[:, s4].contiguous(), 'Yte': Yte}, path_s4)
    print(f"[fsa] da luu\n  {path_ms}  {tuple(Xtr.shape)}\n  {path_s4}  "
          f"{tuple(Xtr[:, s4].shape)}\n[fsa] tong {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
