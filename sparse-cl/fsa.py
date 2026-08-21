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


def inject_adapters(model, rank):
    """Gan adapter sau moi khoi Bottleneck cua ResNet."""
    ads = []
    for m in model.modules():
        if type(m).__name__ == 'Bottleneck':
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
def extract(model, ds, device, bs, desc):
    from tqdm import tqdm
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=min(8, os.cpu_count() or 1), pin_memory=True)
    model.eval()
    feats, labels = [], []
    for x, y in tqdm(loader, desc=desc):
        fs = model(x.to(device, non_blocking=True))
        feats.append(torch.cat([f.mean((2, 3)) for f in fs], 1).cpu())
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
           '--rank', str(getattr(args, 'fsa_rank', 8))]
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
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3, help='classifier')
    p.add_argument('--adapter_lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--bn_mode', default='eval', choices=['eval', 'train'])
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()

    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else 'cpu')
    # 10 epoch giu ten cu (+fsa{seed}) de khong pha cache va lenh tai lap da
    # ghi trong bao cao; cac muc epoch khac them hau to e{N}.
    ep = '' if a.epochs == 10 else f"e{a.epochs}"
    tag_ms = f"{a.model_name}+fsa{a.seed}{ep}+ms"
    tag_s4 = f"{a.model_name}+fsa{a.seed}{ep}"
    path_ms = os.path.join(a.cache_dir, f"{a.dataset}_{tag_ms}_{a.data_augmentation}.pt")
    path_s4 = os.path.join(a.cache_dir, f"{a.dataset}_{tag_s4}_{a.data_augmentation}.pt")
    if os.path.exists(path_ms) and os.path.exists(path_s4):
        print(f"[fsa] cache da co, khong lam gi:\n  {path_ms}\n  {path_s4}")
        return

    import timm
    set_seed(a.seed)
    model = timm.create_model(a.model_name, pretrained=True, features_only=True,
                              out_indices=(1, 2, 3, 4)).to(dev)
    dims = model.feature_info.channels()
    print(f"[fsa] {a.model_name} | stage dims {dims}")

    ads = inject_adapters(model, a.rank)
    for q in model.parameters():
        q.requires_grad = False
    ap = [q for ad in ads for q in ad.parameters()]
    for q in ap:
        q.requires_grad = True
    model.to(dev)
    print(f"[fsa] {len(ads)} adapter | {sum(q.numel() for q in ap)/1e3:.1f}K tham so hoc")

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

    head = nn.Linear(dims[-1], cpt).to(dev)
    opt = torch.optim.Adam([{'params': head.parameters(), 'lr': a.lr},
                            {'params': ap, 'lr': a.adapter_lr}])
    loader = DataLoader(tr0, batch_size=a.batch_size, shuffle=True,
                        num_workers=min(8, os.cpu_count() or 1), pin_memory=True)

    t_start = time.time()
    for ep in range(a.epochs):
        # BN cua backbone giu eval: adapter khong co BN nen khong mat gi,
        # va running stats khong troi.
        model.train() if a.bn_mode == 'train' else model.eval()
        head.train()
        tot = cor = 0
        loss_sum = 0.0
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = torch.tensor([lut[int(v)] for v in y], device=dev)
            logits = head(model(x)[-1].mean((2, 3)))
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.numel()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.numel()
        print(f"  ep {ep+1:2d}/{a.epochs}  loss {loss_sum/tot:.4f}  "
              f"acc {100*cor/tot:.2f}  {time.time()-t_start:.0f}s", flush=True)

    for q in model.parameters():
        q.requires_grad = False

    os.makedirs(a.cache_dir, exist_ok=True)
    Xtr, Ytr = extract(model, tr_full, dev, a.batch_size, 'trich train')
    Xte, Yte = extract(model, te_full, dev, a.batch_size, 'trich test ')
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
