"""Du lieu: trich feature MOT LAN cho ca dataset, cache, roi cat theo task.

Diem quan trong: feature KHONG phu thuoc seed (cung anh + cung ViT = cung vector),
chi cach CHIA TASK moi phu thuoc seed. Nen cache theo dataset, khong theo task
-> quet seed / so task / cach chia deu mien phi.

Cach chia task tai lap y het Fly-CL:
    random_initialization(seed)  ->  random.sample(range(C), C)
de so lieu so sanh duoc voi log co san.
"""

import hashlib
import os
import random
import shutil
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# torchvision/timm chi can khi CHUA co cache -> import lazy trong ham,
# de session chi co torch van chay duoc tren cache co san.


# --------------------------------------------------------------------------- #
# CIFAR-100: toronto.edu hay bi chan/cham tren Kaggle -> lay tu mirror
# --------------------------------------------------------------------------- #

_CIFAR100_MIRROR = ("https://huggingface.co/datasets/nakroy/cifar100-python/"
                    "resolve/main/cifar-100-python.tar.gz")
_CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def _ensure_cifar100(root):
    os.makedirs(root, exist_ok=True)
    tar = os.path.join(root, "cifar-100-python.tar.gz")
    if os.path.isdir(os.path.join(root, "cifar-100-python")):
        return
    if os.path.exists(tar) and hashlib.md5(open(tar, 'rb').read()).hexdigest() == _CIFAR100_MD5:
        return
    print(f"[data] tai CIFAR-100 tu mirror: {_CIFAR100_MIRROR}")
    req = urllib.request.Request(_CIFAR100_MIRROR, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r, open(tar, 'wb') as f:
        shutil.copyfileobj(r, f)
    assert hashlib.md5(open(tar, 'rb').read()).hexdigest() == _CIFAR100_MD5, "tai loi"


# --------------------------------------------------------------------------- #

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_transform(dataset, data_augmentation):
    """Giong het Fly-CL: CIFAR resize thang 224, cac dataset khac resize 256 roi crop."""
    from torchvision import transforms
    input_size = 224
    is_cifar = dataset == 'CIFAR-100'
    size = input_size if is_cifar else int(256 / 224 * input_size)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]
    if data_augmentation == 'resnet':
        t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif data_augmentation == 'vit':
        t.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    elif data_augmentation not in (None, 'None'):
        raise ValueError(f"data_augmentation khong ho tro: {data_augmentation}")
    return transforms.Compose(t)


def load_raw(dataset, root, transform):
    from torchvision import datasets
    if dataset == 'CIFAR-100':
        _ensure_cifar100(root)
        tr = datasets.CIFAR100(root, train=True, download=True, transform=transform)
        te = datasets.CIFAR100(root, train=False, download=True, transform=transform)
    elif dataset == 'CUB-200-2011':
        tr = datasets.ImageFolder(f"{root}/cub/train/", transform=transform)
        te = datasets.ImageFolder(f"{root}/cub/test/", transform=transform)
    elif dataset == 'VTAB':
        tr = datasets.ImageFolder(f"{root}/vtab/train/", transform=transform)
        te = datasets.ImageFolder(f"{root}/vtab/test/", transform=transform)
    else:
        raise ValueError(f"dataset khong ho tro: {dataset}")
    return tr, te


# --------------------------------------------------------------------------- #
# Trich feature + cache
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _extract(backbone, ds, device, batch_size, desc):
    from tqdm import tqdm
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=min(8, os.cpu_count() or 1), pin_memory=True)
    feats, labels = [], []
    backbone.eval()
    for x, y in tqdm(loader, desc=desc):
        feats.append(backbone(x.to(device, non_blocking=True)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def cached_features(args, backbone, device):
    """Tra ve (Xtr, Ytr, Xte, Yte) voi nhan GOC (chua doi thu tu lop).

    Tag cache chua moi thu anh huong toi feature. Doi backbone hay normalization
    ma quen doi tag = cache cu thanh rac im lang, nen ep vao ten file.
    """
    os.makedirs(args.cache_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.model_name}_{args.data_augmentation}"
    path = os.path.join(args.cache_dir, f"{tag}.pt")

    if os.path.exists(path):
        d = torch.load(path, map_location='cpu')
        print(f"[data] dung cache: {path}")
    else:
        if backbone is None:
            raise RuntimeError("Chua co cache ma khong truyen backbone de trich.")
        trsf = build_transform(args.dataset, args.data_augmentation)
        tr, te = load_raw(args.dataset, args.root, trsf)
        Xtr, Ytr = _extract(backbone, tr, device, args.batch_size, 'trich train')
        Xte, Yte = _extract(backbone, te, device, args.batch_size, 'trich test ')
        d = {'Xtr': Xtr.half() if args.cache_fp16 else Xtr, 'Ytr': Ytr,
             'Xte': Xte.half() if args.cache_fp16 else Xte, 'Yte': Yte}
        torch.save(d, path)
        print(f"[data] da luu cache: {path}")

    to = lambda t: t.to(device)
    return (to(d['Xtr']).float(), to(d['Ytr']),
            to(d['Xte']).float(), to(d['Yte']))


# --------------------------------------------------------------------------- #
# Chia task
# --------------------------------------------------------------------------- #

def make_class_order(num_classes, seed):
    """Tai lap y het Fly-CL: seed roi random.sample ngay."""
    set_seed(seed)
    return random.sample(list(range(num_classes)), num_classes)


def _remap(labels, class_order):
    """Doi nhan goc -> vi tri trong class_order.

    Sau buoc nay task t chiem dung dai nhan [t*cpt, (t+1)*cpt), nen classifier
    chi can no them hang o cuoi - khong phai anh xa gi luc train/eval.
    """
    lut = torch.full((max(class_order) + 1,), -1, dtype=torch.long, device=labels.device)
    for new, orig in enumerate(class_order):
        lut[orig] = new
    out = lut[labels.long()]
    assert (out >= 0).all(), "co nhan nam ngoai class_order"
    return out


class _TaskSplit:
    """Phan dung chung cho ca duong feature lan duong anh: doi nhan theo
    class_order, tach validation mot lan, cat theo task bang indexing.

    Khong dung DataLoader: o quy mo nay (153 MB) phep tinh qua re nen overhead
    worker/copy chiem phan lon thoi gian.
    """

    def _setup(self, args, device, Xtr, Ytr, Xte, Yte):
        self.args, self.device = args, device
        self.Xtr, self.Xte = Xtr, Xte

        self.class_order = make_class_order(args.num_classes, args.seed)
        self.ytr = _remap(Ytr, self.class_order)
        self.yte = _remap(Yte, self.class_order)

        self.cpt = args.num_classes // args.num_tasks
        assert self.cpt * args.num_tasks == args.num_classes, \
            f"num_classes={args.num_classes} khong chia het cho num_tasks={args.num_tasks}"

        # tach validation mot lan, co dinh qua moi epoch
        self._val_mask = torch.zeros_like(self.ytr, dtype=torch.bool)
        if args.val_ratio > 0:
            g = torch.Generator(device='cpu').manual_seed(args.seed)
            for c in range(args.num_classes):
                idx = (self.ytr == c).nonzero(as_tuple=True)[0]
                n_val = max(1, int(len(idx) * args.val_ratio))
                pick = idx[torch.randperm(len(idx), generator=g).to(idx.device)[:n_val]]
                self._val_mask[pick] = True

    def classes_of(self, task):
        return range(task * self.cpt, (task + 1) * self.cpt)

    def total_classes(self, task):
        return (task + 1) * self.cpt

    def train_split(self, task):
        lo, hi = task * self.cpt, (task + 1) * self.cpt
        m = (self.ytr >= lo) & (self.ytr < hi)
        tr, va = m & ~self._val_mask, m & self._val_mask
        return (self.Xtr[tr], self.ytr[tr]), (self.Xtr[va], self.ytr[va])

    def test_split(self, task):
        lo, hi = task * self.cpt, (task + 1) * self.cpt
        m = (self.yte >= lo) & (self.yte < hi)
        return self.Xte[m], self.yte[m]

    def summary(self):
        return (f"[data] {self.args.dataset}: {len(self.ytr)} train / {len(self.yte)} test, "
                f"{self.args.num_tasks} task x {self.cpt} lop, "
                f"x={tuple(self.Xtr.shape[1:])} {self.Xtr.dtype}, "
                f"val_ratio={self.args.val_ratio}")


class TaskData(_TaskSplit):
    """Feature da trich san, giu tren GPU. Dung khi backbone dong bang."""

    def __init__(self, args, backbone, device):
        Xtr, Ytr, Xte, Yte = cached_features(args, backbone, device)
        self._setup(args, device, Xtr, Ytr, Xte, Yte)


# --------------------------------------------------------------------------- #
# Duong anh: dung khi backbone train duoc -> feature doi moi epoch, het cache
# --------------------------------------------------------------------------- #

_NORM = {
    'vit':    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    'resnet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _pil_bicubic_matrix(in_size, out_size, device):
    """Ma tran resize [out, in] tai lap DUNG bicubic cua PIL.

    Khong dung F.interpolate(mode='bicubic') duoc: PyTorch dung kernel Keys voi
    a=-0.75, PIL dung a=-0.5. Voi phong to 32->224 chenh lech nay lam cosine
    similarity cua feature ViT tut xuong 0.98 - du de moi so sanh voi ket qua
    dung cache (va voi log goc cua Fly-CL) thanh vo nghia.

    Thuat toan theo PIL Resample.c::precompute_coeffs.
    """
    a = -0.5

    def cubic(x):
        x = abs(x)
        if x < 1.0:
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        if x < 2.0:
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a
        return 0.0

    scale = in_size / out_size
    fscale = max(scale, 1.0)          # phong to -> 1.0, thu nho -> co giai antialias
    support = 2.0 * fscale
    m = np.zeros((out_size, in_size), dtype=np.float64)
    for j in range(out_size):
        center = (j + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), in_size)
        w = np.array([cubic((k + 0.5 - center) / fscale) for k in range(lo, hi)])
        m[j, lo:hi] = w / w.sum()
    return torch.tensor(m, dtype=torch.float32, device=device)


def make_preprocess(args, device, in_size=32, out_size=224):
    """uint8 [B,3,32,32] -> float [B,3,224,224] da chuan hoa, lam het tren GPU.

    Tuong duong build_transform() nhung khong qua PIL/CPU: DataLoader worker tren
    Windows dat hon ca forward cua backbone o quy mo nay, de nguyen thi data
    pipeline moi la nut that chu khong phai GPU.

    Resize tach chieu bang hai phep nhan ma tran, dung thu tu cua PIL: NGANG
    truoc roi DOC, va lam tron ve uint8 sau MOI luot (PIL luu anh trung gian o
    dang 8-bit). Lam tron cua PIL la half-up ((int)(v+0.5)) chu khong phai
    half-even nhu torch.round, nen dung floor(v+0.5).
    """
    if args.data_augmentation not in _NORM:
        raise ValueError(f"data_augmentation khong ho tro duong anh: {args.data_augmentation}")
    mean, std = (torch.tensor(v, device=device).view(1, 3, 1, 1)
                 for v in _NORM[args.data_augmentation])
    M = _pil_bicubic_matrix(in_size, out_size, device)
    u8 = lambda t: t.add_(0.5).floor_().clamp_(0, 255)

    def prep(x):
        x = u8(x.float() @ M.t())               # luot ngang: [B,3,in,out]
        x = u8(M @ x).div_(255.0)               # luot doc:   [B,3,out,out]
        return (x - mean) / std

    return prep


class ImageTaskData(_TaskSplit):
    """Giu anh goc uint8 32x32 tren GPU (153 MB cho CIFAR-100), resize/chuan hoa
    theo tung lo. Nhan va cach chia task giong het TaskData nen so sanh truc tiep
    duoc voi cac ket qua backbone dong bang."""

    def __init__(self, args, device):
        if args.dataset != 'CIFAR-100':
            raise NotImplementedError(
                f"duong anh moi cai cho CIFAR-100, chua ho tro {args.dataset}")
        from torchvision import datasets
        _ensure_cifar100(args.root)
        tr = datasets.CIFAR100(args.root, train=True, download=True)
        te = datasets.CIFAR100(args.root, train=False, download=True)

        def to_u8(a):   # HWC uint8 -> CHW uint8 tren GPU
            return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(device)

        self._setup(args, device,
                    to_u8(tr.data), torch.tensor(tr.targets, device=device),
                    to_u8(te.data), torch.tensor(te.targets, device=device))


def batches(X, Y, batch_size, shuffle=True, generator=None):
    """Chia lo bang indexing tren GPU. Tra ve LIST (khong phai generator) de
    Regularizer.estimate() duyet lai duoc nhieu luot."""
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device, generator=generator) if shuffle \
        else torch.arange(n, device=X.device)
    return [(X[idx[i:i + batch_size]], Y[idx[i:i + batch_size]])
            for i in range(0, n, batch_size)]
