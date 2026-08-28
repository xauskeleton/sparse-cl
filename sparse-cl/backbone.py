"""Nap backbone va kiem cache. Tach khoi train.py vi 9 script khac deu dung.

load_backbone chi duoc goi khi CHUA co cache; co cache roi thi khong can timm,
nen chay duoc ca tren may khong GPU.
"""

import os

import torch


class _MultiStage(torch.nn.Module):
    """Global-average-pool tung stage roi ghep lai thanh mot vector."""

    STAGE_DIMS = (256, 512, 1024, 2048)

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return torch.cat([f.mean((2, 3)) for f in self.base(x)], 1)


class _ViTMultiStage(torch.nn.Module):
    """Lay CLS token o bon diem deu theo do sau cua ViT, ghep lai.

    Doi ung voi bon stage cua ResNet: moi tap la mot phan tu do sau. Tap cuoi
    di qua `norm` cuoi cung nen TRUNG KHOP tuyet doi voi feature 768 chieu da
    cache truoc day - dieu kien de so sanh concat voi moc cu cho cong bang.

    Ba tap con lai KHONG qua norm nao, nen thang do lon rat khac nhau. Khong
    chuan hoa o day: `stage_norms` trong trainer do lai va `scale_b` trong
    FlyCL keo ve cung thang, giong het duong ResNet.
    """

    def __init__(self, base, taps=(2, 5, 8, 11)):
        super().__init__()
        self.base = base
        self.taps = set(taps)
        self.n_tap = len(taps)

    def forward(self, x):
        m = self.base
        x = m.patch_embed(x)
        x = m._pos_embed(x)
        x = m.patch_drop(x)
        x = m.norm_pre(x)
        out = []
        last = max(self.taps)
        for i, blk in enumerate(m.blocks):
            x = blk(x)
            if i in self.taps:
                out.append(m.norm(x)[:, 0] if i == last else x[:, 0])
        return torch.cat(out, 1)


def load_backbone(model_name, device):
    """Chi goi khi CHUA co cache. Co cache roi thi khong can ViT -> chay duoc
    ca tren session CPU."""
    import timm
    if model_name == 'vit_base_patch16_224':
        m = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    elif model_name in ('resnet-50', 'resnet50'):
        m = timm.create_model('resnet50', pretrained=True, num_classes=0)
    elif model_name.endswith('+ms') and 'vit' in model_name:
        # ViT khong co stage; cat bon diem deu theo do sau, moi diem 768 -> 3072
        base = timm.create_model(model_name[:-3], pretrained=True, num_classes=0)
        m = _ViTMultiStage(base)
    elif model_name.endswith('+ms'):
        # Ghep global-average-pool cua 4 stage: 256+512+1024+2048 = 3840.
        # Tang cuoi da chuyen biet hoa cho 1000 lop ImageNet; tang giua tong
        # quat hon va mang thong tin BO TRO - do duoc khi ket hop bang phep
        # nhan. Backbone van dong bang nen Q, G van la sufficient statistics.
        base = timm.create_model(model_name[:-3], pretrained=True,
                                 features_only=True, out_indices=(1, 2, 3, 4))
        m = _MultiStage(base)
    elif '.' in model_name:
        # Tag timm day du, vd resnet50.tv2_in1k = checkpoint torchvision
        # IMAGENET1K_V2 ma Fly-CL dung (resnet50-11ad3fa6.pth). Tag di vao ten
        # cache nen khong dam vao cache cua backbone khac.
        m = timm.create_model(model_name, pretrained=True, num_classes=0)
    else:
        raise ValueError(f"backbone khong ho tro: {model_name}")
    cfg = getattr(m, 'default_cfg', None) or {}
    print(f"[model] backbone {model_name} | tag={cfg.get('tag', '?')}")
    return m.eval().to(device)


def cache_exists(args):
    tag = f"{args.dataset}_{args.model_name}_{args.data_augmentation}"
    return os.path.exists(os.path.join(args.cache_dir, f"{tag}.pt"))
