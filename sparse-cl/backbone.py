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


def load_backbone(model_name, device):
    """Chi goi khi CHUA co cache. Co cache roi thi khong can ViT -> chay duoc
    ca tren session CPU."""
    import timm
    if model_name == 'vit_base_patch16_224':
        m = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    elif model_name in ('resnet-50', 'resnet50'):
        m = timm.create_model('resnet50', pretrained=True, num_classes=0)
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
