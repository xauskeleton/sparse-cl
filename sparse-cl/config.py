"""
Cau hinh model.

Kien truc:

    anh -> backbone (dong bang / fine-tune)     -> feat        [B, 768]
        -> projection thua, HOC DUOC            -> expanded    [B, expand_dim]
        -> top-k winner-take-all                -> sparse code [B, expand_dim]
        -> (tuy chon) MLP
        -> classifier                           -> logits      [B, C]

    Train end-to-end bang cross-entropy.

Dung:
    from config import get_parser, validate
    args = validate(get_parser().parse_args())
"""

import argparse


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Continual learning voi bieu dien thua chieu cao hoc duoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------ task / data
    g = p.add_argument_group('Task')
    g.add_argument('--dataset', default='CIFAR-100',
                   choices=['CIFAR-100', 'CUB-200-2011', 'VTAB'])
    g.add_argument('--root', default='../data')
    g.add_argument('--num_classes', type=int, default=100)
    g.add_argument('--num_tasks', type=int, default=10)

    # --------------------------------------------------------------- backbone
    g = p.add_argument_group('Backbone')
    g.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    g.add_argument('--embedding_dim', type=int, default=768)
    g.add_argument('--freeze_backbone', type=_bool, default=True,
                   help='True  = backbone dong bang, feature cache duoc, train nhanh (vai phut).\n'
                        'False = fine-tune. Cham ~20x, mat cache, va backbone tro thanh '
                        'nguon quen chinh. Chi bat khi linear probe cho thay feature khong du.')
    g.add_argument('--backbone_tune_mode', type=str, default='last_block',
                   choices=['full', 'last_block', 'lora'],
                   help='Chi dung khi freeze_backbone=False. full = rui ro quen cao nhat.')
    g.add_argument('--backbone_lr', type=float, default=1e-5,
                   help='Nen nho hon --lr khoang 100 lan.')
    g.add_argument('--lora_rank', type=int, default=8)

    # -------------------------------------------------- projection (nua trai)
    g = p.add_argument_group('Sparse expansion')
    g.add_argument('--expand_dim', type=int, default=10000,
                   help='Chi phi tuyen tinh theo tham so nay (khong phai binh phuong), '
                        'nen noi rong la nut van re. Thu 20000/40000 truoc khi lam sau.')
    g.add_argument('--train_projection', type=_bool, default=True,
                   help='True = hoc phep chieu (diem chinh cua model). '
                        'False = chieu ngau nhien co dinh, dung lam baseline doi chung.')
    g.add_argument('--sparse_mask', type=_bool, default=True,
                   help='True  = mask ngau nhien co dinh, chi hoc GIA TRI tai vi tri khac 0 '
                        '(3M tham so, giu sparse matmul).\n'
                        'False = ma tran dense day du (7.68M tham so, de overfit hon).')
    g.add_argument('--synaptic_degree', type=int, default=300,
                   help='So ket noi khac 0 moi hang. Chi dung khi sparse_mask=True.')
    g.add_argument('--coding_level', type=float, default=0.1,
                   help='Ti le unit thang top-k. Day la nut chinh: thap -> chong quen tot hon '
                        'nhung de chet unit va underfit. Quet 0.02-0.30.')
    g.add_argument('--projection_lr', type=float, default=1e-3)
    g.add_argument('--projection_schedule', type=str, default='continual',
                   choices=['continual', 'task0', 'offline'],
                   help='continual = hoc lien tuc qua moi task (can --cl_reg de kiem soat troi)\n'
                        'task0     = hoc tren task dau roi dong bang\n'
                        'offline   = hoc truoc tren du lieu proxy roi dong bang')
    g.add_argument('--offline_data', type=str, default=None)

    # ------------------------------------------------------------ dead units
    g = p.add_argument_group('Dead-unit guard')
    g.add_argument('--adaptive_threshold', type=_bool, default=True,
                   help='Bias homeostatic: unit lau khong thang thi ha nguong. '
                        'Voi coding_level thap day gan nhu bat buoc.')
    g.add_argument('--load_balance_coef', type=float, default=0.0,
                   help='Loss can bang tai kieu Mixture-of-Experts. 0 = tat, thu 0.01.')
    g.add_argument('--log_unit_usage', type=_bool, default=True,
                   help='Log histogram tan suat thang moi task. Accuracy KHONG bao cho biet '
                        'unit dang chet dan - chi cai nay bao.')

    # ----------------------------------------------------------------- head
    g = p.add_argument_group('Head')
    g.add_argument('--use_mlp', type=_bool, default=False,
                   help='False = top-k -> Linear(expand_dim, C).\n'
                        'True  = top-k -> Linear -> act -> Linear(C). '
                        'Luu y: voi mlp_act=relu thi gradient tang cuoi thanh dense, '
                        'mat tinh cuc bo von la co che chong quen cua kien truc nay.')
    g.add_argument('--mlp_hidden', type=int, default=512)
    g.add_argument('--mlp_act', type=str, default='relu', choices=['relu', 'topk'],
                   help='topk = giu thua xuyen suot (khuyen dung neu da bat use_mlp).')
    g.add_argument('--mlp_coding_level', type=float, default=0.1,
                   help='Ti le top-k cua tang an, chi dung khi mlp_act=topk.')
    g.add_argument('--mlp_dropout', type=float, default=0.0)

    # ------------------------------------------------------ chong quen (CL)
    g = p.add_argument_group('CL regularization')
    g.add_argument('--cl_reg', type=str, default='ewc_dr',
                   choices=['none', 'ewc', 'ewc_dr'],
                   help='Kiem soat troi cua cac tham so hoc lien tuc qua cac task.')
    g.add_argument('--lamda', type=float, default=10000.0,
                   help='Trong so hinh phat. Gia tri nay lay tu EWC-DR (ResNet from scratch); '
                        'o quy mo nay gan nhu chac chan phai quet lai.')
    g.add_argument('--omegamax', type=float, default=1e-4,
                   help='Chan tren do quan trong.')
    g.add_argument('--importance_dense', type=_bool, default=True,
                   help='TAT top-k khi uoc luong do quan trong. Neu de nguyen, ~90% unit '
                        'khong nhan gradient nen omega=0 vi ly do CAU TRUC chu khong phai '
                        'vi khong quan trong. Logits Reversal khong chua duoc chuyen do.')
    g.add_argument('--importance_epochs', type=int, default=1,
                   help='So luot quet de tich luy omega. Tang neu importance_dense=False.')
    g.add_argument('--protect_head', type=_bool, default=False,
                   help='Ap regularizer len ca classifier, khong chi len projection.')

    # -------------------------------------------------------------- training
    g = p.add_argument_group('Training')
    g.add_argument('--seed', type=int, default=1993)
    g.add_argument('--epochs', type=int, default=100)
    g.add_argument('--lr', type=float, default=1e-3, help='lr cua head.')
    g.add_argument('--weight_decay', type=float, default=1e-4,
                   help='Rui ro chinh la overfit chu khong phai compute: 3-7M tham so tren '
                        '~5000 mau/task, va feature da cache nen KHONG augment duoc.')
    g.add_argument('--batch_size', type=int, default=256)
    g.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    g.add_argument('--val_ratio', type=float, default=0.1,
                   help='Tach tu task hien tai de early stopping. 0 = tat.')
    g.add_argument('--early_stop_patience', type=int, default=10)
    g.add_argument('--ce_scope', type=str, default='new', choices=['new', 'all'],
                   help="new = cross-entropy CHI tren logit cua lop moi (quy uoc PyCIL/EWC-DR); "
                        "lop cu khong nhan gradient nen khong bi day xuong.\n"
                        "all = cross-entropy tren toan bo lop da thay; khong co exemplar thi "
                        "lop cu chi xuat hien lam negative -> logit cua chung bi day xuong.")
    g.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'none'])

    # ----------------------------------------------------------------- cache
    g = p.add_argument_group('Feature cache')
    g.add_argument('--cache_features', type=_bool, default=True,
                   help='Trich feature 1 lan cho ca dataset roi cat theo task. '
                        'Tat di thi ViT chay lai moi epoch -> cham ~20x.')
    g.add_argument('--cache_dir', type=str, default='./cache')
    g.add_argument('--cache_fp16', type=_bool, default=True)

    # ------------------------------------------------------------------ misc
    g = p.add_argument_group('Misc')
    g.add_argument('--gpu', type=int, default=0)
    g.add_argument('--data_augmentation', default='vit', choices=[None, 'vit', 'resnet'])
    g.add_argument('--out_dir', type=str, default='./runs')
    g.add_argument('--exp_name', type=str, default=None)

    return p


# --------------------------------------------------------------------------- #
# Kiem tra to hop flag - day la cho bug hay nap nhat
# --------------------------------------------------------------------------- #

def validate(args):

    # --- cache chi hop le khi feature khong doi ---
    if args.cache_features and not args.freeze_backbone:
        raise ValueError(
            "cache_features=True vo nghia khi backbone train duoc (feature doi moi epoch). "
            "Dat --cache_features False va chuan bi cham hon ~20x."
        )

    # --- regularizer can co gi do troi de bao ve ---
    proj_continual = args.train_projection and args.projection_schedule == 'continual'
    nothing_continual = (not proj_continual
                         and not args.use_mlp          # MLP luon hoc lien tuc neu co
                         and args.freeze_backbone
                         and not args.protect_head)
    if args.cl_reg != 'none' and nothing_continual:
        raise ValueError(
            f"cl_reg={args.cl_reg} nhung khong tham so nao hoc lien tuc qua cac task "
            f"(chieu khong hoc tiep, khong co MLP, backbone dong bang, protect_head=False). "
            f"Chi con classifier thay doi -> regularizer luon bang 0. "
            f"Dat --cl_reg none, hoac bat --protect_head True."
        )

    if args.projection_schedule == 'offline' and not args.offline_data:
        raise ValueError("projection_schedule=offline can --offline_data.")

    if args.mlp_act == 'topk' and not args.use_mlp:
        _warn("mlp_act=topk bi bo qua vi use_mlp=False.")

    # --- coding_level >= 1 lam tang mo rong VO NGHIA (du co MLP hay khong) ---
    if args.coding_level >= 1.0:
        if not args.use_mlp:
            raise ValueError(
                "coding_level>=1.0 + use_mlp=False: khong con phi tuyen nao giua hai lop "
                "tuyen tinh, nen W_head @ W_proj sup thanh MOT ma tran [C, 768]. "
                "Model thoai hoa thanh linear probe tren feature goc."
            )
        raise ValueError(
            "coding_level>=1.0: W_mlp @ W_proj sup thanh mot ma tran [H, 768], nen tang "
            "mo rong 10000 chieu bien thanh phan tich thua bac <=768 cua no - khong them "
            "duoc gi ngoai tham so. Model tuong duong MLP thuong tren feature 768 chieu. "
            "Muon chay baseline MLP dense thi dung --expand_dim <H> --use_mlp False "
            "va thay top-k bang ReLU, dung dat coding_level=1."
        )

    # --- canh bao thiet ke ---
    if args.projection_schedule == 'continual' and args.cl_reg == 'none':
        _warn(
            "projection_schedule=continual + cl_reg=none: phep chieu troi tu do qua cac task. "
            "Day la cau hinh dung cho giai doan 1 (do sparsity tu no chong quen bao nhieu, "
            "khong lan voi dong gop cua regularizer)."
        )

    if args.use_mlp and args.mlp_act == 'relu':
        _warn(
            "use_mlp=True voi relu: gradient tang cuoi tro thanh dense, mat tinh cuc bo. "
            "Chay kem --mlp_act topk de tach bach anh huong cua DO SAU va cua SPARSITY."
        )

    if not args.freeze_backbone:
        _warn(
            "freeze_backbone=False: khong tach bach duoc dong gop cua lop chieu voi cua backbone. "
            "Chay linear probe truoc de xac nhan that su can."
        )

    if args.train_projection and args.coding_level < 0.05 \
            and not (args.adaptive_threshold or args.load_balance_coef > 0):
        _warn(
            f"coding_level={args.coding_level} thap + chieu hoc duoc + khong co co che can bang tai "
            f"-> rui ro unit chet cao. Cannhac --adaptive_threshold True."
        )

    if args.train_projection and not args.sparse_mask:
        _warn(
            "sparse_mask=False: 7.68M tham so dense, mat sparse matmul va tang rui ro overfit "
            "tren ~5000 mau/task."
        )

    if not args.train_projection and args.cl_reg != 'none' and args.freeze_backbone \
            and not args.protect_head:
        _warn("Chieu co dinh + backbone dong bang: chi con head la thay doi, cl_reg gan nhu vo dung.")

    args.exp_name = args.exp_name or _auto_name(args)
    return args


# --------------------------------------------------------------------------- #

def _bool(x):
    if isinstance(x, bool):
        return x
    if str(x).lower() in ('true', 't', '1', 'yes'):
        return True
    if str(x).lower() in ('false', 'f', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Gia tri bool khong hop le: {x}")


def _warn(msg):
    print(f"[config] CANH BAO: {msg}")


def _auto_name(args):
    parts = [
        args.dataset,
        args.model_name.split('_')[0],
        f"d{args.expand_dim}",
        f"k{args.coding_level}",
        'proj-learn' if args.train_projection else 'proj-fixed',
        args.projection_schedule,
        f"mlp-{args.mlp_act}" if args.use_mlp else 'linear',
    ]
    if args.cl_reg != 'none':
        parts.append(f"{args.cl_reg}-l{args.lamda:g}")
    parts.append('bb-frozen' if args.freeze_backbone else f"bb-{args.backbone_tune_mode}")
    parts.append(f"s{args.seed}")
    return '_'.join(str(p) for p in parts)


if __name__ == '__main__':
    args = validate(get_parser().parse_args())
    for k, v in sorted(vars(args).items()):
        print(f"{k:24s} {v}")
