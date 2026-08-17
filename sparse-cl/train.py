"""Vong lap continual learning.

    python train.py                       # cau hinh mac dinh
    python train.py --help                # toan bo flag
    python config.py <flags>              # chi kiem tra tinh hop le, khong chay

Chi so bao cao khop dinh nghia cua Fly-CL de so sanh truc tiep voi
Fly-CL-main/log_cifar_seed1993.txt:
    A_t  = trung binh accuracy tren cac task 0..t sau khi hoc xong task t
    A_T  = A_t o task cuoi          ("last stage accuracy")
    A~   = trung binh cua A_t       ("accumulated / overall accuracy")
"""

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import get_parser, validate
from data import ImageTaskData, TaskData, batches, make_preprocess, set_seed
from model import build_model


# --------------------------------------------------------------------------- #

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
    elif model_name == 'resnet50+ms':
        # Ghep global-average-pool cua 4 stage: 256+512+1024+2048 = 3840.
        # Tang cuoi da chuyen biet hoa cho 1000 lop ImageNet; tang giua tong
        # quat hon va co the huu ich hon cho dataset khac mien. Backbone van
        # dong bang nen Q, G van la sufficient statistics.
        base = timm.create_model('resnet50', pretrained=True,
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


def build_optimizer(model, args):
    groups = model.param_groups()
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return torch.optim.SGD(groups, momentum=0.9, weight_decay=args.weight_decay)


# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, X, Y, args):
    model.eval()
    # 1024 chi an toan cho feature; o duong anh moi mau la 3x224x224 nen phai nho lai
    bs = 1024 if args.cache_features else args.batch_size
    correct = 0
    for i in range(0, X.shape[0], bs):
        logits = model(X[i:i + bs], is_feature=args.cache_features)
        correct += (logits.argmax(1) == Y[i:i + bs]).sum().item()
    return 100.0 * correct / max(X.shape[0], 1)


def train_task(model, reg, tr, va, args, task, known_classes):
    """Huan luyen mot task. Tra ve dict chan doan."""
    Xtr, Ytr = tr
    Xva, Yva = va
    opt = build_optimizer(model, args)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.lr_schedule == 'cosine' else None)
    # fp16 can GradScaler (dai dong cua no hep, gradient de underflow);
    # bf16 co cung so mu voi fp32 nen khong can.
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp_dtype is torch.float16)
    gen = torch.Generator(device=Xtr.device).manual_seed(args.seed + task)

    best_acc, best_state, patience = -1.0, None, 0
    clf_sum = pen_sum = n_steps = 0.0
    val_curve, best_epoch = [], 0
    t_ep0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        ep_clf = ep_pen = 0.0
        bs = batches(Xtr, Ytr, args.batch_size, shuffle=True, generator=gen)
        for xb, yb in bs:
            logits = model(xb, is_feature=args.cache_features)

            if args.ce_scope == 'new' and known_classes > 0:
                loss_clf = F.cross_entropy(logits[:, known_classes:], yb - known_classes)
            else:
                loss_clf = F.cross_entropy(logits, yb)

            pen = reg.penalty(model)
            loss = loss_clf + pen + model.aux_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ep_clf += loss_clf.item()
            ep_pen += pen.detach().item()      # float(pen) tren tensor con grad -> UserWarning
        if sched is not None:
            sched.step()

        clf_sum += ep_clf; pen_sum += ep_pen; n_steps += len(bs)

        if Xva.shape[0] > 0:
            acc = evaluate(model, Xva, Yva, args)
            val_curve.append(round(acc, 2))
            if acc > best_acc + 1e-6:
                best_acc, best_epoch, patience = acc, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"  early stop @ epoch {epoch + 1} (val {best_acc:.2f} @ {best_epoch})")
                    break

            # Khi fine-tune backbone mot task mat hang chuc phut ma khong in dong
            # nao - nhin khong phan biet duoc dang chay voi treo.
            if args.log_every and (epoch + 1) % args.log_every == 0:
                el = time.time() - t_ep0
                print(f"  ep {epoch + 1:3d}/{args.epochs}  val {acc:5.2f}  "
                      f"best {best_acc:5.2f} @{best_epoch}  {el:.0f}s  "
                      f"({el / (epoch + 1):.1f}s/ep)")

    if best_state is not None:
        model.load_state_dict(best_state)

    clf_avg = clf_sum / max(n_steps, 1)
    pen_avg = pen_sum / max(n_steps, 1)
    return {
        'loss_clf': round(clf_avg, 4),
        'penalty': round(pen_avg, 4),
        # muc tieu 0.1-1: nho hon -> regularizer vo hinh; lon hon -> mang bi dong bang
        'pen_over_clf': round(pen_avg / max(clf_avg, 1e-8), 4),
        'val_acc': round(best_acc, 2),
        'best_epoch': best_epoch,        # de hieu chinh --epochs cho cac lan sau
        'epochs_run': len(val_curve),
        'val_curve': val_curve,          # ve ra de xem thuc su hoi tu o dau
    }


def diagnostics(model, reg, args):
    d = {}
    # ~0 nghia la phep chieu dung yen -> cau hinh "hoc chieu" dong nhat voi
    # "chieu dong bang", moi so sanh giua chung deu vo nghia. Kiem TRUOC TIEN.
    u = model.usage_stats()
    if not model.no_expand:
        d['proj_drift'] = round(model.proj.drift(), 4)
        d['dead_frac'] = round(u['proj']['dead_frac'], 4)
        d['usage_cv'] = round(u['proj']['usage_cv'], 4)
    if 'mlp' in u:
        d['mlp_dead_frac'] = round(u['mlp']['dead_frac'], 4)
    if reg.omega:
        # >0.5 nghia la moi omega bang nhau -> EWC-DR thoai hoa thanh L2 thuan
        sat = [(v >= args.omegamax * 0.999).float().mean().item() for v in reg.omega.values()]
        d['omega_saturated'] = round(float(np.mean(sat)), 4)
        # phan vi TRUOC khi cat: doc mot lan la biet nen dat omegamax o dau,
        # khong phai quet mu tung gia tri (moi run backbone mat hang gio)
        for k, v in getattr(reg, 'omega_q', {}).items():
            d[f'omega_{k}'] = float(f'{v:.3g}')
    return d


# --------------------------------------------------------------------------- #

def save(args, acc, per_task, metrics, t_start):
    """Ghi ket qua ra JSON. Goi sau MOI task, khong doi het 10 task: mot run
    backbone fine-tune co the mat hang chuc gio, va phien Kaggle bi cat o gio
    thu 12 - ghi mot lan o cuoi thi mat sach. 'tasks_done' cho biet ket qua da
    day du hay moi mot phan."""
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.exp_name}.json")
    tmp = out + '.tmp'
    payload = {
        'args': {k: str(v) for k, v in vars(args).items()},
        'tasks_done': len(per_task),
        'complete': len(per_task) == args.num_tasks,
        'elapsed': round(time.time() - t_start, 1),
        'acc_matrix': acc, 'metrics': metrics, 'per_task': per_task,
    }
    with open(tmp, 'w') as f:                 # ghi tam roi doi ten: bi giet giua
        json.dump(payload, f, indent=2)       # chung thi khong de lai file hong
    os.replace(tmp, out)
    return out


def compute_metrics(acc, T):
    """acc[i][j] = accuracy tren task i sau khi hoc xong task j (chi j >= i)."""
    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(T)]
    last = T - 1
    forgetting = [max(acc[i][j] for j in range(i, last)) - acc[i][last]
                  for i in range(last)] if T > 1 else []
    bwt = [acc[i][last] - acc[i][i] for i in range(last)] if T > 1 else []
    return {
        'A_t': [round(a, 2) for a in A_t],
        'A_T': round(A_t[-1], 2),
        'A_bar': round(float(np.mean(A_t)), 2),
        'forgetting': round(float(np.mean(forgetting)), 2) if forgetting else 0.0,
        'BWT': round(float(np.mean(bwt)), 2) if bwt else 0.0,
    }


def main():
    args = validate(get_parser().parse_args())
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # Tensor core bf16 chi co tu Ampere (SM 8.0). KHONG dung
    # torch.cuda.is_bf16_supported(): mac dinh no tinh ca truong hop emulation nen
    # tra ve True tren T4 (Turing, SM 7.5) - bf16 chay duoc nhung bang duong phan
    # mem, trong khi fp16 tren cung con card co tensor core that (65 vs 8.1 TFLOPS).
    args.amp_dtype = None
    if args.amp and torch.cuda.is_available():
        major = torch.cuda.get_device_capability(device)[0]
        args.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"[model] autocast backbone: {str(args.amp_dtype).split('.')[-1]} "
              f"(SM {major}.x)")

    if args.cache_features:
        # backbone dong bang -> trich feature mot lan roi bo backbone di
        backbone = None if cache_exists(args) else load_backbone(args.model_name, device)
        data = TaskData(args, backbone, device)
        del backbone
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        args.embedding_dim = data.Xtr.shape[1]   # lay tu feature that, tranh lech config
        model, reg = build_model(args)
    else:
        # backbone nam trong do thi tinh toan -> phai giu anh goc
        backbone = load_backbone(args.model_name, device)
        n = sum(p.numel() for p in backbone.parameters())
        print(f"[model] backbone {'dong bang' if args.freeze_backbone else 'fine-tune'}: "
              f"{n/1e6:.2f}M tham so")
        data = ImageTaskData(args, device)
        args.embedding_dim = backbone.num_features
        model, reg = build_model(args, backbone=backbone)
        model.prep = make_preprocess(args, device)

    print(data.summary())
    model.to(device)
    print(f"[model] {args.exp_name}")

    T = args.num_tasks
    acc = [[0.0] * T for _ in range(T)]
    per_task, known = [], 0
    t_start = time.time()

    for task in range(T):
        total = data.total_classes(task)
        model.expand_head(total)
        model.to(device)
        tr, va = data.train_split(task)
        print(f"\n=== task {task}  lop [{known}, {total})  "
              f"{tr[0].shape[0]} train / {va[0].shape[0]} val ===")

        tic = time.time()
        info = train_task(model, reg, tr, va, args, task, known)
        info['train_time'] = round(time.time() - tic, 1)

        # dong bang phep chieu sau task dau (nhanh A). Voi expand_dim=0 thi khong co
        # phep chieu nao -> khong bao, neu khong log noi da lam mot viec khong xay ra.
        if args.projection_schedule in ('task0', 'offline') and task == 0 \
                and not model.no_expand:
            model.freeze_projection()
            print("  [proj] da dong bang phep chieu sau task 0")

        if reg.enabled:
            reg.set_alpha(known, total)
            reg.estimate(model, batches(*tr, args.batch_size, shuffle=False), device)

        for i in range(task + 1):
            Xi, Yi = data.test_split(i)
            acc[i][task] = evaluate(model, Xi, Yi, args)

        info.update(diagnostics(model, reg, args))
        info['A_t'] = round(float(np.mean([acc[i][task] for i in range(task + 1)])), 2)
        per_task.append(info)
        print(f"  A_t={info['A_t']:.2f} | " +
              " ".join(f"{k}={v}" for k, v in info.items()
                       if k not in ('A_t', 'val_curve')))       # val_curve chi luu vao JSON
        known = total
        save(args, acc, per_task, compute_metrics(acc, task + 1), t_start)

    m = compute_metrics(acc, T)
    m['total_time'] = round(time.time() - t_start, 1)

    print("\n" + "=" * 60)
    print("Accuracy matrix (hang = task, cot = sau khi hoc xong task j)")
    for i in range(T):
        print("  " + " ".join(f"{acc[i][j]:6.2f}" if j >= i else "     ." for j in range(T)))
    print(f"\nA_t         : {m['A_t']}")
    print(f"A_T         : {m['A_T']}   (accuracy sau task cuoi)")
    print(f"A_bar       : {m['A_bar']}   (accumulated - so voi Fly-CL 92.99)")
    print(f"Forgetting  : {m['forgetting']}")
    print(f"BWT         : {m['BWT']}")
    print(f"Tong thoi gian: {m['total_time']}s")

    print(f"Da luu: {save(args, acc, per_task, m, t_start)}")


if __name__ == '__main__':
    main()
