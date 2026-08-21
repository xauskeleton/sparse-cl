"""Vong lap continual learning dung chung cho moi phuong phap.

Chi so bao cao khop dinh nghia cua Fly-CL de so truc tiep voi log cua ho:
    A_t  = trung binh accuracy tren cac task 0..t sau khi hoc xong task t
    A_T  = A_t o task cuoi              ("last stage accuracy")
    A_bar= trung binh cua A_t           ("accumulated accuracy")
"""

import json
import os
import time

import numpy as np
import torch

from backbone import cache_exists, load_backbone
from data_loader import TaskData, set_seed
from models import get_model

STAGE_DIMS = (256, 512, 1024, 2048)


def compute_metrics(acc, T):
    A_t = [float(np.mean([acc[i][t] for i in range(t + 1)])) for t in range(T)]
    last = T - 1
    fg = [max(acc[i][j] for j in range(i, last)) - acc[i][last] for i in range(last)]
    return {'A_t': [round(a, 2) for a in A_t],
            'A_T': round(A_t[-1], 2),
            'A_bar': round(float(np.mean(A_t)), 2),
            'forgetting': round(float(np.mean(fg)), 2) if fg else 0.0}


def _ensure_fsa(args, device):
    """--training_method aper: thich nghi backbone o task 0 roi dong bang.

    Khac AnaCP mot cho: ho chay FSA moi lan; o day feature duoc cache nen chi
    chay mot lan roi dung lai (500s -> 0s cho cac run sau). Tag cache chua SEED
    vi FSA phu thuoc 10 lop nao roi vao task 0.
    """
    if args.training_method in (None, 'none', 'None'):
        return args.model_name
    if args.training_method != 'aper':
        raise ValueError(f"training_method khong ho tro: {args.training_method}")

    base = args.model_name.replace('+ms', '')
    tag = f"{base}+fsa{args.seed}" + ('+ms' if args.model_name.endswith('+ms') else '')
    probe = type(args)(**vars(args))
    probe.model_name = tag
    if not cache_exists(probe):
        from fsa import run_fsa
        print(f"[fsa] chua co cache {tag}, chay FSA truoc...")
        run_fsa(args, device)
    return tag


def train_cil(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args.seed)

    args.model_name = _ensure_fsa(args, device)
    backbone = None if cache_exists(args) else load_backbone(args.model_name, device)
    data = TaskData(args, backbone, device)
    del backbone

    args.in_dim = data.Xtr.shape[1]
    # Chuan hoa thang giua cac stage, tinh mot lan tren toan bo train set.
    if args.in_dim == sum(STAGE_DIMS):
        off = np.cumsum((0,) + STAGE_DIMS)
        args.stage_norms = [data.Xtr[:, off[i]:off[i + 1]].norm(dim=1).mean().item()
                            for i in range(4)]
    else:
        args.stage_norms = [1.0] * 4

    model = get_model(args)
    print(f"[trainer] {args.method} | backbone {args.model_name} | d={args.in_dim} "
          f"| {args.num_tasks} task | seed {args.seed}", flush=True)

    T = args.num_tasks
    acc = [[0.0] * T for _ in range(T)]
    t0 = time.time()

    for task in range(T):
        (Xtr, Ytr), (Xva, Yva) = data.train_split(task)
        Xtr, Ytr = torch.cat([Xtr, Xva]), torch.cat([Ytr, Yva])
        model.update(Xtr, Ytr)
        for i in range(task + 1):
            Xte, Yte = data.test_split(i)
            acc[i][task] = (model.predict(Xte) == Yte).float().mean().item() * 100
        print(f"  task {task}: A_t="
              f"{np.mean([acc[i][task] for i in range(task + 1)]):.2f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    m = compute_metrics(acc, T)
    if hasattr(model, 'diagnostics'):
        m.update({k: round(v, 4) for k, v in model.diagnostics().items()})

    print("\nAccuracy matrix (hang = task, cot = sau khi hoc xong task j)")
    for i in range(T):
        print("  " + " ".join(f"{acc[i][j]:6.2f}" if j >= i else "     ."
                              for j in range(T)))
    print(f"\nA_t        : {m['A_t']}")
    print(f"A_T        : {m['A_T']}")
    print(f"A_bar      : {m['A_bar']}")
    print(f"Forgetting : {m['forgetting']}")
    for k, v in m.items():
        if k not in ('A_t', 'A_T', 'A_bar', 'forgetting'):
            print(f"{k:11s}: {v}")
    print(f"Tong       : {time.time() - t0:.0f}s")

    out = os.path.join(args.log_dir, args.dataset, args.model_name, str(args.seed))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{args.method}.json"), 'w') as f:
        json.dump({'args': {k: str(v) for k, v in vars(args).items()},
                   'acc_matrix': acc, 'metrics': m}, f, indent=2)
    return m
