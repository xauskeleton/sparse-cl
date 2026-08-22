"""Entry point duy nhat.

    python run.py --method flycl --grid 0:1,300:1
    python run.py --method flycl --training_method aper --branches 5 --ridge_lower 5
    python run.py --method anacp_cp --pos post --alpha 1
    python run.py --mode sgd --cl_reg ewc_dr

Xem scripts/ de biet sieu tham so dung cho tung bang trong bao cao.

LUU Y ve ten co: `--model_name` o day la BACKBONE (nhu moi bang cua repo nay va
nhu ten file cache), con phuong phap chon bang `--method`. AnaCP dung
`--model_name` cho phuong phap va `--backbone` cho backbone - khac cho nay.
"""

import argparse

from config import get_parser


def build_parser():
    p = get_parser()
    p.add_argument('--mode', default='cil', choices=['cil', 'sgd'],
                   help='cil = nghiem dong; sgd = nhanh huan luyen bang gradient')
    p.add_argument('--method', default='flycl',
                   choices=['flycl', 'flycl_lp', 'anacp_cp', 'anacp_full',
                            'anacp_ref'])
    p.add_argument('--training_method', default='none', choices=['none', 'aper'],
                   help='aper = First-Session Adaptation truoc khi trich feature')
    p.add_argument('--log_dir', default='./logs')

    g = p.add_argument_group('ridge')
    g.add_argument('--ridge_lower', type=int, default=3)
    g.add_argument('--ridge_upper', type=int, default=13)

    g = p.add_argument_group('flycl')
    g.add_argument('--deg_s4', type=int, default=300)
    g.add_argument('--b_stage', default='3', help='stage nao lam khoi b, vd 3 hoac 2,3')
    g.add_argument('--branches', type=int, default=1)
    g.add_argument('--grid', default='0:1,300:1',
                   help='danh sach deg_s3:w_s3. 0:1 = Fly-CL goc')

    g = p.add_argument_group('flycl_lp: hoc phep chieu mot lan o task 0')
    g.add_argument('--lp_rank', type=int, default=64, help='hang cua adapter')
    g.add_argument('--lp_epochs', type=int, default=5)
    g.add_argument('--lp_lr', type=float, default=1e-3)
    g.add_argument('--lp_bs', type=int, default=256)
    g.add_argument('--lp_wd', type=float, default=1e-4)
    g.add_argument('--lp_pres', type=float, default=0.0,
                   help='he so phat giu feature gan ban goc')

    g = p.add_argument_group('anacp')
    g.add_argument('--pos', default='post', choices=['none', 'pre', 'post', 'whiten'])
    g.add_argument('--spread', default='repo', choices=['none', 'repo', 'paper'])
    g.add_argument('--dewhiten', default='inv', choices=['inv', 'correct'])
    g.add_argument('--alpha', type=float, default=1.0)
    g.add_argument('--shrink', type=float, default=1e-6)
    g.add_argument('--nomu', type=int, default=0)
    g.add_argument('--input_norm', type=int, default=0)
    g.add_argument('--cp_ridge_lower', type=int, default=-2)
    g.add_argument('--cp_ridge_upper', type=int, default=10)
    g.add_argument('--heads', type=int, default=1)
    g.add_argument('--nl1', default='topk', choices=['topk', 'gelu'])
    g.add_argument('--nl2', default='topk', choices=['topk', 'gelu', 'none'])
    g.add_argument('--expand2', type=int, default=0)
    g.add_argument('--replay', type=int, default=100)
    g.add_argument('--D', type=int, default=5000, help='anacp_ref')
    g.add_argument('--reg', type=float, default=1e2, help='anacp_ref')
    g.add_argument('--num_heads', type=int, default=3, help='anacp_ref')
    g.add_argument('--samples_per_class', type=int, default=100, help='anacp_ref')
    g.add_argument('--anacp_path', default='../upstream/AnaCP')
    return p


def main():
    a = build_parser().parse_args()
    a.cache_features = a.freeze_backbone = True
    a.model_name_method = a.method
    a.b_stage = [int(x) for x in a.b_stage.split(',')]

    if a.mode == 'sgd':
        from trainer_sgd import train_sgd
        return train_sgd(a)

    from trainer import train_cil
    if a.method != 'flycl':
        a.deg_s3, a.w_s3 = 0, 1.0
        return train_cil(a)

    # flycl: --grid cho phep chay nhieu cau hinh trong mot lan nap feature
    base = None
    for spec in a.grid.split(','):
        x, y = spec.split(':')
        a.deg_s3, a.w_s3 = int(x), float(y)
        print(f"\n=== deg_s3={a.deg_s3} w_s3={a.w_s3:g} ===")
        m = train_cil(a)
        base = m['A_bar'] if base is None else base
        print(f"delta A_bar so voi dong dau: {m['A_bar'] - base:+.2f}")


if __name__ == '__main__':
    main()
