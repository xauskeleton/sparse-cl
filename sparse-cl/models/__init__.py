def get_model(args):
    """Chon phuong phap theo --model_name. Moi model cai dung hai ham:

        update(X, Y)   X la feature da trich cua task hien tai, Y la nhan goc
        predict(X)     tra ve nhan du doan

    Nho giao dien nay ma trainer.py chi co mot vong lap duy nhat cho moi phuong
    phap - giong cach AnaCP to chuc models/ cua ho.
    """
    name = args.model_name_method.lower()

    if name == 'flycl':
        from .flycl import FlyCL
        return FlyCL(num_classes=args.num_classes, expand_dim=args.expand_dim,
                     coding_level=args.coding_level, deg_s4=args.deg_s4,
                     deg_s3=args.deg_s3, w_s3=args.w_s3, b_stage=args.b_stage,
                     branches=args.branches, ridge_lower=args.ridge_lower,
                     ridge_upper=args.ridge_upper, seed=args.seed,
                     stage_norms=args.stage_norms, device=args.device)

    if name == 'flycl_lp':
        from .flycl_lp import FlyCLLearnedProj
        return FlyCLLearnedProj(
            lp_rank=args.lp_rank, lp_epochs=args.lp_epochs, lp_lr=args.lp_lr,
            lp_bs=args.lp_bs, lp_wd=args.lp_wd, lp_pres=args.lp_pres,
            classes_per_task=args.num_classes // args.num_tasks,
            in_dim=args.in_dim,
            num_classes=args.num_classes, expand_dim=args.expand_dim,
            coding_level=args.coding_level, deg_s4=args.deg_s4,
            deg_s3=args.deg_s3, w_s3=args.w_s3, b_stage=args.b_stage,
            branches=args.branches, ridge_lower=args.ridge_lower,
            ridge_upper=args.ridge_upper, seed=args.seed,
            stage_norms=args.stage_norms, device=args.device)

    if name == 'anacp_cp':
        from .anacp_cp import AnaCPProjection
        return AnaCPProjection(
            num_classes=args.num_classes, expand_dim=args.expand_dim,
            coding_level=args.coding_level, synaptic_degree=args.synaptic_degree,
            pos=args.pos, spread=args.spread, dewhiten=args.dewhiten,
            alpha=args.alpha, shrink=args.shrink, nomu=args.nomu,
            input_norm=args.input_norm, ridge_lower=args.ridge_lower,
            ridge_upper=args.ridge_upper, cp_ridge_lower=args.cp_ridge_lower,
            cp_ridge_upper=args.cp_ridge_upper,
            classes_per_task=args.num_classes // args.num_tasks,
            in_dim=args.in_dim, seed=args.seed, device=args.device)

    if name == 'anacp_full':
        from .anacp_full import AnaCPFull
        return AnaCPFull(
            num_classes=args.num_classes, expand_dim=args.expand_dim,
            expand2=args.expand2 or args.expand_dim, coding_level=args.coding_level,
            synaptic_degree=args.synaptic_degree, heads=args.heads, nl1=args.nl1,
            nl2=args.nl2, replay=args.replay, spread=args.spread,
            dewhiten=args.dewhiten, alpha=args.alpha, shrink=args.shrink,
            ridge_lower=args.ridge_lower, ridge_upper=args.ridge_upper,
            classes_per_task=args.num_classes // args.num_tasks,
            in_dim=args.in_dim, seed=args.seed, device=args.device)

    if name == 'anacp_ref':
        from .anacp_ref import load_reference
        return load_reference(args)

    raise ValueError(f"Unknown model name: {name}")
