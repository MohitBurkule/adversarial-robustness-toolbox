"""
H371 - Weight Displacement Trajectory: Standard vs Adversarial Training

Measures how much weights actually move during adversarial vs standard training.
No prior paper has directly measured per-epoch weight displacement, step length,
gradient direction reversals, and per-layer breakdown across training regimes
from the same random initialisation.

Three models from identical init:
  (a) Standard SGD
  (b) FGSM-AT (eps=0.1)
  (c) PGD-AT (eps=0.1, steps=10, alpha=0.01)

Metrics per epoch:
  - displacement ||W_t - W_0||_F
  - step length ||W_t - W_{t-1}||_F
  - mean gradient norm ||g||_F
  - gradient direction reversals (mean cosine sim of consecutive batch grads)

Post-training:
  - per-layer displacement for each model
  - total path length (sum of step lengths)
  - L2 distance between FGSM-AT and standard final weights
"""
import os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_TRAIN = 6000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01


def flatten_params(model):
    return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])


def per_layer_displacement(model, init_state):
    """Return dict of layer_name -> L2 displacement from init."""
    result = {}
    for name, p in model.named_parameters():
        init_p = init_state[name].cpu()
        disp = (p.detach().cpu() - init_p).norm().item()
        result[name] = disp
    return result


def flatten_grads(model):
    gs = []
    for p in model.parameters():
        if p.grad is not None:
            gs.append(p.grad.detach().cpu().flatten())
        else:
            gs.append(torch.zeros(p.numel()))
    return torch.cat(gs)


def train_epoch(model, Xtr, Ytr, optimizer, mode="standard"):
    """Train one epoch, return (mean_grad_norm, mean_cosine_sim_consecutive_grads)."""
    model.train()
    n = Xtr.size(0)
    perm = torch.randperm(n, device=Xtr.device)

    grad_norms = []
    prev_grad = None
    cosines = []

    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = Xtr[idx], Ytr[idx]

        if mode == "fgsm":
            xb = C.fgsm(model, xb, yb, eps=EPS)
        elif mode == "pgd":
            xb = C.pgd(model, xb, yb, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)

        optimizer.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        optimizer.step()

        cur_grad = flatten_grads(model)
        grad_norms.append(cur_grad.norm().item())

        if prev_grad is not None:
            cos = F.cosine_similarity(prev_grad.unsqueeze(0), cur_grad.unsqueeze(0)).item()
            cosines.append(cos)
        prev_grad = cur_grad

    mean_grad_norm = np.mean(grad_norms)
    mean_cosine = np.mean(cosines) if cosines else float('nan')
    return mean_grad_norm, mean_cosine


def run_training(Xtr, Ytr, init_state, mode_name):
    """Run full training, return (model, epoch_records)."""
    meta = C.dataset_meta(DS)
    model = C.build_model("cnn", meta, width=32)
    # Load identical init weights
    model.load_state_dict(copy.deepcopy(init_state))
    model.to(C.DEVICE)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    w0 = flatten_params(model)
    w_prev = w0.clone()

    records = []
    for ep in range(EPOCHS):
        mean_grad_norm, mean_cosine = train_epoch(model, Xtr, Ytr, optimizer, mode=mode_name)
        scheduler.step()

        w_t = flatten_params(model)
        displacement = (w_t - w0).norm().item()
        step_length = (w_t - w_prev).norm().item()
        w_prev = w_t.clone()

        records.append({
            'epoch': ep + 1,
            'displacement': displacement,
            'step_length': step_length,
            'mean_grad_norm': mean_grad_norm,
            'mean_cosine': mean_cosine,
        })
        print(f"  [{mode_name}] ep {ep+1}/{EPOCHS}: disp={displacement:.4f} step={step_length:.4f} "
              f"grad_norm={mean_grad_norm:.4f} cos_sim={mean_cosine:.4f}")

    return model, records


def main():
    t0 = time.time()
    C.set_seed(SEED)

    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN)
    meta = C.dataset_meta(DS)

    # Build model and save init weights
    init_model = C.build_model("cnn", meta, width=32)
    init_state = copy.deepcopy(init_model.state_dict())

    lines = []
    lines.append("H371: Weight Displacement Trajectory — Standard vs Adversarial Training")
    lines.append("=" * 70)
    lines.append(f"N_TRAIN={N_TRAIN}, EPOCHS={EPOCHS}, LR={LR}, BATCH={BATCH}, SEED={SEED}")
    lines.append(f"EPS={EPS}, PGD_STEPS={PGD_STEPS}, PGD_ALPHA={PGD_ALPHA}")
    lines.append("")

    all_records = {}
    models = {}

    for mode_name in ["standard", "fgsm", "pgd"]:
        print(f"\n--- Training: {mode_name} ---")
        model, records = run_training(Xtr, Ytr, init_state, mode_name)
        all_records[mode_name] = records
        models[mode_name] = model

    # Per-epoch summary table
    lines.append("Per-epoch metrics:")
    lines.append(f"{'mode':<10} {'epoch':<6} {'displacement':<14} {'step_length':<14} {'grad_norm':<14} {'cos_sim':<10}")
    for mode_name in ["standard", "fgsm", "pgd"]:
        for r in all_records[mode_name]:
            lines.append(f"{mode_name:<10} {r['epoch']:<6} {r['displacement']:<14.4f} {r['step_length']:<14.4f} "
                         f"{r['mean_grad_norm']:<14.4f} {r['mean_cosine']:<10.4f}")
    lines.append("")

    # Total path length and final displacement
    lines.append("Summary:")
    for mode_name in ["standard", "fgsm", "pgd"]:
        recs = all_records[mode_name]
        total_path = sum(r['step_length'] for r in recs)
        final_disp = recs[-1]['displacement']
        mean_cos = np.mean([r['mean_cosine'] for r in recs])

        # Clean and robust accuracy
        model = models[mode_name]
        model.eval()
        _, clean_acc = C.logits_and_acc(model, Xte, Yte)
        asr_info = C.attack_success(model, Xte, Yte, attack="pgd", eps=EPS, steps=10)
        pgd_asr = asr_info['asr']

        lines.append(f"  {mode_name}: total_path={total_path:.4f}, final_disp={final_disp:.4f}, "
                     f"mean_cos_sim={mean_cos:.4f}, clean_acc={clean_acc:.4f}, pgd_asr={pgd_asr:.4f}")
    lines.append("")

    # Per-layer displacement
    lines.append("Per-layer displacement (final):")
    for mode_name in ["standard", "fgsm", "pgd"]:
        layer_disp = per_layer_displacement(models[mode_name], init_state)
        lines.append(f"  {mode_name}:")
        for lname, d in layer_disp.items():
            lines.append(f"    {lname}: {d:.4f}")
    lines.append("")

    # L2 distance between final weights of standard vs FGSM-AT
    w_std = flatten_params(models["standard"])
    w_fgsm = flatten_params(models["fgsm"])
    w_pgd = flatten_params(models["pgd"])
    d_std_fgsm = (w_std - w_fgsm).norm().item()
    d_std_pgd = (w_std - w_pgd).norm().item()
    d_fgsm_pgd = (w_fgsm - w_pgd).norm().item()
    lines.append(f"Weight-space L2 distances:")
    lines.append(f"  standard <-> fgsm: {d_std_fgsm:.4f}")
    lines.append(f"  standard <-> pgd:  {d_std_pgd:.4f}")
    lines.append(f"  fgsm <-> pgd:      {d_fgsm_pgd:.4f}")

    elapsed = time.time() - t0
    lines.append(f"\nElapsed: {elapsed:.1f}s")

    output = "\n".join(lines)
    print("\n" + output)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "fashion_mnist")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h371_weight_displacement_trajectory_output.txt")
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
