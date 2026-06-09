"""
Extended training-time-attack experiments on Fashion-MNIST.

Hypotheses tested (all run from a single training pass to share compute):
  V1  AccumSign (true label)         - baseline novel attack from prev run
  V2  TrajEnsemble (true label)      - baseline novel attack from prev run
  V3  AccumSign (predicted label)    - avoid noisy true-label grads when model is wrong
  V4  LateTraj (true, last 50%)      - skip early-training noise
  V5  LossWeighted AccumSign         - weight each step's sign by current CE loss
  V6  MarginLoss AccumSign           - use CW-style margin instead of CE
  V7  PerPixelVote                   - per-pixel sign accepted only if >=70% of steps agree
  V8  TrajEnsemble (predicted label) - per-epoch FGSM using argmax labels
  V9  Adam-style accum               - moment/RMSProp normalised gradient accumulation
Baselines: clean, vanilla FGSM on final model.
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda")
EPS = 15.0 / 255.0
BATCH = 128
EPOCHS = 5
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3); self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*12*12, 128); self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25); self.do2 = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2); x = self.do1(x); x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.do2(x); return self.fc2(x)


def fgsm_grad(model, x, y, loss_fn=None):
    """Return sign(grad), grad, loss on (x,y) — model.eval() expected."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if loss_fn is None:
        loss = F.cross_entropy(logits, y)
    else:
        loss = loss_fn(logits, y)
    g = torch.autograd.grad(loss, x)[0]
    return g.sign().detach(), g.detach(), loss.detach()


def margin_loss(logits, y):
    """CW-style untargeted margin: maximize (max_other - true)."""
    one_hot = F.one_hot(y, logits.size(1)).bool()
    true_logit = logits[one_hot]
    other_max = logits.masked_fill(one_hot, -1e9).max(1).values
    return (other_max - true_logit).mean()  # higher = more wrong-confident


def main():
    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST("./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train, BATCH, shuffle=True, num_workers=2)

    # fixed eval batch
    idx = torch.randperm(len(test))[:512]
    fixed_x = torch.stack([test[i][0] for i in idx]).to(DEVICE)
    fixed_y = torch.tensor([test[i][1] for i in idx]).to(DEVICE)

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # accumulators
    accum_true = torch.zeros_like(fixed_x)        # V1: sum sign(grad) true label
    accum_pred = torch.zeros_like(fixed_x)        # V3: sum sign(grad) predicted label
    accum_late = torch.zeros_like(fixed_x)        # V4: late half
    accum_lw   = torch.zeros_like(fixed_x)        # V5: loss-weighted (raw grad * loss)
    accum_marg = torch.zeros_like(fixed_x)        # V6: margin loss sign
    vote_pos   = torch.zeros_like(fixed_x)        # V7: per-pixel +1 counts
    vote_neg   = torch.zeros_like(fixed_x)        # V7: per-pixel -1 counts
    m_grad = torch.zeros_like(fixed_x); v_grad = torch.zeros_like(fixed_x)  # V9
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    traj_true, traj_pred = [], []                 # V2, V8
    # V10: per-sample wrong-class confusion histogram across training (epochs+steps)
    n_classes = 10
    confusion_hist = torch.zeros(fixed_x.size(0), n_classes, device=DEVICE)

    n_steps_total = sum(1 for _ in train_loader) * EPOCHS
    late_start = n_steps_total // 2
    step = 0

    t0 = time.time()
    for ep in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

            # ---- collect on fixed eval batch ----
            model.eval()
            # true-label grad
            s_true, g_true, l_true = fgsm_grad(model, fixed_x, fixed_y)
            accum_true += s_true
            accum_lw   += g_true * l_true.item()   # loss-weighted raw grad
            vote_pos   += (s_true > 0).float()
            vote_neg   += (s_true < 0).float()
            if step >= late_start:
                accum_late += s_true
            # Adam-style on raw gradient
            m_grad = beta1 * m_grad + (1 - beta1) * g_true
            v_grad = beta2 * v_grad + (1 - beta2) * (g_true ** 2)
            # pred-label grad
            with torch.no_grad():
                pred = model(fixed_x).argmax(1)
            s_pred, _, _ = fgsm_grad(model, fixed_x, pred)
            accum_pred += s_pred
            # V10: record wrong-class confusion (only when prediction != truth)
            wrong = pred != fixed_y
            if wrong.any():
                idx_w = torch.arange(fixed_x.size(0), device=DEVICE)[wrong]
                confusion_hist[idx_w, pred[wrong]] += 1.0
            # margin-loss grad
            s_marg, _, _ = fgsm_grad(model, fixed_x, fixed_y, loss_fn=margin_loss)
            accum_marg += s_marg
            step += 1
        # per-epoch FGSM perturbations
        model.eval()
        x_adv_t = (fixed_x + EPS * fgsm_grad(model, fixed_x, fixed_y)[0]).clamp(0, 1)
        traj_true.append((x_adv_t - fixed_x).detach())
        with torch.no_grad():
            pred = model(fixed_x).argmax(1)
        x_adv_p = (fixed_x + EPS * fgsm_grad(model, fixed_x, pred)[0]).clamp(0, 1)
        traj_pred.append((x_adv_p - fixed_x).detach())
        print(f"  epoch {ep+1}/{EPOCHS} done  ({time.time()-t0:.1f}s)")

    model.eval()

    # ===== construct attack samples =====
    def acc_on(x_adv):
        with torch.no_grad():
            return (model(x_adv).argmax(1) == fixed_y).float().mean().item()

    with torch.no_grad():
        clean_acc = acc_on(fixed_x)
    # baseline FGSM on final model
    sf, _, _ = fgsm_grad(model, fixed_x, fixed_y)
    fgsm_final = acc_on((fixed_x + EPS * sf).clamp(0, 1))

    # V1 / V3 / V4 / V5 / V6 / V9
    def pert_sign(accum):
        return (fixed_x + EPS * accum.sign()).clamp(0, 1)

    v1 = acc_on(pert_sign(accum_true))
    v3 = acc_on(pert_sign(accum_pred))
    v4 = acc_on(pert_sign(accum_late))
    v5 = acc_on(pert_sign(accum_lw))
    v6 = acc_on(pert_sign(accum_marg))

    # V7 per-pixel vote with threshold (>=70% agreement); otherwise leave pixel alone
    total = vote_pos + vote_neg
    pos_frac = vote_pos / total.clamp_min(1)
    direction = torch.where(pos_frac >= 0.7, torch.ones_like(pos_frac),
                  torch.where(pos_frac <= 0.3, -torch.ones_like(pos_frac),
                              torch.zeros_like(pos_frac)))
    v7 = acc_on((fixed_x + EPS * direction).clamp(0, 1))

    # V9 Adam-style
    m_hat = m_grad / (1 - beta1 ** step)
    v_hat = v_grad / (1 - beta2 ** step)
    adam_dir = m_hat / (v_hat.sqrt() + eps_adam)
    v9 = acc_on((fixed_x + EPS * adam_dir.sign()).clamp(0, 1))

    # V2 / V8 trajectory ensemble (per-sample worst)
    def traj_best(traj):
        best_loss = torch.full((fixed_x.size(0),), -1e9, device=DEVICE)
        best_adv = fixed_x.clone()
        for p in traj:
            cand = (fixed_x + p).clamp(0, 1)
            with torch.no_grad():
                losses = F.cross_entropy(model(cand), fixed_y, reduction="none")
            mask = losses > best_loss
            best_loss = torch.where(mask, losses, best_loss)
            best_adv[mask] = cand[mask]
        return acc_on(best_adv)

    v2 = traj_best(traj_true)
    v8 = traj_best(traj_pred)

    # V10: targeted FGSM toward early-confusion class. Pick top-1 confused class per sample;
    # fallback (sample never misclassified) -> 2nd-highest final-model logit.
    with torch.no_grad():
        final_logits = model(fixed_x)
        final_2nd = final_logits.masked_fill(
            F.one_hot(fixed_y, n_classes).bool(), -1e9).argmax(1)
    never_wrong = confusion_hist.sum(1) == 0
    target = confusion_hist.argmax(1)
    target[never_wrong] = final_2nd[never_wrong]
    # targeted FGSM: minimise loss on target -> step in -sign(grad_x L(model(x), target))
    xt = fixed_x.clone().detach().requires_grad_(True)
    loss_t = F.cross_entropy(model(xt), target)
    g_t = torch.autograd.grad(loss_t, xt)[0]
    v10 = acc_on((fixed_x - EPS * g_t.sign()).clamp(0, 1))

    # V10 control: random-target FGSM (same mechanism, random non-true class)
    rand_target = torch.randint(0, n_classes, (fixed_x.size(0),), device=DEVICE)
    same = rand_target == fixed_y
    rand_target[same] = (rand_target[same] + 1) % n_classes
    xr = fixed_x.clone().detach().requires_grad_(True)
    loss_r = F.cross_entropy(model(xr), rand_target)
    g_r = torch.autograd.grad(loss_r, xr)[0]
    v10_rand = acc_on((fixed_x - EPS * g_r.sign()).clamp(0, 1))

    # V10 control: targeted FGSM toward final-model 2nd-best logit (no training history used)
    xs = fixed_x.clone().detach().requires_grad_(True)
    loss_s = F.cross_entropy(model(xs), final_2nd)
    g_s = torch.autograd.grad(loss_s, xs)[0]
    v10_2nd = acc_on((fixed_x - EPS * g_s.sign()).clamp(0, 1))

    # How often does early-confusion target agree with final 2nd-best?
    agree_frac = (target == final_2nd).float().mean().item()

    results = {
        "clean": clean_acc,
        "FGSM_final_model": fgsm_final,
        "V1_accum_true": v1,
        "V2_traj_true": v2,
        "V3_accum_pred": v3,
        "V4_late_traj": v4,
        "V5_loss_weighted": v5,
        "V6_margin_loss": v6,
        "V7_per_pixel_vote": v7,
        "V8_traj_pred": v8,
        "V9_adam_accum": v9,
        "V10_targeted_early_confusion": v10,
        "V10b_targeted_random": v10_rand,
        "V10c_targeted_final_2nd_best": v10_2nd,
        "V10_target_agrees_with_final_2nd": agree_frac,
    }
    print("\n=== Accuracy under attack (lower = stronger attack) ===")
    width = max(len(k) for k in results)
    for k, v in results.items():
        marker = "  <-- baseline" if k == "FGSM_final_model" else ""
        beats = "  *beats FGSM*" if v < fgsm_final and k.startswith("V") else ""
        print(f"  {k:<{width}}  {v:.4f}{marker}{beats}")
    with open("results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
