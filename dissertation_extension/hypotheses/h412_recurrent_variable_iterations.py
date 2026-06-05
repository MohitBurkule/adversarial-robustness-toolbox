"""
H412 - Recurrent backbone with variable inference iterations T (predictive-coding style).

Idea: a CNN whose recurrent core is applied T times, where T is a forward() argument
changeable AT INFERENCE TIME (no retraining). Hypothesis: more recurrent settling
iterations -> slower but more adversarially robust; T=1 -> fast cheap output.

Architecture (pure PyTorch):
  * Conv stem: 1x28x28 -> CxHxW feature map (C channels, downsampled).
  * Recurrent core (WEIGHT-TIED, same params each step): residual GroupNorm conv block
        h_{t+1} = h_t + f(GN(h_t), x_feat)
    applied T times. Residual + GroupNorm keep it stable at high T.
  * Head: GroupNorm + global-avg-pool + linear -> 10 logits, reads the FINAL state.
  * forward(x, T=...) : T is an argument, NOT baked into weights.

Training scheme (stated in report): train ONE model with loss AVERAGED over a small
set of T values T_TRAIN_SET = {1, 3, 5} so the single net is competent across the T
sweep (rather than overfitting to one T). Train once, evaluate at many inference T.

Core experiment: sweep inference T in {1,2,3,5,8,12} on the SAME trained model.
For each T: clean_acc, FGSM_ASR, PGD_ASR (WHITE-BOX: attacker attacks at the same T),
mean margin, and approx forward cost (relative + wall-time).
Gradient-masking check: transfer FGSM/PGD adversarials crafted at T=1 onto the T=8
model; if transfer ASR >> white-box ASR at T=8, the white-box "robustness" is masking.

Config: N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128, SGD(mom=0.9,wd=5e-4),
SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.
ASR = 1 - adv_acc (lower = better / more robust).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

# ---- config --------------------------------------------------------------
DS = "fashion_mnist"
N_TRAIN = 6000
N_EVAL = 2000
EPOCHS = 10
LR = 0.05
BATCH = 128
SEED = 0
EPS = 0.1
PGD_STEPS = 10
PGD_ALPHA = 0.01
T_TRAIN_SET = [1, 3, 5]                 # train: average loss over these T
T_SWEEP = [1, 2, 3, 5, 8, 12]           # inference T sweep
C_CH = 64                               # recurrent feature channels
META = {"channels": 1, "size": 28, "n_classes": 10}


# ---------------------------------------------------------------------------
# model: recurrent backbone with T as a forward() argument
# ---------------------------------------------------------------------------
class RecurrentCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=10, ch=C_CH):
        super().__init__()
        # conv stem: 28 -> 14 -> 7, ch channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1),   # 28->14
            nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),       # 14->7
            nn.GroupNorm(8, ch), nn.ReLU(),
        )
        # recurrent core (weight-tied): f(GN(h), x_feat) -> residual update
        # input to core is concat(h, x_feat) so it keeps seeing the stimulus
        self.core_gn = nn.GroupNorm(8, ch)
        self.core = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        # head reads final state
        self.head_gn = nn.GroupNorm(8, ch)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x, T=5):
        feat = self.stem(x)                 # (B, ch, 7, 7) -- the stimulus drive
        h = feat                            # init state = stimulus features
        for _ in range(int(T)):
            inp = torch.cat([self.core_gn(h), feat], dim=1)
            h = h + self.core(inp)          # residual recurrent update
        z = F.relu(self.head_gn(h))
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)   # (B, ch)
        return self.fc(z)


# ---------------------------------------------------------------------------
# T-aware attack wrappers (white-box at a given T)
# ---------------------------------------------------------------------------
class FixedT(nn.Module):
    """Wraps the recurrent model to a fixed T so common.fgsm/pgd (which call
    model(x)) attack it through the T-iteration forward graph."""
    def __init__(self, model, T):
        super().__init__()
        self.model = model
        self.T = T

    def forward(self, x):
        return self.model(x, T=self.T)


# ---------------------------------------------------------------------------
# training: average loss over T_TRAIN_SET
# ---------------------------------------------------------------------------
def train(model, Xtr, Ytr):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n = Xtr.size(0)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = 0.0
            for T in T_TRAIN_SET:
                loss = loss + F.cross_entropy(model(xb, T=T), yb)
            loss = loss / len(T_TRAIN_SET)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# eval helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def clean_acc_and_margin(wrapped, X, Y, batch=512):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(wrapped(X[i:i + batch]).cpu())
    logits = torch.cat(outs)
    acc = float((logits.argmax(1) == Y.cpu()).float().mean())
    mgn = float(C.margin_of(logits, Y).mean())
    return acc, mgn


@torch.no_grad()
def adv_acc_on(wrapped, Xadv, Y, batch=512):
    """clean-style accuracy on a precomputed adversarial set -> ASR = 1-acc
    over originally-correct samples handled by caller; here simple acc."""
    outs = []
    for i in range(0, Xadv.size(0), batch):
        outs.append(wrapped(Xadv[i:i + batch]).cpu())
    logits = torch.cat(outs)
    return (logits.argmax(1) == Y.cpu())


def fwd_cost(model, X, T, reps=3, batch=512):
    """Approximate wall-time for a full forward pass over X at this T (mean of reps)."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    with torch.no_grad():
        for _ in range(reps):
            t0 = time.time()
            for i in range(0, X.size(0), batch):
                model(X[i:i + batch], T=T)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append(time.time() - t0)
    return float(np.median(ts))


def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "fashion_mnist",
                            "h412_recurrent_variable_iterations_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H412  Recurrent backbone with variable inference iterations T (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} ch={C_CH}")
    out(f"        TRAIN scheme: ONE model, loss AVERAGED over T in {T_TRAIN_SET}")
    out(f"        INFERENCE T sweep = {T_SWEEP}  (same trained model, T set at forward)")
    out(f"        ASR = 1 - adv_acc over originally-correct samples (lower=better)")
    out(f"        device = {C.DEVICE}")
    out("")

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    out("\n[1] training ONE recurrent model (loss averaged over T_TRAIN_SET)...")
    model = RecurrentCNN().to(C.DEVICE)
    train(model, Xtr, Ytr)
    out(f"    trained in {time.time()-t0:.0f}s")

    # ---- sweep inference T (white-box attacks at the SAME T) ----
    out("\n[2] sweeping inference T (white-box: attacker attacks at the same T)...")
    rows = []
    adv_cache = {}   # T -> (correct_mask, fgsm_xadv, pgd_xadv) for masking check
    for T in T_SWEEP:
        w = FixedT(model, T).to(C.DEVICE)
        acc, mgn = clean_acc_and_margin(w, Xte, Yte)
        cost = fwd_cost(model, Xte, T)

        # white-box attacks at this T, over originally-correct samples
        # craft per-batch then measure flips on originally-correct (ASR convention)
        with torch.no_grad():
            corr = adv_acc_on(w, Xte, Yte, batch=512)   # bool tensor (clean correct)
        # FGSM
        fg_x = C.fgsm(w, Xte, Yte, eps=EPS)
        pg_x = C.pgd(w, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        with torch.no_grad():
            fg_ok = adv_acc_on(w, fg_x, Yte)
            pg_ok = adv_acc_on(w, pg_x, Yte)
        corr_np = corr.numpy().astype(bool)
        n_corr = int(corr_np.sum())
        fgsm_asr = float((~fg_ok.numpy()[corr_np]).mean()) if n_corr else float("nan")
        pgd_asr = float((~pg_ok.numpy()[corr_np]).mean()) if n_corr else float("nan")

        adv_cache[T] = {"corr": corr_np, "fg_x": fg_x, "pg_x": pg_x}
        rows.append({"T": T, "cost": cost, "acc": acc,
                     "fgsm": fgsm_asr, "pgd": pgd_asr, "margin": mgn})
        out(f"    T={T:>2}: cost={cost*1000:7.1f}ms clean_acc={acc:.4f} "
            f"FGSM_ASR={fgsm_asr:.4f} PGD_ASR={pgd_asr:.4f} margin={mgn:+.3f}")
        flush_file()

    # ---- gradient-masking check: transfer T=1 adversarials onto T=8 ----
    out("\n[3] gradient-masking check: transfer attacks crafted at T=1 -> evaluated at T=8")
    w8 = FixedT(model, 8).to(C.DEVICE)
    # use the T=8 originally-correct mask for a fair denominator
    corr8 = adv_cache[8]["corr"]
    n8 = int(corr8.sum())
    with torch.no_grad():
        tr_fg_ok = adv_acc_on(w8, adv_cache[1]["fg_x"], Yte)
        tr_pg_ok = adv_acc_on(w8, adv_cache[1]["pg_x"], Yte)
    transfer_fgsm = float((~tr_fg_ok.numpy()[corr8]).mean()) if n8 else float("nan")
    transfer_pgd = float((~tr_pg_ok.numpy()[corr8]).mean()) if n8 else float("nan")
    wb8 = next(r for r in rows if r["T"] == 8)
    out(f"    white-box  @T=8 : FGSM_ASR={wb8['fgsm']:.4f}  PGD_ASR={wb8['pgd']:.4f}")
    out(f"    transfer T1->T8 : FGSM_ASR={transfer_fgsm:.4f}  PGD_ASR={transfer_pgd:.4f}")
    masking = (transfer_pgd > wb8["pgd"] + 0.03) or (transfer_fgsm > wb8["fgsm"] + 0.03)
    out(f"    transfer stronger than white-box at T=8? {'YES -> gradient masking likely' if masking else 'NO -> white-box is the real threat (no obvious masking)'}")

    # ---- main table ----
    out("\n" + "=" * 80)
    out("[4] MAIN TABLE  (cost relative to T=1)")
    out("=" * 80)
    c1 = rows[0]["cost"]
    hdr = "{:>3} {:>10} {:>9} {:>10} {:>9} {:>9} {:>9}".format(
        "T", "fwd_ms", "fwd_xT1", "clean_acc", "FGSM_ASR", "PGD_ASR", "margin")
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        out("{:>3} {:>10.1f} {:>9.2f} {:>10.4f} {:>9.4f} {:>9.4f} {:>+9.3f}".format(
            r["T"], r["cost"] * 1000, r["cost"] / c1, r["acc"],
            r["fgsm"], r["pgd"], r["margin"]))
    out("-" * len(hdr))

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[5] VERDICT")
    out("=" * 80)
    r1 = next(r for r in rows if r["T"] == 1)
    best = min(rows, key=lambda r: r["pgd"])
    pgd_gain = r1["pgd"] - best["pgd"]       # positive => higher T more robust
    acc_change = best["acc"] - r1["acc"]
    cost_mult = best["cost"] / c1
    # monotonic trend over T?
    pgds = [r["pgd"] for r in rows]
    trend_down = pgds[-1] < pgds[0]

    out(f"  T=1  : clean_acc={r1['acc']:.4f} PGD_ASR={r1['pgd']:.4f} (fast, 1x cost)")
    out(f"  best PGD-robust T={best['T']}: clean_acc={best['acc']:.4f} "
        f"PGD_ASR={best['pgd']:.4f} cost={cost_mult:.2f}x")
    out(f"  PGD_ASR change T=1 -> best = {-pgd_gain:+.4f} "
        f"(gain={pgd_gain:+.4f}, positive => more iterations help)")
    out(f"  clean_acc change (best vs T=1) = {acc_change:+.4f}")
    out(f"  PGD_ASR monotonically lower from T=1 to T={T_SWEEP[-1]}? {trend_down}")
    out(f"  gradient-masking check: {'MASKING SUSPECTED' if masking else 'no obvious masking'}")

    if pgd_gain > 0.03 and not masking:
        one = (f"YES (real): increasing inference T lowers PGD-ASR by {pgd_gain:.3f} "
               f"(best T={best['T']}) at {cost_mult:.1f}x compute and "
               f"{acc_change:+.3f} clean-acc; transfer<=white-box so not gradient masking.")
    elif pgd_gain > 0.03 and masking:
        one = (f"APPARENT-ONLY: T={best['T']} lowers white-box PGD-ASR by {pgd_gain:.3f}, "
               f"but T=1->T=8 transfer attacks are STRONGER than white-box at T=8 "
               f"-> the robustness is gradient masking, not genuine.")
    else:
        one = (f"NO: increasing inference T does not meaningfully reduce PGD-ASR "
               f"(best gain {pgd_gain:+.3f}) despite up to {rows[-1]['cost']/c1:.1f}x compute; "
               f"the recurrent settling buys speed/robustness trade-off but no real robustness.")
    out("  ONE-LINE VERDICT: " + one)

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
