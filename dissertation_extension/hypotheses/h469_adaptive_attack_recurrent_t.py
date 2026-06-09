"""
H469 - Adaptive attack on H412 variable-T recurrent (gap M4).

Critique of H412:
    H412 found PGD ASR *grew* with inference T (T=1 -> 0.842, T=12 -> 0.900) and
    its only masking check transferred FGSM/PGD crafted at T=1 onto T=8. That
    transfer was WEAKER than the white-box attack at T=8 (0.81 vs 0.89),
    so H412 concluded "no obvious masking." But H412 only tested ONE
    adversary stance: white-box at the same T as inference. The variable-T
    forward is, in fact, a randomisation surface at inference time. A real
    adversary should be evaluated under several stances:

      (a) PGD-T1  : craft the perturbation at T=1 (fastest, transfer baseline).
      (b) PGD-Tk  : craft at the exact inference T (white-box, H412's stance).
      (c) PGD-EOT : EOT-style PGD whose gradient is averaged across all T in
                    {1..8} at each step (Athalye 2018 -- the canonical adaptive
                    attack against randomised / stochastic inference).
      (d) cross-T transfer matrix: source-T x target-T grid, to ask whether the
          "robustness gradient w.r.t. T" is gradient masking that EOT punctures.

    Prior art:
      - Athalye et al. 2018 "Obfuscated Gradients" (EOT-PGD): the right adaptive
        attack against any randomised inference; H412 did not run it.
      - Linsley et al. 2020 (h-GRU / horizontal recurrence): recurrent vision
        nets show *worse* robustness at longer T because deep recurrence
        amplifies adversarial gradients (matches H412 monotonic ASR curve).
      - Liang et al. 2022 "Are Recurrent Networks More Robust?": the apparent
        robustness of recurrent ConvNets vanishes under EOT / iteration-aware
        attacks - exactly the audit H412 missed.
      - Choksi et al. 2021 (Predify): predictive-coding recurrent loops give
        modest empirical robustness ONLY when the attacker doesn't unroll
        through the loop. Adaptive PGD through the loop closes the gap.
      - Tramer et al. 2020 "On Adaptive Attacks to Adversarial Example
        Defenses": the methodology applied here.

Concretely H469:
    1. Retrain the H412 RecurrentCNN (same arch, same training scheme:
       loss averaged over T_TRAIN_SET = {1,3,5}, eps=0.1, 10 epochs, N=6000).
    2. Adaptive attacks A in {PGD-T1, PGD-Tmatch, PGD-EOT-{1..8}}:
       - PGD-T1: PGD attacks model at FIXED T=1 (transfer baseline, cheap).
       - PGD-Tmatch: PGD attacks model at FIXED T equal to inference T.
       - PGD-EOT-{1..8}: at every PGD step the loss is the mean of CE losses
         computed at every T in {1,..,8} on the same x -- one backward pass
         per T, accumulated. This is the standard adaptive attack against a
         T-randomised model.
    3. Cross-T matrix: rows = adversary T at craft time in {1,2,3,5,8,EOT},
       cols = inference T in {1,2,3,5,8}. Cell = ASR (over originally-correct
       at that inference T). EOT row is the adaptive attack.
    4. Verdict (only flags real masking, not the trivial "EOT is harder"
       outcome):
         - If PGD-EOT ASR at the BEST H412 inference T is strictly higher than
           white-box PGD-Tmatch at the same T by >= 0.03, the H412 conclusion
           "no obvious masking" is OVERTURNED.
         - If PGD-T1 transferred onto Tmatch is the strongest column-attack,
           H412's own check was correct and EOT adds nothing.
         - Otherwise: neither adaptive nor transfer beats white-box -> H412
           verdict (no masking, just no robustness either) stands.

Config: identical to H412. N_TRAIN=6000, EPOCHS=10, LR=0.05, BATCH=128,
SGD(mom=0.9,wd=5e-4), SEED=0, EPS=0.1, PGD_STEPS=10, PGD_ALPHA=0.01.
ASCII only. Flush after every block. Do not execute here.
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

# ---- config (mirrors H412) ------------------------------------------------
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
T_TRAIN_SET = [1, 3, 5]
T_EOT_SET = [1, 2, 3, 4, 5, 6, 7, 8]        # adaptive: average gradient across
T_INFER_SWEEP = [1, 2, 3, 5, 8]             # inference Ts evaluated
T_CRAFT_FIXED = [1, 2, 3, 5, 8]             # fixed-T craft stances
C_CH = 64


# ---------------------------------------------------------------------------
# H412 model copied verbatim
# ---------------------------------------------------------------------------
class RecurrentCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=10, ch=C_CH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, ch), nn.ReLU(),
        )
        self.core_gn = nn.GroupNorm(8, ch)
        self.core = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.head_gn = nn.GroupNorm(8, ch)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x, T=5):
        feat = self.stem(x)
        h = feat
        for _ in range(int(T)):
            inp = torch.cat([self.core_gn(h), feat], dim=1)
            h = h + self.core(inp)
        z = F.relu(self.head_gn(h))
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z)


class FixedT(nn.Module):
    def __init__(self, model, T):
        super().__init__()
        self.model = model
        self.T = T

    def forward(self, x):
        return self.model(x, T=self.T)


# ---------------------------------------------------------------------------
# training: same recipe as H412
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
# adaptive PGD: gradient averaged across the EOT T-distribution
# ---------------------------------------------------------------------------
def pgd_eot(model, x, y, eps, steps, alpha, T_set, random_start=True):
    """EOT-style PGD: at every step, accumulate gradient of CE(model(x, T=t), y)
    over t in T_set (the inference T-distribution the defender exposes).

    Note: PGD is conducted in batches; this function expects x already batched.
    """
    x0 = x.clone().detach()
    xa = x0.clone()
    if random_start:
        xa = xa + torch.empty_like(xa).uniform_(-eps, eps)
        xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = 0.0
        for T in T_set:
            loss = loss + F.cross_entropy(model(xa, T=T), y)
        loss = loss / len(T_set)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
    return xa.detach()


def pgd_eot_batched(model, X, Y, eps, steps, alpha, T_set, batch=128):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(pgd_eot(model, X[i:i + batch], Y[i:i + batch],
                            eps=eps, steps=steps, alpha=alpha, T_set=T_set))
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# eval helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_at(model, X, T, batch=512):
    outs = []
    for i in range(0, X.size(0), batch):
        outs.append(model(X[i:i + batch], T=T).argmax(1).cpu())
    return torch.cat(outs)


def correct_mask_at(model, X, Y, T):
    return (predict_at(model, X, T) == Y.cpu()).numpy().astype(bool)


def asr_on_corr(model, Xadv, Y, T, corr_mask):
    pred = predict_at(model, Xadv, T)
    flipped = (pred != Y.cpu()).numpy()
    n = int(corr_mask.sum())
    return float(flipped[corr_mask].mean()) if n else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "fashion_mnist",
        "h469_adaptive_attack_recurrent_t_output.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H469  Adaptive attack on H412 variable-T recurrent CNN (Fashion-MNIST)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} LR={LR} "
        f"BATCH={BATCH} SGD(mom=0.9,wd=5e-4) SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS} PGD_ALPHA={PGD_ALPHA} ch={C_CH}")
    out(f"        TRAIN: loss averaged over T in {T_TRAIN_SET} (same as H412)")
    out(f"        INFER Ts evaluated: {T_INFER_SWEEP}")
    out(f"        CRAFT Ts (fixed):   {T_CRAFT_FIXED}")
    out(f"        CRAFT EOT T-set:    {T_EOT_SET}")
    out(f"        ASR = 1 - acc over originally-correct samples (lower=better/robust)")
    out(f"        device = {C.DEVICE}")
    out("")
    flush_file()

    C.set_seed(SEED)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")
    flush_file()

    out("\n[1] training ONE recurrent model (H412 recipe)...")
    flush_file()
    model = RecurrentCNN().to(C.DEVICE)
    train(model, Xtr, Ytr)
    out(f"    trained in {time.time()-t0:.0f}s")
    flush_file()

    # ---- clean correctness masks per inference T ----
    out("\n[2] clean accuracy per inference T (for ASR denominators)...")
    corr_by_T = {}
    clean_acc_by_T = {}
    for T in T_INFER_SWEEP:
        m = correct_mask_at(model, Xte, Yte, T)
        corr_by_T[T] = m
        clean_acc_by_T[T] = float(m.mean())
        out(f"    T={T:>2}: clean_acc={clean_acc_by_T[T]:.4f}  "
            f"n_correct={int(m.sum())}/{Xte.size(0)}")
    flush_file()

    # ---- craft adversarials: 5 fixed-T stances + 1 EOT stance ----
    out("\n[3] crafting adversarial sets (one per source attack)...")
    flush_file()
    adv_sets = {}   # name -> Xadv tensor on DEVICE
    for Tc in T_CRAFT_FIXED:
        wc = FixedT(model, Tc).to(C.DEVICE)
        tA = time.time()
        Xa = C.pgd(wc, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
        adv_sets[f"PGD-T{Tc}"] = Xa
        out(f"    crafted PGD-T{Tc}    in {time.time()-tA:5.1f}s")
        flush_file()
    tA = time.time()
    Xa_eot = pgd_eot_batched(model, Xte, Yte, eps=EPS, steps=PGD_STEPS,
                             alpha=PGD_ALPHA, T_set=T_EOT_SET, batch=128)
    adv_sets[f"PGD-EOT{min(T_EOT_SET)}-{max(T_EOT_SET)}"] = Xa_eot
    out(f"    crafted PGD-EOT{min(T_EOT_SET)}-{max(T_EOT_SET)} in {time.time()-tA:5.1f}s "
        f"({len(T_EOT_SET)}x grad cost per step)")
    flush_file()

    # ---- cross-T transfer / adaptive matrix ----
    out("\n[4] cross-T attack matrix  (rows = craft stance, cols = inference T)")
    out("    cell = ASR over originally-correct@T  (lower = more robust)")
    flush_file()
    src_names = [f"PGD-T{Tc}" for Tc in T_CRAFT_FIXED] + \
                [f"PGD-EOT{min(T_EOT_SET)}-{max(T_EOT_SET)}"]
    # header
    col_w = 10
    hdr = "{:<22}".format("attack \\ infer T")
    for T in T_INFER_SWEEP:
        hdr += "{:>{w}}".format(f"T={T}", w=col_w)
    out(hdr)
    out("-" * len(hdr))
    matrix = {}   # (src, T_infer) -> asr
    for src in src_names:
        row = "{:<22}".format(src)
        for T in T_INFER_SWEEP:
            asr = asr_on_corr(model, adv_sets[src], Yte, T, corr_by_T[T])
            matrix[(src, T)] = asr
            row += "{:>{w}.4f}".format(asr, w=col_w)
        out(row)
        flush_file()
    out("-" * len(hdr))

    # ---- white-box "PGD-Tmatch" row (diagonal): cell where craft T == infer T
    out("\n[5] white-box PGD-Tmatch baseline (H412's stance, diagonal cells)")
    wb_match = {}
    for T in T_INFER_SWEEP:
        if T in T_CRAFT_FIXED:
            wb_match[T] = matrix[(f"PGD-T{T}", T)]
            out(f"    PGD-Tmatch @ infer T={T} : ASR={wb_match[T]:.4f}")
    flush_file()

    # ---- best-transfer per inference T (max ASR across SRC rows for that col)
    out("\n[6] best-attack-per-T (column max across all craft stances incl. EOT)")
    best_attack = {}
    for T in T_INFER_SWEEP:
        best_src = max(src_names, key=lambda s: matrix[(s, T)])
        best_attack[T] = (best_src, matrix[(best_src, T)])
        wb = wb_match.get(T, float("nan"))
        delta = matrix[(best_src, T)] - wb
        out(f"    infer T={T:>2}: best={best_src:<22} ASR={matrix[(best_src, T)]:.4f} "
            f"(white-box Tmatch={wb:.4f}, delta={delta:+.4f})")
    flush_file()

    # ---- verdict ----
    out("\n" + "=" * 80)
    out("[7] VERDICT")
    out("=" * 80)
    # is EOT strictly stronger than white-box at any inference T?
    eot_name = f"PGD-EOT{min(T_EOT_SET)}-{max(T_EOT_SET)}"
    eot_gains = {T: matrix[(eot_name, T)] - wb_match.get(T, float("nan"))
                 for T in T_INFER_SWEEP if T in wb_match}
    max_eot_gain_T = max(eot_gains, key=lambda T: eot_gains[T])
    max_eot_gain = eot_gains[max_eot_gain_T]

    # is fixed-T transfer (best non-Tmatch) stronger than white-box?
    transfer_gains = {}
    for T in T_INFER_SWEEP:
        if T not in wb_match:
            continue
        best_non_match_src = max(
            (s for s in src_names if s != f"PGD-T{T}" and s != eot_name),
            key=lambda s: matrix[(s, T)],
            default=None)
        if best_non_match_src is not None:
            transfer_gains[T] = matrix[(best_non_match_src, T)] - wb_match[T]
    max_xfer_T = max(transfer_gains, key=lambda T: transfer_gains[T]) \
        if transfer_gains else None
    max_xfer_gain = transfer_gains[max_xfer_T] if max_xfer_T is not None else float("nan")

    out(f"  white-box PGD-Tmatch (H412 stance) ASR per T:")
    for T in T_INFER_SWEEP:
        if T in wb_match:
            out(f"      T={T:>2}: {wb_match[T]:.4f}")
    out(f"  best EOT-vs-Tmatch gain   : T={max_eot_gain_T} delta={max_eot_gain:+.4f}")
    if max_xfer_T is not None:
        out(f"  best transfer-vs-Tmatch gain: T={max_xfer_T} delta={max_xfer_gain:+.4f}")

    THRESH = 0.03
    if max_eot_gain >= THRESH and max_eot_gain >= max_xfer_gain:
        verdict = (
            f"MASKING DETECTED (overturns H412): PGD-EOT-{min(T_EOT_SET)}-{max(T_EOT_SET)} "
            f"raises ASR at infer T={max_eot_gain_T} by {max_eot_gain:+.3f} over "
            f"white-box PGD-Tmatch. The variable-T forward was hiding a slice of "
            f"the loss surface from per-T white-box attacks; EOT averaging exposes "
            f"it. H412 'no obvious masking' was a false negative.")
    elif max_xfer_T is not None and max_xfer_gain >= THRESH:
        verdict = (
            f"TRANSFER MASKING: a fixed-T transfer attack beats white-box "
            f"PGD-Tmatch at infer T={max_xfer_T} by {max_xfer_gain:+.3f}. "
            f"H412's single T1->T8 transfer probe missed the stronger source-T. "
            f"EOT itself does not buy more (max EOT gain {max_eot_gain:+.3f}).")
    elif max_eot_gain >= THRESH or max_xfer_gain >= THRESH:
        verdict = (
            f"BORDERLINE: at least one adaptive/transfer stance crosses +0.03 ASR "
            f"over white-box PGD-Tmatch (EOT {max_eot_gain:+.3f}, transfer "
            f"{max_xfer_gain:+.3f}). Worth a second seed; H412 verdict tentative.")
    else:
        verdict = (
            f"H412 CONFIRMED: neither PGD-EOT-{min(T_EOT_SET)}-{max(T_EOT_SET)} "
            f"(max gain {max_eot_gain:+.3f}) nor any fixed-T transfer "
            f"(max gain {max_xfer_gain:+.3f}) beats white-box PGD-Tmatch by "
            f">= {THRESH}. The variable-T forward is NOT gradient masking; the "
            f"model is genuinely just non-robust at every T (PGD ASR ~0.85-0.90).")
    out("\n  ONE-LINE VERDICT: " + verdict)
    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
