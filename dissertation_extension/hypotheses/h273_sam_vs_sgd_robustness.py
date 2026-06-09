"""
H273 - Sharpness-Aware Minimisation (SAM) vs SGD: does flatness imply robustness?

SAM finds flatter loss minima by perturbing weights toward the worst-case direction
before each gradient step. Flatter minima have been linked to better generalisation.
This hypothesis tests whether flatness (measured by sharpness proxy) also predicts
adversarial robustness, or whether the two are orthogonal properties.

Three models:
  (a) Baseline SGD  (rho=0, standard training)
  (b) SAM rho=0.05  (mild flatness encouragement)
  (c) SAM rho=0.20  (aggressive flatness encouragement)

Sharpness proxy: perturb all weights by iid N(0, 0.01^2) noise, measure average
loss increase over 20 random perturbations. Lower = flatter basin.

Key question: does SAM's flatter minima translate to lower PGD attack success rate?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS            = "fashion_mnist"
SEED          = 0
N_TRAIN       = 10000
EPOCHS        = 10
LR            = 0.05
MOMENTUM      = 0.9
BATCH         = 128
EPS           = 0.1
PGD_STEPS     = 10
PGD_ALPHA     = 0.01
SHARP_SAMPLES = 20
SHARP_SIGMA   = 0.01
OUT_FILE      = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist", "h273_sam_vs_sgd_robustness_output.txt"
)


# ---------------------------------------------------------------------------
# SAM Optimizer
# ---------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("Use first_step / second_step explicitly.")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_sgd(model, X, Y):
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=5e-4)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    SGD epoch {ep+1}/{EPOCHS}  loss={total_loss:.3f}")


def train_sam(model, X, Y, rho):
    sam = SAM(model.parameters(), torch.optim.SGD, rho=rho,
              lr=LR, momentum=MOMENTUM, weight_decay=5e-4)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = X[idx].to(C.DEVICE), Y[idx].to(C.DEVICE)
            # First pass: gradient for ascent step
            sam.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            sam.first_step(zero_grad=True)
            # Second pass: gradient at perturbed weights
            F.cross_entropy(model(xb), yb).backward()
            sam.second_step(zero_grad=True)
            total_loss += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"    SAM(rho={rho}) epoch {ep+1}/{EPOCHS}  loss={total_loss:.3f}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def measure_sharpness(model, X, Y):
    """Average loss increase after N(0, SHARP_SIGMA^2) weight perturbations."""
    model.eval()
    with torch.no_grad():
        base, nb = 0.0, 0
        for i in range(0, len(X), 256):
            xb, yb = X[i:i+256].to(C.DEVICE), Y[i:i+256].to(C.DEVICE)
            base += F.cross_entropy(model(xb), yb).item()
            nb += 1
        base /= nb

    orig = [p.data.clone() for p in model.parameters()]
    deltas = []
    for _ in range(SHARP_SAMPLES):
        with torch.no_grad():
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * SHARP_SIGMA)
        noisy, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb, yb = X[i:i+256].to(C.DEVICE), Y[i:i+256].to(C.DEVICE)
                noisy += F.cross_entropy(model(xb), yb).item()
                nb += 1
        noisy /= nb
        deltas.append(noisy - base)
        with torch.no_grad():
            for p, op in zip(model.parameters(), orig):
                p.data.copy_(op)
    return float(np.mean(deltas)), float(np.std(deltas))


def evaluate(model, Xte, Yte, label):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    _, clean_acc = C.logits_and_acc(model, Xte, Yte)
    Xfgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    _, fgsm_acc = C.logits_and_acc(model, Xfgsm, Yte)
    Xpgd = C.pgd(model, Xte, Yte, eps=EPS, steps=PGD_STEPS, alpha=PGD_ALPHA)
    _, pgd_acc = C.logits_and_acc(model, Xpgd, Yte)
    mean_margin = float(np.mean(C.margin(model, Xte, Yte)))
    fgsm_asr, pgd_asr = 1.0 - fgsm_acc, 1.0 - pgd_acc
    print(f"  [{label}] clean={clean_acc:.4f}  FGSM_ASR={fgsm_asr:.4f}  "
          f"PGD_ASR={pgd_asr:.4f}  margin={mean_margin:.4f}")
    return dict(clean_acc=clean_acc, fgsm_asr=fgsm_asr,
                pgd_asr=pgd_asr, mean_margin=mean_margin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    C.set_seed(SEED)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("Loading data...")
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS)
    idx = torch.randperm(len(Xtr))[:N_TRAIN]
    Xtr, Ytr = Xtr[idx], Ytr[idx]
    Xte, Yte = Xte[:500], Yte[:500]
    print(f"  train={len(Xtr)}  test={len(Xte)}")

    configs = [
        ("sgd_baseline", "sgd",  0.00),
        ("sam_rho005",   "sam",  0.05),
        ("sam_rho020",   "sam",  0.20),
    ]
    results = {}

    for name, method, rho in configs:
        print(f"\n--- Training {name} ---")
        C.set_seed(SEED)
        model = C.build_model("cnn", {"channels": 1, "size": 28, "n_classes": 10},
                              width=32, seed=SEED)
        model.to(C.DEVICE)
        if method == "sgd":
            train_sgd(model, Xtr, Ytr)
        else:
            train_sam(model, Xtr, Ytr, rho=rho)

        r = evaluate(model, Xte, Yte, label=name)
        print("  Measuring sharpness...")
        sm, ss = measure_sharpness(model, Xtr, Ytr)
        print(f"  [{name}] sharpness={sm:.4f}±{ss:.4f}")
        r["sharpness_mean"] = sm
        r["sharpness_std"]  = ss
        results[name] = r

    elapsed = time.time() - t0
    sgd   = results["sgd_baseline"]
    sam5  = results["sam_rho005"]
    sam20 = results["sam_rho020"]

    lines = [
        "H273 - SAM vs SGD Robustness\n",
        "=" * 70 + "\n\n",
        f"N_train={N_TRAIN}  epochs={EPOCHS}  lr={LR}  eps={EPS}  pgd_steps={PGD_STEPS}\n",
        f"Sharpness: {SHARP_SAMPLES} random N(0,{SHARP_SIGMA}^2) perturbations, avg loss increase\n\n",
        f"{'Model':<20} {'CleanAcc':>10} {'FGSM_ASR':>10} {'PGD_ASR':>10} "
        f"{'Margin':>10} {'Sharpness':>16}\n",
        "-" * 76 + "\n",
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<20} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} "
            f"{r['pgd_asr']:>10.4f} {r['mean_margin']:>10.4f} "
            f"  {r['sharpness_mean']:>8.4f}±{r['sharpness_std']:.4f}\n"
        )
    lines.append(f"\nElapsed: {elapsed:.1f}s\n\n")
    lines.append("ANALYSIS\n--------\n")
    lines.append(f"SAM rho=0.05 sharpness delta vs SGD: {sam5['sharpness_mean']-sgd['sharpness_mean']:+.4f}\n")
    lines.append(f"SAM rho=0.20 sharpness delta vs SGD: {sam20['sharpness_mean']-sgd['sharpness_mean']:+.4f}\n")
    lines.append(f"SAM rho=0.05 PGD_ASR  delta vs SGD: {sam5['pgd_asr']-sgd['pgd_asr']:+.4f}\n")
    lines.append(f"SAM rho=0.20 PGD_ASR  delta vs SGD: {sam20['pgd_asr']-sgd['pgd_asr']:+.4f}\n")
    verdict = ("flatness correlates with lower PGD_ASR => flatness may imply robustness."
               if sam20["pgd_asr"] < sgd["pgd_asr"]
               else "flatness does NOT reduce PGD_ASR => flatness and robustness are orthogonal.")
    lines.append(f"Verdict: {verdict}\n")

    with open(OUT_FILE, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {OUT_FILE}")
