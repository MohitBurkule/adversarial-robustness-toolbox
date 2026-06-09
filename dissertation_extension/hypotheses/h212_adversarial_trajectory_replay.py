"""
H212 - Adversarial Trajectory Replay: do adversarial gradients transfer better?

Extension of H210 with a key difference: Model A is trained with PGD-AT
(gradient steps computed on adversarial inputs only). The net displacement
D_adv = W_final_AT - W_init captures geometry learned from adversarial examples.

Hypothesis: adversarial gradients explore a more "universal" weight-space
direction — one that captures the decision boundary geometry of the data
manifold rather than class-specific memorisation — so D_adv should transfer
to a different init better than D_clean (from H210).

Three comparisons:
  A_clean : normally-trained model (baseline)
  A_at    : PGD-AT model
  B_clean : random init + D_clean (H210 result: ~11% clean acc)
  B_at    : random init + D_adv   (this hypothesis)
  D_rand  : random init + random displacement ||D_adv|| (control)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
EPOCHS = 8
BATCH = 256
N_EVAL = 500
META = {"channels": 1, "size": 28, "n_classes": 10}

torch.manual_seed(SEED)
np.random.seed(SEED)

print("H212: Adversarial Trajectory Replay")
print("=" * 60)

Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
Xte_eval, Yte_eval = Xte[:N_EVAL], Yte[:N_EVAL]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = torch.utils.data.DataLoader(
    list(zip(Xtr, Ytr)), batch_size=BATCH, shuffle=True,
    generator=torch.Generator().manual_seed(SEED))

def train_clean(seed=SEED):
    model = C.build_model("cnn", META, width=32, seed=seed)
    W0 = [p.data.clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    model.eval()
    D = [p.data.clone() - w0 for p, w0 in zip(model.parameters(), W0)]
    return model, D

def train_at(seed=SEED):
    model = C.build_model("cnn", META, width=32, seed=seed)
    W0 = [p.data.clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_adv = xb.clone().detach()
            for _ in range(3):
                xb_adv = xb_adv.requires_grad_(True)
                loss = F.cross_entropy(model(xb_adv), yb)
                loss.backward()
                xb_adv = (xb_adv.detach() + (EPS/3)*xb_adv.grad.sign())
                xb_adv = torch.max(torch.min(xb_adv, xb+EPS), xb-EPS).clamp(0,1)
            opt.zero_grad()
            F.cross_entropy(model(xb_adv.detach()), yb).backward()
            opt.step()
    model.eval()
    D = [p.data.clone() - w0 for p, w0 in zip(model.parameters(), W0)]
    return model, D

def apply_displacement(D, seed=42):
    model = C.build_model("cnn", META, width=32, seed=seed)
    with torch.no_grad():
        for p, d in zip(model.parameters(), D):
            p.data.add_(d)
    model.eval()
    return model

def random_displacement(D, seed=42, ctrl_seed=999):
    D_norm = sum((d**2).sum().item() for d in D)**0.5
    torch.manual_seed(ctrl_seed)
    model = C.build_model("cnn", META, width=32, seed=seed)
    with torch.no_grad():
        rands = [torch.randn_like(d) for d in D]
        r_norm = sum((r**2).sum().item() for r in rands)**0.5
        scale = D_norm / (r_norm + 1e-10)
        for p, r in zip(model.parameters(), rands):
            p.data.add_(r * scale)
    model.eval()
    return model

def evaluate(model, name):
    with torch.no_grad():
        clean_acc = (model(Xte_eval).argmax(1) == Yte_eval).float().mean().item()
    margins = C.margin(model, Xte_eval)
    X_pgd = C.pgd(model, Xte_eval, Yte_eval, eps=EPS, steps=10, alpha=0.01)
    with torch.no_grad():
        pgd_s = (model(X_pgd).argmax(1) != Yte_eval).cpu().float().numpy()
    try: auroc = roc_auc_score(pgd_s, -np.array(margins))
    except: auroc = float('nan')
    print(f"  {name:<50} clean={clean_acc:.3f}  PGD_ASR={pgd_s.mean():.3f}  AUROC={auroc:.3f}")
    return clean_acc, pgd_s.mean()

print("[Model A_clean] Normal training...")
t0=time.time()
A_clean, D_clean = train_clean()
D_clean_norm = sum((d**2).sum().item() for d in D_clean)**0.5
print(f"  done {time.time()-t0:.1f}s  ||D_clean||={D_clean_norm:.3f}")

print("[Model A_at] PGD-AT training...")
t0=time.time()
A_at, D_adv = train_at()
D_adv_norm = sum((d**2).sum().item() for d in D_adv)**0.5
print(f"  done {time.time()-t0:.1f}s  ||D_adv||={D_adv_norm:.3f}")
print(f"  ||D_adv|| / ||D_clean|| ratio: {D_adv_norm/D_clean_norm:.3f}")

B_clean = apply_displacement(D_clean, seed=42)
B_at    = apply_displacement(D_adv,   seed=42)
B_rand  = random_displacement(D_adv,  seed=42)

print("\n--- Results ---")
evaluate(A_clean, "A_clean (normal training)")
evaluate(A_at,    "A_at (PGD-AT training)")
ca_Bc, asr_Bc = evaluate(B_clean, "B_clean (rand init + D_clean, H210 baseline)")
ca_Ba, asr_Ba = evaluate(B_at,    "B_at    (rand init + D_adv  ← THIS HYPOTHESIS)")
evaluate(B_rand,  "B_rand  (rand init + random displ ||D_adv||)")

print("\n--- Interpretation ---")
print(f"Adversarial displacement transfer: clean={ca_Ba:.3f}  PGD_ASR={asr_Ba:.3f}")
print(f"Clean displacement transfer:       clean={ca_Bc:.3f}  PGD_ASR={asr_Bc:.3f}")
if ca_Ba > ca_Bc + 0.05:
    print("CONFIRMED: Adversarial trajectory transfers better than clean trajectory.")
    print("Adversarial gradients encode more universal geometry.")
elif abs(ca_Ba - ca_Bc) < 0.05:
    print("NO DIFFERENCE: Adversarial vs clean displacement transfers equally poorly.")
    print("Both fail — the init is what matters, not the gradient direction.")
else:
    print("REVERSED: Clean trajectory transfers better than adversarial.")
