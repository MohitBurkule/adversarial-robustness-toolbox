"""
H184 - Is "human-like illusion training" real robustness or gradient masking?

The Theme-C campaign found that 2-AFC perceptual models trained on HUMAN illusion
labels have PGD attack-success rate ~0.000 (vs ~0.70 for PHYSICAL-label models),
which looks like a robustness win for "training the machine to be confused like a
human". A PGD ASR of exactly 0 is the textbook signature of gradient masking, so
before claiming robustness we apply the standard battery of masking checks
(Athalye et al. 2018; cf. the AutoAttack labels used elsewhere in this project):

  * white-box FGSM and PGD ASR (gradient attacks),
  * black-box random-search ASR (gradient-free; cannot be fooled by masking),
  * transfer ASR (perturbations crafted on the OTHER scheme's model),
  * mean input-gradient L2 norm (masking => vanishing input gradients).

Attacks are untargeted and measured as flips of the model's OWN clean prediction,
so the comparison is independent of which label scheme the model was trained on.

If the human-scheme model's black-box / transfer ASR is high while its white-box
PGD ASR is ~0 and its gradients have collapsed, the "robustness" is masking, not a
genuinely flatter decision surface.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

KINDS = ["ebbinghaus", "muller_lyer", "ponzo"]
SEEDS = [0, 1, 2]
SZ = 32
EPS = 0.1
META = {"channels": 1, "size": SZ, "n_classes": 2}


def train_scheme_model(kind, scheme, seed):
    Xtr, PHY, HUM, LAB = C.make_illusion_dataset(kind, n=2500, label_scheme=scheme, sz=SZ, seed=seed)
    model = C.build_model("cnn", META, width=16, seed=seed)
    C.train_model(model, Xtr, LAB, epochs=6, opt="adam", lr=0.002, ncls=2)
    return model


def random_search_asr(model, X, Yref, eps, K=50, seed=0):
    """Gradient-free black-box L-inf attack: success if any of K random
    perturbations flips the model's own clean prediction."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    with torch.no_grad():
        base = model(X).argmax(1)
        correct = base == Yref
        flipped = torch.zeros_like(correct)
        for _ in range(K):
            noise = (torch.rand(X.shape, generator=g, device=X.device) * 2 - 1) * eps
            xa = (X + noise).clamp(0, 1)
            flipped |= (model(xa).argmax(1) != Yref)
    corr = correct.cpu().numpy().astype(bool)
    fl = (flipped & correct).cpu().numpy()
    return float(fl[corr].mean()) if corr.sum() else float("nan")


def input_grad_norm(model, X, Y, batch=256):
    norms = []
    for i in range(0, X.size(0), batch):
        x = X[i:i + batch].clone().detach().requires_grad_(True)
        y = Y[i:i + batch]
        loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        norms.append(grad.flatten(1).norm(dim=1).detach().cpu())
    return float(torch.cat(norms).mean())


def run_kind_seed(kind, seed):
    mphys = train_scheme_model(kind, "physical", seed)
    mhum = train_scheme_model(kind, "human", seed)

    # fresh physical test set; attack flips of each model's OWN clean prediction
    Xte, PHY, HUM, _ = C.make_illusion_dataset(kind, n=800, label_scheme="physical", sz=SZ, seed=seed + 777)

    out = {"kind": kind, "seed": seed}
    for tag, m, other in [("physical", mphys, mhum), ("human", mhum, mphys)]:
        with torch.no_grad():
            pred = m(Xte).argmax(1)
        acc_phys = float((pred == PHY).float().mean())   # agreement with physical truth
        fgsm = C.attack_success(m, Xte, pred, attack="fgsm", eps=EPS)["asr"]
        pgd = C.attack_success(m, Xte, pred, attack="pgd", eps=EPS, steps=10)["asr"]
        bb = random_search_asr(m, Xte, pred, EPS, K=50, seed=seed)
        gnorm = input_grad_norm(m, Xte, pred)
        # transfer: craft PGD on `other`, apply to m, count flips of m's clean pred
        with torch.no_grad():
            other_pred = other(Xte).argmax(1)
        xadv_other = C.pgd(other, Xte, other_pred, EPS, 10)
        with torch.no_grad():
            transfer = float((m(xadv_other).argmax(1) != pred).float().mean())
        out[tag] = {"phys_acc": acc_phys, "fgsm_asr": fgsm, "pgd_asr": pgd,
                    "blackbox_asr": bb, "transfer_asr": transfer, "grad_norm": gnorm}
    return out


def main():
    print("=" * 74)
    print("H184 - Illusion 'human-like' robustness: real or gradient masking?")
    print("=" * 74)
    print(f"Device={C.DEVICE}  kinds={KINDS}  eps={EPS}")
    rows = []
    for kind in KINDS:
        for s in SEEDS:
            t0 = time.time()
            r = run_kind_seed(kind, s)
            r["runtime_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"\n[{kind} s{s}] ({r['runtime_s']}s)")
            for tag in ("physical", "human"):
                d = r[tag]
                print(f"  {tag:8s}: phys_acc={d['phys_acc']:.3f}  FGSM={d['fgsm_asr']:.3f}  "
                      f"PGD={d['pgd_asr']:.3f}  blackbox={d['blackbox_asr']:.3f}  "
                      f"transfer={d['transfer_asr']:.3f}  grad_norm={d['grad_norm']:.2e}")

    print("\n" + "=" * 74)
    print("MEANS across kinds x seeds")
    for tag in ("physical", "human"):
        def m(k):
            v = [r[tag][k] for r in rows if r[tag][k] == r[tag][k]]
            return sum(v) / len(v) if v else float("nan")
        print(f"  {tag:8s}: phys_acc={m('phys_acc'):.3f}  FGSM={m('fgsm_asr'):.3f}  "
              f"PGD={m('pgd_asr'):.3f}  blackbox={m('blackbox_asr'):.3f}  "
              f"transfer={m('transfer_asr'):.3f}  grad_norm={m('grad_norm'):.2e}")
    print("=" * 74)
    print("Masking verdict: if the HUMAN-scheme model has PGD~0 but blackbox/transfer")
    print("ASR clearly > 0 and a collapsed input-gradient norm, its apparent")
    print("robustness is gradient masking, not a genuinely flatter surface. If")
    print("blackbox/transfer are ALSO ~0, the robustness survives gradient-free")
    print("attack and the human-confusion training is a real effect.")
    print("=" * 74)


if __name__ == "__main__":
    main()
