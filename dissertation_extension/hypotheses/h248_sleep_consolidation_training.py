"""
H248 - Sleep Consolidation Training: alternating wake/sleep phases.

Alternate "wake" phases (normal SGD on full data, 3 epochs) and "sleep"
phases (SGD on easy samples only = top-50% margin, 1 epoch).
Compare to: standard training, PGD-AT.
Does sleep-wake alternation delay robust overfitting?
Measure: clean accuracy and PGD ASR at each epoch checkpoint.
Also test: sleep on anti-adversarial samples (x - eps*sign(grad)).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
N_EVAL = 300
TOTAL_EPOCHS = 30
BATCH_SIZE = 128
WAKE_EPOCHS = 3
SLEEP_EPOCHS = 1

def eval_asr(model, X, Y):
    Xadv = C.pgd(model, X, Y, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).float().mean().item()

def eval_acc(model, X, Y, batch=256):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb, yb = X[i:i+batch], Y[i:i+batch]
            correct += (model(xb).argmax(1).cpu() == yb.cpu()).sum().item()
    return correct / len(X)

def sgd_epoch(model, X, Y, optimizer, anti_adv=False):
    """One epoch of SGD. If anti_adv, perturb inputs toward correct class."""
    model.train()
    n = len(X)
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb, yb = X[idx], Y[idx]
        if anti_adv:
            # Anti-adversarial: move toward gradient of correct class
            model.eval()
            xb_aa = xb.clone().requires_grad_(True)
            logits = model(xb_aa)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            with torch.no_grad():
                xb = torch.clamp(xb - EPS * xb_aa.grad.sign(), 0, 1)
            model.train()
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)

def train_with_checkpoints(model, Xtr, Ytr, Xte, Yte, mode='standard',
                            total_epochs=TOTAL_EPOCHS, lr=0.01,
                            check_every=5):
    """Train model and record clean_acc / PGD_ASR every check_every epochs."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=1e-4)
    records = []
    epoch = 0

    if mode == 'standard':
        while epoch < total_epochs:
            sgd_epoch(model, Xtr, Ytr, optimizer)
            epoch += 1
            if epoch % check_every == 0:
                ca = eval_acc(model, Xte, Yte)
                asr = eval_asr(model, Xte, Yte)
                records.append((epoch, ca, asr))
                print(f"    [{mode}] Epoch {epoch}: clean={ca:.3f}, ASR={asr:.3f}")

    elif mode == 'pgd_at':
        while epoch < total_epochs:
            model.train()
            n = len(Xtr)
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                xb, yb = Xtr[idx], Ytr[idx]
                xb_adv = C.pgd(model, xb, yb, eps=EPS, steps=5, alpha=EPS/4)
                model.train()
                optimizer.zero_grad()
                F.cross_entropy(model(xb_adv), yb).backward()
                optimizer.step()
            epoch += 1
            if epoch % check_every == 0:
                ca = eval_acc(model, Xte, Yte)
                asr = eval_asr(model, Xte, Yte)
                records.append((epoch, ca, asr))
                print(f"    [{mode}] Epoch {epoch}: clean={ca:.3f}, ASR={asr:.3f}")

    elif mode in ('sleep_wake', 'sleep_wake_anti'):
        anti_adv = (mode == 'sleep_wake_anti')
        while epoch < total_epochs:
            # Wake phase: full data
            for _ in range(WAKE_EPOCHS):
                if epoch >= total_epochs:
                    break
                sgd_epoch(model, Xtr, Ytr, optimizer)
                epoch += 1
                if epoch % check_every == 0:
                    ca = eval_acc(model, Xte, Yte)
                    asr = eval_asr(model, Xte, Yte)
                    records.append((epoch, ca, asr))
                    print(f"    [{mode}] Epoch {epoch}: clean={ca:.3f}, ASR={asr:.3f}")

            # Sleep phase: easy samples (top-50% margin)
            for _ in range(SLEEP_EPOCHS):
                if epoch >= total_epochs:
                    break
                margins = np.array(C.margin(model, Xtr[:2000]))
                threshold = np.median(margins)
                easy_mask = torch.from_numpy(margins >= threshold)
                Xtr_easy = Xtr[:2000][easy_mask]
                Ytr_easy = Ytr[:2000][easy_mask]
                if len(Xtr_easy) == 0:
                    break
                sgd_epoch(model, Xtr_easy, Ytr_easy, optimizer,
                          anti_adv=anti_adv)
                epoch += 1
                if epoch % check_every == 0:
                    ca = eval_acc(model, Xte, Yte)
                    asr = eval_asr(model, Xte, Yte)
                    records.append((epoch, ca, asr))
                    print(f"    [{mode}] Epoch {epoch}: clean={ca:.3f}, ASR={asr:.3f}")

    return records

def main():
    print("=== H248: Sleep Consolidation Training ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    modes = ['standard', 'pgd_at', 'sleep_wake', 'sleep_wake_anti']
    all_records = {}

    for mode in modes:
        print(f"\n[Training mode: {mode}]")
        C.set_seed(SEED)
        model = C.build_model("cnn", meta, seed=SEED)
        records = train_with_checkpoints(model, Xtr, Ytr, Xte_e, Yte_e,
                                         mode=mode,
                                         total_epochs=TOTAL_EPOCHS,
                                         check_every=5)
        all_records[mode] = records

    print(f"\n--- Summary ---")
    print(f"{'Mode':>20} | Final clean acc | Final PGD ASR")
    print("-" * 55)
    for mode in modes:
        recs = all_records[mode]
        if recs:
            _, final_ca, final_asr = recs[-1]
            print(f"  {mode:>18} | {final_ca:>15.3f} | {final_asr:>13.3f}")

    print("\nTrajectory comparison (epoch, clean_acc, PGD_ASR):")
    for mode in modes:
        print(f"  {mode}:")
        for ep, ca, asr in all_records[mode]:
            print(f"    epoch {ep:3d}: clean={ca:.3f}, ASR={asr:.3f}")

    print("\nInterpretation: sleep_wake should show slower ASR increase vs standard "
          "while maintaining reasonable clean accuracy; pgd_at should show lowest ASR.")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
