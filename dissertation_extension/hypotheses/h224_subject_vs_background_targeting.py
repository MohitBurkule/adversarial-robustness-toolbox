"""
H224 - Do adversarial perturbations concentrate on subject (salient) vs background pixels?

Compute GradCAM-style saliency on last conv layer of SmallCNN.
For each test sample:
  - subject_mask = top 30% pixels by saliency
  - background_mask = bottom 30% pixels
  - compute δ_fgsm energy fractions on subject vs background
Measure AUROC(energy -> FGSM_success), Spearman rho(subject_energy, margin).
Compare: do successful attacks have higher subject_energy than failed attacks?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1

os.makedirs("results/fashion_mnist", exist_ok=True)


def compute_gradcam_saliency(model, X, batch=32):
    """
    Compute GradCAM-style saliency for each sample in X.
    Hook on model.features[-2] (the last Conv2d in the features Sequential).
    SmallCNN.features = [Conv,BN,Act,MaxPool, Conv,BN,Act,MaxPool, Conv,BN,Act,MaxPool]
    indices:              0   1  2  3          4   5  6  7          8   9  10  11
    Last Conv2d is at index 8.
    Returns saliency maps of shape (N, 28, 28) as numpy.
    """
    model.eval()
    N = X.size(0)
    all_saliency = []

    # We'll process one sample at a time for GradCAM to get per-sample gradients
    # But for speed, process in small batches
    for start in range(0, N, batch):
        xb = X[start:start+batch].clone().requires_grad_(False)
        bs = xb.size(0)
        activations = []
        gradients = []

        def fwd_hook(module, inp, out):
            activations.append(out)

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Register hooks on the last conv layer (index 8 in features)
        last_conv = model.features[8]
        fh = last_conv.register_forward_hook(fwd_hook)
        bh = last_conv.register_backward_hook(bwd_hook)

        # Forward pass
        xb_req = xb.requires_grad_(True)
        logits = model(xb_req)
        # Use gradient of max logit w.r.t. feature maps
        max_logit_sum = logits.max(1).values.sum()

        # Backward
        model.zero_grad()
        max_logit_sum.backward()

        fh.remove()
        bh.remove()

        acts = activations[0].detach()   # (bs, C, h, w)
        grads = gradients[0].detach()    # (bs, C, h, w)

        # Global average pool the gradients over spatial dims -> weights
        weights = grads.mean(dim=[2, 3], keepdim=True)  # (bs, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (bs, 1, h, w)
        cam = F.relu(cam)

        # Upsample to 28x28
        cam_up = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
        cam_up = cam_up.squeeze(1).cpu()  # (bs, 28, 28)

        # Normalise per-sample to [0,1]
        for i in range(bs):
            c = cam_up[i]
            mn, mx = c.min(), c.max()
            if mx - mn > 1e-8:
                cam_up[i] = (c - mn) / (mx - mn)
            else:
                cam_up[i] = torch.zeros_like(c)

        all_saliency.append(cam_up.numpy())

    return np.concatenate(all_saliency, axis=0)  # (N, 28, 28)


def energy_fractions(delta_np, saliency_np, top_frac=0.3, bot_frac=0.3):
    """
    For each sample compute energy fraction on subject (top saliency) and
    background (bottom saliency) pixels.
    delta_np: (N, 28, 28) perturbation magnitudes
    saliency_np: (N, 28, 28) saliency values in [0,1]
    Returns subject_energy (N,), background_energy (N,)
    """
    N = delta_np.shape[0]
    n_pix = 28 * 28
    n_top = int(n_pix * top_frac)
    n_bot = int(n_pix * bot_frac)

    subj_e = np.zeros(N)
    back_e = np.zeros(N)

    for i in range(N):
        d = delta_np[i].flatten()     # (784,)
        s = saliency_np[i].flatten()  # (784,)
        d2 = d ** 2
        total = d2.sum()
        if total < 1e-12:
            continue
        order = np.argsort(s)[::-1]   # descending saliency
        subj_idx = order[:n_top]
        back_idx = order[-n_bot:]
        subj_e[i] = d2[subj_idx].sum() / total
        back_e[i] = d2[back_idx].sum() / total

    return subj_e, back_e


def spearman_rho(a, b):
    from scipy.stats import spearmanr
    try:
        r, _ = spearmanr(a, b)
        return float(r)
    except Exception:
        return float("nan")


def main():
    print("=" * 70)
    print("H224 - Subject vs Background adversarial perturbation targeting")
    print("=" * 70)

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_eval=N_EVAL, seed=SEED)
    print(f"Device={C.DEVICE}  N_eval={N_EVAL}  eps={EPS}")

    # Train model
    t0 = time.time()
    model = C.build_model("cnn", meta, width=32, seed=SEED)
    C.train_model(model, Xtr, Ytr, epochs=10)
    print(f"Training done in {time.time()-t0:.1f}s")

    # Compute logit margin
    logit_margin = C.margin(model, Xte, Yte)

    # Compute FGSM adversarial examples
    print("Computing FGSM perturbations...")
    Xadv_fgsm = C.fgsm(model, Xte, Yte, eps=EPS)
    delta = (Xadv_fgsm - Xte).cpu().detach()

    # FGSM success labels
    with torch.no_grad():
        fgsm_preds = model(Xadv_fgsm).argmax(1).cpu()
    fgsm_success = (fgsm_preds != Yte.cpu()).numpy().astype(int)
    print(f"FGSM ASR={fgsm_success.mean():.3f}")

    # Compute GradCAM saliency
    print("Computing GradCAM saliency...")
    t0 = time.time()
    saliency = compute_gradcam_saliency(model, Xte, batch=32)
    print(f"GradCAM done in {time.time()-t0:.1f}s")

    # Perturbation magnitude (absolute value of delta)
    delta_mag = delta.abs().squeeze(1).numpy()  # (N, 28, 28)

    # Compute energy fractions
    subj_e, back_e = energy_fractions(delta_mag, saliency, top_frac=0.30, bot_frac=0.30)

    # Metrics
    mean_subj = float(subj_e.mean())
    mean_back = float(back_e.mean())

    auroc_subj = C.safe_auroc(fgsm_success, subj_e)
    auroc_back = C.safe_auroc(fgsm_success, back_e)
    rho_subj_margin = spearman_rho(subj_e, logit_margin)

    # Compare successful vs failed attacks
    success_mask = fgsm_success.astype(bool)
    fail_mask = ~success_mask
    mean_subj_success = float(subj_e[success_mask].mean()) if success_mask.sum() > 0 else float("nan")
    mean_subj_fail = float(subj_e[fail_mask].mean()) if fail_mask.sum() > 0 else float("nan")
    mean_back_success = float(back_e[success_mask].mean()) if success_mask.sum() > 0 else float("nan")
    mean_back_fail = float(back_e[fail_mask].mean()) if fail_mask.sum() > 0 else float("nan")

    print("\n" + "=" * 70)
    print("RESULTS")
    print(f"  Mean subject_energy (top 30% salient pixels): {mean_subj:.4f}")
    print(f"  Mean background_energy (bot 30% salient px):  {mean_back:.4f}")
    print(f"  Subject/background energy ratio:              {mean_subj/mean_back:.3f}")
    print()
    print(f"  AUROC(subject_energy -> FGSM_success):    {auroc_subj:.4f}")
    print(f"  AUROC(background_energy -> FGSM_success): {auroc_back:.4f}")
    print(f"  Spearman rho(subject_energy, margin):     {rho_subj_margin:.4f}")
    print()
    print(f"  Subject energy: success={mean_subj_success:.4f}  fail={mean_subj_fail:.4f}")
    print(f"  Background energy: success={mean_back_success:.4f}  fail={mean_back_fail:.4f}")
    print()
    print("INTERPRETATION")
    if mean_subj > mean_back:
        print("  Perturbations concentrate MORE on subject (salient) pixels.")
        print("  This supports the view that adversarial attacks exploit semantically")
        print("  meaningful features (subject) rather than background noise.")
    else:
        print("  Perturbations concentrate MORE on background pixels.")
        print("  Adversarial attacks may exploit low-saliency texture/background features.")
    if auroc_subj > auroc_back:
        print(f"  Subject energy is a better predictor of FGSM success (AUROC diff={auroc_subj-auroc_back:.4f}).")
    else:
        print(f"  Background energy is a better predictor of FGSM success (AUROC diff={auroc_back-auroc_subj:.4f}).")
    print("=" * 70)


if __name__ == "__main__":
    main()
