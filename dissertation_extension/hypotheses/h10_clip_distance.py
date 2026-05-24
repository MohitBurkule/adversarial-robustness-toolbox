"""
Hypothesis H10: CLIP-style image-text cosine distance as a model-free
per-sample adversarial vulnerability predictor on Fashion-MNIST.

Setup:
  * Train a small CNN victim on Fashion-MNIST (10 epochs, Adam).
  * Use CLIP ViT-B/32 (via HuggingFace transformers) to embed each test image
    (upsampled to 224x224, replicated to 3 channels) and the 10 class-name
    prompts ("a photo of a {class}").
  * Per-sample CLIP features:
        - clip_true_cos   : cos sim with true-class text embedding
        - clip_2nd_cos    : cos sim with next-highest class text embedding
        - clip_margin     : clip_true_cos - clip_2nd_cos
  * Baseline features from the victim:
        - victim_margin   : logit margin (top - 2nd) on the *true* class
        - mean_pix, std_pix : raw image statistics
  * Vulnerability targets:
        - flipped_FGSM      eps = 15/255
        - flipped_PGD       eps = 15/255
        - min_eps_FGSM      via binary search (continuous target)
  * Univariate AUROC for every (feature, target).
  * Multivariate logistic regression: does clip_margin add over victim_margin?

Run from repo root with the project venv:
    /mnt/1tbone/trashy/adversarial-robustness-toolbox/.venv/bin/python \
        dissertation_extension/hypotheses/h10_clip_distance.py

Notes / caveats:
  * CLIP was trained on natural RGB photos; Fashion-MNIST is 28x28 greyscale.
    The absolute similarities are low and noisy; what matters is *relative*
    ordering across samples within a class.
  * The "a photo of a {label}" prompt is a sensible default but not tuned.
  * CLIP embedding is the dominant compute cost (~10k test images through
    ViT-B/32). Use --subset to shrink for a quick smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_TEST = 15.0 / 255.0
PGD_STEPS = 20
PGD_STEP_SIZE = 2.0 / 255.0
EPOCHS = 10
BATCH = 128

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# Victim model (matches diagnostic_test.py)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, n)
        self.do1 = nn.Dropout(0.25)
        self.do2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = self.do1(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.do2(x)
        return self.fc2(x)


def train_victim(train_set):
    torch.manual_seed(0)
    np.random.seed(0)
    loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS}  ({time.time()-t0:.1f}s)", flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    g = x.grad.sign().detach()
    return (x.detach() + eps * g).clamp(0.0, 1.0)


def pgd(model, x, y, eps, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    x0 = x.clone().detach()
    xa = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    xa = xa.clamp(0.0, 1.0)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        F.cross_entropy(model(xa), y).backward()
        with torch.no_grad():
            xa = xa + step_size * xa.grad.sign()
            xa = torch.max(torch.min(xa, x0 + eps), x0 - eps).clamp(0.0, 1.0)
    return xa.detach()


def flipped_mask(model, x_adv, y_true, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, x_adv.size(0), batch):
            pred = model(x_adv[i:i+batch]).argmax(1)
            out.append(pred != y_true[i:i+batch])
    return torch.cat(out).cpu().numpy().astype(np.int32)


def min_eps_fgsm(model, x, y, max_eps=64/255, n_steps=8):
    """Per-sample binary search for the smallest FGSM epsilon that flips."""
    N = x.size(0)
    lo = torch.zeros(N, device=x.device)
    hi = torch.full((N,), max_eps, device=x.device)
    # pre-compute gradient sign once (FGSM has a single gradient direction)
    xg = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xg), y).backward()
    sign = xg.grad.sign().detach()
    # iterate
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        x_adv = (x + mid.view(-1, 1, 1, 1) * sign).clamp(0.0, 1.0)
        with torch.no_grad():
            flipped = torch.zeros(N, dtype=torch.bool, device=x.device)
            for i in range(0, N, 512):
                pred = model(x_adv[i:i+512]).argmax(1)
                flipped[i:i+512] = pred != y[i:i+512]
        hi = torch.where(flipped, mid, hi)
        lo = torch.where(flipped, lo, mid)
    # check whether max_eps even flips; if not, set to max_eps (right-censored)
    return hi.cpu().numpy()


# ---------------------------------------------------------------------------
# CLIP embeddings (HuggingFace transformers)
# ---------------------------------------------------------------------------
def compute_clip_features(test_x_28, test_y, batch=64):
    """
    test_x_28 : (N, 1, 28, 28) tensor in [0, 1]
    Returns:
        cos_sim_all : (N, 10) numpy array
    """
    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP ViT-B/32 ...", flush=True)
    model_id = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    # Pre-compute text embeddings
    prompts = [f"a photo of a {c}" for c in CLASS_NAMES]
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        text_emb = clip.get_text_features(**text_inputs)
        text_emb = F.normalize(text_emb, dim=-1)  # (10, D)

    # Manual image preprocessing to avoid PIL roundtrip:
    # CLIP processor uses bicubic resize to 224 + normalisation with
    # mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758).
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)

    N = test_x_28.size(0)
    all_sims = np.zeros((N, 10), dtype=np.float32)
    print(f"Embedding {N} images through CLIP ...", flush=True)
    for i in range(0, N, batch):
        chunk = test_x_28[i:i+batch].to(DEVICE)            # (b, 1, 28, 28)
        chunk = chunk.repeat(1, 3, 1, 1)                   # (b, 3, 28, 28)
        chunk = F.interpolate(chunk, size=(224, 224), mode="bicubic", align_corners=False)
        chunk = chunk.clamp(0.0, 1.0)
        chunk = (chunk - mean) / std
        with torch.no_grad():
            img_emb = clip.get_image_features(pixel_values=chunk)
            img_emb = F.normalize(img_emb, dim=-1)          # (b, D)
            sims = img_emb @ text_emb.T                     # (b, 10)
        all_sims[i:i+batch] = sims.cpu().numpy()
        if (i // batch) % 20 == 0:
            print(f"  {i+chunk.size(0)}/{N}", flush=True)

    # free CLIP
    del clip
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return all_sims


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="Use only this many test samples (smoke test).")
    ap.add_argument("--out", default="h10_results.json")
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    tfm = transforms.ToTensor()
    train_set = datasets.FashionMNIST("./data", train=True,  download=True, transform=tfm)
    test_set  = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)

    print("Training victim CNN ...", flush=True)
    victim = train_victim(train_set)

    # Stack test set
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    if args.subset is not None:
        test_x = test_x[:args.subset]
        test_y = test_y[:args.subset]
    N = test_x.size(0)
    print(f"Test set size: {N}", flush=True)

    test_x_dev = test_x.to(DEVICE)
    test_y_dev = test_y.to(DEVICE)

    # --- Baseline features ---
    print("Computing victim logit margins ...", flush=True)
    with torch.no_grad():
        logits_all = []
        for i in range(0, N, 512):
            logits_all.append(victim(test_x_dev[i:i+512]))
        logits = torch.cat(logits_all)            # (N, 10)
    true_logit = logits[torch.arange(N), test_y_dev]
    logits_masked = logits.clone()
    logits_masked[torch.arange(N), test_y_dev] = -1e9
    second_logit = logits_masked.max(1).values
    victim_margin = (true_logit - second_logit).cpu().numpy()

    flat = test_x.view(N, -1).numpy()
    mean_pix = flat.mean(axis=1)
    std_pix  = flat.std(axis=1)

    # --- CLIP features ---
    cos_all = compute_clip_features(test_x, test_y)   # (N, 10)
    clip_true_cos = cos_all[np.arange(N), test_y.numpy()]
    cos_masked = cos_all.copy()
    cos_masked[np.arange(N), test_y.numpy()] = -np.inf
    clip_2nd_cos  = cos_masked.max(axis=1)
    clip_margin   = clip_true_cos - clip_2nd_cos
    # CLIP zero-shot argmax accuracy as a sanity check
    clip_argmax = cos_all.argmax(axis=1)
    clip_zs_acc = float((clip_argmax == test_y.numpy()).mean())
    print(f"CLIP zero-shot accuracy on Fashion-MNIST: {clip_zs_acc:.3f}", flush=True)

    # --- Attacks / targets ---
    print("FGSM eps=15/255 ...", flush=True)
    x_fgsm = fgsm(victim, test_x_dev, test_y_dev, EPS_TEST)
    flipped_fgsm = flipped_mask(victim, x_fgsm, test_y_dev)

    print("PGD eps=15/255 ...", flush=True)
    # PGD in chunks to avoid OOM on backward
    x_pgd_chunks = []
    for i in range(0, N, 256):
        x_pgd_chunks.append(pgd(victim, test_x_dev[i:i+256], test_y_dev[i:i+256], EPS_TEST))
    x_pgd = torch.cat(x_pgd_chunks)
    flipped_pgd = flipped_mask(victim, x_pgd, test_y_dev)

    print("FGSM min-eps binary search ...", flush=True)
    min_eps = min_eps_fgsm(victim, test_x_dev, test_y_dev)

    # --- Feature table ---
    feats = {
        "victim_margin":  victim_margin,
        "mean_pix":       mean_pix,
        "std_pix":        std_pix,
        "clip_true_cos":  clip_true_cos,
        "clip_2nd_cos":   clip_2nd_cos,
        "clip_margin":    clip_margin,
    }
    targets_binary = {
        "flipped_FGSM": flipped_fgsm,
        "flipped_PGD":  flipped_pgd,
        # threshold min_eps below median => "easy to attack" => vulnerable
        "low_min_eps_FGSM": (min_eps < np.median(min_eps)).astype(np.int32),
    }

    # --- Univariate AUROC (vulnerability = positive class) ---
    print("\nUnivariate AUROC (positive = vulnerable):", flush=True)
    auroc = {}
    for tname, tvec in targets_binary.items():
        auroc[tname] = {}
        if tvec.sum() == 0 or tvec.sum() == len(tvec):
            print(f"  [{tname}] degenerate target ({tvec.sum()}/{len(tvec)} pos)")
            continue
        for fname, fvec in feats.items():
            # Higher margin => MORE robust => flip "score" sign so larger = more vulnerable.
            # We compute AUROC of (-feat) vs target; equivalent reading: deviation from 0.5.
            try:
                a = roc_auc_score(tvec, -fvec)
            except ValueError:
                a = float("nan")
            auroc[tname][fname] = a
            print(f"  {tname:>18s}  {fname:>15s}  AUROC(-feat)={a:.3f}")

    # --- Multivariate: does clip_margin add over victim_margin? ---
    print("\nLogistic regression: does clip_margin add over victim_margin?", flush=True)
    multivar = {}
    for tname, tvec in targets_binary.items():
        if tvec.sum() == 0 or tvec.sum() == len(tvec):
            continue
        Xb = np.stack([victim_margin], axis=1)
        Xf = np.stack([victim_margin, clip_margin], axis=1)
        sc = StandardScaler()
        Xb_s = sc.fit_transform(Xb)
        Xf_s = StandardScaler().fit_transform(Xf)
        lr_b = LogisticRegression(max_iter=1000).fit(Xb_s, tvec)
        lr_f = LogisticRegression(max_iter=1000).fit(Xf_s, tvec)
        a_b = roc_auc_score(tvec, lr_b.predict_proba(Xb_s)[:, 1])
        a_f = roc_auc_score(tvec, lr_f.predict_proba(Xf_s)[:, 1])
        coefs = dict(zip(["victim_margin", "clip_margin"], lr_f.coef_[0].tolist()))
        multivar[tname] = {
            "auroc_victim_only": a_b,
            "auroc_victim_plus_clip": a_f,
            "delta": a_f - a_b,
            "stdised_coefs": coefs,
        }
        print(f"  [{tname}] victim-only AUROC={a_b:.3f}  +clip_margin AUROC={a_f:.3f} "
              f"(delta={a_f-a_b:+.3f})  coefs={coefs}")

    # --- Save ---
    out = {
        "n_test": int(N),
        "epochs": EPOCHS,
        "eps_test": EPS_TEST,
        "clip_zero_shot_accuracy": clip_zs_acc,
        "univariate_auroc": auroc,
        "multivariate": multivar,
        "min_eps_quartiles": np.quantile(min_eps, [0.25, 0.5, 0.75]).tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
