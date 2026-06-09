"""
H221 - Temporal averaging (simulated video) as adversarial defence.

Create "video" sequences: for each image x, create 3-frame clip:
  [x + N(0,0.05²), x, x + N(0,0.05²)]

Train 2 models:
  1. image_model: standard CNN on single frames
  2. video_model: same CNN but input is mean of 3-frame clip

Attack both:
  - frame_independent: FGSM on single frame
  - temporally_consistent: same δ applied to all 3 frames

Key question: does temporal averaging act as natural smoothing against
frame-independent attacks?
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

os.makedirs("results/fashion_mnist", exist_ok=True)

DS = "fashion_mnist"
N_EVAL = 300
SEED = 0
EPS = 0.1
NOISE_STD = 0.05
EPOCHS = 10
BATCH = 128


# ---------------------------------------------------------------------------
# Video averaging: make 3-frame clip and return mean
# ---------------------------------------------------------------------------
def make_video_input(x, noise_std=NOISE_STD, seed=None):
    """x: (B,C,H,W) -> averaged 3-frame clip (B,C,H,W)."""
    n1 = torch.randn_like(x) * noise_std
    n2 = torch.randn_like(x) * noise_std
    frames = torch.stack([x + n1, x, x + n2], dim=0)  # (3,B,C,H,W)
    return frames.mean(dim=0).clamp(0, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_image_model(model, Xtr, Ytr, epochs=EPOCHS):
    C.train_model(model, Xtr, Ytr, epochs=epochs, opt="sgd", lr=0.05, ncls=10)


def train_video_model(model, Xtr, Ytr, epochs=EPOCHS):
    """Train on mean of 3-frame clips."""
    opt = C.make_optimizer(model, "sgd", 0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx]
            yb = Ytr[idx]
            x_avg = make_video_input(xb)
            opt.zero_grad()
            loss = F.cross_entropy(model(x_avg), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()


# ---------------------------------------------------------------------------
# Attack helpers
# ---------------------------------------------------------------------------
def fgsm_delta(model, x, y, eps):
    """Return signed-gradient delta (not clamped to [0,1])."""
    xc = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(xc), y)
    g, = torch.autograd.grad(loss, xc)
    return (eps * g.sign()).detach()


def asr_frame_independent_image(model, Xte, Yte, eps=EPS):
    """Standard FGSM on image model."""
    res = C.attack_success(model, Xte, Yte, attack="fgsm", eps=eps)
    return res["asr"]


def asr_frame_independent_video(model, Xte, Yte, eps=EPS, noise_std=NOISE_STD):
    """
    Frame-independent attack on video model:
    - compute FGSM delta on the clean centre frame
    - apply delta to centre frame only, noisy frames are unperturbed
    - video model sees mean([n1+x, x+δ, n2+x])
    """
    model.eval()
    n_correct = 0
    n_flipped = 0
    for i in range(0, Xte.size(0), 256):
        x = Xte[i:i+256]
        y = Yte[i:i+256]
        # check clean correctness (on video input)
        with torch.no_grad():
            x_avg_clean = make_video_input(x)
            clean_pred = model(x_avg_clean).argmax(1)
            correct_mask = (clean_pred == y)

        # FGSM delta on clean frame (through model)
        delta = fgsm_delta(model, x, y, eps)

        # build adversarial video: perturb centre frame, noisy frames unchanged
        n1 = torch.randn_like(x) * noise_std
        n2 = torch.randn_like(x) * noise_std
        frames_adv = torch.stack([
            (x + n1).clamp(0, 1),
            (x + delta).clamp(0, 1),
            (x + n2).clamp(0, 1),
        ], dim=0)
        x_avg_adv = frames_adv.mean(dim=0).clamp(0, 1)

        with torch.no_grad():
            adv_pred = model(x_avg_adv).argmax(1)
            flipped = (adv_pred != y)

        n_correct += int(correct_mask.sum())
        n_flipped += int((flipped & correct_mask).sum())

    return float(n_flipped / n_correct) if n_correct > 0 else float("nan")


def asr_temporally_consistent_image(model, Xte, Yte, eps=EPS):
    """Same δ on all 3 frames, but image model only sees single frame."""
    # For image model: model sees single frame. Temporally consistent = same as standard FGSM.
    return asr_frame_independent_image(model, Xte, Yte, eps)


def asr_temporally_consistent_video(model, Xte, Yte, eps=EPS, noise_std=NOISE_STD):
    """
    Temporally consistent attack on video model:
    - compute FGSM delta on clean frame
    - apply SAME delta to all 3 frames (noisy + delta)
    - video model sees mean([n1+x+δ, x+δ, n2+x+δ])
    """
    model.eval()
    n_correct = 0
    n_flipped = 0
    for i in range(0, Xte.size(0), 256):
        x = Xte[i:i+256]
        y = Yte[i:i+256]
        with torch.no_grad():
            x_avg_clean = make_video_input(x)
            correct_mask = (model(x_avg_clean).argmax(1) == y)

        delta = fgsm_delta(model, x, y, eps)

        n1 = torch.randn_like(x) * noise_std
        n2 = torch.randn_like(x) * noise_std
        frames_adv = torch.stack([
            (x + n1 + delta).clamp(0, 1),
            (x + delta).clamp(0, 1),
            (x + n2 + delta).clamp(0, 1),
        ], dim=0)
        x_avg_adv = frames_adv.mean(dim=0).clamp(0, 1)

        with torch.no_grad():
            adv_pred = model(x_avg_adv).argmax(1)
            flipped = (adv_pred != y)

        n_correct += int(correct_mask.sum())
        n_flipped += int((flipped & correct_mask).sum())

    return float(n_flipped / n_correct) if n_correct > 0 else float("nan")


def main():
    print("=" * 74)
    print("H221 - Temporal averaging (simulated video) as adversarial defence")
    print("=" * 74)
    print(f"Device={C.DEVICE}  dataset={DS}  N_EVAL={N_EVAL}  eps={EPS}  noise_std={NOISE_STD}")

    C.set_seed(SEED)
    meta = C.dataset_meta(DS)
    Xtr, Ytr, Xte, Yte = C.load_dataset(DS, n_eval=N_EVAL, seed=SEED)

    # --- Train image model ---
    print("\n[image_model] Training standard CNN on single frames...")
    t0 = time.time()
    C.set_seed(SEED)
    image_model = C.build_model("cnn", meta, width=32, seed=SEED)
    train_image_model(image_model, Xtr, Ytr)
    print(f"  Done in {time.time()-t0:.1f}s")

    # --- Train video model ---
    print("\n[video_model] Training CNN on averaged 3-frame clips...")
    t0 = time.time()
    C.set_seed(SEED)
    video_model = C.build_model("cnn", meta, width=32, seed=SEED)
    train_video_model(video_model, Xtr, Ytr)
    print(f"  Done in {time.time()-t0:.1f}s")

    # --- Clean accuracy ---
    with torch.no_grad():
        logits_img, clean_acc_img = C.logits_and_acc(image_model, Xte, Yte)
        # video model evaluated on averaged clips
        Xte_avg = make_video_input(Xte)
        logits_vid, clean_acc_vid = C.logits_and_acc(video_model, Xte_avg, Yte)

    print(f"\nClean accuracy - image_model: {clean_acc_img:.3f}  video_model: {clean_acc_vid:.3f}")

    # --- Attack evaluations ---
    print("\nComputing attack success rates...")

    # image model
    print("  image_model: frame_independent (FGSM on single frame)...")
    asr_img_fi = asr_frame_independent_image(image_model, Xte, Yte)
    print(f"    ASR={asr_img_fi:.3f}")

    print("  image_model: temporally_consistent (same δ, single frame sees it)...")
    asr_img_tc = asr_temporally_consistent_image(image_model, Xte, Yte)
    print(f"    ASR={asr_img_tc:.3f}")

    # video model
    print("  video_model: frame_independent (δ only on centre frame)...")
    asr_vid_fi = asr_frame_independent_video(video_model, Xte, Yte)
    print(f"    ASR={asr_vid_fi:.3f}")

    print("  video_model: temporally_consistent (same δ on all frames)...")
    asr_vid_tc = asr_temporally_consistent_video(video_model, Xte, Yte)
    print(f"    ASR={asr_vid_tc:.3f}")

    print("\n" + "=" * 74)
    print(f"{'Model':<16} {'CleanAcc':>9} {'FrameIndep_ASR':>15} {'TempConsist_ASR':>16}")
    print("-" * 58)
    print(f"{'image_model':<16} {clean_acc_img:>9.3f} {asr_img_fi:>15.3f} {asr_img_tc:>16.3f}")
    print(f"{'video_model':<16} {clean_acc_vid:>9.3f} {asr_vid_fi:>15.3f} {asr_vid_tc:>16.3f}")

    print("\n" + "=" * 74)
    print("Interpretation:")
    d_fi = asr_vid_fi - asr_img_fi
    d_tc = asr_vid_tc - asr_img_tc
    print(f"  Frame-independent ASR delta (video-image): {d_fi:+.3f}")
    print(f"  Temporally-consistent ASR delta (video-image): {d_tc:+.3f}")
    print("  Negative delta on frame-independent = temporal averaging acts as smoothing.")
    print("  Similar delta on temporally-consistent = attack not neutralised by averaging.")
    print("=" * 74)


if __name__ == "__main__":
    main()
