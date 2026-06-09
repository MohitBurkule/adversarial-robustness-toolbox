"""
H419 - Attention vs Convolution adversarial profile: ViT-tiny vs SmallCNN (Fashion-MNIST).

Hypothesis: the global receptive field of self-attention creates a different
adversarial attack surface than the local receptive fields of convolutional
layers. Specifically, a ViT operating on 4x4 patches of 28x28 Fashion-MNIST
(49 tokens) may be attacked at a different success rate and margin-distribution
shape than a matched SmallCNN, even when both achieve similar clean accuracy.
Stochastic token dropout (dropping tokens at random during training) is tested
as a cheap attention-specific regulariser that may broaden the attack surface
further or provide implicit adversarial smoothing.

References:
  Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words: Transformers for
    Image Recognition at Scale." ICLR 2021. (arXiv:2010.11929)
  Mahmood et al. (2021) "On the Robustness of Vision Transformers to Adversarial
    Examples." ICCV 2021. (arXiv:2104.02610)

Conditions (trained from scratch, same N_TRAIN / EPOCHS / opt):
  A. SmallCNN baseline  (width=32, BN, ReLU)
  B. ViT-tiny           (patch=4, dim=64, heads=4, depth=4, MLP ratio=2)
  C. ViT-tiny + stochastic token dropout (p_drop=0.1 per token per forward)

Metrics: clean acc, FGSM ASR, PGD ASR, mean decision margin, margin std-dev.

Config: N_TRAIN=6000, N_EVAL=2000, EPOCHS=15, LR=1e-3 (AdamW), SEED=0,
        EPS=0.1, PGD_STEPS=10, patch=4, dim=64, heads=4, depth=4.
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

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
DS         = "fashion_mnist"
N_TRAIN    = 6000
N_EVAL     = 2000
EPOCHS     = 15
LR         = 1e-3
BATCH      = 128
SEED       = 0
EPS        = 0.1
PGD_STEPS  = 10

PATCH      = 4      # patch size (4x4); 28/4 = 7, so 49 tokens
DIM        = 64     # embedding dim
HEADS      = 4
DEPTH      = 4
MLP_RATIO  = 2      # MLP hidden dim = DIM * MLP_RATIO
TOK_DROP_P = 0.1    # per-token drop probability for condition C (training only)

META = {"channels": 1, "size": 28, "n_classes": 10}

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "fashion_mnist",
    "h419_attention_mini_transformer_output.txt",
)

# ---------------------------------------------------------------------------
# ViT-tiny
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and linearly project."""
    def __init__(self, in_ch=1, patch=4, dim=64):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, dim)  where N = (H/patch)*(W/patch)
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """
    Minimal Vision Transformer for 28x28 single-channel inputs.

    patch=4 -> 7x7=49 tokens; prepend CLS token -> sequence length 50.
    stoch_drop_p: probability to zero-out each non-CLS token during training.
    """
    def __init__(self, in_ch=1, img_size=28, patch=4, dim=64, heads=4,
                 depth=4, mlp_ratio=2, n_classes=10, stoch_drop_p=0.0):
        super().__init__()
        assert img_size % patch == 0, "img_size must be divisible by patch"
        n_patches = (img_size // patch) ** 2

        self.patch_embed = PatchEmbed(in_ch, patch, dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.stoch_drop_p = stoch_drop_p

        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, heads, dim * mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

        # weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        tokens = self.patch_embed(x)                      # (B, N, dim)
        cls    = self.cls_token.expand(B, -1, -1)         # (B, 1, dim)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, N+1, dim)
        tokens = tokens + self.pos_embed

        # stochastic token dropout: zero non-CLS tokens with prob p (train only)
        if self.training and self.stoch_drop_p > 0.0:
            # mask shape (B, N, 1); CLS at index 0 is always kept
            mask = torch.bernoulli(
                torch.full((B, tokens.size(1) - 1, 1), 1.0 - self.stoch_drop_p,
                           device=x.device)
            )
            tokens[:, 1:, :] = tokens[:, 1:, :] * mask

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])      # classify from CLS token


# ---------------------------------------------------------------------------
# training (reuses common.train_model pattern but AdamW + cosine)
# ---------------------------------------------------------------------------

def train_model_adamw(model, Xtr, Ytr, epochs, lr, batch, seed):
    C.set_seed(seed)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n     = Xtr.size(0)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def eval_condition(model, Xte, Yte):
    _, acc   = C.logits_and_acc(model, Xte, Yte)
    fg       = C.attack_success(model, Xte, Yte, attack="fgsm", eps=EPS)
    pg       = C.attack_success(model, Xte, Yte, attack="pgd",  eps=EPS,
                                 steps=PGD_STEPS)
    mar      = C.margin(model, Xte, Yte)
    return {
        "acc":       acc,
        "fgsm_asr":  fg["asr"],
        "pgd_asr":   pg["asr"],
        "margin_mean": float(mar.mean()),
        "margin_std":  float(mar.std()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    def flush_file():
        with open(OUT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")

    out("=" * 80)
    out("H419  Attention vs Convolution adversarial profile: ViT-tiny vs SmallCNN")
    out("       Dataset: Fashion-MNIST  (28x28, 1-channel, 10 classes)")
    out("=" * 80)
    out(f"config: N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} EPOCHS={EPOCHS} "
        f"LR={LR} (AdamW wd=1e-2) BATCH={BATCH} SEED={SEED}")
    out(f"        EPS={EPS} PGD_STEPS={PGD_STEPS}")
    out(f"        ViT: patch={PATCH} dim={DIM} heads={HEADS} depth={DEPTH} "
        f"mlp_ratio={MLP_RATIO}")
    out(f"        token-dropout p={TOK_DROP_P} (condition C only, training only)")
    out(f"        device={C.DEVICE}")
    out("")
    out("References:")
    out("  Dosovitskiy et al. 2020 'An Image is Worth 16x16 Words' (arXiv:2010.11929)")
    out("  Mahmood et al. 2021 'On the Robustness of Vision Transformers to Adversarial"
        " Examples' (arXiv:2104.02610)")
    out("")

    # ---- data ----------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = C.load_dataset(
        DS, n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    out(f"data: Xtr={tuple(Xtr.shape)}  Xte={tuple(Xte.shape)}")

    conditions = [
        ("A. SmallCNN",          "cnn",  False),
        ("B. ViT-tiny",          "vit",  False),
        ("C. ViT-tiny+tok-drop", "vit",  True),
    ]

    results = {}

    for label, arch, tok_drop in conditions:
        out("")
        out(f"[{label}] training...")
        t1 = time.time()
        C.set_seed(SEED)

        if arch == "cnn":
            model = C.SmallCNN(in_ch=1, size=28, n_classes=10,
                               width=32, act="relu", bn=True).to(C.DEVICE)
            train_model_adamw(model, Xtr, Ytr, EPOCHS, LR, BATCH, SEED)
        else:
            p = TOK_DROP_P if tok_drop else 0.0
            model = ViTTiny(
                in_ch=1, img_size=28, patch=PATCH, dim=DIM, heads=HEADS,
                depth=DEPTH, mlp_ratio=MLP_RATIO, n_classes=10,
                stoch_drop_p=p,
            ).to(C.DEVICE)
            train_model_adamw(model, Xtr, Ytr, EPOCHS, LR, BATCH, SEED)

        r = eval_condition(model, Xte, Yte)
        results[label] = r
        out(f"    clean_acc={r['acc']:.4f}  FGSM_ASR={r['fgsm_asr']:.4f}  "
            f"PGD_ASR={r['pgd_asr']:.4f}")
        out(f"    margin  mean={r['margin_mean']:+.4f}  "
            f"std={r['margin_std']:.4f}  ({time.time()-t1:.0f}s)")
        flush_file()

    # ---- summary table -------------------------------------------------------
    out("")
    out("=" * 80)
    out("SUMMARY TABLE")
    out("=" * 80)
    hdr = ("{:<26} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
        "condition", "clean_acc", "FGSM_ASR", "PGD_ASR",
        "margin_mean", "margin_std"))
    out(hdr)
    out("-" * len(hdr))
    for label, _, _ in conditions:
        r = results[label]
        out("{:<26} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f} {:>10.4f}".format(
            label, r["acc"], r["fgsm_asr"], r["pgd_asr"],
            r["margin_mean"], r["margin_std"]))
    out("-" * len(hdr))

    # ---- verdict -------------------------------------------------------------
    out("")
    out("=" * 80)
    out("VERDICT")
    out("=" * 80)

    cnn_r = results["A. SmallCNN"]
    vit_r = results["B. ViT-tiny"]
    vtd_r = results["C. ViT-tiny+tok-drop"]

    d_pgd_vit = vit_r["pgd_asr"] - cnn_r["pgd_asr"]
    d_pgd_vtd = vtd_r["pgd_asr"] - cnn_r["pgd_asr"]
    d_mar_vit  = vit_r["margin_mean"] - cnn_r["margin_mean"]

    out(f"  ViT vs CNN  PGD_ASR delta : {d_pgd_vit:+.4f} "
        f"({'ViT harder to attack' if d_pgd_vit < 0 else 'ViT easier to attack'})")
    out(f"  ViT+drop vs CNN PGD delta : {d_pgd_vtd:+.4f} "
        f"({'token-drop helps' if d_pgd_vtd < d_pgd_vit else 'token-drop no extra help'})")
    out(f"  ViT vs CNN margin_mean    : {d_mar_vit:+.4f} "
        f"({'ViT larger margins' if d_mar_vit > 0 else 'ViT smaller margins'})")

    out("")
    # One-line verdict
    threshold = 0.03     # meaningful difference in ASR
    if abs(d_pgd_vit) < threshold:
        verdict = ("NEUTRAL: ViT-tiny and SmallCNN show similar PGD robustness; "
                   "global attention does not substantially change the adversarial "
                   "attack surface at this scale.")
    elif d_pgd_vit < 0:
        verdict = ("POSITIVE: ViT-tiny is more robust to PGD than SmallCNN; "
                   "global receptive field may reduce gradient alignment across "
                   "the input, consistent with Mahmood et al. 2021.")
    else:
        verdict = ("NEGATIVE: ViT-tiny is less robust to PGD than SmallCNN; "
                   "global attention may expose a wider linearisable attack surface "
                   "despite the same classification accuracy.")

    tok_note = ("Token-dropout provides additional robustness." if d_pgd_vtd < d_pgd_vit - threshold
                else "Token-dropout does not add meaningful extra robustness.")
    out(f"  ONE-LINE VERDICT: {verdict} {tok_note}")

    out("")
    out(f"done in {time.time() - t0:.1f}s")
    flush_file()
    print(f"\n[saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
