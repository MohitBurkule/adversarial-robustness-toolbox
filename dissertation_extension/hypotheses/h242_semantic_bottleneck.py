"""
H242 - Semantic Bottleneck: Vector Quantisation bottleneck as adversarial defence.

Add a VQ bottleneck after the penultimate layer. Codebook sizes C ∈ [16, 64, 256].
VQ layer: straight-through estimator, replace feature with nearest codebook entry.
Train end-to-end. Measure: clean accuracy and PGD ASR at each C.
Hypothesis: smaller codebook → more robust but lower accuracy (Pareto trade-off).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from campaign import common as C

SEED = 0
EPS = 0.1
N_EVAL = 300
EPOCHS = 15
CODEBOOK_SIZES = [16, 64, 256]
BATCH_SIZE = 128

class VQLayer(nn.Module):
    """Vector Quantisation with straight-through estimator."""
    def __init__(self, n_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.codebook = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.normal_(self.codebook.weight, mean=0, std=1.0)

    def forward(self, z):
        # z: (B, D)
        # distances to codebook entries
        distances = (
            (z ** 2).sum(dim=1, keepdim=True)
            - 2 * z @ self.codebook.weight.T
            + (self.codebook.weight ** 2).sum(dim=1)
        )  # (B, n_embeddings)
        indices = distances.argmin(dim=1)  # (B,)
        z_q = self.codebook(indices)  # (B, D)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # VQ commitment loss
        vq_loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        return z_q_st, vq_loss


class VQModel(nn.Module):
    """CNN with VQ bottleneck before final classifier."""
    def __init__(self, base_model, feature_dim, n_embeddings, n_classes):
        super().__init__()
        # Separate backbone and head from base_model
        # We'll use the base_model as feature extractor and add our own head
        self.backbone = base_model
        self.vq = VQLayer(n_embeddings, feature_dim)
        self.classifier = nn.Linear(feature_dim, n_classes)
        self._feature_dim = feature_dim

    def forward(self, x, return_vq_loss=False):
        # Extract features via backbone's feature layers
        features = self._get_features(x)
        z_q, vq_loss = self.vq(features)
        logits = self.classifier(z_q)
        if return_vq_loss:
            return logits, vq_loss
        return logits

    def _get_features(self, x):
        # Run all layers of backbone except the last linear
        model = self.backbone
        # Try standard CNN pattern: iterate named children
        layers = list(model.children())
        # Find the last linear layer index
        last_linear_idx = None
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                last_linear_idx = i

        if last_linear_idx is not None:
            for i, layer in enumerate(layers[:last_linear_idx]):
                x = layer(x)
            # Flatten if needed
            if x.dim() > 2:
                x = x.flatten(1)
        else:
            # Fallback: run whole model and use as features
            x = model(x)
        return x


def get_feature_dim(model, meta):
    """Get the feature dimension before the last linear layer."""
    dummy = torch.zeros(1, meta['channels'], meta['size'], meta['size'])
    dummy = dummy.to(next(model.parameters()).device)
    layers = list(model.children())
    last_linear_idx = None
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            last_linear_idx = i

    if last_linear_idx is not None:
        with torch.no_grad():
            x = dummy
            for layer in layers[:last_linear_idx]:
                x = layer(x)
            if x.dim() > 2:
                x = x.flatten(1)
        return x.shape[1]
    else:
        return meta.get('n_classes', 10)


def train_vq_model(vq_model, Xtr, Ytr, epochs, lr=0.01):
    device = next(vq_model.parameters()).device
    optimizer = optim.SGD(vq_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    vq_model.train()
    n = len(Xtr)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = Xtr[idx]
            yb = Ytr[idx]
            optimizer.zero_grad()
            logits, vq_loss = vq_model(xb, return_vq_loss=True)
            ce_loss = F.cross_entropy(logits, yb)
            loss = ce_loss + vq_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")


def eval_model(model, X, Y, batch=256):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch]
            yb = Y[i:i+batch]
            preds = model(xb).argmax(1)
            correct += (preds.cpu() == yb.cpu()).sum().item()
    return correct / len(X)


def pgd_asr(model, X, Y):
    Xadv = C.pgd(model, X, Y, eps=EPS, steps=10, alpha=0.01)
    model.eval()
    with torch.no_grad():
        preds = model(Xadv).argmax(1).cpu()
    return (preds != Y.cpu()).float().mean().item()


def main():
    print("=== H242: Semantic Bottleneck (VQ) ===")
    t0 = time.time()

    C.set_seed(SEED)
    meta = C.dataset_meta("fashion_mnist")
    Xtr, Ytr, Xte, Yte = C.load_dataset("fashion_mnist")
    Xte_e, Yte_e = Xte[:N_EVAL], Yte[:N_EVAL]

    # [1] Train baseline CNN (no VQ)
    print("\n[1] Training baseline CNN (no VQ)...")
    model_base = C.build_model("cnn", meta, seed=SEED)
    C.train_model(model_base, Xtr, Ytr, epochs=EPOCHS)
    base_clean = eval_model(model_base, Xte_e, Yte_e)
    base_asr = pgd_asr(model_base, Xte_e, Yte_e)
    print(f"    Baseline: clean={base_clean:.3f}, PGD_ASR={base_asr:.3f}")

    # Determine feature dim
    feature_dim = get_feature_dim(model_base, meta)
    print(f"    Feature dim before classifier: {feature_dim}")

    # [2] Train VQ models at each codebook size
    print("\n[2] Training VQ models...")
    results = [{'codebook': 'none', 'clean': base_clean, 'pgd_asr': base_asr}]

    for cb_size in CODEBOOK_SIZES:
        print(f"\n  Codebook size C={cb_size}...")
        C.set_seed(SEED)
        base = C.build_model("cnn", meta, seed=SEED)
        vq_model = VQModel(base, feature_dim, cb_size, meta['n_classes'])

        try:
            train_vq_model(vq_model, Xtr, Ytr, epochs=EPOCHS)
            clean_acc = eval_model(vq_model, Xte_e, Yte_e)
            asr = pgd_asr(vq_model, Xte_e, Yte_e)
        except Exception as e:
            print(f"    ERROR: {e}")
            clean_acc = float('nan')
            asr = float('nan')

        print(f"  C={cb_size}: clean={clean_acc:.3f}, PGD_ASR={asr:.3f}")
        results.append({'codebook': cb_size, 'clean': clean_acc, 'pgd_asr': asr})

    print(f"\n--- Summary ---")
    print(f"{'Codebook':>10} | {'Clean Acc':>10} | {'PGD ASR':>8}")
    print("-" * 36)
    for r in results:
        print(f"  {str(r['codebook']):>8}  | {r['clean']:>10.3f} | {r['pgd_asr']:>8.3f}")
    print("Interpretation: smaller codebook should reduce PGD ASR at cost of "
          "clean accuracy (Pareto trade-off).")
    print(f"Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
