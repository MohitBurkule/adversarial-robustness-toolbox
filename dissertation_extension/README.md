# Dissertation extension — Robustness of Adversarial Defences (Burkule 2022, extended 2026)

Extension to the original MSc thesis. Replicates the headline result (FGSM defence
generalises best to unseen attacks), then explores whether training-time information
gives stronger attacks or new per-sample vulnerability diagnostics.

## Headline findings

1. **Replication (Fashion-MNIST, simple 5-layer CNN, ε=15/255 L∞)**:
   - Vanilla CNN: 91.6% clean → 28.5% FGSM, 3.2% PGD, 8.1% BIM.
   - FGSM-adv-trained: 88.9% clean → 79.5% FGSM, 74.1% PGD, 76.0% BIM.
   - Defence trained only against FGSM lifts PGD robustness from 3% → 74%.
     Reproduces Burkule 2022 finding.
2. **Training-time attacks (V1–V10, see `extend_v2.py`)**: 9 novel attack variants
   that use signals collected during training (per-step sign-gradients, per-epoch
   FGSM perturbations, accumulated Adam-style gradients, early-confusion-class
   targeted FGSM, etc.). Two variants — V2 TrajEnsemble (per-sample best across
   epoch checkpoints) and V9 Adam-accum — marginally beat vanilla FGSM on the
   undefended model (30.9% vs 32.0% accuracy under attack). None transfer well to
   the FGSM-adv-trained model. Closest published parallel: CVPR 2025
   "Enhancing Adversarial Transferability with Checkpoints of a Single Model's
   Training" — exact same idea (V2).
3. **Class-identity persistence (`analyse_fgsm_targets.py`)**:
   For samples that take ≥1 epoch to learn, the wrong class predicted in epoch 0
   matches the final model's 2nd-best logit class 70–100% of the time. For samples
   learned at epoch 0 the match drops to 4–5%. Final-margin sharply discriminates
   these regimes.
4. **Diagnostic test (`diagnostic_test.py`)**: defines a per-sample binary feature
   `S(x) = 1[top_training_confusion(x) == final_2nd_best(x)]`. Univariate AUROC
   0.57–0.75 for predicting FGSM/transfer-attack success. **Multivariate Δ AUROC
   = 0.0000–0.0008** across MNIST, Fashion-MNIST, self/transfer, all 4 margin
   quartiles → fully collinear with final-margin + cartography features.
   **Negative result: real geometric phenomenon, no new predictive value.**
5. **Curriculum AT (Phase B of `longer_and_curriculum.py`)**: adversarial training
   restricted to bottom-50%-margin samples gets 79.3% FGSM acc (≈ adv-all's
   79.5%) but only 63.4% PGD (vs adv-all's 74.1%). Binary hard/easy masking is
   worse than uniform adv augmentation. Soft reweighting à la GAIRAT (ICLR 2021)
   is the principled fix.

## Files

| Script | Purpose | Output |
|---|---|---|
| `replicate_and_extend.py` | Baseline + FGSM-defence replication + V1/V2 novel attacks | `results.json` |
| `extend_v2.py` | V1–V10 attack-variant comparison on one training run | `results_v2.json` |
| `analyse_fgsm_targets.py` | Where does FGSM land? 2nd-best vs early-confusion class, bucketed by learning-epoch | `analysis_targets.json` |
| `longer_and_curriculum.py` | 20-epoch trajectory analysis + Phase B curriculum AT (vanilla / adv-all / adv-hard-only) | `long_curriculum_results.json` |
| `diagnostic_test.py` | S(x) diagnostic multivariate AUROC test on MNIST + Fashion-MNIST, with margin-quartile conditional analysis | `logs/diag2.log` |

## Reproduction

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install adversarial-robustness-toolbox scikit-learn
.venv/bin/python replicate_and_extend.py   # ~2 min on GPU
.venv/bin/python extend_v2.py              # ~3 min
.venv/bin/python analyse_fgsm_targets.py   # ~1 min
.venv/bin/python longer_and_curriculum.py  # ~5 min
.venv/bin/python diagnostic_test.py        # ~5 min
```

Verified on RTX 3060 Ti (CUDA 12.4, PyTorch 2.6, Python 3.12).

## Relation to prior work

| What we did | Closest published work |
|---|---|
| V2 trajectory-checkpoint attack | "Enhancing Adversarial Transferability with Checkpoints of a Single Model's Training" (CVPR 2025) |
| V1/V9 historical-gradient attack | HGAA — "Historical Gradient Adversarial Attack" (Sensors 2024) |
| Per-sample difficulty as AT weight | GAIRAT (ICLR 2021); Probabilistic Margins (2106.07904); HAM (TIFS 2024) |
| Bucketing by learning-epoch | Baldock — "Deep Learning Through the Lens of Example Difficulty" (NeurIPS 2021); Toneva — "Empirical Study of Example Forgetting" (ICLR 2019); Maini — SSFT (NeurIPS 2022) |
| Per-sample diagnostic via training dynamics | Dataset Cartography (Swayamdipta EMNLP 2020); Nayak Holistic Vulnerability (IJCNN 2022) |
| Class-identity persistence (`S(x)`) | **No direct match** — closest is Cross-Class Features in AT (ICML 2025), but that paper operates on feature-attribution overlap, not per-sample wrong-class identity. |
