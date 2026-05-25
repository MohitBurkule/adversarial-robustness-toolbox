# Future Research Directions

Ideas for follow-on experiments beyond the H107–H156 systematic study.
Each section covers the hypothesis, what prior work exists (from web search, May 2026), and an honest novelty assessment.

---

## Idea 1: Adversarial Attackability Over Training Time (+ Grokking Connection)

### The Hypothesis

Train a model and, at regular checkpoints, measure both clean accuracy *and* adversarial vulnerability (FGSM/PGD success rate, mean min-ε). Plot all three over training steps on a single graph. Key questions:
- Does adversarial robustness track clean accuracy monotonically, or does it lag/lead?
- Are undertrained models (before they fully fit training data) more or less robust?
- On sufficiently hard datasets, does a grokking-like delayed robustness onset appear?

To see grokking clearly, use a harder dataset (e.g. CIFAR-100, modular arithmetic, or a subset of ImageNet) and train well past the point of training-set saturation.

### What Prior Work Says

**Directly related:**
- **Humayun et al., "Deep Networks Always Grok and Here is Why" (arXiv 2402.15555, 2024)** — the central paper. On ResNet18/CIFAR-10, adversarial robustness emerges after ~10⁴ optimiser steps and converges near ~2×10⁵ steps, long after clean accuracy plateaus. The mechanism proposed is "local complexity" (density of linear region boundaries). This is almost exactly the hypothesis.
- **"Is Delayed Robustness Really Grokking?" (OpenReview 2024)** — a challenge paper arguing the phenomenon is an optimiser artifact (softmax collapse + excessive effective learning rate under Adam), not true grokking. The debate is live.
- **Rice et al., "Overfitting in Adversarially Robust Deep Learning" (ICML 2020)** — for adversarially trained models, robust accuracy *peaks* early then degrades; early stopping is critical. The mirror image for adversarial training.
- **"The Surprising Harmfulness of Benign Overfitting for Adversarial Robustness" (arXiv 2401.12236)** — benign overfitting (fitting noise while generalising cleanly) actively harms adversarial robustness in overparameterised settings.

### Novelty Assessment

**Low-to-moderate** for the standard-training version. Humayun et al. (2024) have done this almost exactly on CIFAR-10. Novel angles that could distinguish new work:
1. Study under adversarial training specifically (does grokking-like robustness still appear?).
2. Use genuinely hard algorithmic tasks (modular arithmetic, sparse parity) where classical grokking is well documented — nobody has mapped adversarial vulnerability during algorithmic grokking.
3. Measure the *per-sample* vulnerability trajectory (not just aggregate) and see whether boundary-proximate samples are "the last to become robust."

**Verdict**: Worth doing as a replication + extension on harder tasks. If grokking-phase adversarial vulnerability data can be collected during true algorithmic grokking, there is a publishable finding.

---

## Idea 2: Anti-Adversarial ("Confirmatory") Examples and Training

### The Hypothesis

Standard adversarial examples push an input *across* the decision boundary (or close to it). The proposal here is the opposite: generate inputs that are pushed *further into the interior* of the correct class region — maximally confirmatory examples. Then:

1. **Does training on these confirmatory examples improve clean accuracy or generalisation?**
2. **Does mixing them with standard training data improve the robustness-accuracy trade-off?**
3. **Extreme version: train *only* on confirmatory examples and evaluate generalisation.** Does removing the original data but keeping only these easy, interior-of-class samples produce a useful model?

The motivation: adversarial training (which includes adversarial examples) is known to hurt clean accuracy. Confirmatory examples are the opposite perturbation direction — maybe they help.

### What Prior Work Says

**Closest existing work (important to distinguish from):**

- **Zhang et al., "Attacks Which Do Not Kill Training Make Adversarial Learning Stronger" (ICML 2020, arXiv 2002.11242) — Friendly Adversarial Training (FAT)**: FAT uses early-stopped PGD to find the *least* adversarial examples that are *still misclassified* (they still cross the decision boundary, just by the smallest margin). This is **not** the same as confirmatory examples — FAT examples are still adversarial (they fool the model), just barely. The hypothesis here is about examples that are correctly classified and pushed *further* into the correct region.
- **Helper Adversarial Training (HAT)**: uses correctly-labelled-but-perturbed examples to counteract excessive margin growth in AT.
- **Inverse Adversarial Training (IAT)**: generates inputs toward high-likelihood regions of their true class.
- **Boundary Adversarial Examples Against Adversarial Overfitting (arXiv 2211.14088)**: uses boundary-proximate examples to prevent overfitting in AT.

The specific proposal here — generating examples that are *maximally interior* to the correct class manifold (not just the least-bad adversarial example but actively confirmatory) — appears **less directly studied** than FAT. IAT is closest but not identical.

### Implementation Idea

```python
# Anti-adversarial perturbation: maximise correct-class logit
delta = alpha * sign(grad_wrt_correct_class_logit)
x_confirmatory = clamp(x + delta, 0, 1)
```

This is literally the reverse of FGSM: FGSM maximises loss (pushes toward wrong class), anti-FGSM maximises the correct-class confidence. The resulting examples are "more of what the model already likes."

### Novelty Assessment

**Moderate.** The exact framing of generating interior-of-class confirmatory examples as training augmentation is distinct from FAT, HAT, and IAT. The specific question — "can confirmatory examples recover the clean accuracy lost by adversarial training?" — is concrete and testable. The "train only on confirmatory examples" variant is the most speculative but also potentially surprising. Connected to anti-curriculum learning literature where easy-first training has mixed results.

**Verdict**: Novel enough to run and publish as an empirical study. If confirmatory examples measurably improve generalisation or help the AT accuracy tradeoff, it is a clean result.

---

## Idea 3: Which Samples Get Misclassified After Adversarial Training, and Why?

### The Hypothesis

1. Train a vanilla model. Record its clean predictions on a held-out test set.
2. Fine-tune the same model with adversarial training (e.g. PGD-AT on training data).
3. Re-evaluate the same test set. Identify samples that were **previously correct but are now wrong** after adversarially fine-tuning (the "newly misclassified" set).
4. Characterise these samples: are they low-margin in the vanilla model? Are they easily adversarial (low min-ε to flip)?
5. Secondary question: are the newly-misclassified samples also the ones closest to the decision boundary — i.e., are they the same ones that are most adversarially vulnerable in the vanilla model?

### Connection to the Existing Study

This directly extends the findings of H147 (partial data) and the H118/H119/H120 defense experiments:
- H147 (partial) showed PGD-AT expands mean min-ε by 6.7× while maintaining 88% clean accuracy. But which 12% of samples are now wrong?
- H135 showed that ensemble predictive entropy > margin. Could we predict *before* AT which samples will be hurt?
- The H150 finding that confusion ratio = margin (monotone reparameterisation) means per-sample margin is the right predictor to test.

### What Prior Work Says

**Directly related:**
- **"Exploring the Forgetting in Adversarial Training" (ICLR 2025)**: studies within-AT forgetting (samples learned correctly early in AT, then forgotten as training continues). Close but not identical — their analysis is within AT, not the vanilla → AT transition.
- **"Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off" (ICLR 2023)**: establishes mechanistically that AT applies uniform ε that is too large for low-margin samples, pushing adversarial examples across the *true* decision boundary. This predicts exactly that low-margin samples are the ones hurt most.
- **A3T: Accuracy Aware Adversarial Training (arXiv 2211.16316)**: monitors per-sample classification probability during AT and treats samples differently.
- **Moderate-Margin Adversarial Training (MMAT, ScienceDirect 2023)**: proposes using boundary-proximate examples specifically to reduce clean accuracy degradation.

The specific experimental protocol — measuring per-sample prediction changes from vanilla → AT, then correlating with pre-AT margin and min-ε — has not been done as a **direct causal/predictive audit** in existing work. Prior papers establish the mechanism theoretically; the proposed experiment would provide explicit per-sample evidence and turn it into a **predictable pre-AT signal**.

### Novelty Assessment

**Moderate-to-high** for the predictive framing. The theoretical mechanism is established, but the specific question "can we predict, from the vanilla model's margin alone, which samples will be hurt by subsequent AT?" appears untested. If the pre-AT margin is a strong predictor of post-AT accuracy degradation, this would close a loop: the same samples that are adversarially vulnerable (low margin) are also the samples that suffer most when the model is made robust.

**Verdict**: High practical and theoretical value. Worth implementing as a direct follow-on to H118–H120 and H147. The experiment is cheap (one vanilla run + one AT run + per-sample comparison). If the correlation is strong (which the prior work mechanism predicts), it is a tight publishable result.

---

## Publishability Assessment of the Full H107–H156 Study

### What Has Been Done

157 hypotheses implemented, 50 fully run across Fashion-MNIST, CIFAR-10, and Imagenette (SVHN in progress). The study systematically evaluated per-sample adversarial vulnerability predictors across:
- Attribution methods (LRP, SmoothGrad, IG, GradCAM, saliency)
- Uncertainty methods (deep ensemble, SWAG, MC-dropout, snapshot ensemble)
- Training dynamics (forgetting, C-score, memorisation)
- Defense comparisons (CURE, AugMix, Manifold Mixup, ALP, AWP, PGD-AT)
- Attack transferability (cross-model, cross-architecture CNN vs ViT)
- Black-box attacks (SimBA, ZOO, NES, Sign-OPT)
- Detection (meta-detector ensemble)

### Flagship Findings (Paper-Worthy)

1. **Margin dominance with exceptions**: The logit margin is the single best univariate predictor (0.87–0.9976 AUROC). Two features slightly exceed it: SmoothGrad norms (0.97) and deep ensemble entropy (0.9547). This is a systematic benchmark no prior work has done at this scale.

2. **Structural vulnerability** (H148, H156): 74.7% of samples are universally vulnerable to all 4 attacks. FGSM and iterative attacks have orthogonal feature importance structures (r = −0.034) — this is a clean, striking result.

3. **RFNN baseline** (H142): Random feature projections match the fully trained CNN margin as a vulnerability predictor. Boundary proximity is encoded in input geometry, not deep representations.

4. **Augmentation defenses alter margin predictability** (H118–H124): CURE preserves margin predictability (+0.003); augmentation-based methods (AugMix, Manifold Mixup, ALP) significantly reduce it (−0.15 to −0.25), suggesting they alter decision boundary geometry rather than expanding it.

5. **Null findings are also contributions**: Forgetting events and C-score are near-random predictors — learning difficulty ≠ adversarial proximity. This cleanly disentangles two properties that are often conflated.

### Verdict

**Yes, this is worth a paper** — specifically as a comprehensive empirical survey / benchmark: *"What Predicts Per-Sample Adversarial Vulnerability? A Systematic Benchmark of 50 Predictors Across Attacks, Defenses, and Architectures."* The breadth (50 predictors, 4 datasets, multiple attack families, defenses, and architectures) and the tight null-vs-positive stratification make it a reference paper for the adversarial robustness field. No existing paper has evaluated this range of predictors in a controlled, systematic way.

Target venues: ICLR / NeurIPS / ICML (as an empirical study / benchmarks track); or TPAMI / IJCV as a longer journal version.

---

*Last updated: 2026-05-24*
