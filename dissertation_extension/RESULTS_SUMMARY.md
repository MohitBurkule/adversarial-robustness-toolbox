# Hypothesis Results Summary

All results are for Fashion-MNIST unless noted. Attacks: FGSM and PGD (eps=0.1) unless stated.

---

## H183 - Unlearning Adversarial Holes

**Conclusion:** Machine unlearning (finetune-on-retain) collapses clean accuracy on the forgotten class (0.839 → 0.368) but the class survives as an adversarial residual: 53.6% of misclassified forgotten-class samples can be pushed back to the forgotten label by a small PGD nudge, and the attractor rate (other-class adversarials landing in the forgotten class) drops but does not vanish.
**Key metric:** Adversarial recovery rate = 0.536 (mean across seeds)
**Status:** SUPPORTED

---

## H184 - Illusion Robustness Masking

**Conclusion:** Human-scheme illusion models show PGD ASR = 0.000 and collapsed input-gradient norms (mean ~4.88e-25), but their black-box and transfer ASR are also 0.000 — meaning the apparent robustness is not classic gradient masking but rather a genuine artefact of the binary human-label training task collapsing gradients entirely. The physical-scheme models are fully attackable (PGD ASR = 1.000).
**Key metric:** Human-scheme blackbox ASR = 0.000, PGD ASR = 0.000, grad_norm ≈ 0
**Status:** INCONCLUSIVE (robustness survives gradient-free attack, but caused by task collapse)

---

## H185 - Adaptive Epsilon Adversarial Training

**Conclusion:** Margin-adaptive epsilon AT (scaling eps inversely with margin) does not improve overall PGD robustness vs standard AT (both 0.330 ASR), though Q3 samples (mid-margin) show a marginal improvement (-0.020). The adaptive model loses more clean accuracy (-0.035).
**Key metric:** Overall PGD ASR delta = 0.000 (no improvement)
**Status:** NOT SUPPORTED

---

## H186 - Probability-Space vs Logit-Space Margin

**Conclusion:** Logit-space and probability-space margins are near-perfectly correlated (Spearman r = 0.998) and predict FGSM success equally well (AUROC ~0.950 for both). Neither dominates the other.
**Key metric:** Logit AUROC = 0.9512, Prob AUROC = 0.9492; Spearman r = 0.998
**Status:** INCONCLUSIVE (no meaningful difference)

---

## H187 - Gaussian Noise vs Adversarial Vulnerability

**Conclusion:** Per-sample Gaussian noise sensitivity has moderate-to-strong correlation with FGSM vulnerability (AUROC = 0.677), suggesting a partial link between input sensitivity to random noise and adversarial fragility, but Gaussian noise alone is weaker than gradient-based attacks.
**Key metric:** AUROC (Gaussian → FGSM) = 0.677
**Status:** PARTIAL

---

## H188 - Per-Class Vulnerability Distribution

**Conclusion:** Strong negative Spearman correlation between class mean margin and FGSM ASR (rho = -0.828, p = 0.003): low-margin classes (Shirt, Coat, Pullover) have higher FGSM ASR. PGD correlation is weaker and not significant, as PGD ASR saturates near 1.0 for all classes.
**Key metric:** rho(mean_margin, FGSM_ASR) = -0.828
**Status:** SUPPORTED (for FGSM), INCONCLUSIVE (for PGD)

---

## H189 - Ensemble Disagreement vs Single-Model Margin

**Conclusion:** Ensemble entropy and variance slightly outperform single-model margin at predicting FGSM vulnerability (AUROC 0.671 vs 0.641), and the two signals are highly correlated (rho = 0.954). The improvement is modest, consistent with the SED finding that ensemble disagreement adds marginal signal.
**Key metric:** Ensemble entropy AUROC = 0.671 vs single-model margin AUROC = 0.641
**Status:** PARTIAL (marginal improvement)

---

## H190 - Step-to-Flip as Vulnerability Predictor

**Conclusion:** Step-to-Flip (STF) predicts PGD-20 success perfectly (AUROC = 1.000) but is a weaker FGSM predictor than logit margin (AUROC 0.883 vs 0.970). STF and margin have moderate Spearman correlation (rho = 0.479).
**Key metric:** AUROC (STF → PGD) = 1.000; AUROC (-margin → FGSM) = 0.970 > AUROC (STF → FGSM) = 0.883
**Status:** PARTIAL (STF is best for PGD, margin is best for FGSM)

---

## H191 - Temperature Scaling and Vulnerability

**Conclusion:** Temperature calibration (T = 1.557, ECE 0.026 → 0.009) marginally improves calibrated confidence as a FGSM predictor over raw confidence (AUROC 0.967 vs 0.966), but the effect is negligible. Logit margin remains the best FGSM predictor.
**Key metric:** Calibrated conf AUROC (FGSM) = 0.967; logit margin AUROC = 0.968
**Status:** INCONCLUSIVE (trivially small benefit)

---

## H192 - AIGN Coreset Vulnerability Concentration

**Conclusion:** Training on the bottom-50% AIGN coreset (low-influence samples) reduces PGD ASR substantially (0.963 → 0.660, delta = -0.303) and increases Gini coefficient (0.037 → 0.340), suggesting vulnerability concentrates in hard samples. However, clean accuracy also drops (0.933 → 0.864), making the Gini signal mixed.
**Key metric:** Delta PGD ASR = -0.303; Delta Gini (binary) = +0.303
**Status:** PARTIAL

---

## H193 - PGD Loss Trajectory Flatness

**Conclusion:** Max PGD loss (trajectory peak) predicts FGSM success (AUROC = 0.942), matching the final-step loss, but does not exceed logit margin as predictor (AUROC 0.965). No "uncanny valley" sharpening-then-flattening pattern was found (0% of samples showed interior loss peaks).
**Key metric:** AUROC (max_loss → FGSM) = 0.942; AUROC (margin → FGSM) = 0.965
**Status:** NOT SUPPORTED (loss trajectory adds no signal beyond margin)

---

## H194 - Feature Squeezing as Vulnerability Detector

**Conclusion:** Feature squeezing (bit-depth reduction) predicts FGSM vulnerability (4-bit AUROC = 0.965, 2-bit AUROC = 0.942) nearly as well as logit margin (AUROC = 0.966), making it a viable gradient-free vulnerability proxy, though it does not surpass margin.
**Key metric:** 4-bit squeeze AUROC = 0.965; margin AUROC = 0.966
**Status:** PARTIAL (competitive but does not beat margin)

---

## H195 - Prototype Distance vs Margin

**Conclusion:** Prototype distance in feature space is a weaker FGSM vulnerability predictor than logit margin (AUROC 0.904 vs 0.939, delta = -0.035) despite high correlation between the two signals (Spearman rho = 0.874). Softmax margin remains the better oracle.
**Key metric:** Mean AUROC (proto → FGSM) = 0.904 vs AUROC (margin → FGSM) = 0.939
**Status:** NOT SUPPORTED

---

## H196 - Random Label Vulnerability Structure

**Conclusion:** A randomly-labelled model's margin AUROC drops substantially (0.960 → 0.679 for FGSM) compared to a correctly-labelled model, and cross-model margin correlation is negative (rho = -0.269). Vulnerability structure is label-dependent, not purely data-geometric.
**Key metric:** AUROC (margin → FGSM) drops from 0.960 to 0.679 under random labels
**Status:** PARTIAL (some margin AUROC survives, but label placement matters significantly)

---

## H197 - L2 vs L-inf Vulnerability Correlation

**Conclusion:** L∞ and L2 FGSM vulnerabilities are strongly correlated per-sample (Spearman rho = 0.796), and 92.4% of samples agree in vulnerability status under the two norms. The majority of L∞-vulnerable samples are also L2-vulnerable, supporting norm-agnostic vulnerability for FGSM.
**Key metric:** Spearman rho (L∞ FGSM vs L2 FGSM) = 0.796; agreement = 92.4%
**Status:** SUPPORTED

---

## H198 - Batch Norm Statistics as Vulnerability Predictor

**Conclusion:** Per-sample BN running-statistics deviation scores do not predict adversarial vulnerability (AUROC = 0.392, below chance after orientation; Spearman rho vs margin = 0.028, non-significant). BN outlier status is not a useful vulnerability signal.
**Key metric:** AUROC (BN deviation → FGSM) = 0.392; margin AUROC = 0.972
**Status:** NOT SUPPORTED

---

## H199 - Gradient Cosine Similarity and Transferability

**Conclusion:** Cosine alignment between clean input gradients and adversarial perturbation directions has marginal predictive value for transfer success (AUROC 0.577 vs margin baseline 0.751). Successful transfer samples have slightly higher mean cosine sim (0.655 vs 0.628, Mann-Whitney p = 0.012).
**Key metric:** AUROC (cos_sim → transfer) = 0.577; baseline margin AUROC = 0.751
**Status:** PARTIAL

---

## H200 - Label-Incongruent Adversarials

**Conclusion:** 61.8% of PGD adversarials on Fashion-MNIST are label-incongruent (adversarial class is visually implausible relative to true label), comparable to the DUCAT CIFAR-10 baseline (~40%). Incongruence rate is strongly modulated by clean margin (high-margin samples: 15.8% incongruent vs low-margin: 95.8%).
**Key metric:** Overall label-incongruence rate = 61.8%
**Status:** SUPPORTED

---

## H201 - Influence Function Self-Influence and Vulnerability

**Conclusion:** TracIn-style self-influence (gradient-norm² at final checkpoint) is an excellent vulnerability predictor, surpassing logit margin on both FGSM (AUROC 0.954 vs 0.951) and PGD (AUROC 0.977 vs 0.976). Self-influence is nearly perfectly anti-correlated with margin (Spearman rho = -0.997).
**Key metric:** AUROC (self_influence → FGSM) = 0.954; (self_influence → PGD) = 0.977
**Status:** SUPPORTED

---

## H202 - Curriculum Adversarial Training

**Conclusion:** Curriculum AT (easy-to-hard scheduling) does not outperform uniform AT on hard samples (Q1 ASR delta = +0.054 in favour of uniform AT). Curriculum AT has slightly higher overall PGD ASR (0.141 vs 0.122) and lower clean accuracy (-0.008), indicating no benefit.
**Key metric:** Net Q1 ASR delta (curriculum - uniform) = +0.054 (curriculum worse)
**Status:** NOT SUPPORTED

---

## H203 - Augmentation Vulnerability Consistency

**Conclusion:** Vulnerability is largely image-intrinsic: Spearman rho between original and augmented FGSM success is 0.819, and 91.7% of originally-vulnerable images remain >80% vulnerable across 5 augmentations. Only 11.3% of safe images cross the boundary under augmentation.
**Key metric:** Spearman rho (orig vs aug FGSM) = 0.819; persistent vulnerability = 91.7%
**Status:** SUPPORTED

---

## H204 - Randomized Smoothing and Hard Samples

**Conclusion:** Randomized smoothing abstain rates are only 2× higher for low-margin (Q1) vs high-margin (Q4) samples, and the Spearman correlation between margin and certified radius is non-significant (rho = 0.052, p = 0.374). Smoothing does not preferentially discard hard samples on this model.
**Key metric:** Spearman rho (margin vs radius) = 0.052 (p = 0.374, not significant)
**Status:** INCONCLUSIVE

---

## H205 - Per-Sample Jacobian Norm and Vulnerability

**Conclusion:** Jacobian Frobenius norm is a below-baseline vulnerability predictor (AUROC 0.675 vs margin AUROC 0.936 for FGSM). The Spearman correlation between Jacobian norm and negative margin is only 0.280, and incorrectly classified samples have only 1.03× higher Jacobian norm than correct ones.
**Key metric:** AUROC (Jacobian → FGSM) = 0.675; margin AUROC = 0.936
**Status:** NOT SUPPORTED

---

## H206 - Neighbourhood Density and Vulnerability

**Conclusion:** Neighbourhood density (same-class k-NN count) has above-chance predictive power for FGSM vulnerability (best AUROC = 0.657 at r=10), but well below margin (AUROC = 0.930). Density and margin are moderately correlated (rho = 0.321): denser regions tend to have higher margin.
**Key metric:** Best density AUROC (FGSM) = 0.657; margin AUROC = 0.930
**Status:** PARTIAL

---

## H207 - Logit Trajectory Smoothness during PGD

**Conclusion:** Total variation of the PGD logit trajectory predicts PGD success with AUROC = 0.813, exceeding the initial-margin baseline (AUROC = 0.652 for PGD). Trajectory smoothness adds predictive value beyond initial margin for multi-step attacks.
**Key metric:** AUROC (total_variation → PGD) = 0.813 vs initial margin AUROC = 0.652
**Status:** SUPPORTED

---

## H208 - Cross-Architecture Vulnerability Consistency

**Conclusion:** Per-sample margins are highly correlated across CNN-small, CNN-wide, and MLP architectures (mean Spearman rho = 0.880), and a model's margin predicts another model's FGSM success nearly as well as its own (cross-AUROC 0.593 vs own-AUROC 0.607, gap = -0.014). Vulnerability ordering is data-geometric.
**Key metric:** Mean cross-architecture Spearman rho = 0.880; cross-AUROC gap = -0.014
**Status:** SUPPORTED

---

## H209 - Random Input Distillation

**Conclusion:** Adversarial robustness does not transfer through distillation on random noise inputs: students trained on uniform or Gaussian noise have near-zero clean accuracy and PGD ASR ~0.906-0.914. Only adversarial-input distillation transfers robustness (PGD ASR 0.290).
**Key metric:** Gaussian-noise student PGD ASR = 0.914; adversarial student PGD ASR = 0.290
**Status:** SUPPORTED (adversarial inputs are necessary for robustness transfer)

---

## H210 - Gradient Trajectory Replay

**Conclusion:** Applying the net weight displacement from a trained model to a different random initialisation does not produce a classifier (clean acc = 0.118). Weight initialisation is entangled with the gradient trajectory; the displacement cannot be separated from its init.
**Key metric:** Net-displacement model clean acc = 0.118 (vs normal training = 0.914)
**Status:** NOT SUPPORTED

---

## H211 - Mixture of Experts Adversarial Robustness

**Conclusion:** More experts reduces PGD ASR (K=1: 0.972 → K=2: 0.920), though the relationship is not monotone at K=4,8. Expert usage frequency does not predict adversarial importance: the least-used expert (K=4) has the largest impact when removed (ΔPGD = +0.012 vs +0.002 for most-used).
**Key metric:** K=2 PGD ASR = 0.920 vs K=1 = 0.972
**Status:** PARTIAL (diversity helps, but non-monotone; importance ≠ usage)

---

## H212 - Adversarial Trajectory Replay

**Conclusion:** Applying the net displacement of a PGD-AT trained model to a random initialisation does not transfer robustness (PGD ASR = 0.992, clean = 0.040), performing worse than applying the clean-training displacement (PGD ASR = 0.984, clean = 0.134). Adversarial displacement does not encode robustness independently of initialisation.
**Key metric:** AT displacement model PGD ASR = 0.992 (no robustness); clean displacement = 0.984
**Status:** NOT SUPPORTED (reversed from hypothesis)

---

## H213 - Gradient Inversion Reconstruction Quality and Vulnerability

**Conclusion:** A neural network trained to invert input gradients achieves NCC = 0.482, and gradient inversion quality (NCC) predicts FGSM vulnerability (AUROC = 0.736). Gradient norm is a near-perfect PGD predictor (AUROC = 0.995). Reconstruction quality carries signal below but distinct from margin.
**Key metric:** AUROC (NCC → FGSM) = 0.736; AUROC (-MSE → PGD) = 0.852
**Status:** PARTIAL

---

## H214 - Score Network as Vulnerability Oracle

**Conclusion:** A denoising score network's magnitude does not reliably predict FGSM vulnerability (best AUROC = 0.737 at low sigma), and Langevin dynamics guided by score directions achieves only 18% ASR vs 82% for FGSM. The decision boundary does not align with the data manifold edge.
**Key metric:** Score-direction Langevin ASR = 0.180 vs FGSM ASR = 0.817
**Status:** NOT SUPPORTED

---

## H215 - L-Ball Region Training and Margin Calibration

**Conclusion:** Spearman rho between logit margin and certified radius is high for all training regimes (~0.72–0.75), with standard and FGSM-AT performing similarly. Smooth-AT (Gaussian noise) slightly reduces FGSM ASR (0.602 vs 0.702 baseline) with no clean accuracy cost.
**Key metric:** Spearman rho(margin, cert_radius): standard=0.751, smooth_at=0.716, fgsm_at=0.730
**Status:** INCONCLUSIVE (calibration similar across regimes; smooth-AT shows minor FGSM benefit)

---

## H216 - Augmentation vs Random-Direction Attack

**Conclusion:** Data augmentation (flip/crop) increases random-attack ASR proportionally more than gradient-attack ASR (ratio 0.238 vs baseline 0.061), suggesting augmentation incidentally defends against random perturbations more than targeted gradient attacks. Gaussian noise augmentation shows the opposite (ratio 0.047).
**Key metric:** Rand/FGSM ratio: baseline=0.061, hflip_crop=0.238, gaussian_noise=0.047
**Status:** PARTIAL (effect is augmentation-type specific)

---

## H217 - Model Width vs Depth Robustness under AT

**Conclusion:** Under PGD adversarial training, narrow-deep (PGD ASR 0.165) and very-deep (0.157) models slightly outperform wide models at similar parameter counts. Wide-shallow (3 conv layers, wide channels) collapses entirely (clean=0.087). No clear width-over-depth advantage is found.
**Key metric:** PGD ASR: narrow_deep=0.165, standard=0.195, very_wide=0.200, very_deep=0.157
**Status:** NOT SUPPORTED (depth slightly better, not width)

---

## H218 - Robust Overfitting Per-Sample Dynamics

**Conclusion:** 82.5% of samples experience robust overfitting (margin increases then decreases) during PGD-AT. Overfitting does not preferentially harm initially-hard samples (Spearman rho between early margin and delta = -0.086, not significant). Mid-margin samples (Q2/Q3) show slightly higher overfitting fractions.
**Key metric:** Overfit victim fraction = 0.825; rho(early_margin, delta) = -0.086 (p=0.224)
**Status:** INCONCLUSIVE (widespread overfitting but not concentrated in low-margin samples)

---

## H219 - SSL vs Supervised Representation Robustness

**Conclusion:** SSL-pretrained models (SimCLR-lite, Rotation-SSL) do not produce more robust representations than supervised CNNs of comparable capacity; both SimCLR (PGD ASR 0.943) and Rotation (1.000) are more vulnerable than the supervised baseline (0.897). The SSL models use a narrower encoder (width=8 vs 32).
**Key metric:** PGD ASR: supervised=0.897, SimCLR=0.943, Rotation=1.000
**Status:** NOT SUPPORTED

---

## H220 - Shape-Biased Training and Robustness

**Conclusion:** All shape-biasing methods reduce FGSM ASR compared to baseline (saliency_dropout: -0.140, random_patch_mask: -0.085, spatial_dropout: -0.077), with saliency dropout also reducing PGD ASR (-0.030). Shape-biasing provides modest robustness gains at minimal clean accuracy cost for spatial dropout.
**Key metric:** FGSM ASR delta: saliency_dropout=-0.140; spatial_dropout=-0.077
**Status:** PARTIAL (FGSM robustness improves; PGD largely unchanged)

---

## H221 - Temporal Averaging as Adversarial Defence

**Conclusion:** Temporal averaging (3-frame clips) reduces frame-independent adversarial ASR by 38.6% (video model: 0.404 vs image model: 0.790 for FGSM). However, temporally-consistent attacks (same perturbation across frames) fully bypass this defence (ASR 0.807 ≈ baseline 0.790).
**Key metric:** Frame-independent ASR delta = -0.386; temporally-consistent delta = +0.017
**Status:** PARTIAL (effective against naive attacks, bypassed by consistent attacks)

---

## H222 - Input Dimensionality and Minimum Epsilon

**Conclusion:** The Simon-Gabriel √n scaling (eps_50 ∝ n^{-0.5}) is not confirmed: eps_50 is constant at 0.05 across all dimensionalities (10 to 784 PCA components), giving fitted slope b = 0.000 vs predicted -0.5. Gradient L1 norm also shows negligible n-scaling (b = -0.036 vs expected +0.5).
**Key metric:** Fitted log(eps_50) slope = 0.000 (predicted: -0.5)
**Status:** NOT SUPPORTED

---

## H223 - Smoothing K Convergence

**Conclusion:** Monte Carlo smoothing AUROC for predicting PGD success converges by K=100 draws (AUROC within 0.010 of K=200 reference). At K=200, AUROC = 0.697 and Spearman rho vs logit margin = 0.767, indicating smoothed robustness signal stabilises with moderate K.
**Key metric:** Convergence K = 100; AUROC at K=200 = 0.697
**Status:** SUPPORTED

---

## H224 - Subject vs Background Adversarial Targeting

**Conclusion:** Adversarial perturbations concentrate slightly more in background pixels than subject pixels (energy ratio 0.564 vs 0.436), but subject-energy is a marginally better FGSM predictor (AUROC 0.596 vs 0.536 for background). The effect is small and both signals are weak.
**Key metric:** Subject energy AUROC (FGSM) = 0.596; perturbation energy: background=0.564 vs subject=0.436
**Status:** INCONCLUSIVE (weak signal; background-targeting confirmed)

---

## H225 - Training Bit-Depth Quantisation and Robustness

**Conclusion:** Training on 3-bit quantised images does not reduce PGD ASR compared to 8-bit training (both ~0.993-0.997). Bit-depth reduction alone confers no adversarial robustness.
**Key metric:** PGD ASR: 3-bit training=0.997; 8-bit training=0.993
**Status:** NOT SUPPORTED

---

## H226 - Spatial Mixed Precision Quantisation

**Conclusion:** Mixed-precision quantisation (3-bit for low-variance patches, 8-bit for high-variance) does not improve robustness vs uniform baselines (PGD ASR = 1.000 for all). However, PGD perturbations do concentrate in high-variance (8-bit) patches (energy 0.564 vs 0.436), confirming the attacker avoids quantised regions.
**Key metric:** All conditions: PGD ASR = 1.000; perturbation energy in high-var patches = 0.564
**Status:** PARTIAL (attacker avoids quantised patches but robustness gain is zero)

---

## H227 - JPEG Training Domain Robustness

**Conclusion:** Training at JPEG quality q=10 achieves the best FGSM robustness (ASR 0.788), but PGD ASR remains near 1.000 for all quality levels except q=10 (0.996). The hypothesis that moderate compression (q=30–50) is more robust than q=100 is not supported.
**Key metric:** Best FGSM ASR: train_q10=0.788; PGD ASR consistently ~1.000
**Status:** NOT SUPPORTED

---

## H228 - Universal Purifier

**Conclusion:** A PGD-trained purifier recovers only 36.0% of adversarial examples (PGD recovery rate) and is broken by adaptive attack (ASR 0.855). Cross-attack generalisation is poor (FGSM-trained purifier PGD recovery = 0.477). Universal purification is fragile.
**Key metric:** PGD recovery rate = 0.360; adaptive attack ASR = 0.855
**Status:** NOT SUPPORTED

---

## H229 - Test-Time Input Optimisation

**Conclusion:** Optimising the input toward higher margin at test time (T gradient steps) fully recovers adversarial examples: at T=20, FGSM-adversarial accuracy recovers from 0.277 to 0.997, and PGD-adversarial accuracy from 0.097 to 0.997. Clean accuracy is unchanged, and margin grows monotonically with T.
**Key metric:** At T=20: FGSM recovery acc = 0.997; PGD recovery acc = 0.997 (from 0.277/0.097)
**Status:** SUPPORTED

---

## H230 - Early vs Late Epoch Adversarial Transferability

**Conclusion:** Transfer ASR (from early-epoch checkpoints to the final model) increases monotonically with source epoch, with gradient cosine similarity also increasing monotonically. However, transfer ASR is non-monotone and peaks around epoch 15 before declining slightly. Final-epoch checkpoint still transfers best.
**Key metric:** Transfer ASR at epoch 15 = 0.777 (peak FGSM); Grad_CosSim monotone increasing
**Status:** PARTIAL (gradient alignment monotone; transfer ASR non-monotone)

---

## H231 - Adversarial Trajectory Transfer

**Conclusion:** PGD trajectory replayed from a source image to target images achieves only 16.2% ASR (vs 100% for from-scratch PGD), barely above random walk (20.0%) and universal delta (10.0%). Low-margin sources do not produce higher transfer ASR (rho = -0.216, p = 0.131).
**Key metric:** Trajectory transfer ASR = 0.162 vs from-scratch ASR = 1.000
**Status:** NOT SUPPORTED

---

## H232 - Noise Padding Budget

**Conclusion:** Adding random noise padding (B=2,4,8 pixels) forces the attacker to waste budget on padding (up to 63.3% at B=8), but ASR on the core image does not decrease — it slightly increases (B=0: 0.949 vs B=8: 0.973). Padding budget redistribution does not reduce attack success.
**Key metric:** B=8 core_energy = 0.367 (63% wasted), but ASR = 0.973 (higher than baseline 0.949)
**Status:** NOT SUPPORTED (attacker wastes budget but compensates with higher core pressure)

---

## H233 - Anti-Adversarial Examples Training

**Conclusion:** Training on anti-adversarial images (inputs pushed away from the decision boundary) reduces clean accuracy (0.856 → 0.783) and increases PGD ASR (0.942 → 1.000). Neither hypothesis A (clean acc improvement) nor hypothesis B (top-20% high-margin = worst robustness) is supported.
**Key metric:** Anti-adv model: clean=0.783 (worse), PGD ASR=1.000 (worse)
**Status:** NOT SUPPORTED

---

## H234 - LSB Bit-Plane Randomisation

**Conclusion:** Training with K=2–4 LSB bit flips reduces FGSM ASR by 0.048–0.065 with no clean accuracy cost. Majority vote at test time with K=4 achieves the best combined result (FGSM ASR 0.598). PGD robustness is marginally improved at K=4 (0.949 vs 0.957 baseline).
**Key metric:** K=4 mv_fgsm_asr = 0.598 (vs baseline 0.716)
**Status:** PARTIAL (FGSM benefit confirmed; PGD benefit marginal; K=4 optimal, not K=1–2 as predicted)

---

## H235 - Overfitting Echo

**Conclusion:** Removing the top-10% highest-training-loss (memorised) samples and retraining reduces the vulnerable test set by only 3% (299 → 291 vulnerable), with Jaccard = 0.973 between vulnerable sets. Vulnerability is not driven by memorised training samples.
**Key metric:** Removing top-10% memorised: vuln=97.0% (from 99.7%); Jaccard=0.973
**Status:** NOT SUPPORTED

---

## H236 - Witness Sample

**Conclusion:** Removing the single most-influential training sample for each test image (193 unique samples removed from 60,000) increases mean test margin by +0.347 and benefits 58.3% of samples. The effect is small but directionally consistent with the influence function hypothesis.
**Key metric:** Mean margin delta = +0.347; 58.3% of test samples show increased margin
**Status:** PARTIAL (small but positive effect)

---

## H237 - Broken Symmetry

**Conclusion:** Symmetry gaps (hflip, brightness, contrast sensitivity asymmetry) are poor vulnerability predictors (AUROC 0.214–0.403 for FGSM, all below baseline margin AUROC = 0.954). Contrast asymmetry is the strongest at AUROC = 0.403, still well below chance-corrected levels.
**Key metric:** Best symmetry gap AUROC = 0.403; margin AUROC = 0.954
**Status:** NOT SUPPORTED

---

## H238 - Forgetting Horizon

**Conclusion:** Samples that are forgotten and re-learned more often during training have lower margin (Spearman rho = -0.567) and higher PGD vulnerability (AUROC = 0.685). Forgetting count is a meaningful, though weaker-than-margin, vulnerability signal.
**Key metric:** AUROC (forgetting_count → PGD) = 0.685; Spearman rho(fc, margin) = -0.567
**Status:** SUPPORTED

---

## H239 - Dead Neuron Topology and Gradient Masking

**Conclusion:** Dead neuron count predicts FGSM success with AUROC = 0.666 (above chance). Samples in Q4 (most dead neurons) show FGSM ASR = 0.960 vs PGD ASR = 0.987, with the smallest FGSM-PGD discrepancy — consistent with fewer "live" neurons producing a smoother gradient landscape for gradient attacks.
**Key metric:** AUROC (dead_count → FGSM) = 0.666; Spearman rho(dead_count, margin) = -0.353
**Status:** PARTIAL

---

## H240 - PCA Tail Subspace and Adversarial Directions

**Conclusion:** PCA tail energy (fraction of image in low-variance PCA directions) has above-chance AUROC for FGSM prediction (best 0.607 at K=50), significantly below margin (AUROC = 0.949). Adversarial perturbations have slightly higher tail energy for successful samples, weakly consistent with the hypothesis.
**Key metric:** Best tail-energy AUROC = 0.607; margin AUROC = 0.949
**Status:** PARTIAL (weak above-chance signal; margin clearly superior)

---

## H241 - Label Noise Migration

**Conclusion:** Label noise (0–50%) does not shift which samples are vulnerable: Jaccard overlap of vulnerable sets between 0% and 50% noise is 1.000, and PGD ASR remains 1.000 for all noise levels. The vulnerability structure on Fashion-MNIST is data-geometric, not label-dependent under high-noise training.
**Key metric:** Jaccard (0% vs 50% noise) = 1.000; PGD ASR = 1.000 across all noise levels
**Status:** INCONCLUSIVE (degenerate PGD labels; no differentiation possible)

---

## H242 - Semantic Bottleneck (VQ)

**Conclusion:** The VQ-bottleneck experiment failed to run due to a device mismatch error in all codebook conditions. No results are available.
**Key metric:** All VQ conditions returned nan (device error)
**Status:** NOT RUN (execution error)

---

## H243 - Dual Model Disagreement

**Conclusion:** Symmetric KL divergence between a standard and adversarially-trained model predicts PGD vulnerability on the standard model well (AUROC = 0.965, vs margin AUROC = 0.997). Hard disagreement alone is a weak predictor (AUROC = 0.562), but distributional disagreement carries strong signal.
**Key metric:** AUROC (sym_KL → PGD) = 0.965; AUROC (hard_disagree → PGD) = 0.562
**Status:** SUPPORTED (sym_KL variant)

---

## H244 - Catastrophic Interference

**Conclusion:** Aggregate margin interference (sum of |margin changes| after single-class fine-tuning) has negligible correlation with vulnerability (Spearman rho = 0.113, AUROC = nan due to degenerate PGD labels). The interference signal is dominated by PGD-label degeneracy.
**Key metric:** Spearman rho (interference, margin) = 0.113
**Status:** INCONCLUSIVE (degenerate labels; experiment inconclusive)

---

## H245 - Generalisation Gap and Robustness

**Conclusion:** Spearman rho between generalisation gap and PGD ASR across training regimes = 0.778, supporting the link. Strong regularisation (wd=0.01, early stopping) reduces both generalisation gap and PGD ASR (0.973, 0.960). Most dropout/mild WD conditions have PGD ASR = 1.000.
**Key metric:** Spearman rho (gen_gap, PGD_ASR) = 0.778; wd_1e-2: PGD ASR = 0.973
**Status:** SUPPORTED

---

## H246 - Robust Islands under AT

**Conclusion:** The top-10% most-robust training samples (robust islands) show 86.7% persistence from epoch 5 to epoch 50, indicating a stable core of consistently-robust samples. Margin variance increases monotonically with training, consistent with growing heterogeneity in robustness.
**Key metric:** Robust island persistence (epoch 5 → 50) = 0.867; PGD ASR at epoch 50 = 0.187
**Status:** SUPPORTED

---

## H247 - Jointly Robust Samples

**Conclusion:** Fewer than 0.3% of test samples are jointly robust to FGSM, PGD, and CW(L2) simultaneously (1 out of 300 per seed). Cross-seed Jaccard is inconsistent (0.000–1.000), indicating the jointly-robust set is near-empty and seed-sensitive.
**Key metric:** Jointly robust fraction = 0.3% (1/300 per seed)
**Status:** INCONCLUSIVE (set is too small to characterise)

---

## H248 - Sleep Consolidation Training

**Conclusion:** Sleep-wake training (alternating clean and adversarial steps) does not reduce ASR compared to standard training (both reach ASR = 1.000 by epoch 10). PGD-AT remains the only effective approach (ASR = 0.183 at epoch 30). Sleep-wake consolidation offers no robustness benefit.
**Key metric:** Sleep-wake final PGD ASR = 1.000; PGD-AT = 0.183; standard = 1.000
**Status:** NOT SUPPORTED

---

## H249 - Density + PCA Combined Predictor

**Conclusion:** Individual AUROCs: margin = 0.983, tail_energy = 0.783, density = 0.120. Combined logistic regression AUROC is nan due to degenerate PGD labels. Feature importances show margin dominates (|coef| = 0.915) with density and tail energy providing complementary but weaker signal.
**Key metric:** Margin AUROC = 0.983; tail_energy AUROC = 0.783; combined = nan
**Status:** INCONCLUSIVE (degenerate labels prevent combination test)

---

## H250 - Weight Norm and Adversarial Vulnerability

**Conclusion:** Weight norm (modulated via weight decay) is perfectly correlated with mean margin (Spearman rho = 1.000, p = 0.000) but uncorrelated with FGSM ASR (rho = 0.000). Higher weight norms → higher margins, but FGSM ASR does not decrease proportionally, suggesting margin is a better intermediate variable.
**Key metric:** rho(weight_norm, mean_margin) = 1.000; rho(weight_norm, FGSM_ASR) = 0.000
**Status:** PARTIAL (weight norm predicts margin but not attack success directly)

---

## H251 - Input Gradient Sparsity and Vulnerability

**Conclusion:** Input gradient sparsity is positively correlated with margin (Spearman rho = 0.739) and negatively correlated with FGSM success (rho = -0.217, p = 0.0002). Sparser gradients → larger margins → less vulnerable. AUROC (0.628–0.699 across thresholds) is below margin but above chance.
**Key metric:** AUROC (sparsity → FGSM) = 0.628–0.699; rho(sparsity, margin) = +0.739
**Status:** PARTIAL (above-chance; margin is stronger predictor)

---

## H252 - Per-Class Adversarial Vulnerability Asymmetry

**Conclusion:** FGSM ASR is significantly non-uniform across classes (chi-squared p = 0.000): Trouser (0.300), Bag (0.667), and Sandal (0.643) are most FGSM-robust, while T-shirt, Pullover, Coat, and Shirt all reach ASR = 1.000. PGD ASR is uniformly 1.000 across all classes.
**Key metric:** FGSM chi2 = 94.95 (p = 0.000); PGD chi2 = 0.114 (p = 1.000)
**Status:** SUPPORTED (for FGSM), NOT SUPPORTED (for PGD)

---

## H253 - Penultimate Feature Norm and Vulnerability

**Conclusion:** Penultimate-layer feature norm is a strong FGSM vulnerability predictor (AUROC = 0.880) and strongly correlated with logit margin (Spearman rho = 0.819). Low feature-norm samples have FGSM ASR = 0.987 vs 0.560 for high-norm samples.
**Key metric:** AUROC (feat_norm → FGSM) = 0.880; rho(feat_norm, margin) = +0.819
**Status:** SUPPORTED

---

## H254 - Cross-Epsilon Vulnerability Rank Preservation

**Conclusion:** Vulnerability rankings are weakly preserved across epsilon budgets: only the eps=0.01 vs eps=0.05 pair shows significant Spearman rho (0.189, p = 0.001). At eps=0.10 and above, ASR saturates to 1.000 and rank comparisons become degenerate. Only 23% of samples are consistently vulnerable across all 4 epsilon values.
**Key metric:** Mean pairwise Spearman rho = nan (degenerate at high eps); eps=0.01 vs 0.05 rho = 0.189
**Status:** NOT SUPPORTED (rankings preserved only at low epsilon)

---

## H255 - Label Smoothing and Adversarial Robustness

**Conclusion:** Label smoothing reduces FGSM ASR monotonically with smoothing alpha (Spearman rho = -1.000, p = 0.000): at alpha=0.2, FGSM ASR drops from 0.850 to 0.677 (-17.3%) with no clean accuracy cost. PGD ASR is unchanged (rho = -0.258, not significant). Margin also decreases with smoothing (rho = -1.000).
**Key metric:** FGSM ASR delta at alpha=0.2 vs 0.0: -0.173; rho(alpha, FGSM_ASR) = -1.000
**Status:** SUPPORTED (for FGSM), NOT SUPPORTED (for PGD)

---

## H256 - Input Complexity and Vulnerability

**Conclusion:** Input complexity features (pixel std, entropy, JPEG size) have weak-to-zero predictive power for FGSM or PGD vulnerability (AUROC 0.424–0.565, best for pixel entropy → FGSM = 0.565). None approach the margin baseline.
**Key metric:** Best AUROC (pixel_entropy → FGSM) = 0.565; margin AUROC ~0.95
**Status:** NOT SUPPORTED

---

## H257 - Batch Norm Statistics Shift under Attack

**Conclusion:** Aggregate BN activation shift between clean and adversarial inputs is small (mean = 0.082) and uncorrelated with PGD vulnerability (Spearman rho = 0.032, p = 0.579) or margin (rho = -0.028). BN shift is not a meaningful vulnerability signal.
**Key metric:** rho(BN shift, PGD success) = 0.032 (p = 0.579); AUROC = 0.535
**Status:** NOT SUPPORTED

---

## H258 - Random Seed Vulnerability Consistency

**Conclusion:** Vulnerability is highly consistent across 5 random seeds: mean pairwise Jaccard = 0.956, and 87.7% of test samples are universally vulnerable (all 5 seeds). Vulnerability frequency across seeds strongly predicts "always vulnerable" (AUROC = 1.000), while single-model margin is a poor predictor of cross-seed universality (AUROC = 0.135).
**Key metric:** Mean pairwise Jaccard = 0.956; universally vulnerable = 87.7%
**Status:** SUPPORTED (vulnerability is data-geometric, seed-independent)

---

## H259 - Perturbation Detectability via Pixel Statistics

**Conclusion:** Adversarial examples (FGSM and PGD) are trivially detectable when the perturbation delta L2 norm is included as a feature (AUROC = 1.000). Pixel statistics alone (mean, std, L2 norm) have near-chance AUROC (0.506 best). Detection requires access to the perturbation, not just the output image.
**Key metric:** Detector AUROC with delta_l2 = 1.000; pixel stats alone AUROC ≈ 0.50
**Status:** SUPPORTED (with delta access), NOT SUPPORTED (pixel stats alone)

---

## H260 - Targeted vs Untargeted Attack Difficulty

**Conclusion:** Targeted and untargeted PGD attacks achieve nearly identical ASR on Fashion-MNIST (untargeted 0.920, targeted 0.917), with only 1 sample differing between the two. Low-margin samples are equally susceptible to both attack types. The margin-gap correlation is non-significant (rho = 0.099, p = 0.087).
**Key metric:** Untargeted ASR = 0.920; Targeted ASR = 0.917; gap = 0.003
**Status:** INCONCLUSIVE (task too easy at eps=0.1 for any gap to emerge)

---

## H261 - Targeted Noise Hardening

**Conclusion:** NOT RUN — script written but not yet executed.
**Key metric:** N/A
**Status:** NOT RUN

---

## H262 - Gaussian Noise Sigma Sweep

**Dataset:** Fashion-MNIST  
**Conclusion:** Gaussian input noise augmentation during training provides marginal robustness gains at small sigma, peaks around sigma=0.1-0.2, then degrades clean accuracy without further robustness benefit. Best sigma gives FGSM_ASR reduction of ~15pp vs baseline but PGD_ASR barely moves. Noise augmentation addresses FGSM more than PGD.
**Key metric:** sigma=0.0: FGSM_ASR=0.844, PGD_ASR=1.000; best sigma reduces FGSM_ASR meaningfully, PGD_ASR unchanged
**Status:** PARTIALLY SUPPORTED (FGSM only)

---

## H263 - Noise vs AT Boundary Distance

**Dataset:** Fashion-MNIST  
**Conclusion:** Minimum perturbation epsilon (min_eps) correlates strongly with margin — Spearman rho=0.935 for standard model, 0.853 for AT model. Noise-augmented models show weaker correlation (rho=0.560). Noise and adversarial attacks probe similar geometric structure but noise augmentation only partially captures the boundary geometry that AT does.
**Key metric:** Spearman rho(min_eps, margin): standard=0.935, AT=0.853, noise-aug=0.560
**Status:** SUPPORTED

---

## H264 - Noise Curriculum Hardening

**Dataset:** Fashion-MNIST  
**Conclusion:** Curriculum increasing sigma during training (easy→hard noise) provides marginal improvement over fixed noise. Best curriculum (sigma 0→0.3) gives FGSM_ASR 0.650 vs baseline 0.842. PGD_ASR barely changes across conditions. No significant advantage over fixed sigma.
**Key metric:** Curriculum best FGSM_ASR=0.650 vs baseline 0.842; PGD_ASR ~unchanged
**Status:** NOT SUPPORTED (no advantage over fixed sigma)

---

## H265 - Per-Sample Noise Threshold

**Dataset:** Fashion-MNIST  
**Conclusion:** Per-sample noise destruction threshold (sigma_dest) correlates with input-space margin — Spearman rho=0.657. Samples that require more noise to destroy their prediction are also harder to attack adversarially. Noise and adversarial probes share the same geometric signal.
**Key metric:** Spearman rho(sigma_dest, margin)=0.657 (p<<0.001)
**Status:** SUPPORTED

---

## H266 - Noise Augmentation Class Specificity

**Dataset:** Fashion-MNIST  
**Conclusion:** Noise augmentation (sigma=0.2) reduces FGSM_ASR for most classes but effect is highly class-specific. T-shirt: -42pp, Trouser: -28pp. Some classes (Coat, class 6) show no improvement or slight degradation. Noise helps classes with smooth decision boundaries; fails for classes already near the boundary regardless.
**Key metric:** FGSM_ASR delta range: -42pp (T-shirt) to ~0pp (Coat); PGD_ASR delta range: -66pp (Trouser) to -17pp (T-shirt)
**Status:** SUPPORTED (class-specific effect confirmed)

---

## H267 - Input Noise vs Weight Noise

**Dataset:** Fashion-MNIST  
**Conclusion:** Input noise augmentation outperforms weight noise (Gaussian weight perturbation during training) for adversarial robustness. Weight noise: FGSM_ASR=0.896 (barely better than baseline 0.930). Input noise: FGSM_ASR=0.650. Weight noise does not transfer to input-space robustness.
**Key metric:** Input noise PGD_ASR=0.896; weight noise PGD_ASR=0.996; baseline PGD_ASR=1.000
**Status:** SUPPORTED (input noise > weight noise)

---

## H268 - Noise Certified vs Empirical Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** Certified robustness radius (randomised smoothing) correlates with empirical min_eps — Spearman rho=0.681. Certified and empirical robustness rank samples similarly: samples certified robust are empirically harder to attack. Certification provides a useful proxy but not a tight bound.
**Key metric:** Spearman rho(cert_radius, min_eps)=0.681 (p<<0.001)
**Status:** SUPPORTED

---

## H269 - Hidden Noise Layer Position

**Dataset:** Fashion-MNIST  
**Conclusion:** Inserting noise after the first block (noise_b1) reduces decision margin the most (ΔMargin=-3.73) and lowers PGD ASR slightly, suggesting early-layer noise disrupts gradient flow more than later positions. Mid/late-layer noise (b2, b3) yields smaller improvements and can increase FGSM ASR.
**Key metric:** noise_b1 ΔMargin=-3.73, ΔPGD_ASR=-0.015; noise_b2 ΔMargin=-0.85, ΔPGD_ASR=-0.005
**Status:** PARTIALLY SUPPORTED

---

## H270 - Manifold Mixup vs Input Mixup

**Dataset:** Fashion-MNIST  
**Conclusion:** Adversarial mixup (adv_mixup) dramatically outperforms both input and manifold mixup, cutting FGSM_ASR from 0.748 to 0.136 and PGD_ASR from 1.000 to 0.218. Manifold mixup offers only a marginal FGSM_ASR reduction over input mixup (+0.030 gap) while neither provides meaningful PGD robustness.
**Key metric:** adv_mixup ΔFGSM_ASR=-0.612, ΔPGD_ASR=-0.781; manifold vs input ΔFGSM_ASR=+0.030
**Status:** NOT SUPPORTED (manifold mixup does not beat input mixup; adv_mixup is the real winner)

---

## H271 - Learned vs Fixed Hidden Noise

**Dataset:** Fashion-MNIST  
**Conclusion:** Learned noise alone provides negligible benefit over fixed noise (similar clean accuracy and only minor PGD_ASR reduction). However, learned noise combined with adversarial training (learned_noise_at) is highly effective, reducing FGSM_ASR from 0.760 to 0.049 and PGD_ASR from 0.998 to 0.766, far surpassing fixed noise alone.
**Key metric:** learned_noise_at ΔFGSM_ASR=-0.711, ΔPGD_ASR=-0.232; learned sigma grows deeper (block3=0.016 vs block1=0.001)
**Status:** PARTIALLY SUPPORTED

---

## H272 - Hidden Noise Boundary Distance

**Dataset:** Fashion-MNIST  
**Conclusion:** Input-space noise increases mean boundary distance more than hidden noise (ΔMeanBD=+0.031 vs +0.001), suggesting input noise physically pushes samples further from decision boundaries. Neither variant shows a meaningful correlation between clean margin and boundary distance (Spearman rho near zero), so the hypothesis that hidden noise improves boundary distance more than input noise is not supported.
**Key metric:** input_noise MeanBD=0.054 vs hidden_noise MeanBD=0.024 vs baseline 0.022; Spearman rho baseline=0.055, input_noise=-0.199, hidden_noise=0.055
**Status:** NOT SUPPORTED

---

## H273 - SAM vs SGD Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** SAM (rho=0.05) does not improve adversarial robustness vs SGD. SAM: FGSM_ASR=0.708, PGD_ASR=0.904 vs SGD baseline FGSM_ASR=0.726, PGD_ASR=0.878. SAM marginally reduces FGSM_ASR but increases PGD_ASR — finds flatter weight-space minima but not more robust ones.
**Key metric:** SAM PGD_ASR=0.904 vs SGD PGD_ASR=0.878 (worse); margin drops 6.59→5.57
**Status:** NOT SUPPORTED

---

## H274 - SWA Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** Single-trajectory SWA (averaging checkpoints across training) improves FGSM_ASR by ~6pp (0.822→0.764) but not PGD. Multi-seed SWA collapses model to ~9% accuracy — BN statistics incompatible across seeds.
**Key metric:** SWA trajectory FGSM_ASR=0.764 vs baseline 0.822; multi-seed SWA clean_acc=0.086
**Status:** PARTIALLY SUPPORTED (single-trajectory only; multi-seed fails)

---

## H275 - Mode Connectivity Standard vs AT

**Dataset:** Fashion-MNIST  
**Conclusion:** Linear interpolation between standard and AT models shows increasing robustness as alpha increases (alpha=0 is standard, alpha=1 is AT). FGSM_ASR decreases linearly: 0.748→0.174 as alpha 0→1.0. No sharp barrier — adversarial loss landscape is convex along this path, suggesting AT models are not in a geometrically isolated basin from standard models on Fashion-MNIST.
**Key metric:** FGSM_ASR at alpha=0: 0.748; alpha=0.5: 0.520; alpha=1.0: 0.174. Monotone decrease.
**Status:** PARTIALLY SUPPORTED (mode connectivity exists; no barrier found)

---

## H276 - Spectral Norm Implicit AT

**Dataset:** Fashion-MNIST  
**Conclusion:** Spectral normalisation on weights HURTS adversarial robustness. SN-only: PGD_ASR=0.964 vs baseline 0.898 (+6.6pp worse). Combined SN+WD also worse. Spectral norm constrains Lipschitz constant but does not align gradients with input perturbations.
**Key metric:** SN-only PGD_ASR delta = +0.066 (worse); margin drops 5.69→4.27
**Status:** NOT SUPPORTED (hurts robustness)

---

## H277 - Dataset Distillation (Clean Gradient Matching)

**Dataset:** Fashion-MNIST  
**Conclusion:** Distilling 60k training images into 10 synthetic images (1 per class) via gradient matching fails completely. Model trained on 10 synthetic images: clean_acc=0.220 vs baseline 0.933. Synthetic images do not capture sufficient class structure from 200 outer steps of gradient matching.
**Key metric:** Distilled model clean_acc=0.220; baseline 0.933; PGD_ASR=0.927
**Status:** NOT SUPPORTED (distillation fails at K=1 image/class)

---

## H278 - Adversarial Dataset Distillation

**Dataset:** Fashion-MNIST  
**Conclusion:** Matching adversarial gradients (FGSM on real data) during distillation. Results similar to H277 — 10 synthetic images insufficient regardless of gradient type. Adversarial gradient matching does not meaningfully improve over clean gradient matching at K=1.
**Key metric:** Similar failure mode to H277; match_loss converges to ~0.53 but resulting images insufficient
**Status:** NOT SUPPORTED

---

## H279 - Robust Basin Geometry

**Dataset:** Fashion-MNIST  
**Conclusion:** AT model occupies a different loss basin from standard model. Per-sample Spearman rho(margin, pgd_vulnerable): standard=-0.447, AT=-0.768. AT model has much stronger margin-vulnerability correlation, consistent with AT pushing model into a geometrically more structured basin where margin is a reliable vulnerability predictor.
**Key metric:** rho(margin, pgd_vulnerable): standard=-0.447, AT=-0.768
**Status:** SUPPORTED

---

## H280 - Adversarial Gradient Structure

**Dataset:** Fashion-MNIST  
**Conclusion:** Adversarial gradients on correct predictions are more structured (higher sign consistency, lower entropy) than on misclassified samples. Dense gradient regions show higher vulnerability (rho=+0.74 with margin). Gradient sparsity and sign entropy are predictive of adversarial success.
**Key metric:** Spearman rho(grad_sparsity, margin)=+0.74
**Status:** SUPPORTED

---

## H281 - Gradient Norm Regularisation

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising parameter gradient norm during training provides no robustness benefit. FGSM_ASR and PGD_ASR unchanged across lambda values. Weight-space gradient norm penalty does not transfer to input-space robustness.
**Key metric:** All lambda conditions: FGSM_ASR ~0.75, PGD_ASR ~0.99 (no improvement)
**Status:** NOT SUPPORTED

---

## H282 - Gradient Sign Entropy

**Dataset:** Fashion-MNIST  
**Conclusion:** Sign alignment (fraction of gradient components with same sign as FGSM direction) negatively correlates with FGSM success — Spearman rho=-0.437 (standard), -0.458 (AT). Higher sign consistency = higher vulnerability. Consistent with FGSM exploiting coordinated gradient sign structure.
**Key metric:** Spearman rho(sign_alignment, FGSM_success)=-0.437 (standard), -0.458 (AT)
**Status:** SUPPORTED

---

## H283 - Backward Gradient Normalisation

**Dataset:** Fashion-MNIST  
**Conclusion:** Using L2-normalised gradient direction (Smooth-FGSM) instead of sign during AT changes the robustness geometry. Smooth-FGSM: L2_ASR=0.683 vs baseline 0.877 (good L2 robustness) but Linf_ASR=0.901 (poor Linf robustness). The direction of gradient normalisation during training determines which threat model the model becomes robust to.
**Key metric:** Smooth-FGSM L2_ASR=0.683 (vs baseline 0.877); Linf_ASR=0.901 (vs FGSM-AT 0.382)
**Status:** SUPPORTED (gradient direction determines robustness geometry)

---

## H284 - Gradient Breadth vs Vulnerability

**Dataset:** Fashion-MNIST  
**Conclusion:** Gradient breadth (number of significant gradient components) correlates with adversarial vulnerability. Broader gradients (more components activated) = more vulnerable. Supports hypothesis that concentrated gradients are harder to exploit with fixed-budget attacks.
**Key metric:** Spearman rho(breadth, vulnerability) significant; negative rho between breadth/entropy and margin
**Status:** SUPPORTED

---

## H285 - Gradient Breadth Regularisation

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising gradient entropy/breadth during training provides no consistent robustness improvement. Best lambda gives marginal FGSM_ASR reduction but PGD_ASR unchanged. Breadth regularisation in weight space does not translate to input-space concentration.
**Key metric:** Best: FGSM_ASR 0.704 vs baseline 0.752; PGD_ASR unchanged ~0.95
**Status:** NOT SUPPORTED (marginal at best)

---

## H286 - Inter-Batch Gradient Variance Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising variance between consecutive batch gradients (||g1-g2||²) gives marginal improvement at lambda=0.001 (FGSM_ASR -2.3pp) but hurts at lambda>=0.01. Sweet spot is lambda=0.001. Novel approach not previously studied for adversarial robustness.
**Key metric:** lambda=0.001: FGSM_ASR=0.424 vs baseline 0.447 (-2.3pp); lambda=0.01: FGSM_ASR=0.501 (worse)
**Status:** MARGINALLY SUPPORTED (lambda=0.001 only)

---

## H287 - Hutchinson Hessian Trace Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising the Hessian trace (via Hutchinson estimator) hurts adversarial robustness. Higher lambda = worse ASR. Hessian trace estimate does not decrease with lambda (146-155 range across lambdas) suggesting training does not reach flatter minima. 3-4× compute overhead for negative return.
**Key metric:** lambda=0.01: FGSM_ASR=0.488, PGD_ASR=0.558 vs baseline 0.457, 0.542 (worse)
**Status:** NOT SUPPORTED (hurts robustness)

---

## H288 - Input Gradient Penalty Comparison (7 conditions)

**Dataset:** Fashion-MNIST  
**Conclusion:** Comprehensive comparison of 7 implicit robustness methods. Input gradient norm penalty (Ross 2018) is best: PGD_ASR=0.514, clean_acc=0.888. SAM hurts robustness vs SGD baseline. Adam + batch variance penalty is worst (PGD_ASR=0.634). Adam does not subsume variance penalty — it actively interferes.
**Key metric:** Best: SGD + input grad penalty PGD_ASR=0.514, clean_acc=0.888; Worst: Adam+batch var PGD_ASR=0.634
**Status:** INPUT GRAD PENALTY SUPPORTED; SAM NOT SUPPORTED; BATCH VAR + ADAM NOT SUPPORTED

---

## H289 - Hidden Layer Adversarial Vulnerability

**Dataset:** Fashion-MNIST  
**Conclusion:** Early layers are most brittle. block0_out requires only L-inf ε=0.083 in activation space to flip predictions, vs ε=0.578 at head_linear_out. Vulnerability decreases monotonically with depth. Input-space (ε=0.171) sits between block1 and block2.
**Key metric:** Ranking: block0 (0.083) < block1 (0.093) < input (0.171) < block2 (0.205) < head_linear (0.578)
**Status:** SUPPORTED (early layers most brittle)

---

## H290 - Hidden Layer Adversarial Training

**Dataset:** Fashion-MNIST  
**Conclusion:** Block0 activation-space AT transfers well to input-space robustness — PGD_ASR=0.424 vs baseline 0.972, almost matching full input-space AT (0.358). Block1-AT weaker (PGD_ASR=0.708). All-layers-AT simultaneously collapses model to ~10% accuracy — too much stacked perturbation destroys the representation.
**Key metric:** Block0-AT PGD_ASR=0.424; Input FGSM-AT PGD_ASR=0.358; all-layers-AT clean_acc=0.105
**Status:** SUPPORTED (early-layer AT transfers; stacking all layers fails)

---

## H291 - Layer Vulnerability Profile

**Dataset:** Fashion-MNIST  
**Conclusion:** L2 activation radius correlates strongly with input-space margin at every layer (rho~0.90). Cross-layer radii are highly correlated (rho 0.87–0.97) — if a sample is brittle at one layer it is brittle at all. Jacobian norm does not significantly predict per-sample radius (rho -0.18 to +0.12, all p>0.1).
**Key metric:** rho(L2 radius, margin) ~0.90 at all layers; cross-layer rho 0.87–0.97; Jacobian-radius rho non-significant
**Status:** SUPPORTED (global brittleness structure; Jacobian norm not predictive per-sample)

---

## H292 - Rolling K-Batch Gradient Variance Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Extending H286's 2-batch variance penalty to K=10 batches improves robustness. K=10, lambda=0.01: PGD_ASR=0.509, clean_acc=0.890 (clean accuracy also improves). K=5 adds little over K=2. Sweet spot K=10, lambda=0.01.
**Key metric:** K=10, lambda=0.01: PGD_ASR=0.509, clean_acc=0.890 vs K=2 baseline PGD_ASR=0.540
**Status:** SUPPORTED (K=10 meaningfully better than K=2)

---

## H293 - Gradient Acceleration Penalty vs Momentum

**Dataset:** Fashion-MNIST  
**Conclusion:** Gradient acceleration penalty (||g_t - 2g_{t-1} + g_{t-2}||²) cannot substitute for momentum and adds nothing on top of it. No-momentum + any lambda: PGD_ASR ~0.71 (same as no-momentum baseline). Momentum+acceleration: PGD_ASR=0.524 vs momentum alone 0.529 (marginal). Confirms acceleration penalty ≈ momentum but momentum does it better at 1/25 compute cost.
**Key metric:** No-mom+accel: PGD_ASR 0.71 (no improvement); mom+accel: 0.524 vs mom-only 0.529
**Status:** NOT SUPPORTED (momentum already subsumes acceleration penalty)

---

## H294 - Higher-Order Input Gradient Penalties

**Dataset:** Fashion-MNIST  
**Conclusion:** Order-2 input gradient penalty (penalising gradient field smoothness) adds value when combined with order-1 at the right lambda. Best: order-1+2 (lambda1=0.01, lambda2=0.001): FGSM_ASR=0.435 (best of all implicit methods), PGD_ASR=0.520. Order-2 alone at lambda=0.01 also improves over baseline. Higher lambda2 overshoots.
**Key metric:** Order-1+2 best: FGSM_ASR=0.435, PGD_ASR=0.520; order-1 only: FGSM_ASR=0.454, PGD_ASR=0.521
**Status:** SUPPORTED (order-2 adds marginal value over order-1 at correct lambda)

---

## H295 - Combined Input + Activation Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Combining input gradient penalty (lam1) and activation gradient penalty (lam2) provides no benefit over the baseline; all combined configurations either match or worsen ASR compared to the no-penalty baseline. Activation-only penalty (lam2=0.01) is the marginal best variant but still offers no improvement over input-only penalty from prior hypotheses.  
**Key metric:** Best variant lam2=0.01: FGSM_ASR=0.757, PGD_ASR=0.913; combined lam1=lam2=0.01: FGSM_ASR=0.765, PGD_ASR=0.916 (matches baseline)  
**Status:** NOT SUPPORTED (no additive benefit from combining input + activation gradient penalties)

---

## H296 - First-Layer-Only Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Restricting gradient regularisation to the first layer only provides negligible robustness improvement. Best lambda (0.01) gives a slight FGSM reduction (0.758 vs 0.768 baseline), but PGD_ASR remains near or above baseline, and higher lambda degrades performance.  
**Key metric:** lam=0.01: FGSM_ASR=0.758, PGD_ASR=0.917 vs baseline FGSM_ASR=0.768, PGD_ASR=0.925  
**Status:** NOT SUPPORTED (first-layer-only penalty is insufficient for meaningful robustness gains)

---

## H297 - Adversarial Input Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising the gradient at adversarial inputs (rather than clean inputs) produces modest FGSM improvements over the clean-gradient variant, with best lam=0.01 achieving FGSM_ASR=0.748. However, PGD_ASR reductions are marginal and the clean-gradient penalty offers no benefit, suggesting adversarial-point regularisation is only weakly effective.  
**Key metric:** adv_grad lam=0.01: FGSM_ASR=0.748, PGD_ASR=0.903 vs baseline FGSM_ASR=0.768, PGD_ASR=0.922  
**Status:** PARTIALLY SUPPORTED (adversarial-point gradient penalty helps FGSM marginally; PGD improvement is minor)

---

## H298 - Gradient Cosine Alignment Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising misalignment between input gradients and attack directions via cosine similarity offers no meaningful robustness improvement. Best lambda (0.01) gives FGSM_ASR=0.753 (marginal) and PGD_ASR=0.918 (essentially unchanged), while lam=0.1 degrades both metrics. The 6× compute overhead is not justified.  
**Key metric:** lam=0.01: FGSM_ASR=0.753, PGD_ASR=0.918 vs baseline FGSM_ASR=0.763, PGD_ASR=0.917  
**Status:** NOT SUPPORTED (cosine alignment penalty adds no meaningful robustness at significant compute cost)

---

## H299 - Gradient Sign Consistency Training

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising sign inconsistency in input gradients across training steps yields very minor FGSM improvements at lam=0.1 (0.765 vs 0.775 baseline) and a slight PGD reduction, but no setting achieves a substantial robustness gain. The effect is weak and inconsistent across lambdas.  
**Key metric:** lam=0.1: FGSM_ASR=0.765, PGD_ASR=0.913 vs baseline FGSM_ASR=0.775, PGD_ASR=0.933  
**Status:** NOT SUPPORTED (gradient sign consistency training produces only negligible robustness gains)

---

## H300 - Gradient Clipping Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** Training-time gradient clipping does not improve adversarial robustness and can actively worsen it. Global-norm clipping at T=0.5 reduces FGSM_ASR slightly (0.722) but sharply increases PGD_ASR to 0.966, indicating false improvement via gradient masking. Value and per-layer clipping show no consistent benefit.  
**Key metric:** global_norm T=0.5: FGSM_ASR=0.722, PGD_ASR=0.966 (gradient masking); value T=0.5: FGSM_ASR=0.749, PGD_ASR=0.920  
**Status:** NOT SUPPORTED (gradient clipping does not robustify; global-norm clipping induces gradient masking)

---

## H301 - Lion Optimizer Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** Lion optimizer reduces FGSM_ASR substantially (0.731 vs 0.760 for SGD) but raises PGD_ASR (0.940) and inflates the margin metric to anomalous values (~15.8), strongly suggesting gradient masking. Lion with weight decay (lion_wd) reduces PGD_ASR to 0.910 but still shows inflated margins, so apparent FGSM gains are likely artefactual.  
**Key metric:** lion: FGSM_ASR=0.731, PGD_ASR=0.940, margin=15.791; lion_wd: FGSM_ASR=0.716, PGD_ASR=0.910, margin=14.469  
**Status:** NOT SUPPORTED (Lion's apparent FGSM gains are consistent with gradient masking, not true robustness)

---

## H302 - Gradient Centralization

**Dataset:** Fashion-MNIST  
**Conclusion:** Gradient centralization (zero-meaning weight gradients) provides no robustness benefit and worsens PGD_ASR across all variants. GC alone raises PGD_ASR from 0.930 (baseline) to 0.954; GC+Adam reaches 0.973. Clean accuracy is unaffected but adversarial resilience degrades.  
**Key metric:** gc: FGSM_ASR=0.764, PGD_ASR=0.954 vs baseline FGSM_ASR=0.765, PGD_ASR=0.930  
**Status:** NOT SUPPORTED (gradient centralization worsens PGD robustness with no FGSM improvement)

---

## H303 - Adversarial Logit Pairing

**Dataset:** Fashion-MNIST  
**Conclusion:** Adversarial Logit Pairing (ALP) dramatically reduces both FGSM and PGD attack success rates at the cost of clean accuracy. At lam=1.0, PGD_ASR drops to 0.336 (from 0.916 baseline) and clean accuracy falls to 0.807. The FGSM-AT variant achieves similar robustness (PGD_ASR=0.331) with clean=0.804. Higher lambda degrades clean accuracy further with no robustness gain.  
**Key metric:** lam=1.0: clean=0.807, FGSM_ASR=0.301, PGD_ASR=0.336; lam=0.1: clean=0.859, FGSM_ASR=0.348, PGD_ASR=0.388  
**Status:** SUPPORTED (ALP substantially improves robustness, with expected clean accuracy trade-off)

---

## H304 - TRADES Objective

**Dataset:** Fashion-MNIST  
**Conclusion:** TRADES achieves strong adversarial robustness with a controlled clean accuracy trade-off. At beta=1, PGD_ASR drops to 0.340 and FGSM_ASR to 0.338 while clean accuracy remains at 0.840. Higher beta values further reduce clean accuracy (0.794 at beta=6) without meaningfully improving robustness beyond beta=1. Confirms TRADES as an effective adversarial training objective.  
**Key metric:** beta=1: clean=0.840, FGSM_ASR=0.338, PGD_ASR=0.340; beta=6: clean=0.794, FGSM_ASR=0.339, PGD_ASR=0.332  
**Status:** SUPPORTED (TRADES substantially reduces ASR at all tested beta values, replicating established results)

---

## H305 - Input Gradient L1 vs L2 Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** L1 input gradient penalty achieves slightly stronger PGD robustness at high lambda (0.1: PGD_ASR=0.881, FGSM_ASR=0.734) compared to L2 at the same lambda (PGD_ASR=0.905, FGSM_ASR=0.751), but both remain well above adversarial training baselines. Neither norm delivers transformative robustness; L1 sparsifies gradients marginally more effectively.  
**Key metric:** L1 lam=0.1: FGSM_ASR=0.734, PGD_ASR=0.881; L2 lam=0.1: FGSM_ASR=0.751, PGD_ASR=0.905  
**Status:** PARTIALLY SUPPORTED (L1 marginally outperforms L2 at high lambda, but overall robustness gains are modest)

---

## H306 - Mixup + Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Combining Mixup with gradient penalty does not synergise; mixup alone reduces FGSM_ASR to 0.723 but sharply raises PGD_ASR to 0.973 (gradient masking), and the combined variant similarly shows FGSM_ASR=0.720 with PGD_ASR=0.953. Gradient-penalty-only roughly matches baseline. The apparent FGSM improvement from mixup is not accompanied by PGD robustness.  
**Key metric:** mixup_grad_penalty: FGSM_ASR=0.720, PGD_ASR=0.953; grad_penalty_only: FGSM_ASR=0.761, PGD_ASR=0.921  
**Status:** NOT SUPPORTED (Mixup + gradient penalty shows likely gradient masking; no genuine robustness improvement)

---

## H307 - Label Smoothing + Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Label smoothing improves FGSM_ASR markedly (0.680 alone, 0.692 combined) while keeping PGD_ASR near baseline (~0.907-0.911) and slightly boosting clean accuracy. Adding gradient penalty to label smoothing provides no additional benefit. The FGSM reduction from label smoothing may reflect softened margins rather than true input-space robustness.  
**Key metric:** label_smooth_only: FGSM_ASR=0.680, PGD_ASR=0.911, clean=0.882; combined: FGSM_ASR=0.692, PGD_ASR=0.907, clean=0.886  
**Status:** PARTIALLY SUPPORTED (label smoothing reduces FGSM ASR but PGD robustness is unchanged; gradient penalty adds nothing)

---

## H308 - Dropout Gradient Interaction

**Dataset:** Fashion-MNIST  
**Conclusion:** Increasing dropout rate monotonically reduces both FGSM and PGD attack success rates while slightly lowering clean accuracy. Dropout=0.5 cuts PGD ASR from 0.931 to 0.842 and shrinks input gradient norms from 0.012 to 0.009, suggesting dropout genuinely smooths the loss surface rather than merely masking gradients.  
**Key metric:** dropout=0.5: PGD_ASR=0.842, FGSM_ASR=0.683, clean=0.871, grad_norm=0.0085  
**Status:** SUPPORTED

---

## H309 - Batch Size Gradient Noise

**Dataset:** Fashion-MNIST  
**Conclusion:** Smaller batch sizes (more gradient noise) yield substantially lower PGD ASR but also lower clean accuracy when controlling for total compute. Batch=16 achieves PGD_ASR=0.718 vs 0.981 at batch=512, but at the cost of 10 percentage points of clean accuracy — the robustness is largely an artefact of under-training rather than noise-induced smoothing.  
**Key metric:** batch=16: PGD_ASR=0.718, clean=0.784; batch=512: PGD_ASR=0.981, clean=0.879  
**Status:** NOT SUPPORTED (robustness from small batches is confounded with under-training)

---

## H310 - Focal Loss Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** Focal loss with gamma=1-2 provides a marginal reduction in FGSM ASR (0.738-0.742 vs 0.769 for CE) but no meaningful PGD improvement (~0.916-0.919). Gamma=0.5 catastrophically fails (clean acc collapses to 0.105), and gamma=5 slightly worsens both metrics. Focal loss does not buy adversarial robustness.  
**Key metric:** gamma=2: FGSM_ASR=0.738, PGD_ASR=0.919, clean=0.888  
**Status:** NOT SUPPORTED

---

## H311 - Symmetric Cross Entropy

**Dataset:** Fashion-MNIST  
**Conclusion:** Symmetric cross entropy (SCE) with reverse-KL weight beta=0.1-1.0 provides a small PGD ASR reduction (0.893-0.906 vs 0.917 baseline) while maintaining clean accuracy. The effect is modest and unlikely to be practically significant.  
**Key metric:** beta=0.5: PGD_ASR=0.891, FGSM_ASR=0.770, clean=0.878  
**Status:** NOT SUPPORTED (marginal PGD improvement, not practically significant)

---

## H312 - Gradient Penalty Schedule

**Dataset:** Fashion-MNIST  
**Conclusion:** No gradient penalty schedule (fixed, linear warmup, cosine, step) meaningfully improves robustness over the no-penalty baseline. All schedules yield PGD ASR in the 0.921-0.933 range with comparable clean accuracy. Scheduling the gradient penalty is ineffective.  
**Key metric:** best=cosine: PGD_ASR=0.921, clean=0.881; baseline(none): PGD_ASR=0.922  
**Status:** NOT SUPPORTED

---

## H313 - Per-Class Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Per-class gradient penalty (scaling lambda inversely with per-class vulnerability) provides negligible improvement over uniform penalty — PGD ASR 0.896 vs 0.900 and FGSM ASR 0.745 vs 0.748. The two approaches are statistically indistinguishable.  
**Key metric:** per_class: PGD_ASR=0.896, FGSM_ASR=0.745; uniform: PGD_ASR=0.900, FGSM_ASR=0.748  
**Status:** NOT SUPPORTED

---

## H314 - Natural Gradient Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** Diagonal natural gradient (DiagNG) achieves dramatically lower ASR (FGSM=0.368, PGD=0.402) but at the cost of collapsed clean accuracy (0.637) and an anomalous margin of 3737, indicating severe gradient masking / broken optimization rather than genuine robustness. SGD remains the most reliable optimizer.  
**Key metric:** DiagNG: PGD_ASR=0.402, clean=0.637, margin=3737; SGD: PGD_ASR=0.933, clean=0.885  
**Status:** NOT SUPPORTED (apparent robustness is gradient masking from broken optimization)

---

## H315 - Gradient Regularisation Frequency

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising high-frequency components of input gradients (via DFT filtering) at lambda=0.01 and 0.10 produces no measurable change in FGSM or PGD ASR compared to the no-penalty baseline (~0.919 PGD ASR across all settings).  
**Key metric:** lambda=0.10: PGD_ASR=0.918, FGSM_ASR=0.768; baseline: PGD_ASR=0.920, FGSM_ASR=0.762  
**Status:** NOT SUPPORTED

---

## H316 - Adversarial Weight Perturbation

**Dataset:** Fashion-MNIST  
**Conclusion:** FGSM-AT alone (gamma=0) achieves strong robustness (PGD_ASR=0.328, clean=0.804). Adding adversarial weight perturbation (AWP) with gamma=0.001 degrades both clean accuracy (0.750) and robustness (PGD_ASR=0.375). Higher gamma=0.01 collapses training entirely (clean=0.109). AWP hurts rather than helps in this setting.  
**Key metric:** FGSM-AT: PGD_ASR=0.328, clean=0.804; AWP(0.001): PGD_ASR=0.375, clean=0.750  
**Status:** NOT SUPPORTED (AWP degrades both accuracy and robustness vs plain FGSM-AT)

---

## H317 - Gradient Penalty Warmup/Cooldown

**Dataset:** Fashion-MNIST  
**Conclusion:** No warmup/cooldown schedule for gradient penalty improves robustness over the no-penalty baseline. All variants (fixed, cosine anneal, warmup, cooldown, warmup+cooldown) cluster around PGD_ASR=0.914-0.932 with near-identical clean accuracy. Cosine annealing gives the best PGD_ASR (0.914) but the improvement is negligible.  
**Key metric:** cosine_anneal: PGD_ASR=0.914, clean=0.880; baseline(none): PGD_ASR=0.922  
**Status:** NOT SUPPORTED

---

## H318 - Stochastic Gradient Sign Training

**Dataset:** Fashion-MNIST  
**Conclusion:** Randomly flipping gradient signs during training (p=0.05-0.20) provides a small reduction in FGSM ASR (0.755-0.759 vs 0.772 baseline) and PGD ASR (0.914-0.923 vs 0.924 baseline). The effect is modest and diminishes at higher flip rates. This is not a viable robustness strategy.  
**Key metric:** p_flip=0.05: PGD_ASR=0.914, FGSM_ASR=0.755, clean=0.883  
**Status:** NOT SUPPORTED (marginal improvements, not practically meaningful)

---

## H319 - Gradient Penalty Only on Misclassified

**Dataset:** Fashion-MNIST  
**Conclusion:** Applying gradient penalty selectively to misclassified samples yields nearly identical robustness to uniform penalty and the no-penalty baseline (PGD_ASR=0.918-0.928, FGSM_ASR=0.770-0.777). Selective application does not concentrate the regularisation benefit.  
**Key metric:** misclassified: PGD_ASR=0.920, FGSM_ASR=0.770; none: PGD_ASR=0.918  
**Status:** NOT SUPPORTED

---

## H320 - Feature Gradient Alignment

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising the alignment between feature activations and input gradients at lambda=0.10 yields a small PGD ASR improvement (0.905 vs 0.915 baseline) while maintaining clean accuracy (0.883). The effect is modest but consistent across lambda values, suggesting some signal. The FGSM improvement is negligible.  
**Key metric:** lambda=0.10: PGD_ASR=0.905, FGSM_ASR=0.764, clean=0.883, margin=6.246  
**Status:** PARTIALLY SUPPORTED (small but consistent PGD improvement; not large enough to be practically useful)

---

## H321 - CutMix + Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** CutMix alone substantially reduces both FGSM_ASR (0.638 vs baseline 0.753) and PGD_ASR (0.869 vs 0.913), but adding a gradient penalty on top provides no further benefit and slightly worsens PGD robustness. Gradient penalty alone is ineffective, slightly increasing PGD_ASR above baseline.  
**Key metric:** cutmix_only: FGSM_ASR=0.638, PGD_ASR=0.869; cutmix+gp(0.01): FGSM_ASR=0.642, PGD_ASR=0.883; baseline: FGSM_ASR=0.753, PGD_ASR=0.913  
**Status:** PARTIALLY SUPPORTED (CutMix alone improves robustness but gradient penalty does not add further benefit)

---

## H322 - Elastic Net Gradient Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Neither L1-only, L2-only, nor elastic-net combinations of gradient penalties produce consistent robustness gains over baseline. The best result (L1 lam=0.01: FGSM_ASR=0.745, PGD_ASR=0.908) is only marginally better than baseline (0.761/0.924) and the improvement does not hold across configurations.  
**Key metric:** best: L1only(0.01): FGSM_ASR=0.745, PGD_ASR=0.908; baseline: FGSM_ASR=0.761, PGD_ASR=0.924  
**Status:** NOT SUPPORTED (elastic-net gradient penalties yield no reliable robustness improvement)

---

## H323 - Jacobian Frobenius Penalty

**Dataset:** Fashion-MNIST  
**Conclusion:** Penalising the full Jacobian Frobenius norm yields strong robustness gains: at lam=0.01, FGSM_ASR drops to 0.619 and PGD_ASR to 0.676, compared to baseline 0.774/0.927. Plain input-gradient penalty at equivalent strength provides no improvement, confirming that the full Jacobian constraint is the operative factor.  
**Key metric:** jacobian(lam=0.01): FGSM_ASR=0.619, PGD_ASR=0.676, clean=0.881; baseline: FGSM_ASR=0.774, PGD_ASR=0.927  
**Status:** SUPPORTED (Jacobian Frobenius penalty substantially reduces adversarial success rates)

---

## H324 - Virtual Adversarial Training

**Dataset:** Fashion-MNIST  
**Conclusion:** FGSM adversarial training dramatically reduces both FGSM_ASR (0.294) and PGD_ASR (0.337) versus baseline (0.774/0.922), confirming strong robustness from adversarial training. Virtual adversarial training (VAT, eps=0.1) provides only marginal improvement over baseline (FGSM_ASR=0.749, PGD_ASR=0.913).  
**Key metric:** FGSM-AT: FGSM_ASR=0.294, PGD_ASR=0.337, clean=0.811; VAT(eps=0.1): FGSM_ASR=0.749, PGD_ASR=0.913; baseline: FGSM_ASR=0.774, PGD_ASR=0.922  
**Status:** PARTIALLY SUPPORTED (FGSM-AT is highly effective; VAT with small perturbation radius provides minimal benefit)

---

## H325 - Confidence Regularisation

**Dataset:** Fashion-MNIST  
**Conclusion:** Entropy-based confidence regularisation at high lambda (lam=1.0) reduces FGSM_ASR to 0.637 and PGD_ASR to 0.896, but at the cost of shrinking the decision margin (1.27 vs baseline 6.13), suggesting gradient masking. Confidence penalty variants do not improve robustness and worsen PGD_ASR at high lambda.  
**Key metric:** entropy(lam=1.0): FGSM_ASR=0.637, PGD_ASR=0.896, margin=1.268; baseline: FGSM_ASR=0.761, PGD_ASR=0.911  
**Status:** NOT SUPPORTED (apparent FGSM improvement at high lambda is accompanied by margin collapse indicative of gradient masking)

---

## H326 - Gradient Penalty + Adversarial Training Combined

**Dataset:** Fashion-MNIST  
**Conclusion:** FGSM adversarial training alone achieves strong robustness (FGSM_ASR=0.295, PGD_ASR=0.331). Adding gradient penalty at any lambda does not improve over FGSM-AT alone and slightly reduces the margin, indicating no synergy between the two defences.  
**Key metric:** FGSM-AT: FGSM_ASR=0.295, PGD_ASR=0.331; FGSM-AT+gp(0.01): FGSM_ASR=0.306, PGD_ASR=0.334; baseline: FGSM_ASR=0.772, PGD_ASR=0.922  
**Status:** NOT SUPPORTED (gradient penalty adds no benefit on top of adversarial training)

---

## H327 - Randomised Smoothing Training

**Dataset:** Fashion-MNIST  
**Conclusion:** Training with Gaussian noise augmentation progressively reduces both FGSM_ASR and PGD_ASR as sigma increases, with sigma=0.3 achieving FGSM_ASR=0.623 and PGD_ASR=0.668 at the cost of clean accuracy dropping to 0.818. Both FGSM and PGD ASR decrease together, indicating genuine robustness rather than gradient masking.  
**Key metric:** sigma=0.3: FGSM_ASR=0.623, PGD_ASR=0.668, clean=0.818; sigma=0.2: FGSM_ASR=0.645, PGD_ASR=0.734, clean=0.857; baseline: FGSM_ASR=0.767, PGD_ASR=0.916  
**Status:** SUPPORTED (noise augmentation training provides genuine robustness gains, trading off clean accuracy)

---

## H328 - Input Gradient Clipping Training

**Dataset:** Fashion-MNIST  
**Conclusion:** Clipping input gradients during training at any tested threshold (tau=0 to 2.0) provides no robustness benefit; FGSM_ASR and PGD_ASR remain at or above baseline across all conditions. The approach fails to constrain adversarial vulnerability.  
**Key metric:** tau=0(all clipped): FGSM_ASR=0.756, PGD_ASR=0.905; tau=2.0: FGSM_ASR=0.779, PGD_ASR=0.932; baseline: FGSM_ASR=0.773, PGD_ASR=0.925  
**Status:** NOT SUPPORTED (gradient clipping during training does not reduce adversarial success rates)

---

## H329 - Weight Decay vs Gradient Interaction

**Dataset:** Fashion-MNIST  
**Conclusion:** Moderate weight decay (wd=1e-4) yields a slight reduction in FGSM_ASR (0.751) and PGD_ASR (0.909) while maintaining clean accuracy, but the gains are small. High weight decay (wd=1e-2) collapses the margin (3.61) and raises PGD_ASR (0.936), indicating over-regularisation. Input gradient norms remain essentially flat across all weight decay values.  
**Key metric:** wd=1e-4: FGSM_ASR=0.751, PGD_ASR=0.909, margin=6.317; wd=1e-2: FGSM_ASR=0.784, PGD_ASR=0.936, margin=3.606; baseline wd=0: FGSM_ASR=0.772, PGD_ASR=0.923  
**Status:** NOT SUPPORTED (weight decay does not meaningfully interact with gradient-based robustness; moderate values give negligible improvement)

---

## H330 - Proximal Gradient Training

**Dataset:** Fashion-MNIST  
**Conclusion:** Proximal gradient steps (L1 soft-thresholding on weights) at small lambda cause margin collapse without meaningful robustness gain, and at larger lambda cause instability or clean accuracy collapse. The approach does not yield reliable adversarial robustness.  
**Key metric:** SGD+prox(1e-3): FGSM_ASR=0.615, PGD_ASR=0.678, clean=0.760, margin=0.808; Adam+prox(1e-3): clean=0.105 (collapsed); baseline: FGSM_ASR=0.761, PGD_ASR=0.911  
**Status:** NOT SUPPORTED (proximal gradient training is unstable and does not produce reliable robustness gains)

---

## H331 - Gradient Penalty Per-Epoch Analysis

**Dataset:** Fashion-MNIST  
**Conclusion:** Tracking robustness epoch-by-epoch shows that gradient penalty (lam=0.01) does not separate from the baseline trajectory: both conditions converge to near-identical FGSM_ASR (~0.764 vs 0.759) and PGD_ASR (~0.924 vs 0.920) by epoch 10, and input gradient norms are virtually the same throughout training.  
**Key metric:** gp ep=10: FGSM_ASR=0.764, PGD_ASR=0.924, inp_grad=0.0072; baseline ep=10: FGSM_ASR=0.759, PGD_ASR=0.920, inp_grad=0.0069  
**Status:** NOT SUPPORTED (gradient penalty provides no epoch-level benefit over baseline; trajectories are indistinguishable)

---

## H332 - Gradient Penalty Transfer Attack Robustness

**Dataset:** Fashion-MNIST  
**Conclusion:** Gradient penalty training (lam=0.01) provides no improvement in transfer attack robustness compared to the standard baseline: transfer FGSM_ASR (0.456) and transfer PGD_ASR (0.455) are identical to baseline (0.456/0.454). FGSM adversarial training reduces both white-box and transfer ASR substantially (transfer FGSM_ASR=0.212, transfer PGD_ASR=0.215).  
**Key metric:** baseline: transfer_FGSM_ASR=0.456, transfer_PGD_ASR=0.454; gp(0.01): transfer_FGSM_ASR=0.456, transfer_PGD_ASR=0.455; FGSM-AT: transfer_FGSM_ASR=0.212, transfer_PGD_ASR=0.215  
**Status:** NOT SUPPORTED (gradient penalty confers no transfer robustness advantage over standard training)

---

## H358 - Virtual Adversarial Training
**Dataset:** Fashion-MNIST
**Conclusion:** VAT with high regularisation strength (lambda=10) provides a moderate reduction in adversarial success rates (PGD ASR drops from 0.911 to 0.771, FGSM ASR from 0.752 to 0.657), but robustness remains high and clean accuracy is maintained near baseline. Low lambda values produce negligible benefit.
**Key metric:** lambda=10: clean=0.874, FGSM_ASR=0.657, PGD_ASR=0.771; baseline: FGSM_ASR=0.752, PGD_ASR=0.911
**Status:** PARTIALLY SUPPORTED

---

## H359 - Adversarial Logit Pairing
**Dataset:** Fashion-MNIST
**Conclusion:** ALP collapses the model at moderate-to-high lambda values: lambda=1.0 and 10.0 both drive clean accuracy to near-chance (0.108) with adversarial ASR of 0.892, indicating training instability and model failure rather than robustness. Only very weak regularisation (lambda=0.1) shows partial benefit while retaining reasonable clean accuracy.
**Key metric:** lambda=0.1: clean=0.742, PGD_ASR=0.758; lambda=1.0: clean=0.108 (collapsed)
**Status:** NOT SUPPORTED

---

## H360 - Class-Conditional Gradient Alignment
**Dataset:** Fashion-MNIST
**Conclusion:** Class-conditional gradient alignment (CCGA) produces no meaningful improvement over the baseline: FGSM ASR and PGD ASR remain essentially unchanged (0.914 in both conditions) and clean accuracy is nearly identical. The regularisation has no measurable effect on robustness.
**Key metric:** baseline: clean=0.877, PGD_ASR=0.914; CCGA: clean=0.884, PGD_ASR=0.914
**Status:** NOT SUPPORTED

---

## H361 - Perceptually Aligned Gradient
**Dataset:** Fashion-MNIST
**Conclusion:** Perceptually aligned gradient penalty produces negligible changes in adversarial success rates across all tested lambda values — PGD ASR decreases only marginally from 0.915 to 0.907 at lambda=0.1 while clean accuracy is maintained. The intervention offers no practically meaningful robustness improvement.
**Key metric:** lambda=0.1: clean=0.878, FGSM_ASR=0.766, PGD_ASR=0.907; baseline: PGD_ASR=0.915
**Status:** NOT SUPPORTED

---

## H362 - Gradient Diversity Ensemble
**Dataset:** Fashion-MNIST
**Conclusion:** Ensembling alone (without gradient diversity penalty) reduces PGD ASR from 0.928 to 0.758, and adding the diversity penalty provides additional FGSM reduction (ASR drops to 0.610 at lambda=0.1). The benefit appears to come primarily from the ensemble architecture rather than the diversity regularisation term itself.
**Key metric:** ensemble+lambda=0.1: clean=0.888, FGSM_ASR=0.610, PGD_ASR=0.741; single model: PGD_ASR=0.928
**Status:** PARTIALLY SUPPORTED

---

## H363 - Reverse-KL TRADES
**Dataset:** Fashion-MNIST
**Conclusion:** Reverse-KL TRADES achieves strong robustness: at beta=1, PGD ASR drops dramatically to 0.324 from the unregularised baseline, with clean accuracy of 0.835. Higher beta values trade further clean accuracy for diminishing robustness gains, suggesting beta=1 is the optimal operating point.
**Key metric:** beta=1: clean=0.835, FGSM_ASR=0.326, PGD_ASR=0.324; beta=6: clean=0.723, PGD_ASR=0.383
**Status:** SUPPORTED

---

## H364 - Jacobian Nuclear Norm Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Jacobian nuclear norm regularisation progressively reduces PGD ASR as lambda increases (0.925 → 0.816 at lambda=0.01), with FGSM ASR also falling from 0.769 to 0.681. Clean accuracy is well-preserved, but absolute robustness remains poor with the majority of adversarial examples still succeeding.
**Key metric:** lambda=0.01: clean=0.882, FGSM_ASR=0.681, PGD_ASR=0.816; baseline: PGD_ASR=0.925
**Status:** PARTIALLY SUPPORTED

---

## H365 - Confidence-Weighted Gradient Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Confidence-weighted gradient penalty provides substantial robustness improvement at lambda=0.1: PGD ASR falls from 0.913 to 0.653 and FGSM ASR from 0.767 to 0.584, while clean accuracy is maintained at 0.878. The regularisation scales effectively with strength without collapsing the model.
**Key metric:** lambda=0.1: clean=0.878, FGSM_ASR=0.584, PGD_ASR=0.653; baseline: PGD_ASR=0.913
**Status:** SUPPORTED

---

## H366 - Smooth Activation Swap (ReLU vs SiLU)
**Dataset:** Fashion-MNIST
**Conclusion:** Replacing ReLU with SiLU provides no robustness benefit without adversarial training — both have similar high PGD ASR (~0.924-0.926). With FGSM adversarial training, both activations achieve similar robustness (PGD ASR ~0.334), indicating that activation smoothness is not the limiting factor for adversarial robustness.
**Key metric:** relu+AT: clean=0.802, PGD_ASR=0.334; silu+AT: clean=0.804, PGD_ASR=0.334; silu baseline: PGD_ASR=0.926
**Status:** NOT SUPPORTED

---

## H367 - Gradient Penalty with Mixup (AugMax-style)
**Dataset:** Fashion-MNIST
**Conclusion:** Adversarial mixup (with or without gradient penalty) achieves strong robustness with PGD ASR of ~0.364, though at some cost to clean accuracy (~0.84). Standard mixup and gradient penalty alone provide no robustness benefit. The robustness gain is driven by the adversarial training component of the mixup strategy.
**Key metric:** adv_mixup: clean=0.845, FGSM_ASR=0.336, PGD_ASR=0.364; std_mixup: PGD_ASR=0.966
**Status:** PARTIALLY SUPPORTED

---

## H368 - Bregman Divergence AT (Itakura-Saito)
**Dataset:** Fashion-MNIST
**Conclusion:** Bregman divergence adversarial training with the Itakura-Saito divergence is highly unstable: beta=1 achieves partial robustness (PGD ASR=0.478) but with poor clean accuracy (0.709), while beta=3 and beta=6 collapse entirely to near-chance performance with undefined margin values. The approach is not viable in this form.
**Key metric:** beta=1: clean=0.709, PGD_ASR=0.478; beta=3: clean=0.105 (collapsed, margin=nan)
**Status:** NOT SUPPORTED

---

## H369 - Multi-Scale Gradient Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Multi-scale gradient penalty provides no robustness improvement over the baseline — PGD ASR remains near 0.922-0.930 across all lambda values, with FGSM ASR similarly unchanged or slightly worsening. The penalty marginally increases the decision margin but has no effect on attack success rates.
**Key metric:** lambda=0.01: clean=0.871, FGSM_ASR=0.769, PGD_ASR=0.922; baseline: PGD_ASR=0.923
**Status:** NOT SUPPORTED

---

## H370 - Adversarial Training with ELLE Local Linearity Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** The ELLE local linearity penalty applied on top of FGSM adversarial training provides no additional robustness benefit: PGD ASR remains ~0.334 and FGSM ASR ~0.308 across all lambda values, essentially identical to the FGSM-AT baseline without the penalty. ELLE does not improve upon standard adversarial training.
**Key metric:** lambda=0.0: clean=0.803, PGD_ASR=0.333; lambda=0.1: clean=0.798, PGD_ASR=0.337 (no improvement)
**Status:** NOT SUPPORTED

---

## H333 - Gradient Similarity Across Classes
**Dataset:** Fashion-MNIST
**Conclusion:** Within-class gradient similarity is positive and higher in adversarially trained models (W-A=0.2135 vs 0.0961 baseline), confirming that adversarial training increases within-class gradient alignment rather than reducing it; across-class similarity remains near zero in both conditions, suggesting cross-class gradient repulsion is not the mechanism by which AT achieves robustness.
**Key metric:** baseline: within_sim=0.0922, across_sim=-0.0040, W-A=0.0961; fgsm_at: within_sim=0.2027, across_sim=-0.0108, W-A=0.2135, PGD_ASR=0.3300
**Status:** PARTIALLY SUPPORTED (within-class similarity correlates with robustness but direction is opposite to naive expectation)

---

## H334 - Gradient Penalty on Logit Margin Loss
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising the gradient of the logit-margin loss provides no robustness benefit: all margin_lam and ce_lam conditions yield PGD_ASR in the range 0.91–0.94 and FGSM_ASR in 0.75–0.79, virtually identical to the unpenalised baseline (PGD_ASR=0.918).
**Key metric:** baseline (margin_lam=0): FGSM_ASR=0.7530, PGD_ASR=0.9180; ce_lam=0.1: FGSM_ASR=0.7505, PGD_ASR=0.9115 (marginal)
**Status:** NOT SUPPORTED (logit-margin gradient penalty does not improve adversarial robustness)

---

## H335 - Input Gradient Norm Normalisation
**Dataset:** Fashion-MNIST
**Conclusion:** Normalising input gradients during training and adding an explicit norm penalty produce no meaningful change in adversarial susceptibility: all three conditions give nearly identical FGSM_ASR (~0.757–0.768) and PGD_ASR (~0.918–0.930), with gradient norms essentially unchanged.
**Key metric:** standard: FGSM_ASR=0.7680, PGD_ASR=0.9205, grad_norm=1.8448; normalised+penalty: FGSM_ASR=0.7575, PGD_ASR=0.9180, grad_norm=1.8671
**Status:** NOT SUPPORTED (gradient norm normalisation does not reduce adversarial vulnerability)

---

## H336 - Inter-Class Gradient Orthogonality Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising cosine similarity between per-class mean gradients (encouraging inter-class orthogonality) does not reduce attack success rates: FGSM_ASR and PGD_ASR are nearly identical across all lambda values, though the penalty slightly increases the logit margin.
**Key metric:** lam=0: FGSM_ASR=0.7660, PGD_ASR=0.9175, margin=5.896; lam=0.01: FGSM_ASR=0.7625, PGD_ASR=0.9190, margin=6.245
**Status:** NOT SUPPORTED (inter-class gradient orthogonality penalty yields no robustness improvement)

---

## H337 - Gradient Magnitude Entropy Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Encouraging high-entropy (diffuse) input gradient magnitudes provides a modest FGSM_ASR reduction at lam=1.0 (0.714 vs 0.784 baseline) and reduces PGD_ASR to 0.877 while maintaining clean accuracy (0.879), suggesting gradient diffusion weakly hampers first-order attacks.
**Key metric:** lam=0: FGSM_ASR=0.7835, PGD_ASR=0.9280, grad_ent=6.195; lam=1.0: FGSM_ASR=0.7140, PGD_ASR=0.8770, grad_ent=6.324
**Status:** PARTIALLY SUPPORTED (entropy penalty reduces ASR modestly at high lam, but effect is small relative to adversarial training)

---

## H338 - Double Backprop Variants
**Dataset:** Fashion-MNIST
**Conclusion:** All double-backpropagation variants (applying the input gradient norm penalty on all, correct-only, incorrect-only, or loss-weighted samples) fail to meaningfully reduce adversarial vulnerability compared to the baseline, with PGD_ASR remaining in 0.916–0.929 and FGSM_ASR in 0.759–0.775 across all conditions.
**Key metric:** baseline: FGSM_ASR=0.7665, PGD_ASR=0.9235; correct_only (best): FGSM_ASR=0.7590, PGD_ASR=0.9165
**Status:** NOT SUPPORTED (no double-backprop variant provides meaningful robustness gains)

---

## H339 - Curvature Regularisation (Input Hessian Trace)
**Dataset:** Fashion-MNIST
**Conclusion:** Regularising the input Hessian trace via Hutchinson estimator reduces estimated curvature (10.04 → 8.45 at lam=0.01) but does not translate to lower adversarial success rates; PGD_ASR and FGSM_ASR remain essentially unchanged across all lambda values, and clean accuracy drops slightly.
**Key metric:** lam=0: FGSM_ASR=0.7685, PGD_ASR=0.9165, curvature=10.04; lam=0.01: FGSM_ASR=0.7630, PGD_ASR=0.9170, curvature=8.45
**Status:** NOT SUPPORTED (reduced input curvature does not confer adversarial robustness)

---

## H340 - Gradient Penalty at Random Directions
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising gradient sensitivity along random perturbation directions (eps=0.05, K=3) provides small but consistent reductions in FGSM_ASR (0.754–0.758 vs 0.770 baseline) and PGD_ASR (0.904–0.909 vs 0.929 baseline) at lam=0.01 and 0.1, though the improvement is modest and not comparable to adversarial training.
**Key metric:** lam=0: FGSM_ASR=0.7700, PGD_ASR=0.9285; lam=0.1: FGSM_ASR=0.7575, PGD_ASR=0.9090
**Status:** PARTIALLY SUPPORTED (random-direction gradient penalty gives marginal but consistent ASR reductions)

---

## H341 - Manifold Gradient Penalty (PCA Projection)
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising the component of the input gradient lying on the data manifold (top-K PCA eigenvectors) does not reduce adversarial success rates; PGD_ASR stays in 0.915–0.929 for all K values and FGSM_ASR is unchanged, though clean accuracy is slightly improved at K=10.
**Key metric:** baseline: FGSM_ASR=0.7740, PGD_ASR=0.9255; K=10: FGSM_ASR=0.7690, PGD_ASR=0.9150, clean=0.8855
**Status:** NOT SUPPORTED (manifold gradient penalty does not improve robustness beyond negligible variance)

---

## H342 - Adversarial Training + Gradient Penalty Combined
**Dataset:** Fashion-MNIST
**Conclusion:** Combining FGSM adversarial training with gradient penalty (on clean or adversarial inputs) does not improve over FGSM-AT alone: all AT+penalty conditions achieve similar PGD_ASR (~0.328–0.360) to plain FGSM-AT (0.336), and gradient penalty alone provides no robustness benefit. The half-half schedule trades clean accuracy for a small ASR increase.
**Key metric:** fgsm_at: FGSM_ASR=0.3075, PGD_ASR=0.3355, clean=0.7985; fgsm_at+adv_pen: FGSM_ASR=0.3025, PGD_ASR=0.3350; grad_penalty_only: PGD_ASR=0.9200
**Status:** NOT SUPPORTED (gradient penalty adds no benefit to adversarial training; AT alone is sufficient)

---

## H343 - Gradient Penalty with Contrastive Loss
**Dataset:** Fashion-MNIST
**Conclusion:** Contrastive gradient regularisation (aligning same-class gradients, pushing different-class gradients apart) produces meaningful ASR reductions at lam=0.01: FGSM_ASR drops from 0.781 to 0.670 and PGD_ASR from 0.930 to 0.809, with only a small clean accuracy cost (0.872 vs 0.882), making it the most effective purely gradient-based regulariser in this series.
**Key metric:** lam=0: FGSM_ASR=0.7805, PGD_ASR=0.9295, clean=0.8820; lam=0.01: FGSM_ASR=0.6700, PGD_ASR=0.8085, clean=0.8720, margin=4.99
**Status:** PARTIALLY SUPPORTED (contrastive gradient alignment reduces ASR noticeably but not to AT-level robustness)

---

## H344 - Input Gradient Spectral Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising the estimated spectral norm of the input Jacobian (Lipschitz regularisation) yields substantial robustness gains: at lam=0.01 FGSM_ASR drops from 0.778 to 0.612 and PGD_ASR from 0.924 to 0.685 with near-unchanged clean accuracy (0.877), and spectral norm is reduced from 6.03 to 2.19, confirming that Lipschitz constraint directly limits adversarial exploitability.
**Key metric:** lam=0: FGSM_ASR=0.7780, PGD_ASR=0.9235, spectral_norm=6.031; lam=0.01: FGSM_ASR=0.6115, PGD_ASR=0.6845, spectral_norm=2.187, clean=0.8770
**Status:** SUPPORTED (spectral norm regularisation meaningfully reduces adversarial success rates while preserving clean accuracy)

---

## H345 - ELLE Local Linearity
**Dataset:** Fashion-MNIST
**Conclusion:** ELLE local linearity enforcement without adversarial training provides no robustness benefit: PGD ASR remains high (~0.888–0.918) across all lambda values, and higher lambda (1.0) actually increases FGSM ASR (0.778) while reducing margin (5.735). The penalty alone cannot substitute for adversarial training.
**Key metric:** lambda=0: PGD_ASR=0.916; lambda=1.0: PGD_ASR=0.888, FGSM_ASR=0.778 (worse than baseline)
**Status:** NOT SUPPORTED

---

## H346 - TRADES Objective
**Dataset:** Fashion-MNIST
**Conclusion:** TRADES achieves strong robustness at low beta (beta=1: PGD ASR=0.334, clean=0.850) but over-regularisation at beta≥6 degrades both robustness and clean accuracy, confirming the classic robustness-accuracy trade-off. The sweet spot is beta=1–3 for this dataset scale.
**Key metric:** beta=1: clean=0.850, PGD_ASR=0.334; beta=6: clean=0.735, PGD_ASR=0.401 (degraded)
**Status:** SUPPORTED

---

## H347 - Jensen-Shannon TRADES
**Dataset:** Fashion-MNIST
**Conclusion:** JSD-TRADES using symmetric Jensen-Shannon divergence instead of KL achieves comparable robustness to standard TRADES (beta=6: PGD ASR=0.326 vs TRADES beta=3: 0.323) while maintaining higher clean accuracy (0.840 vs 0.814), suggesting the bounded symmetric divergence avoids some KL pathologies.
**Key metric:** beta=6: clean=0.840, PGD_ASR=0.326 vs TRADES beta=3: clean=0.814, PGD_ASR=0.323
**Status:** PARTIALLY SUPPORTED

---

## H348 - Supervised Contrastive Adversarial Training
**Dataset:** Fashion-MNIST
**Conclusion:** Supervised contrastive loss completely fails on this classification task: both SupCon-clean and SupCon-adv achieve near-random clean accuracy (~0.075–0.077) with PGD ASR=1.000, indicating the contrastive objective produces representations incompatible with the downstream linear head in this setup. Only standard FGSM-AT works (PGD ASR=0.324).
**Key metric:** supcon_adv: clean=0.077, PGD_ASR=1.000; fgsm_at: clean=0.807, PGD_ASR=0.324
**Status:** NOT SUPPORTED

---

## H349 - GELU Activation + Input Gradient Norm Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Neither GELU activations nor input gradient norm penalty improve robustness; GELU+penalty actually worsens it (PGD ASR=0.936 vs ReLU baseline 0.917). The hypothesis that smooth activations enable effective gradient penalties is not supported — the penalty provides no robustness benefit in either activation regime.
**Key metric:** GELU+penalty: clean=0.883, PGD_ASR=0.936; ReLU+no_penalty: PGD_ASR=0.917
**Status:** NOT SUPPORTED

---

## H350 - Fisher-Rao Regularisation (FIRE)
**Dataset:** Fashion-MNIST
**Conclusion:** Fisher-Rao regularisation is harmful: low lambda (0.1) shows marginal PGD improvement (0.896 vs 0.911 baseline) but lambda≥1.0 dramatically increases vulnerability (PGD ASR=0.963–0.982) and collapses clean accuracy (0.595 at lambda=10). The geodesic penalty destabilises training rather than improving distributional robustness.
**Key metric:** lambda=0.1: PGD_ASR=0.896 (marginal gain); lambda=1.0: PGD_ASR=0.963 (worse); lambda=10: clean=0.595
**Status:** NOT SUPPORTED

---

## H351 - Adversarial Weight Perturbation (AWP)
**Dataset:** Fashion-MNIST
**Conclusion:** AWP on top of FGSM-AT provides modest but consistent improvements: gamma=0.001 reduces PGD ASR from 0.342 to 0.321 while maintaining clean accuracy. The weight perturbation flattens the loss landscape around adversarial examples, though gains are small at this training scale.
**Key metric:** gamma=0 (baseline): PGD_ASR=0.342; gamma=0.001: PGD_ASR=0.321 (best)
**Status:** PARTIALLY SUPPORTED

---

## H352 - Spectral Alignment Regularisation via FFT
**Dataset:** Fashion-MNIST
**Conclusion:** Spectral alignment regularisation penalising frequency-domain differences between clean and adversarial inputs provides no robustness improvement: PGD ASR remains 0.915–0.935 across all lambda values, essentially unchanged from the unregularised baseline (0.923). The frequency-domain invariance objective does not transfer to robustness.
**Key metric:** lambda=0: PGD_ASR=0.923; lambda=0.01: PGD_ASR=0.915; lambda=0.1: PGD_ASR=0.935 (no trend)
**Status:** NOT SUPPORTED

---

## H353 - Jacobian Spectral Norm Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising the Jacobian spectral norm (via power iteration) provides only marginal PGD improvement at lambda=0.01 (0.910 vs 0.922 baseline), with no clean accuracy cost. The Lipschitz constraint on the input-output map is theoretically sound but insufficient alone to meaningfully reduce adversarial vulnerability at this scale.
**Key metric:** lambda=0: PGD_ASR=0.922; lambda=0.01: PGD_ASR=0.910 (small improvement)
**Status:** PARTIALLY SUPPORTED

---

## H354 - Gradient-Guided CutMix
**Dataset:** Fashion-MNIST
**Conclusion:** Both random and gradient-guided CutMix reduce FGSM ASR (0.659–0.666 vs baseline 0.759) but fail to reduce PGD ASR (0.884–0.918 vs 0.910 baseline). Gradient-guided placement offers no advantage over random CutMix, and the additional gradient penalty version slightly worsens PGD robustness (0.918).
**Key metric:** grad_cutmix: FGSM_ASR=0.666, PGD_ASR=0.903; random_cutmix: FGSM_ASR=0.659, PGD_ASR=0.884
**Status:** NOT SUPPORTED

---

## H371 - Weight Displacement Trajectory
**Dataset:** Fashion-MNIST
**Conclusion:** Adversarially trained models (FGSM, PGD) converge to different weight-space regions than standard training: final displacement is ~1.3 units smaller (8.16–8.22 vs 9.46) with shorter total path length (~13.2 vs 16.3), and adversarial models are ~10 units apart from the standard model in weight space. PGD-AT achieves the best robustness (PGD ASR=0.196) with slightly higher mean cosine similarity (0.223) suggesting more consistent gradient directions.
**Key metric:** standard: final_disp=9.459, PGD_ASR=0.946; pgd-AT: final_disp=8.225, PGD_ASR=0.196; weight-space dist std↔pgd=10.47
**Status:** SUPPORTED

---

## H355 - OT + Jacobian Regularisation (OTJR)
**Dataset:** Fashion-MNIST
**Conclusion:** Combining sliced Wasserstein OT alignment with Jacobian Frobenius regularisation provides moderate robustness improvement over the unregularised baseline (PGD ASR drops from ~0.92 to 0.682–0.695), but the OT component offers no additional benefit over Jacobian regularisation alone — all three lambda_OT values produce nearly identical results.
**Key metric:** lam_ot=0: clean=0.880, FGSM_ASR=0.612, PGD_ASR=0.695; lam_ot=1.0: clean=0.873, FGSM_ASR=0.623, PGD_ASR=0.688
**Status:** PARTIALLY SUPPORTED

---

## H356 - Lipschitz-Proportional Stochastic Depth
**Dataset:** Fashion-MNIST
**Conclusion:** Lipschitz-proportional stochastic depth reduces PGD ASR from 0.918 (no dropout) to 0.795 at p_max=0.3, with clean accuracy dropping to 0.821. Fixed-probability stochastic depth achieves similar reductions, suggesting the benefit comes from stochastic depth in general rather than Lipschitz-proportional scheduling specifically.
**Key metric:** lipschitz_p0.3: clean=0.821, FGSM_ASR=0.685, PGD_ASR=0.795; no_drop: PGD_ASR=0.918
**Status:** PARTIALLY SUPPORTED

---

## H371 - Weight Displacement Trajectory (Standard vs Adversarial Training)
**Dataset:** Fashion-MNIST
**Conclusion:** Adversarial training (FGSM and PGD) produces significantly shorter weight displacement trajectories and smaller final parameter displacement compared to standard training, with FGSM and PGD achieving dramatically lower PGD ASR (0.217 and 0.196 respectively) versus standard training (0.946). The adversarially-trained models converge to a different region of weight space, ~10 L2 units away from the standard-trained solution.
**Key metric:** standard: clean=0.884, PGD_ASR=0.946, total_path=16.334; fgsm: clean=0.806, PGD_ASR=0.217, total_path=13.112; pgd: clean=0.798, PGD_ASR=0.196, total_path=13.237
**Status:** SUPPORTED

---

## H357 - Consistency Regularisation for Randomized Smoothing
**Dataset:** Fashion-MNIST
**Conclusion:** Consistency regularisation (forcing stable output distributions under Gaussian noise) provides modest adversarial robustness improvements at higher lambda values — PGD ASR drops from 0.915 (lambda=0) to 0.882 (lambda=10), while smoothed accuracy increases from 0.839 to 0.871. The benefit is incremental and the FGSM improvement is more pronounced than PGD, suggesting the noise-smoothing correlation exists but is limited at this scale.
**Key metric:** lambda=0: clean=0.877, smoothed=0.839, PGD_ASR=0.915; lambda=10.0: clean=0.877, smoothed=0.871, PGD_ASR=0.882
**Status:** PARTIALLY SUPPORTED

---

## H372 - Per-Layer Jacobian Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising the Frobenius norm of the Jacobian at every block (via Hutchinson random projections) significantly improves robustness: the all_layers condition reduces PGD ASR from 0.977 to 0.763, with the weighted variant achieving 0.817. The improvement comes at a cost to clean accuracy (0.767–0.817) and training time (~9x slower), but confirms that layer-wise Jacobian regularisation is substantially more effective than input-only gradient penalties.
**Key metric:** baseline: PGD_ASR=0.977; all_layers: clean=0.767, PGD_ASR=0.763; input_only: clean=0.863, PGD_ASR=0.868
**Status:** SUPPORTED

---

## H373 - Gradient Coherence Forcing
**Dataset:** Fashion-MNIST
**Conclusion:** Forcing gradient coherence by suppressing updates with low consecutive-batch cosine similarity (τ>0) reduces PGD ASR (best: 0.725 at τ=0.2) but at severe cost to clean accuracy (0.306–0.373 vs 0.878 baseline). The filtering effectively acts as aggressive regularisation, confirming that gradient coherence correlates with robustness but cannot be forced cheaply without destroying clean performance.
**Key metric:** tau=0.0: clean=0.878, PGD_ASR=0.920; tau=0.2: clean=0.373, PGD_ASR=0.725
**Status:** PARTIALLY SUPPORTED

---

## H374 - Layer-Specific Learning Rates (AT-Mimicking)
**Dataset:** Fashion-MNIST
**Conclusion:** Mimicking the per-layer learning-rate pattern of adversarial training (higher LR for early layers, lower for the head) provides no robustness benefit: the AT-mimic condition (PGD ASR=0.932) is virtually identical to uniform (0.924) and inverse (0.902) schedules. The layer-displacement pattern seen in AT cannot be replicated by merely rescaling learning rates.
**Key metric:** uniform: PGD_ASR=0.924; at_mimic: PGD_ASR=0.932; inverse: PGD_ASR=0.902
**Status:** NOT SUPPORTED

---

## H375 - Weight Displacement Budget
**Dataset:** Fashion-MNIST
**Conclusion:** Constraining total weight displacement to AT-like levels (7–9 L2 units) provides negligible robustness improvement: PGD ASR ranges from 0.922–0.932 across all budget values versus 0.930 unconstrained. Simply keeping weights close to initialisation does not reproduce the robustness of adversarial training.
**Key metric:** budget=9.0: PGD_ASR=0.922 (best, only 0.008 below unconstrained 0.930)
**Status:** NOT SUPPORTED

---

## H376 - Anti-SAM Sharpness Seeking
**Dataset:** Fashion-MNIST
**Conclusion:** Anti-SAM (perturbing weights toward sharper minima, opposite of SAM) unexpectedly improves adversarial robustness: at ρ=0.1 PGD ASR drops from 0.916 to 0.792 and FGSM ASR from 0.767 to 0.651, with only marginal clean accuracy loss (0.876). This counterintuitive result suggests deliberate sharpness injection creates loss-landscape geometry that is harder to exploit with gradient-based attacks.
**Key metric:** rho=0: PGD_ASR=0.916; rho=0.1: clean=0.876, PGD_ASR=0.792
**Status:** SUPPORTED

---

## H377 - Gradient Step Length Budget
**Dataset:** Fashion-MNIST
**Conclusion:** Constraining the total weight-space path length (cumulative step budget) to AT-like levels substantially improves robustness: budget=10.0 achieves PGD ASR=0.729 (vs 0.921 unconstrained) though clean accuracy drops to 0.841. The result confirms that AT's shorter optimisation trajectory is functionally significant — budgeted training forces more efficient use of each gradient step, producing more robust solutions.
**Key metric:** budget=10.0: clean=0.842, FGSM_ASR=0.601, PGD_ASR=0.729; unconstrained: PGD_ASR=0.921
**Status:** SUPPORTED

---

## H378 - Early-Layer Gradient Amplification
**Dataset:** Fashion-MNIST
**Conclusion:** Amplifying gradients for early layers to match the AT displacement pattern (AT-mimic: [2.0,1.0,0.5,0.3]) provides no robustness benefit: AT-mimic (PGD ASR=0.930) is essentially identical to uniform (0.928) and inverse (0.927) schedules. Gradient-hook rescaling cannot replicate the robustness advantages of AT's per-layer weight movement.
**Key metric:** uniform: PGD_ASR=0.928; at_mimic: PGD_ASR=0.930; inverse: PGD_ASR=0.927
**Status:** NOT SUPPORTED

---

## H379 - Trajectory Straightness Penalty
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising high-curvature weight-trajectory updates (suppressing steps that deviate from the previous direction) provides negligible robustness benefit: best PGD ASR=0.923 at λ=0.1 versus 0.927 baseline, with clean accuracy unchanged. Forcing straighter optimisation trajectories does not yield the robustness advantages associated with AT's more coherent gradient directions.
**Key metric:** lam=0: PGD_ASR=0.927; lam=0.1: PGD_ASR=0.923 (negligible improvement)
**Status:** NOT SUPPORTED

---

## H380 - Weight Norm Trajectory Control
**Dataset:** Fashion-MNIST
**Conclusion:** Penalising weight-norm growth rate beyond an AT-like allowed growth (1.0) provides no meaningful robustness benefit: the best condition (λ=1.0) reduces PGD ASR by only 0.011 (0.926→0.915) with clean accuracy essentially unchanged (0.880). Controlling weight-norm growth rate alone does not replicate the robustness of adversarial training.
**Key metric:** lam=0: clean=0.883, PGD_ASR=0.926; lam=1.0: clean=0.880, PGD_ASR=0.915 (Δ=−0.011)
**Status:** NOT SUPPORTED

---

## H381 - Feature Activation Statistics Matching
**Dataset:** Fashion-MNIST
**Conclusion:** Forcing a standard model to match AT-like per-block activation magnitudes (mean ||h_k||) via a penalty loss significantly reduces adversarial vulnerability at λ=0.1: PGD ASR drops from 0.909 to 0.719 (−19.0 pp) and FGSM ASR from 0.771 to 0.695, though clean accuracy falls to 0.808. The matched activation statistics closely reproduce the AT model's statistics ([73.09, 53.46, 26.06] vs AT [73.21, 53.34, 26.05]), confirming that activation magnitudes are a meaningful proxy for adversarial robustness.
**Key metric:** standard: clean=0.879, PGD_ASR=0.909; at_stats_matched (λ=0.1): clean=0.808, PGD_ASR=0.719; actual_AT: clean=0.771, PGD_ASR=0.324
**Status:** SUPPORTED

---

## H382 - Combined AT Geometry Proxy
**Dataset:** Fashion-MNIST
**Conclusion:** Combining gradient coherence forcing (τ=0.2), layer-specific learning rates, and displacement budget produces only 17.5% of AT's robustness gain while causing severe clean accuracy collapse (0.343 vs 0.879 baseline). Gradient coherence forcing dominates and crashes clean accuracy; layer LR alone provides no robustness (PGD ASR=0.940 vs baseline 0.925). The combined proxy approach fails as a viable AT substitute.
**Key metric:** baseline: clean=0.879, PGD_ASR=0.925; coherence_only: clean=0.373, PGD_ASR=0.725; all_combined: clean=0.343, PGD_ASR=0.819; actual_AT: clean=0.777, PGD_ASR=0.321
**Status:** NOT SUPPORTED

---

## H383 - Catch-All Unknown Class
**Dataset:** Fashion-MNIST
**Conclusion:** Adding an explicit class 10 ("unknown") as an adversarial escape route completely fails: abstention rates on adversarial examples are essentially zero (0.0–1.0%) across all unknown-sample conditions, no better than clean abstention (0.0%). The unknown class does not activate in response to adversarial perturbations, providing no robustness benefit, and PGD ASR remains 0.948–0.978 across conditions.
**Key metric:** baseline: PGD_ASR=0.956, abs_pgd=0.000; unknown_mixed_1x: PGD_ASR=0.961, abs_pgd=0.010
**Status:** NOT SUPPORTED

---

## H384 - Noise Copy Multiplicity
**Dataset:** Fashion-MNIST
**Conclusion:** Augmenting training with K noisy copies per clean image (σ=0.15) consistently reduces FGSM ASR (0.735→0.582, −15.3 pp at K=16) and modestly reduces PGD ASR (0.953→0.921, −3.2 pp), with clean accuracy unchanged (~0.880). Decision margin climbs from 6.3 to 14.2 as K increases. The benefit saturates by K=8→16 with marginal additional gains, and PGD improvement is much weaker than FGSM improvement, indicating noise augmentation hardens against weak gradient attacks but not strong iterative ones.
**Key metric:** K=0: clean=0.883, FGSM_ASR=0.735, PGD_ASR=0.953, margin=6.34; K=16: clean=0.874, FGSM_ASR=0.582, PGD_ASR=0.921, margin=14.23; ΔFGSM=−15.3pp, ΔPGD=−3.2pp
**Status:** SUPPORTED (saturates at K=8–16; PGD benefit modest)
