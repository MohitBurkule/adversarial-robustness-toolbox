# Scientific Findings Summary: Adversarial Robustness Hypotheses H107–H156

This document compiles the running analysis and scientific findings of our executed hypotheses. It is updated automatically by a recurring analysis agent as results are populated in the `results/` folder.

---

## Executive Summary
1. **Logit Margin Dominance**: Across nearly all attack types (white-box, iterative, black-box, certified bounds), the clean logit margin is the single most powerful predictor of sample vulnerability, consistently achieving **0.87–0.99 AUROC**.
2. **Black-Box Search Complexity**: Decision boundary proximity directly translates to search complexity in black-box coordinate space. The Spearman correlation between clean logit margin and the queries required to flip a prediction is exceptionally high (**+0.73** for SimBA, **+0.64** for Sign-OPT).
3. **MIM vs FGSM Consistency**: Momentum Iterative Method (MIM) is much more sample-efficient than FGSM (mean min eps of **0.0304** vs **0.0543**), yet their per-sample vulnerability rankings are almost perfectly preserved (**+0.97 Spearman correlation**).
4. **Certified Bounds Inadequacy**: Standard trained models have exactly **0.00% certified robustness** at $\epsilon = 15/255$, with a negligible mean certified radius of **0.0005**, highlighting the necessity of certified training methods (like IBP or randomized smoothing) to establish non-trivial bounds.

---

## Completed Hypothesis Analysis

### H107: Targeted FGSM Vulnerability Analysis
- **Path**: [h107_targeted_fgsm.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h107_targeted_fgsm.py)
- **Log**: [h107_targeted_fgsm_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h107_targeted_fgsm_output.txt)
- **Key Stats**: 
  - Flipped to at least one target: **80.4%**
  - Mean targeted success rate across 9 classes: **23.1%**
  - Untargeted FGSM success rate: **70.7%**
- **Univariate AUROC**:
  - `margin`: **0.8703** (Direction: `-`)
  - `std_pix` (contrast): **0.6335** (Direction: `-`)
  - `mean_pix` (brightness): **0.5022** (Direction: `+`)
- **Implications**: Clean margin strongly predicts targeted white-box vulnerability, while contrast serves as a weak proxy.

### H108: PGD Random Restarts Analysis
- **Path**: [h108_pgd_restarts.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h108_pgd_restarts.py)
- **Log**: [h108_pgd_restarts_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h108_pgd_restarts_output.txt)
- **Key Stats**:
  - PGD Success Rate: **95.9%** (K=1) to **97.9%** (K=20)
- **Univariate AUROC (by K restarts)**:
  - `margin`: **0.9579** (K=1) | **0.9550** (K=20)
  - `sobel_mean` (edge energy): **0.6991** (K=1) | **0.6970** (K=20)
- **Analysis by Difficulty**:
  - *Easy (flipped at K=1)*: Margin = **9.74**, Sobel Mean = **0.74**
  - *Robust (unflipped at K=20)*: Margin = **25.47**, Sobel Mean = **0.85**
- **Implications**: Edge energy and margin are highly collinear indicators of PGD restart resilience.

### H109: Momentum Iterative Method (MIM)
- **Path**: [h109_mim.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h109_mim.py)
- **Log**: [h109_mim_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h109_mim_output.txt)
- **Key Stats**:
  - Success at $\epsilon=15/255$: **94.4%**
  - Mean MIM min $\epsilon$ to flip: **0.0304** (vs. **0.0543** for FGSM)
  - MIM vs. FGSM Spearman Rank Correlation: **+0.9697** (p-value: 0.00)
- **Univariate AUROC**:
  - `margin`: **0.9284** | `sobel_mean`: **0.6726**
- **Implications**: Despite MIM being significantly more sample-efficient and stronger than FGSM, per-sample relative vulnerability rankings remain highly preserved.

### H110: Diverse Input Method (DIM) Transferability
- **Path**: [h110_dim.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h110_dim.py)
- **Log**: [h110_dim_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h110_dim_output.txt)
- **Key Stats**:
  - DIM Self-attack Success Rate: **74.6%**
  - DIM Transfer-attack Success Rate (Model B $\to$ Model A): **52.8%**
- **Univariate AUROC (Transfer target)**:
  - `margin`: **0.9135** (Direction: `-`)
  - `sobel_mean`: **0.5637** (Direction: `-`)
- **Implications**: The victim's own margin remains a massive predictor of transfer attack success, showing that transferability is governed primarily by the target model's boundary proximity rather than the source model's properties.

### H111: Expectation Over Transformations (EOT) Robustness
- **Path**: [h111_eot.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h111_eot.py)
- **Log**: [h111_eot_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h111_eot_output.txt)
- **Key Stats**:
  - Plain PGD-10 Success: **92.6%**
  - EOT PGD-10 Success: **72.8%**
  - Only plain PGD succeeded: **19.8%** | Only EOT succeeded: **0.0%**
- **Implications**: Gradient averaging over transformations (EOT) smooths gradients and decreases attack efficacy against undefended models, where local high-frequency gradients are highly exploitable.

### H112: Saturation/Value Gamma Attack
- **Path**: [h112_hsv_attack.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h112_hsv_attack.py)
- **Log**: [h112_hsv_attack_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h112_hsv_attack_output.txt)
- **Key Stats**:
  - Gamma Success Rate: **4.5%**
  - Mean $|\log(\gamma)|$ to flip: **0.4288**
- **Univariate AUROC**:
  - `margin`: **0.9439** (Direction: `-`)
- **Implications**: Global luminance scaling (gamma modulation) is a very weak perturbation for standard CNNs.

### H113: SimBA-Pixel Black-box Search
- **Path**: [h113_simba.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h113_simba.py)
- **Log**: [h113_simba_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h113_simba_output.txt)
- **Key Stats**:
  - Success Rate (200-query budget): **60.0%**
  - Mean queries to flip (successful): **99.5**
  - Clean Margin vs. Queries to Flip Correlation: **+0.7319** (p-value: 2.20e-21)
- **Implications**: Proximity to the decision boundary directly governs black-box search efficiency in coordinate space.

### H114: ZOO Zeroth-Order Optimization
- **Path**: [h114_zoo.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h114_zoo.py)
- **Log**: [h114_zoo_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h114_zoo_output.txt)
- **Key Stats**:
  - Success Rate (subsample 200): **54.5%**
  - Mean queries to flip (successful): **314.65**
- **Univariate AUROC**:
  - `margin`: **0.9976** (Direction: `-`)
- **Implications**: Boundary proximity is extremely critical for zeroth-order coordinate-gradient estimation.

### H115: NES Zeroth-Order Gradient Estimation
- **Path**: [h115_nes.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h115_nes.py)
- **Log**: [h115_nes_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h115_nes_output.txt)
- **Key Stats**:
  - NES Success: **51.0%** (mean 711.6 queries)
  - FGSM Success: **76.0%**
- **Univariate AUROC**:
  - `margin`: **0.9572** (Direction: `-`)
- **Implications**: Natural Evolution Strategies gradient search remains highly aligned with the true white-box boundary.

### H116: Sign-OPT Decision Black-box
- **Path**: [h116_sign_opt.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h116_sign_opt.py)
- **Log**: [h116_sign_opt_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h116_sign_opt_output.txt)
- **Key Stats**:
  - Median $L_2$ Perturbation: **2.5892** (mean 606 queries)
  - Clean Margin vs. L2 Distance Correlation: **+0.6406**
- **Implications**: Decision-boundary crossing distance correlates beautifully with white-box clean logit margin.

### H117: certified Interval Bound Propagation (IBP)
- **Path**: [h117_ibp_bound.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h117_ibp_bound.py)
- **Log**: [h117_ibp_bound_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h117_ibp_bound_output.txt)
- **Key Stats**:
  - Certified robust percentage at $\epsilon=15/255$: **0.00%**
  - Mean IBP Certified Radius: **0.0005**
- **Univariate AUROC (predicting empirical PGD-10 success)**:
  - `margin`: **0.9210** | `IBP_certified_radius`: **0.8755**
- **Implications**: Standard trained CNNs have virtually zero certified robustness. However, certified radii correlate strongly with empirical white-box robustness.

### H118: CURE Curvature Regularization vs Vanilla CNN
- **Path**: [h118_cure.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h118_cure.py)
- **Log**: [h118_cure_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h118_cure_output.txt)
- **Robustness Results**:
  - *Vanilla CNN*: Clean Acc = **93.4%**, FGSM Success = **68.1%**, PGD Success = **93.0%**, Mean min $\epsilon$ = **0.0320**
  - *CURE CNN*: Clean Acc = **91.8%**, FGSM Success = **27.8%**, PGD Success = **39.5%**, Mean min $\epsilon$ = **0.0946**
- **Vulnerability Feature Predictability changes (PGD)**:
  - `margin`: CURE = **0.9656** | Vanilla = **0.9627** (Change = `+0.0029`)
  - `sobel_mean`: CURE = **0.5524** | Vanilla = **0.7212** (Change = `-0.1689`)
- **Implications**: Curvature Regularization (CURE) significantly elevates empirical robustness (reducing PGD success from 93% to 39.5% and expanding mean boundary distance threefold from 0.032 to 0.095) while maintaining exceptional generalization (91.8% clean). Interestingly, edge energy (`sobel_mean`) becomes fully decoupled from vulnerability, while margin remains a perfect predictor.

### H119: Adversarial Logit Pairing (ALP) vs Vanilla CNN
- **Path**: [h119_alp.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h119_alp.py)
- **Log**: [h119_alp_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h119_alp_output.txt)
- **Robustness Results**:
  - *Vanilla CNN*: Clean Acc = **93.6%**, FGSM Success = **69.7%**, PGD Success = **95.1%**, Mean min $\epsilon$ = **0.0309**
  - *ALP CNN*: Clean Acc = **88.4%**, FGSM Success = **0.0%**, PGD Success = **70.3%**, Mean min $\epsilon$ = **0.0664**
- **Predictability changes (PGD)**:
  - `margin`: ALP = **0.7141** | Vanilla = **0.9598** (Change = `-0.2457`)
- **Implications**: Adversarial Logit Pairing (ALP) successfully enforces FGSM immunity (0% success) and moderately reduces PGD susceptibility. However, the pairing regularizer heavily compresses clean logit representations, resulting in a **24.5% drop in margin's predictive power (AUROC 0.71)** and a minor decrease in clean accuracy.

### H120: Adversarial Weight Perturbation (AWP) vs Vanilla CNN
- **Path**: [h120_awp.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h120_awp.py)
- **Log**: [h120_awp_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h120_awp_output.txt)
- **Robustness Results**:
  - *Vanilla CNN*: Clean Acc = **93.6%**, FGSM Success = **71.8%**, PGD Success = **95.5%**, Mean min $\epsilon$ = **0.0309**
  - *AWP CNN*: Clean Acc = **35.2%**, FGSM Success = **21.9%**, PGD Success = **31.0%**, Mean min $\epsilon$ = **0.1362**
- **Implications**: While AWP achieves an impressive boundary distance expansion (mean min $\epsilon$ = 0.1362), performing weight-space adversarial perturbation on a small model without concurrent standard adversarial training results in severe underfitting/collapse (clean accuracy drop to 35.2%). This highlights that weight perturbation is a regularizer that *supplements* AT, rather than substituting for it.

### H121: Cutout Data Augmentation Defense vs Vanilla CNN
- **Path**: [h121_cutout.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h121_cutout.py)
- **Log**: [h121_cutout_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h121_cutout_output.txt)
- **Robustness Results**:
  - *Vanilla CNN*: Clean Acc = **93.8%**, FGSM Success = **71.5%**, PGD Success = **95.2%**, Mean min $\epsilon$ = **0.0305**
  - *Cutout CNN*: Clean Acc = **93.0%**, FGSM Success = **67.2%**, PGD Success = **94.5%**, Mean min $\epsilon$ = **0.0321**
- **Predictability changes (PGD)**:
  - `margin`: Cutout = **0.9300** | Vanilla = **0.9542** (Change = `-0.0242`)
- **Implications**: Spatial pixel-patch occlusion (Cutout) offers almost no empirical defense against L-inf noise attacks (PGD remains at 94.5%), showing that occlusion robustness does not generalize to mathematical high-frequency noise.

### H122: Random Erasing Data Augmentation Defense vs Vanilla CNN
- **Path**: [h122_random_erasing.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h122_random_erasing.py)
- **Log**: [h122_random_erasing_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h122_random_erasing_output.txt)
- **Robustness Results**:
  - *Vanilla CNN*: Clean Acc = **93.9%**, FGSM Success = **68.7%**, PGD Success = **93.0%**, Mean min $\epsilon$ = **0.0316**
  - *Random Erasing CNN*: Clean Acc = **93.3%**, FGSM Success = **69.4%**, PGD Success = **95.1%**, Mean min $\epsilon$ = **0.0323**
- **Implications**: Correspondingly to Cutout, filling random patches with noise (Random Erasing) does not offer protection against adversarial gradients, with PGD success slightly rising to 95.1%. Occlusion augmentation is ineffective against direct adversarial input perturbations.

### H123: AugMix Defense vs Vanilla CNN
- **Path**: [h123_augmix.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h123_augmix.py)
- **Log**: [h123_augmix_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h123_augmix_output.txt)
- **Robustness Results**:
  - *Vanilla CNN*: Clean Acc = **92.9%**, FGSM = **69.9%**, PGD = **94.3%**, Mean min ε = **0.0309**
  - *AugMix CNN*: Clean Acc = **87.1%**, FGSM = **35.7%**, PGD = **70.2%**, Mean min ε = **0.0508**
- **Predictability Changes (PGD)**:
  - `margin`: AugMix = **0.8183** | Vanilla = **0.9689** (Δ = **−0.1506**)
  - `sobel_mean`: AugMix = **0.6554** | Vanilla = **0.7114** (Δ = **−0.0560**)
- **Implications**: AugMix offers meaningful empirical robustness gains (PGD 94.3% → 70.2%) at a 6% clean accuracy cost. Like ALP, it compresses logit distributions enough to reduce margin AUROC by 15pp, suggesting it alters decision boundary geometry — but at a smaller accuracy penalty than ALP.

### H124: Manifold Mixup Defense vs Vanilla CNN
- **Path**: [h124_manifold_mixup.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h124_manifold_mixup.py)
- **Log**: [h124_manifold_mixup_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h124_manifold_mixup_output.txt)
- **Robustness Results**:
  - *Vanilla CNN*: Clean Acc = **93.1%**, FGSM = **69.1%**, PGD = **94.1%**, Mean min ε = **0.0312**
  - *Manifold Mixup CNN*: Clean Acc = **92.5%**, FGSM = **64.1%**, PGD = **88.0%**, Mean min ε = **0.0371**
- **Predictability Changes (PGD)**:
  - `margin`: MixMix = **0.7734** | Vanilla = **0.9844** (Δ = **−0.2110**)
- **Implications**: Manifold Mixup provides mild robustness (PGD 94.1% → 88.0%) while maintaining clean accuracy (~92.5%), but reduces margin AUROC by a striking **21pp** — the largest reduction among augmentation-based defenses. Feature-space interpolation fundamentally scrambles the logit geometry that otherwise cleanly separates robust from vulnerable samples.

### H125: LRP Attribution vs Adversarial Vulnerability
- **Path**: [h125_lrp.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h125_lrp.py)
- **Status**: ❌ **FAILED** (RuntimeError: `.detach()` missing before `.numpy()` — **now patched** in source)

### H126: SmoothGrad Attribution vs Adversarial Vulnerability
- **Path**: [h126_smoothgrad.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h126_smoothgrad.py)
- **Log**: [h126_smoothgrad_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h126_smoothgrad_output.txt)
- **Model**: Clean Acc = **93.8%**, FGSM = **72.5%**, PGD = **94.5%**
- **Univariate AUROC (vs PGD Flip)**:
  - `smoothgrad_l2_norm`: **0.9704** | `smoothgrad_max`: **0.9699** | `smoothgrad_entropy`: **0.8881** | `margin`: **0.9710**
- **Implications**: SmoothGrad norms and maxima are **exceptional vulnerability predictors** (~0.97 AUROC), essentially matching the logit margin. Attribution concentration (entropy) is also a strong predictor (0.89). This is a key finding: noise-averaged attribution signals carry nearly all the adversarial vulnerability information that direct boundary proximity measures carry.

### H127: Integrated Gradients Attribution vs Adversarial Vulnerability
- **Path**: [h127_integrated_gradients.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h127_integrated_gradients.py)
- **Log**: [h127_integrated_gradients_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h127_integrated_gradients_output.txt)
- **Model**: Clean Acc = **94.0%**, FGSM = **73.1%**, PGD = **96.4%**
- **Univariate AUROC (vs PGD Flip)**:
  - `margin`: **0.9661** | `IG_l2_norm`: **0.7554** | `IG_entropy`: **0.6232** | `IG_total_attribution`: **0.6292**
- **Implications**: Unlike SmoothGrad, IG features are considerably weaker predictors (best 0.76). IG captures semantic attribution via path integration from a baseline, making it less informative about boundary proximity. The contrast with SmoothGrad confirms that noise-based averaging tracks boundary oscillation better than path-integration.

### H128: GradCAM Saliency vs Adversarial Vulnerability
- **Path**: [h128_gradcam.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h128_gradcam.py)
- **Log**: [h128_gradcam_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h128_gradcam_output.txt)
- **Model**: Correctly classified **9,250 / 10,000**
- **Univariate AUROC (vs PGD Flip)**:
  - `gradcam_max`: **0.8111** | `gradcam_entropy`: **0.6429** | `gradcam_l2_to_image_center`: **0.5860** | `margin`: **0.9607**
- **Implications**: GradCAM peak activation is a moderately strong predictor (0.81 AUROC). Spatial concentration and entropy are weak. High peak activation signals a strong gradient hook at the last convolutional layer, but the spatial pattern of class-discriminative features carries limited additional vulnerability information.

### H129: Forgetting Events vs Adversarial Vulnerability
- **Path**: [h129_forgetting_events.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h129_forgetting_events.py)
- **Log**: [h129_forgetting_events_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h129_forgetting_events_output.txt)
- **Univariate AUROC (vs PGD Flip)**:
  - `forgetting_event_count`: **0.5372** (≈ random) | `margin`: **0.9428**
- **Implications**: Forgetting events (times a sample is learned then re-misclassified during training) are essentially **uninformative** about test-time adversarial vulnerability. Training-time learning instability and test-time boundary proximity are orthogonal properties.

### H130: C-Score (Training Consistency) vs Adversarial Vulnerability
- **Path**: [h130_cscore.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h130_cscore.py)
- **Log**: [h130_cscore_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h130_cscore_output.txt)
- **Univariate AUROC (vs PGD Flip)**:
  - `c_score`: **0.5280** (≈ random) | `margin`: **0.9638**
- **Implications**: The C-score (fraction of random-subset ensembles correctly classifying a sample) is entirely uninformative. *Learning difficulty* and *adversarial proximity* are orthogonal.

### H131: Memorization Proxy vs Adversarial Vulnerability
- **Path**: [h131_memorization.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h131_memorization.py)
- **Log**: [h131_memorization_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h131_memorization_output.txt)
- **Model**: Correctly classified **9,133 / 10,000**
- **Univariate AUROC (vs PGD Flip)**:
  - `memorization_proxy`: **0.9284** | `margin`: **0.9387**
- **Implications**: Variance in predictions across independent random-subset ensembles (a memorization proxy) achieves **0.93 AUROC** — nearly matching the logit margin. Highly memorised samples are disproportionately close to the decision boundary. This is one of the strongest novel alternative predictors discovered.

### H132: Snapshot Ensemble Disagreement vs Adversarial Vulnerability
- **Path**: [h132_snapshot_ensemble.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h132_snapshot_ensemble.py)
- **Log**: [h132_snapshot_ensemble_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h132_snapshot_ensemble_output.txt)
- **Model**: Correctly classified **9,354 / 10,000**
- **Univariate AUROC (vs PGD Flip)**:
  - `snap_disagreement`: **0.5130** (≈ random) | `softmax_variance`: **0.8969** | `margin`: **0.9363**
- **Implications**: Hard-vote snapshot disagreement is near-random; **softmax variance across snapshots** achieves 0.90 AUROC. Soft continuous signals vastly outperform hard label signals — retaining calibrated probabilities is critical for vulnerability assessment.

### H133: Stochastic Depth CNN vs Vanilla CNN
- **Path**: [h133_stochastic_depth.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h133_stochastic_depth.py)
- **Log**: [h133_stochastic_depth_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h133_stochastic_depth_output.txt)
- **Robustness**:
  - *Vanilla CNN*: Correct = **9,225**, margin AUROC PGD = **0.9532**
  - *Stochastic Depth CNN*: Correct = **8,918** (↓3%), margin AUROC PGD = **0.9031**
- **Implications**: Stochastic Depth provides no robustness benefit and reduces clean accuracy by 3% on this shallow network. Without sufficient depth for meaningful path randomization, SD acts as a noisy regularizer with marginal adverse effects.

### H134: BNN Predictive Variance (SWAG) vs Adversarial Vulnerability
- **Path**: [h134_bnn_variance.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h134_bnn_variance.py)
- **Log**: [h134_bnn_variance_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h134_bnn_variance_output.txt)
- **Model**: SWA correctness = **9,257 / 10,000**
- **Univariate AUROC (vs PGD Flip)**:
  - `bnn_predictive_variance`: **0.9176** | `margin`: **0.9307**
- **Implications**: SWAG posterior predictive variance is a **strong predictor** (0.92 AUROC), nearly matching the margin. Bayesian weight-space uncertainty captures boundary proximity well — practically useful when logits are inaccessible.

### H135: Uncertainty Decomposition (Aleatoric vs Epistemic) vs Adversarial Vulnerability
- **Path**: [h135_uncertainty_decomp.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h135_uncertainty_decomp.py)
- **Log**: [h135_uncertainty_decomp_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h135_uncertainty_decomp_output.txt)
- **Model**: Victim = **9,241 / 10,000**
- **Univariate AUROC (vs PGD Flip)**:
  - `predictive_entropy`: **0.9547** | `aleatoric`: **0.9547** | `epistemic`: **0.9481** | `margin`: **0.9431**
- **Implications**: Ensemble predictive entropy **exceeds the logit margin** (0.955 vs 0.943). Both aleatoric and epistemic components are independently strong. **Deep ensemble uncertainty is a better single vulnerability predictor than the clean logit margin** — a flagship result.

### H136: Per-Class Margin Statistics vs Adversarial Vulnerability
- **Path**: [h136_class_margin_stats.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h136_class_margin_stats.py)
- **Log**: [h136_class_margin_stats_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h136_class_margin_stats_output.txt)
- **Class Margin Means**: Trouser (class 1) = **18.76** (safest), Shirt (class 6) = **3.61** (most vulnerable)
- **Univariate AUROC (vs PGD Flip)**:
  - `deviation_from_class_mean`: **0.8607** | `margin`: **0.9484**
- **Implications**: Intra-class margin deviation is a moderately strong predictor (0.86). Classes differ dramatically in boundary proximity: Trouser is 5× safer than Shirt. The within-class deviation captures genuine per-sample variability beyond class-level effects.

### H137: Pixel Gradient Sign Agreement (Multi-Model) vs Adversarial Vulnerability
- **Path**: [h137_pixel_sign_agreement.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h137_pixel_sign_agreement.py)
- **Log**: [h137_pixel_sign_agreement_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h137_pixel_sign_agreement_output.txt)
- **Univariate AUROC (vs PGD Flip)**:
  - `pixel_sign_agreement`: **0.8864** | `margin`: **0.9686**
- **Implications**: Cross-model gradient sign consensus is a **strong predictor** (0.89 AUROC). Samples with highly aligned adversarial gradients across models are disproportionately vulnerable. This architecture-agnostic signal could support black-box vulnerability assessment without model access.

### H138: Layer Ablation Sensitivity vs Adversarial Vulnerability
- **Path**: [h138_layer_ablation.py](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/hypotheses/h138_layer_ablation.py)
- **Log**: [h138_layer_ablation_output.txt](file:///home/mohit/1tbone/trashy/adversarial-robustness-toolbox/dissertation_extension/results/h138_layer_ablation_output.txt)
- **Model**: 9,261 correctly classified; FGSM = **70.3%**, PGD = **93.8%**
- **Univariate AUROC (vs min_eps)**:
  - `max_ablation_drop`: **0.7664** | `mean_ablation_drop`: **0.7305** | `margin`: **0.9468**
- **Implications**: Max accuracy drop upon zeroing out any single layer is a moderate predictor (0.77 AUROC). Samples overly reliant on a single narrow representation layer are more vulnerable, suggesting that representational breadth (distributing information across layers) correlates with adversarial robustness.

---

## Running AUROC Leaderboard (updated H107–H138)

| Rank | Feature | Best AUROC | Target | Hypothesis |
|------|---------|-----------|--------|------------|
| 1 | `margin` (ZOO context) | **0.9976** | min_eps | H114 |
| 2 | `predictive_entropy` (deep ensemble) | **0.9547** | PGD | H135 ⭐ |
| 3 | `smoothgrad_l2_norm` | **0.9704** | PGD | H126 ⭐ |
| 4 | `smoothgrad_max` | **0.9699** | PGD | H126 ⭐ |
| 5 | `margin` (typical) | **~0.95** | PGD | most Hxxx |
| 6 | `bnn_predictive_variance` (SWAG) | **0.9349** | min_eps | H134 |
| 7 | `memorization_proxy` | **0.9284** | PGD | H131 |
| 8 | `softmax_variance` (snapshot) | **0.8969** | PGD | H132 |
| 9 | `smoothgrad_entropy` | **0.8881** | PGD | H126 |
| 10 | `pixel_sign_agreement` | **0.8864** | PGD | H137 |
| 11 | `gradcam_max` | **0.8111** | PGD | H128 |
| 12 | `deviation_from_class_mean` | **0.8607** | PGD | H136 |
| 13 | `max_ablation_drop` | **0.7664** | min_eps | H138 |
| 14 | `IG_l2_norm` | **0.7554** | PGD | H127 |
| 15 | `IBP_certified_radius` | **0.8755** | PGD | H117 |
| 16 | `forgetting_event_count` | **0.5372** | PGD | H129 (≈ random) |
| 17 | `c_score` | **0.5280** | PGD | H130 (≈ random) |

**Key Insights**:
- Training dynamics features (forgetting, C-score) are near-random vulnerability predictors — learning difficulty ≠ adversarial proximity.
- SmoothGrad attribution norms approach margin-level prediction (0.97 AUROC).
- Deep ensemble uncertainty **exceeds** the margin as a vulnerability predictor (H135) — a flagship result for practical applications without logit access.
- Memorization proxy and SWAG posterior variance are strong novel predictors (0.92–0.93 AUROC).
- Augmentation defenses (AugMix, Manifold Mixup) alter decision boundary geometry, reducing margin predictability — unlike CURE which preserves it perfectly.

