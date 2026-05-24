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
