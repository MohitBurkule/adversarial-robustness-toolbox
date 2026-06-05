# New Hypotheses H215–H234
## From User Research Ideas — 2026-06-05

Research source: adversarial ML literature 2018-2025 + original user hypotheses.
Each entry: ID, title, core question, prior art, novelty, proposed experiment.

---

## H215 — L-Ball Region Training (Epsilon-Ball Loss Integration)

**Core question:** Can we train by optimising the loss over the *entire* ε-ball around each sample
(not just the worst-case PGD point), and does this produce better-calibrated per-sample margins?

**Prior art:**
- IBP/CROWN-IBP (Gowal et al. 2018, arXiv 1906.06316): propagate interval bounds through network; certifies ε-ball
- Randomized smoothing (Cohen et al. 2019, arXiv 1902.02918): majority vote over Gaussian samples = integration over a ball
- TRADES (Zhang et al. ICML 2019): decomposes clean + boundary loss; implicit ball smoothing
- SOTA: 67% certified CIFAR-10 accuracy at ε=8/255 via CROWN-IBP (ICLR 2020)

**Novelty:** Compare ball-training methods head-to-head on Fashion-MNIST:
  1. PGD-AT (worst-case point)
  2. Randomized smoothing training (expectation over K=20 Gaussian draws)
  3. IBP-lite (propagate linear bounds through first conv layer only)
  Measure: per-sample certified radius vs logit margin; does ball-training produce tighter margin–radius correlation?

**Experiment:**
- Train 3 models: PGD-AT, smooth-AT (add σ=0.1 Gaussian noise to all training inputs K times), IBP-1layer
- Measure: certified radius (randomized smoothing), logit margin, FGSM/PGD ASR
- Key metric: Spearman ρ(margin, certified_radius) — does ball-training tighten this correlation?

---

## H216 — Augmentation Diversity vs Adversarial Robustness

**Core question:** Which random augmentation strategy gives best robustness against a *random* adversarial attack (FGSM with random direction, not gradient-guided)?

**Prior art:**
- AugMix (Hendrycks et al. 2020, arXiv 1912.02781): JSD consistency loss over augmentation mixtures
- AugMax (Wang et al. NeurIPS 2021, arXiv 2110.13771): adversarial augmentation composition; CIFAR-10 ~80% PGD accuracy
- RandAugment (Cubuk et al. ICCV 2020): random magnitude + policy
- PixMix (Hendrycks et al. 2022): dreamlike image mixing for corruption robustness
- Conclusion: augmentation alone < AT; AugMax closes the gap most

**Novelty:** Test augmentation strategies specifically against a *random* (non-gradient) attack:
random_fgsm = x + ε * random_unit_vector (no gradient, just random direction).
This isolates whether augmentation diversity increases the fraction of "robust" directions.

**Experiment:**
- Train 5 models on Fashion-MNIST: baseline, +RandomHFlip+Crop, +AugMix-lite, +Cutout, +RandAugment
- Attack with: FGSM (gradient), random-FGSM (no gradient), PGD-10
- Compare: does diversity augmentation specifically help against random attacks vs gradient attacks?
- Key metric: ASR(random-FGSM) vs ASR(FGSM) ratio — does augmentation flatten the adversarial landscape?

---

## H217 — Model Width/Depth vs Adversarial Vulnerability

**Core question:** Does increasing model capacity (width or depth) reduce adversarial vulnerability, and is width more effective than depth?

**Prior art:**
- Madry et al. ICLR 2018: wider models more robust under AT
- Gowal et al. 2021: WideResNet-70-16 achieves 63.58% AutoAttack CIFAR-10
- Bartoldson et al. ICML 2024 (arXiv 2404.09349): scaling laws for robustness; WR-70-16 suboptimal compute
- ~13.4% reduction in attack success per 10x model size (empirical)

**Novelty:** Ablation on Fashion-MNIST with controlled parameter count:
  - Fix total params ≈ 50K, 200K, 800K
  - Compare wide-shallow vs narrow-deep architectures at each count
  - Measure: margin distribution, PGD-10 ASR, AUROC(-margin → PGD)

**Experiment:**
- 3 widths × 3 depths = 9 architectures on Fashion-MNIST
- Standard training + PGD-AT training (2×9 = 18 models)
- Key metric: is the margin AUROC tautology scale-invariant? Does bigger model = higher AUROC?

---

## H218 — Robust Overfitting: When Does More Training Hurt?

**Core question:** Does robust overfitting occur on Fashion-MNIST under PGD-AT, and can per-sample margin trajectories predict which samples will overfit?

**Prior art:**
- Rice et al. ICML 2020: robust overfitting — PGD-AT accuracy peaks ~epoch 70-100, then declines on CIFAR-10
- Humayun et al. arXiv 2402.15555: grokking dynamics in adversarial training
- Early stopping is optimal mitigation

**Novelty:** Per-sample robust overfitting analysis:
- Track per-sample margin at epochs 1, 5, 10, 15, 20, 30, 50
- Identify which samples' margins *decrease* after peak (overfitting victims)
- Test: do low-margin samples at epoch 1 predict overfitting victims at epoch 50?
- Does the margin-AUROC tautology get stronger or weaker as overfitting progresses?

**Experiment:**
- Train PGD-AT model, save checkpoints every 5 epochs
- Compute per-sample margin at each checkpoint
- Plot: margin trajectory for Q1/Q4 quartiles; overfitting rate per quartile
- Key finding: does per-sample margin predict robust overfitting susceptibility?

---

## H219 — DINOv2 / Self-Supervised ViT Adversarial Robustness

**Core question:** Is a DINOv2-pretrained model harder to attack than a supervised ResNet of similar accuracy?

**Prior art:**
- Mahmood et al. ICCV 2021: ViTs show different transfer properties than CNNs under attacks
- arXiv 2206.06761: SSL ViTs attacked with DINO-specific blended gradient attack
- DINOv2 (arXiv 2304.07193): 1B param SSL; more generalizable features
- Finding: SSL confers robustness advantage; erodes after fine-tuning

**Novelty:** Test on Fashion-MNIST with a small DINO-like SSL pretraining:
- Pretrain small ViT via self-distillation (teacher-student) on Fashion-MNIST train set
- Linear probe for classification
- Compare FGSM/PGD ASR vs supervised CNN of same accuracy
- Key question: do SSL features lie on a "smoother" manifold, making gradient-based attacks less effective?

**Experiment (feasible on FMNIST):**
- Train: small ViT SSL (DINO-style, 5 epochs) → linear probe
- Baseline: CNN trained supervised to same clean accuracy
- Attack both with FGSM/PGD-10 at ε=0.1
- Measure: ASR, margin distribution, score direction alignment (from H214)

---

## H220 — Part-Based / Shape-Biased Models vs Adversarial Attacks

**Core question:** Does enforcing shape/part bias in a CNN (via Stylized-ImageNet-like training or patch masking) improve adversarial robustness on Fashion-MNIST?

**Prior art:**
- Geirhos et al. ICLR 2019 (arXiv 1811.12231): texture-biased CNNs → shape-biased via Stylized-ImageNet; shape bias improves corruption robustness
- Sitawarin et al. NeurIPS 2022 (arXiv 2209.09117): part-based models achieve 15% higher accuracy at same robustness
- Key finding: shape bias ≠ adversarial robustness; helps corruptions only

**Novelty:** On Fashion-MNIST, enforce part-awareness via:
1. Random patch masking during training (forces global reasoning)
2. Saliency-guided dropout (mask top-k% salient pixels during training)
3. Spatial dropout on conv feature maps

Compare: does each strategy reduce adversarial ASR? Does it improve or hurt margin distribution?

**Experiment:**
- 4 models: baseline, random-patch-mask (30% pixels), saliency-mask, spatial-dropout
- Measure: ASR(FGSM/PGD), clean accuracy, margin AUROC
- Key test: does reducing texture reliance improve robustness?

---

## H221 — Video vs Image Adversarial Attack Difficulty

**Core question:** Is it harder to adversarially attack a model trained on short temporal sequences (3-frame clips) than a model trained on individual images?

**Prior art:**
- Jiang et al. ACM MM 2019 (arXiv 1904.05181): black-box video attacks need 100x queries vs images
- Wei et al. IJCV 2022: sparse attacks on video — key frames exploit temporal redundancy
- Temporal consistency constraints make white-box attacks more constrained
- Attack difficulty scales with temporal consistency requirement

**Novelty (feasible experiment):** Simulate "video" on Fashion-MNIST:
- Create 3-frame "clips": [x + Gaussian_noise_t1, x, x + Gaussian_noise_t3]  (static + noise)
- Train temporal CNN (3-frame input, Conv3D or frame-averaging)
- Attack with: (a) frame-independent FGSM, (b) temporally-consistent FGSM (same δ across frames), (c) per-frame PGD
- Measure: ASR for each attack type vs image-only baseline

**Experiment:**
- Image model: standard CNN on single-frame
- Video model: same CNN with 3-frame temporal pooling
- Attack (a),(b),(c) on both; measure ASR and perturbation budget required
- Key finding: does temporal averaging act as natural adversarial smoothing?

---

## H222 — Input Dimensionality and Adversarial Vulnerability

**Core question:** If we reduce input dimensionality via PCA or learned compression, does adversarial vulnerability (as measured by FGSM ASR and minimum ε to attack) decrease as √(n_components)?

**Prior art:**
- Simon-Gabriel et al. ICML 2019 (arXiv 1802.01421): vulnerability ∝ √(input_dimension) in L₁ gradient norm
- Gilmer et al. ICLR 2019 (arXiv 1905.12202): concentration of measure → inherent limits
- Bubeck et al. NeurIPS 2021: single gradient step sufficient on random networks

**Novelty:** Direct experimental test on Fashion-MNIST:
- PCA-compress to n=[28, 50, 100, 200, 400, 784] dimensions
- Train CNN on each compressed representation
- Measure: min ε to achieve 50% ASR (FGSM); gradient L1-norm per sample; margin
- Test Simon-Gabriel's √(n) scaling law empirically

**Experiment:**
- For each n_dim: PCA(n) → retrain CNN → FGSM sweep (ε=0.01..0.5) → find ε_{50%}
- Plot ε_{50%} vs n_dim; fit ε ∝ n^α; test if α ≈ -0.5
- Key finding: does dimensionality reduction provide certified-like protection?

---

## H223 — Noise Augmentation Count for Smoothing Robustness

**Core question:** What is the minimum number of Gaussian noise copies K needed to achieve robust AUROC comparable to margin-based prediction?

**Prior art:**
- Cohen et al. ICML 2019: certified radius from K Monte-Carlo samples; confidence grows with K
- H77, H204 in this codebase: randomized smoothing certified radius analysis
- High-K certification converges but low-K is noisy; K=100 standard for σ=0.25

**Novelty:** Sweep K=1..200 and measure:
1. Certified radius per sample at each K
2. Spearman ρ(smooth_margin_K, logit_margin) — convergence with K
3. AUROC of smooth_margin_K vs PGD success — at what K does it plateau?

**Experiment:**
- Take base classifier; compute smoothed prediction over K Gaussian noisy copies
- For K in [1, 5, 10, 20, 50, 100, 200]: compute per-sample certified radius
- Spearman ρ(certified_radius_K, logit_margin) vs K: convergence curve
- Key finding: what K is "good enough" for robustness prediction (not full certification)?

---

## H224 — Subject vs Background Pixel Targeting in Adversarial Attacks

**Core question:** Do adversarial perturbations concentrate on semantically meaningful (subject) pixels or background pixels?

**Prior art:**
- H149, H12 in codebase: saliency spatial patterns and entropy
- Tsipras et al. 2019: robust features look like object parts
- Finding in codebase: saliency adds negligible AUROC beyond margin on Fashion-MNIST
- Key gap: Fashion-MNIST has no explicit background; need segmentation mask

**Novelty:** Use saliency maps as proxy for subject vs background:
- Define "subject mask" = top-30% saliency pixels; "background" = bottom-30%
- Compute ‖δ_fgsm · subject_mask‖ vs ‖δ_fgsm · background_mask‖ per sample
- Test: does attack energy concentrate on subject or background?
- Does subject-concentration correlate with attack success?

**Experiment:**
- For 300 test samples: compute GradCAM saliency mask (top-30% pixels)
- FGSM perturbation δ = ε * sign(∇_x L)
- Measure: subject_ratio = ‖δ * mask‖ / ‖δ‖ (fraction of perturbation energy on subject)
- Correlate subject_ratio with margin and PGD success
- Key finding: do harder samples have attacks that concentrate on subject?

---

## H225 — LSB Stripping and Bit-Depth Reduction vs Adversarial Perturbations

**Core question:** Since adversarial perturbations often fall in the least significant bits (ε=0.1 ≈ 25/255 ≈ 4-5 bits), does training exclusively on 4-bit quantized images confer robustness?

**Prior art:**
- Feature Squeezing (Xu et al. 2017): bit-depth reduction detected adversarials but broken by BPDA
- Guo et al. 2018: static preprocessing (bit-depth, JPEG, TV-min) defeated by adaptive attacks
- H46, H48, H194 in codebase: feature squeezing analysis on Fashion-MNIST
- Finding: 3-bit quantization recovery rate 15-25%; adaptive attacks restore 90% ASR

**Novelty:** Train-time bit-depth reduction (not just test-time defense):
- Train 5 models on: 8-bit (baseline), 6-bit, 5-bit, 4-bit, 3-bit images
- Test all with standard FGSM/PGD (no adaptive attack)
- Key question: does the model *learn* representations robust to quantization noise, reducing adversarial sensitivity?

**Experiment:**
- Quantize all training images to n-bit before training (n in [3,4,5,6,8])
- Test on 8-bit test images (standard eval); also test on n-bit test images
- Measure: clean accuracy, FGSM ASR, PGD ASR, margin distribution
- Control: test-time-only quantization vs train-time quantization (is train-time necessary?)

---

## H226 — Mixed Spatial Precision: More Bits for High-Complexity Regions

**Core question:** Can variable bit-depth per image region (more bits where there's more structure, fewer bits in smooth regions) reduce adversarial sensitivity while preserving clean accuracy?

**Prior art:**
- JPEG naturally implements this via DCT coefficient quantization (coarse for high-freq, fine for DC)
- Mixed precision in DNN training (FP16/FP32 per layer) improves training stability
- No prior work on input-adaptive spatial bit-depth for adversarial robustness

**Novelty:** Implement image-adaptive quantization:
- Compute local variance in 4×4 patches; high variance → 8 bits; low variance → 3 bits
- "Variable precision image" = concat [8-bit_subject_patches, 3-bit_background_patches]
- Train and test on variable-precision images

**Experiment:**
- Compute patch variance → threshold → bit-depth map per image
- Quantize accordingly; train CNN on variable-precision images
- Attack with standard FGSM/PGD (perturbation applied to 8-bit original)
- Measure: does variable precision training improve robustness on smooth regions?

---

## H227 — Training on JPEG-Compressed Images Without Decompression

**Core question:** Does training a model on JPEG quality-30 or quality-50 images (without JPEG-decoding, treating them as the "true" training domain) improve robustness?

**Prior art:**
- Dziugaite et al. 2016: JPEG as test-time defense (broken by adaptive attacks)
- Feature Squeezing includes JPEG; defeated by BPDA
- JPEG training (train on q=50, test on q=50): implicit regularisation via DCT quantization grid
- Neural image compression (e.g., Balle et al. 2018) in adversarial setting: underexplored

**Novelty:** Train-time JPEG domain shift:
- Encode training images at q ∈ {10, 30, 50, 70, 90} → decode → train CNN
- Test on q=100 (lossless) images
- Hypothesis: JPEG training forces model to use robust (low-frequency) features only

**Experiment:**
- 5 training regimes: q=10,30,50,70,90 + baseline q=100
- Standard training (no AT); measure FGSM/PGD ASR
- Key metric: does q=30-50 training reduce ASR on non-JPEG test images?
- Adaptive attack: re-compute δ with JPEG encoding in loop (BPDA) — does training robustness survive?

---

## H228 — Universal Adversarial Purifier / Foundational Defence Model

**Core question:** Can a single trained denoiser (purifier) remove adversarial perturbations from arbitrary inputs before classification, without needing to know which attack was used?

**Prior art:**
- Defense-GAN (Samangouei et al. 2018): project onto GAN manifold; defeated by adaptive attacks
- HGD (Liao et al. 2018): guided denoiser trained against perturbations
- DiffPure (Shi et al. 2022, arXiv 2205.07460): diffusion-based purification; SOTA; ~80% on CIFAR-10 with AutoAttack
- Adaptive attacks break DiffPure (Croce & Hein 2022); EOT evaluation required

**Novelty (Fashion-MNIST feasible):** Train small denoising autoencoder as purifier:
- Purifier P: CNN encoder-decoder trained to reconstruct clean x from perturbed x + δ_PGD
- Test: does P(x_adv) → x_clean restore classification?
- Universal test: train purifier against PGD-ε=0.1, evaluate on FGSM, C&W, AutoAttack

**Experiment:**
- Train P with MSE(P(x + δ_pgd), x) on training set
- Evaluate: P(x_adv) → clf → is prediction restored? Recovery rate per attack type
- Measure: clean accuracy degradation from purification (accuracy-recovery tradeoff)
- Adaptive attack: optimize δ through P (treat P as differentiable)

---

## H229 — Test-Time Input Optimisation (Fixed Model, Learnable Input)

**Core question:** Given a fixed pretrained model f, can we optimise x' ← argmin_x L(f(x), y) starting from x_test, and does this improve classification and robustness?

**Prior art:**
- Deep Dream (Simonyan & Vedaldi 2015): optimise input to maximise activation
- Test-time training (Sun et al. 2020): adapt model; this is adapt *input* instead
- Mahendran & Vedaldi 2015: image reconstruction from model representation
- No direct prior on "test-time input optimisation" for adversarial defence

**Novelty (fully novel):**
- Fix model f (pretrained, frozen weights)
- At test time: x* = x_test - α * ∇_x L(f(x), ŷ)  for T steps  (move x toward cleaner prediction)
- Question A: does x* have higher margin than x_test?
- Question B: if x_test is adversarial, does input optimisation recover clean prediction?
- Question C: what does x* look like? (feature inversion)

**Experiment:**
- Train base CNN; freeze it
- For each test sample: run gradient descent on input (T=10, 20, 50 steps; α=0.01)
- Measure: margin(x*) vs margin(x), classification accuracy, recovery rate from adversarial
- Visualise x* for random samples — does input optimisation denoise?

---

## H230 — Early vs Late Epoch Adversarial Example Effectiveness

**Core question:** Are adversarial examples generated by an *early*-epoch (partially trained) model as effective against the *final* model as examples generated by the final model itself?

**Prior art:**
- Curriculum AT (Cai et al. 2018): start with small ε, gradually increase; early-epoch adversarials are weaker
- Zhang et al. TRADES: early vs late epoch boundary geometry differs
- Wong et al. fast AT (arXiv 1904.01234): single-step FGSM with random init; cheap early-epoch attacks
- No direct "adversarial example transfer across training epochs" study

**Novelty:**
- Train CNN; save checkpoints at epochs [1, 5, 10, 15, 20]
- For each checkpoint k: generate FGSM/PGD adversarials
- Test those adversarials against the *final* (epoch-20) model
- Measure: ASR(attack_from_epoch_k, evaluated_on_final_model) vs k

**Experiment:**
- 5 source models (epochs 1,5,10,15,20); 1 target (final model)
- For each source: generate 300 adversarials; evaluate on target
- Plot: ASR vs source epoch; Spearman ρ between source epoch and transfer ASR
- Key finding: is early-epoch geometry predictive of final-model vulnerability?

---

## H231 — Adversarial Trajectory Transfer Between Images

**Core question:** Can the adversarial trajectory (sequence of gradient-step perturbations) computed for image X be applied to image X' to create an adversarial example without re-running the attack?

**Prior art:**
- Universal adversarial perturbations (UAP, Moosavi-Dezfooli et al. 2017): single δ fools 85% of images
- Liu et al. 2017: adversarial examples transfer across models (decision boundary geometry shared)
- H210 (this codebase): weight trajectory transfer failed; needed permutation alignment
- Difference: UAP is image-agnostic; trajectory-transfer would preserve per-image order of steps

**Novelty:** Transfer the *sequence* of K perturbation steps from X to X':
- For image X: compute δ_1,...,δ_K (PGD steps = gradient directions at each iteration)
- Apply same sequence to X': x'_k+1 = x'_k + α * δ_k (using X's directions, not X's gradients)
- Compare vs: universal δ (UAP-style), from-scratch PGD on X', random walk
- Measure: ASR of trajectory-transferred adversarials

**Experiment:**
- Source: compute PGD-20 trajectories for 50 images → store δ_1..δ_20 per image
- Transfer: apply each trajectory to 100 *other* images
- Measure: ASR when trajectory applied to non-source image
- Baseline: UAP (single δ across all images), random steps
- Key finding: does adversarial geometry transfer in direction space?

---

## H232 — Random Noise Padding: Attack Budget Allocation

**Core question:** If we pad input images with B pixels of random noise on all sides, do adversarial attacks "waste" perturbation budget on padding pixels, reducing attack effectiveness on the core image?

**Prior art:**
- Xie et al. 2019 (DIM): input diversity (random resize + padding) improves adversarial *transfer*
- Random padding as preprocessing: used for diversity in attacks, not as defence
- Adaptive attacks ignore useless padding pixels (gradient=0 on frozen padding)

**Novelty:**
- Pad test images with B ∈ {0, 2, 4, 8} pixels of random noise (re-randomised per forward pass)
- Adaptive attack: l∞ ball over full padded image (including padding pixels)
- Question: does attacker focus on core image or padding?
- Measure: ‖δ_core‖ vs ‖δ_padding‖ in PGD adversarials; ASR on padded vs unpadded model

**Experiment:**
- Train model on randomly-padded images (padding re-randomised each batch)
- Attack: PGD-10 on full padded image (attacker can perturb padding pixels too)
- Measure: fraction of ‖δ‖ budget used on padding vs core; ASR
- Control: model trained without padding, tested with padding at test time
- Key finding: does stochastic padding provide any adversarial smoothing?

---

## H233 — Anti-Adversarial Examples: Moving WITH the Gradient

**Core question:** Does training on "anti-adversarial" (gradient-maximally-correct) examples improve generalisation or reduce clean robustness?

**Prior art:**
- Salman et al. 2020: adversarially robust representations improve transfer
- Yin et al. 2022 "Friendly Noise": augmentation with adversarial noise in *correct* direction
- Bartlett et al. "benign overfitting" harmfulness for adversarial robustness
- No direct "anti-adversarial" (gradient-ASCENT on loss = gradient-DESCENT for correct class) study

**Anti-adversarial definition:**
x_anti = x - ε * sign(∇_x L(f(x), y))  [moves x to INCREASE margin = toward class centroid]

**Novelty:**
- Generate anti-adversarial examples for all training images
- Hypothesis A: training on anti-adversarials improves generalisation (samples are "more canonical")
- Hypothesis B: training on anti-adversarials reduces adversarial robustness (model not exposed to boundary)
- Hypothesis C: keeping only anti-adversarial samples (lowest-loss subset) = cleaner training data

**Experiment:**
- Generate training set: x_anti = x - ε * sign(∇_x L)  for all training images
- Train 3 models: baseline (x), anti-adversarial (x_anti), mixed (x + x_anti)
- Measure: clean accuracy, FGSM/PGD ASR, margin distribution
- Also test: train on only the top-20% highest-margin (most anti-adversarial) samples

---

## H234 — Bit-Plane Randomisation as Adversarial Noise

**Core question:** Randomly flipping the K least-significant bits of all pixels (at training AND test time) — does this destroy adversarial perturbations and serve as an implicit randomized smoothing?

**Prior art:**
- Feature squeezing (bit-depth reduction) broken by BPDA
- Randomized smoothing (Cohen 2019): Gaussian noise σ ≈ N(0, σ²); certified L2 robustness
- Difference: bit-plane noise is *discrete uniform* not Gaussian; no known certification theorem
- LSB randomization: equivalent to adding ±1/255 per pixel uniformly (very different distribution)

**Novelty:** Train with random LSB flipping:
- At each forward pass: flip bottom-K bits of each pixel value (K=1,2,3)
- This adds discrete uniform noise U(-2^K/255, +2^K/255) per pixel
- At test time: apply same randomisation K times, take majority vote (à la randomized smoothing)

**Experiment:**
- Train 4 models: baseline, LSB-flip-K=1, LSB-flip-K=2, LSB-flip-K=3
- Test: standard FGSM/PGD; adaptive attack (BPDA through bit-flip)
- Measure: clean accuracy, ASR, and whether adversarial perturbations survive bit-flip
- Theoretical note: ε=0.1 ≈ 25/255; K=4-bit perturbation = 16/255 = in adversarial range

---

## Summary Table

| ID | Title | Novel vs Prior Art | Feasibility |
|----|-------|--------------------|-------------|
| H215 | L-Ball Region Training | Compare IBP vs smooth-AT vs PGD-AT on FMNIST | Medium |
| H216 | Augmentation vs Random Attack | Tests augmentation specifically against random-direction FGSM | Easy |
| H217 | Model Width/Depth Robustness Scaling | Width-vs-depth controlled ablation on FMNIST | Easy |
| H218 | Robust Overfitting Per-Sample | Per-sample margin trajectory predicts overfitting | Medium |
| H219 | SSL ViT Robustness (DINO-lite) | Mini-DINO on FMNIST vs supervised CNN | Hard |
| H220 | Part-Based / Shape Bias | Patch masking forces global reasoning | Easy |
| H221 | Video Temporal Averaging | Simulate 3-frame clips on FMNIST | Easy |
| H222 | Dimensionality vs Min-ε to Attack | Test Simon-Gabriel √n scaling law empirically | Easy |
| H223 | Noise Count K for Smoothing | Sweep K=1..200 for AUROC convergence | Easy |
| H224 | Subject vs Background Pixel Targeting | GradCAM mask energy analysis | Easy |
| H225 | Train-Time Bit-Depth Reduction | Train on 3-8 bit; test adversarial robustness | Easy |
| H226 | Spatial Mixed Precision | Adaptive per-patch bit depth | Medium |
| H227 | JPEG Training Domain | Train on JPEG q=30-50; test on q=100 | Easy |
| H228 | Universal Adversarial Purifier | CNN denoiser + adaptive attack evaluation | Medium |
| H229 | Test-Time Input Optimisation | Fixed model, gradient descent on input at test time | Easy |
| H230 | Early vs Late Epoch Adversarial Transfer | Cross-checkpoint adversarial example transfer | Easy |
| H231 | Adversarial Trajectory Transfer | Apply X's PGD step directions to X' | Medium |
| H232 | Random Noise Padding Budget | PGD budget allocation: padding vs core pixels | Easy |
| H233 | Anti-Adversarial Examples | Move with gradient; train on resulting samples | Easy |
| H234 | Bit-Plane LSB Randomisation | Discrete uniform noise (K LSB flip) as smoothing | Easy |
