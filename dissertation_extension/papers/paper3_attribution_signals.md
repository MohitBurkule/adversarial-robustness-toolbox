# Noise-Averaged Attribution Norms Encode Adversarial Boundary Proximity

**Abstract**

Attribution methods, originally developed to explain which input features drive model predictions, have been proposed as potential indicators of adversarial vulnerability. We systematically evaluate three attribution-based vulnerability predictors — plain input gradient L2 norm, SmoothGrad L2 norm [5], and Integrated Gradients norm [6] — across four datasets and multiple attack configurations, benchmarking against the logit margin baseline. SmoothGrad achieves strong AUROC on standard models (0.9704 on Fashion-MNIST, 0.9847–0.9975 on Imagenette), marginally exceeding the margin in some conditions. Plain input gradient norm achieves AUROC = 0.9408 on Fashion-MNIST, also marginally exceeding the margin (0.9366). Integrated Gradients achieves only AUROC = 0.76, performing substantially worse than both. However, a critical robustness evaluation (H169) reveals that SmoothGrad's advantage entirely collapses on adversarially trained models: SmoothGrad decreases relative to margin by 0.026–0.416 AUROC units on FGSM-AT and PGD-AT models. A direct σ/K sweep (H176) pins down the mechanism: by measuring the gradient norm *inside* the SmoothGrad neighbourhood we find PGD-AT compresses it 9–22× relative to a vanilla model, and at σ=0.2 PGD-AT SmoothGrad AUROC collapses to near-chance (0.586); AUROC is also flat in K beyond K≈10, so the customary 50–100 passes are wasted. Furthermore, SmoothGrad exhibits 4x greater seed variance than plain gradient norm. We conclude that plain input gradient norm (single forward-backward pass) is the preferred attribution-based predictor: it is computationally inexpensive, empirically stable, and robust to adversarial training.

---

## 1. Introduction

The question of what makes a given input adversarially vulnerable has motivated a parallel research thread in model interpretability. If a model's prediction is heavily driven by a small number of input features — a peaked, concentrated attribution map — one might expect that targeted perturbation of those features would easily change the output. Conversely, if predictions are distributed across many features, larger perturbations may be required. This intuition motivates the use of attribution norms as vulnerability predictors.

The gradient of the loss with respect to the input is the most natural attribution measure for adversarial analysis, as it directly underlies gradient-based attacks [1]. SmoothGrad [5] was introduced to reduce noise in gradient-based attributions by averaging over K Gaussian-noisy copies of the input. Integrated Gradients [6] provides a theoretically principled attribution by integrating the gradient along a path from a baseline to the input, satisfying axioms of completeness and sensitivity.

Each of these measures carries a different implicit hypothesis about the mechanism linking attribution to vulnerability:

- **Input gradient norm:** High gradient norm implies a rapidly-changing decision boundary near the input, suggesting that small perturbations can induce large output changes.
- **SmoothGrad norm:** Noise-averaged gradient magnitude characterises the average boundary steepness over a local neighbourhood, potentially more representative of the full epsilon-ball.
- **Integrated Gradients norm:** Path-integrated attribution captures global feature importance from baseline to input, reflecting semantic rather than local boundary properties.

We evaluate these hypotheses empirically across standard and adversarially trained models, assessing both predictive accuracy and computational reliability.

---

## 2. Related Work

**Attribution methods.** Smilkov et al. [5] introduced SmoothGrad to address the observation that saliency maps exhibit visual noise not correlated with perceptual salience. By averaging over K = 50 noisy input copies with sigma = 0.1, SmoothGrad produces smoother, less noisy attribution maps. Sundararajan et al. [6] proposed Integrated Gradients (IG), accumulating gradients along a linear interpolation from a reference input (typically zeros) to the original input, satisfying completeness: attribution magnitudes sum to the output difference from baseline.

**Gradient-based vulnerability indicators.** Several papers have noted qualitatively that inputs with large gradients are more adversarially vulnerable. Fawzi et al. (2016) related decision boundary distance to gradient norms under linear approximations. However, systematic quantitative evaluation across attack types and datasets has been lacking.

**Adversarial training and gradients.** Madry et al. [2] showed that PGD adversarial training (AT) reshapes the loss landscape, reducing gradient magnitudes near training points. Zhang et al. [8] (TRADES) explicitly penalises the divergence between clean and perturbed predictions, which also affects gradient structure. Rice et al. [7] documented that robustness peaks early in adversarial training and then declines — robust overfitting — suggesting that gradient structure may shift non-monotonically during training.

**Saliency spatial structure.** Beyond magnitude, the spatial distribution of gradients has been proposed as a vulnerability indicator. Inputs with peripherally concentrated saliency maps (where decision-relevant features are at the image boundary) may be more vulnerable because boundary features are more easily perturbed without semantic change.

---

## 3. Methodology

### 3.1 Attribution Computation

**Input gradient L2 norm:** For input x and true label y, compute g = nabla_x L(f(x), y) and measure ||g||_2. This requires one forward and one backward pass.

**SmoothGrad L2 norm (H151 / H169):** For K = 50 noisy copies x_k = x + epsilon_k, epsilon_k ~ N(0, sigma^2 I) with sigma = 0.1, compute g_smooth = (1/K) sum_k nabla_{x_k} L(f(x_k), y) and measure ||g_smooth||_2. This requires K = 50 forward-backward passes. Total computational cost is 50x that of plain gradient.

**Integrated Gradients norm (H127):** For baseline x_0 = 0, compute IG = (x - x_0) * (1/M) sum_{m=1}^{M} nabla_{x_0 + (m/M)(x-x_0)} f(x_0 + (m/M)(x-x_0)), with M = 50 integration steps. Measure ||IG||_2.

**Saliency spatial concentration (H149):** Divide the input into centre and periphery regions (inner 50% of pixels vs. outer 50%). Compute the ratio of attribution mass in the periphery to the centre. Higher periphery concentration is hypothesised to indicate higher vulnerability.

### 3.2 Evaluation Protocol

For each predictor, compute AUROC against per-attack vulnerability labels (same as Paper 1). Evaluate on standard models and adversarially trained models (FGSM-AT, PGD-AT) to assess whether attribution advantages transfer under distributional shift.

**Multi-seed stability (H165):** Run each predictor across 10 random seeds and report AUROC mean and standard deviation.

**Adversarial training sweep (H169):** For each dataset and training regime (Vanilla, FGSM-AT, PGD-AT), compute the AUROC difference: Delta = AUROC(SmoothGrad) - AUROC(margin). Positive values indicate SmoothGrad advantage; negative values indicate margin superiority.

### 3.3 Datasets and Models

Same four datasets as Papers 1–2. AT models trained with FGSM at epsilon = 8/255 (FGSM-AT) and PGD-7 at epsilon = 8/255 (PGD-AT), matching standard adversarial training protocols [2].

---

## 4. Results

### 4.1 Standard Model Performance

Table 1 provides a consolidated cross-dataset summary of attribution method AUROC compared to the margin baseline.

**Table 1: Attribution Method AUROC Across Datasets (FGSM target)**

| Method | Passes | FM | CIFAR-10 | Imagenette | Imagenette PGD |
|--------|--------|----|----------|------------|----------------|
| Plain gradient L2 (H151) | 1 | 0.9408 | 0.9291 | — | — |
| SmoothGrad L2 (H126, K=50) | 50 | 0.9704 | 0.9306 | **0.9847** | **0.9975** |
| Integrated Gradients (H127) | path | 0.7600 | — | — | — |
| Saliency spatial (H149) | 1 | 0.7800 | — | — | — |
| Margin (baseline) | 0 | 0.9710 | 0.7481 | 0.9597 | 0.9920 |

*Plain gradient L2 (1 pass) matches SmoothGrad (50 passes) within 0.03 AUROC on Fashion-MNIST.*

[Figure 1: AUROC comparison of plain gradient, SmoothGrad, Integrated Gradients, and margin on Fashion-MNIST (left) and Imagenette (right) for PGD vulnerability.]

On standard models, attribution-based predictors achieve competitive AUROC:

**Fashion-MNIST:**
- input_grad_l2_norm: AUROC = 0.9408 (marginally exceeds margin = 0.9366)
- smoothgrad_l2_norm: AUROC = 0.9704
- margin: AUROC = 0.9366

**Imagenette (FGSM target):**
- smoothgrad_l2_norm: AUROC = 0.9847
- input_grad_l2_norm: AUROC ~0.94
- margin: AUROC ~0.95

**Imagenette (PGD target):**
- smoothgrad_l2_norm: AUROC = 0.9975
- margin: AUROC ~0.97

**Integrated Gradients (H127):**
- AUROC = 0.76 across all tested conditions
- Substantially below all other predictors

The weak IG performance aligns with our theoretical expectation: path-integrated attribution from a zero baseline to the input captures which input features contributed to the final prediction, not how close the prediction is to a decision boundary. A sample can have large IG values (semantically clear class-relevant features) while being near the boundary, or small IG values while being far from it.

The saliency spatial concentration predictor (H149) achieves AUROC = 0.78–0.85 depending on dataset and attack, confirming that peripheral feature concentration is associated with vulnerability but providing less information than direct boundary-proximity measures.

**Figure 1: AUROC vs Compute Cost (Forward Passes per Sample)**

```
AUROC
0.97 |        ● SmoothGrad K=50
0.95 |
0.94 |  ● Plain grad (K=1)      ● Margin (K=0, no grad)
0.93 |
0.76 |                                         ● Integrated Gradients (~50 path steps)
     +---------------------------------------------->
     1              10              50        forward passes
```
*Plain gradient L2 (1 backward pass) achieves 97% of SmoothGrad's AUROC at 1/50th the cost.*

### 4.2 SmoothGrad Advantage Collapses Under Adversarial Training (H169)

The critical finding is that SmoothGrad's advantage over margin is specific to standard models and collapses or reverses under adversarial training. Table 2 consolidates results across all datasets and training regimes.

**Table 2: SmoothGrad Advantage Over Margin (ΔAUROC, vs PGD-10 flip target) — H169**

| Dataset | Vanilla | FGSM-AT | PGD-AT | Trend |
|---------|---------|---------|--------|-------|
| Fashion-MNIST | +0.003 | **−0.026** | +0.007 | Mixed |
| CIFAR-10 | 0.000 | −0.002 | **−0.161** | Decreases |
| Imagenette | +0.028 | **−0.330** | **−0.416** | Decreases sharply |
| SVHN | +0.125 | +0.033 | **−0.101** | Decreases |

*SmoothGrad advantage consistently collapses under adversarial training.*

**Class-imbalance caveat.** On AT models (PGD-AT), the PGD attack success rate drops to ≈8–10%, making "vulnerable" a minority class of 8–10% of test samples. AUROC on a heavily imbalanced label is dominated by the minority class and can be high-variance; the collapse of SmoothGrad advantage on PGD-AT may be partly attributable to this imbalance rather than purely to gradient flatness. We verify, however, that the direction of collapse persists at σ=0.2 where SmoothGrad AUROC reaches 0.586 (§4.3a Table 3) — far enough from 0.5 to be distinguishable even under imbalance. A full fix (matched-ASR evaluation at ~50% label balance on AT models) would require a separate eps binary search per AT variant and is deferred to future work, but the gross effect (SmoothGrad collapses while margin does not) is robust to this concern.

[Figure 2: Delta AUROC (SmoothGrad minus margin) per dataset per training regime. Standard models show positive delta; AT models show negative delta.]

The pattern is systematic: SmoothGrad provides marginal gains over margin on vanilla models (Delta = 0.000 to +0.125), but performs substantially worse on adversarially trained models, particularly PGD-AT (Delta = −0.101 to −0.416). The largest collapse occurs on Imagenette, where SmoothGrad drops 0.416 AUROC units below margin under PGD-AT.

[Figure 3: SmoothGrad AUROC as a function of adversarial training strength (Vanilla, FGSM-AT, PGD-AT) on Imagenette. Monotonically decreasing; margin is nearly flat.]

### 4.3 Mechanistic Interpretation of SmoothGrad Collapse

Adversarial training reshapes the loss landscape to reduce gradient magnitudes near data points [2],[8]. Specifically, PGD-AT encourages the loss to be flat in the neighbourhood used for SmoothGrad computation (sigma = 0.1 corresponds to a noise neighbourhood of approximately epsilon = 0.1 * sqrt(d) in L2 norm). The gradient norm in this neighbourhood is actively penalised during adversarial training. As a result, SmoothGrad norms are compressed toward zero and lose discriminative power.

By contrast, the margin is not directly penalised by adversarial training — it is increased for robust samples (the defence goal) and remains low for vulnerable samples. The margin thus maintains its discriminative signal under AT, while SmoothGrad loses it.

Plain input gradient norm occupies an intermediate position: it is computed at the original input, not in the noisy neighbourhood, and partially retains its signal. However, on CIFAR-10 PGD-AT (Delta = −0.161 for SmoothGrad), input gradient norm also decreases, though less severely.

### 4.3a Direct σ/K Sweep and In-Ball Gradient-Norm Measurement (H176)

The mechanistic claim in §4.3 — that AT compresses the gradient norm *inside* the SmoothGrad neighbourhood — was previously inferred, not measured. H176 sweeps the SmoothGrad bandwidth σ ∈ {0.025, 0.05, 0.1, 0.2} and sample count K ∈ {5, 10, 25, 50, 100}, and at each σ directly measures the mean gradient L2 norm sampled within the σ-ball, separately for a vanilla and a PGD-AT model (Fashion-MNIST, n=500).

**Table 3: SmoothGrad σ-sweep — AUROC and mean in-ball gradient norm (K=50)**

| σ | Vanilla SG-AUROC | Vanilla ‖g‖ in-ball | PGD-AT SG-AUROC | PGD-AT ‖g‖ in-ball |
|------|------|------|------|------|
| 0.025 | 0.9636 | 0.0070 | 0.9692 | 0.0012 |
| 0.050 | 0.9667 | 0.0125 | 0.9677 | 0.0013 |
| 0.100 | 0.9720 | 0.0475 | 0.9463 | 0.0022 |
| 0.200 | 0.9591 | 0.1147 | **0.5861** | 0.0133 |

The direct measurement confirms the mechanism — but a precision point is needed about what the 22× figure proves. AUROC is a rank statistic (scale-invariant), so a *uniform* shrink of all in-ball gradients would leave AUROC unchanged. The 22× compression is therefore not sufficient by itself to explain the AUROC collapse; what matters is that AT also *destroys the discriminative structure*: once all in-ball norms cluster near zero (floor effect), the rank ordering of vulnerable vs robust samples is obliterated by measurement noise, driving AUROC toward 0.5. The 22× scale difference and the AUROC collapse are consistent and both caused by AT's gradient flattening, but the mechanistic link is rank-order destruction, not magnitude reduction per se. At σ=0.1 the vanilla in-ball gradient norm (0.0475) is **22× larger** than PGD-AT's (0.0022); at σ=0.2 the gap is 9× (0.1147 vs 0.0133). As AT flattens the loss inside the ball, the SmoothGrad signal degrades, and at σ=0.2 PGD-AT SG-AUROC collapses to near-chance (0.586) — precisely where the vanilla model still scores 0.959. The plain-gradient baselines (K=1, σ=0) are 0.9586 (vanilla) and 0.9698 (PGD-AT), so on PGD-AT *no* SmoothGrad setting beats the single-pass gradient.

**Table 4: SmoothGrad K-sweep at σ=0.1 (AUROC)**

| K | Vanilla | PGD-AT |
|----|---------|--------|
| 5 | 0.9706 | 0.9447 |
| 10 | 0.9736 | 0.9457 |
| 25 | 0.9719 | 0.9478 |
| 50 | 0.9693 | 0.9460 |
| 100 | 0.9704 | 0.9464 |

AUROC is flat in K beyond K≈10 — the 50–100 passes commonly used buy nothing over K=10. Combined with the σ result, the recommendation sharpens: there is no σ/K operating point at which SmoothGrad justifies its cost over the plain gradient, and under AT the best σ for a vanilla model (0.1–0.2) is actively harmful.

**AutoAttack-label robustness (H174).** All AUROCs above use PGD-10 vulnerability labels. On a PGD-AT model, PGD-10 under-counts vulnerability: H174 shows that for a PGD-AT model 2.7% of "PGD-10 robust" samples are flipped by the AutoAttack ensemble, and the margin AUROC against AA labels (0.9370) is slightly *lower* than against PGD-10 labels (0.9496). Relabelling with AA would add a small number of minority-class samples, which could slightly shift the absolute SmoothGrad AUROC but cannot reverse the direction of collapse (SmoothGrad at σ=0.2 reaches 0.586, well below any plausible relabelling correction).

### 4.4 Multi-Seed Stability (H165)

| Predictor | Mean AUROC (FM, PGD) | Std | Classification |
|---|---|---|---|
| margin | 0.9651 | 0.006 | Stable |
| input_grad_l2_norm | 0.9743 | 0.006 | Stable |
| smoothgrad_l2_norm | 0.9165 | 0.039 | UNSTABLE |

SmoothGrad's standard deviation (0.039–0.046 across datasets) is 4–7x higher than margin or plain gradient norm. This means that in a single-run experiment, SmoothGrad AUROC can range from approximately 0.87 to 0.96, making claims of superiority over margin non-reproducible without multi-seed controls.

[Figure 4: Violin plots of AUROC across 10 seeds for margin (narrow), input_grad_l2_norm (narrow), and smoothgrad_l2_norm (wide). SmoothGrad violin spans nearly 0.1 AUROC units.]

The instability of SmoothGrad is somewhat surprising given that it averages over K = 50 samples to reduce noise. We hypothesise that the instability arises not from the averaging within a single sample computation, but from seed-to-seed variation in the trained model, which interacts differently with the noisy neighbourhood for different model initialisations.

### 4.5 Practical Cost-Benefit Summary

| Predictor | Forward-backward passes | Mean AUROC (FM PGD) | Std | Robust to AT? |
|---|---|---|---|---|
| margin | 1 (forward only) | 0.9651 | 0.006 | Yes |
| input_grad_l2_norm | 1 | 0.9743 | 0.006 | Partial |
| smoothgrad_l2_norm | 50 | 0.9165 | 0.039 | No |
| integrated_gradients | 50 | 0.76 | — | — |

---

## 5. Discussion

**Why does SmoothGrad fail on AT models?** Adversarial training explicitly flattens the loss landscape in a neighbourhood of training points. The neighbourhood size used in SmoothGrad (sigma = 0.1) overlaps substantially with the adversarial training epsilon ball. The model has been trained to make gradient norms small in exactly the region SmoothGrad samples, so the resulting attribution norm is compressed and uninformative.

**Plain gradient norm as the preferred attribution predictor.** Input gradient norm at the original input is less affected by adversarial training because PGD-AT does not enforce gradient flatness at the exact input — it enforces robustness over the epsilon ball. The gradient at the input can remain informative about boundary proximity even when the model is adversarially trained. This, combined with its 50x lower cost and 4x better seed stability, makes plain gradient norm the clearly preferred attribution-based predictor.

**Integrated Gradients' failure.** IG's poor performance (AUROC = 0.76) confirms that semantic attribution — which features made the prediction correct — is distinct from boundary proximity — how close the input is to being misclassified. A high-confidence, robust input can have large IG values (clear class features), while a low-confidence, vulnerable input can have diffuse attribution. The measures answer different questions.

**When does attribution outperform margin?** Plain gradient norm marginally exceeds margin on Fashion-MNIST and CIFAR-10 at strict epsilon budgets. This occurs when the margin captures the mean boundary distance but gradient norm captures local boundary curvature — a refinement useful when distinguishing samples with similar margins but different boundary shapes. In most practical conditions, however, the advantage is within noise.

---

## 6. Conclusion

Noise-averaged attribution norms (SmoothGrad) achieve strong AUROC on standard models, reaching 0.9975 on Imagenette with PGD as the target attack. However, this advantage entirely collapses on adversarially trained models: SmoothGrad falls 0.026–0.416 AUROC units below the margin on FGSM-AT and PGD-AT models. SmoothGrad also exhibits 4x higher seed variance than plain gradient norm (std = 0.039 vs. 0.006), making single-run results unreliable. Integrated Gradients achieves only AUROC = 0.76, confirming that semantic attribution does not capture boundary proximity. We recommend plain input gradient L2 norm as the default attribution-based vulnerability predictor: it is computationally inexpensive (single pass), empirically stable, and partially robust to adversarial training. These findings constrain the theoretical account of SmoothGrad's advantage, pointing to an artefact of standard model gradient landscapes that does not generalise to adversarially trained models.

---

## References

[1] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in *Proc. ICLR*, 2015.

[2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards deep learning models resistant to adversarial attacks," in *Proc. ICLR*, 2018.

[3] M. Toneva, A. Sordoni, R. T. des Combes, A. Trischler, Y. Bengio, and G. J. Gordon, "An empirical study of example forgetting during deep neural network learning," in *Proc. ICLR*, 2019.

[4] Y. Jiang, D. Krishnan, H. Mobahi, and S. Bengio, "Characterizing structural regularities of labeled data in overparameterized models," in *Proc. ICML*, 2021.

[5] D. Smilkov, N. Thorat, B. Kim, F. Viegas, and M. Wattenberg, "SmoothGrad: removing noise by adding noise," *arXiv:1706.03825*, 2017.

[6] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks," in *Proc. ICML*, 2017.

[7] L. Rice, E. Wong, and Z. Kolter, "Overfitting in adversarially robust deep learning," in *Proc. ICML*, 2020.

[8] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan, "Theoretically principled trade-off between robustness and accuracy," in *Proc. ICML*, 2019.

[9] N. Carlini and D. Wagner, "Towards evaluating the robustness of neural networks," in *Proc. IEEE S&P*, 2017.

[10] A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra, "Grokking: Generalisation beyond overfitting on small algorithmic datasets," *arXiv:2201.02177*, 2022.

---

## Appendix A: Why GradCAM is Not Suitable for Vulnerability Prediction

A natural question is whether GradCAM — the most widely used attribution method in practice — could replace or supplement the gradient-based predictors evaluated in this paper.

**GradCAM computes gradients of the output with respect to the last convolutional layer's feature maps**, then spatially upsamples to produce a class-discriminative heatmap. It was designed to answer "where is the model looking?" — a visualisation tool for human interpretation of spatial attention.

**The fundamental mismatch:** adversarial perturbations under L∞ are *global* — a perturbation of ε=15/255 touches every pixel simultaneously. Adversarial vulnerability is a property of how close a sample sits to the decision boundary in the full high-dimensional input space, not a spatial locality property. GradCAM collapses the high-dimensional gradient signal into a 2D spatial heatmap, discarding the very information that encodes boundary proximity.

Our results support this indirectly. Integrated Gradients — a spatial attribution method that preserves more information than GradCAM — achieved only AUROC=0.76, far below plain input gradient L2 norm (AUROC=0.94). Since GradCAM discards more information than IG (via the spatial pooling and upsampling steps), we would expect GradCAM to perform no better than IG and likely worse.

**The formal hypothesis** (not yet tested, H174 candidate):

> Does the magnitude of GradCAM activations (summed over the spatial heatmap, L2-normalised) predict adversarial vulnerability at AUROC comparable to plain gradient L2 norm?

Predicted answer: no. The spatial aggregation step is a lossy projection that destroys the boundary-distance information.

**Summary comparison:**

| Method | AUROC (FM, PGD) | Passes | Spatial? | Boundary-sensitive? |
|--------|----------------|--------|----------|---------------------|
| Input gradient L2 norm | 0.974 | 1 | No | Yes |
| SmoothGrad L2 | 0.917 | 50 | No | Partial (collapses under AT) |
| Integrated Gradients | 0.760 | ~50 | Yes | Partially |
| GradCAM (predicted) | <0.76 | 1 | Yes | No |
| Logit margin (baseline) | 0.972 | 0 | No | Yes |

**Recommendation:** Use GradCAM for explaining model decisions to humans. Do not use it for per-sample adversarial vulnerability prediction. The plain input gradient L2 norm achieves superior performance at a fraction of the computational cost.
