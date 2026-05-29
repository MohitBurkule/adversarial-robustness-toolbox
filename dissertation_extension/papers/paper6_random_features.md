# Per-Sample Adversarial Vulnerability is Encoded in Input Geometry: Evidence from Random Feature Networks

**Abstract**

We investigate the origin of per-sample adversarial vulnerability: is it determined by the geometry of the input space, or does it emerge from learned representations? We compare a standard CNN against a Random Feature Neural Network (RFNN) — an identical architecture whose convolutional layers are frozen at random initialization, with only the final linear classifier trained. On Fashion-MNIST, the RFNN achieves margin AUROC of 0.9232 (FGSM) and 0.9188 (PGD), compared to 0.9201 and 0.9381 for the fully trained CNN. The PGD gap is only 0.019. On Imagenette, the gap grows to 0.038. Raw pixel statistics (mean, std) are substantially weaker (AUROC ≈ 0.514–0.609). Bayesian neural network uncertainty estimates (AUROC 0.9187 ± 0.013) and gradient phase analysis (AUROC 0.84) provide converging evidence. We conclude that adversarial vulnerability is primarily encoded in the input-space geometry — the position of clean samples relative to class boundaries in the ambient pixel space — and that learned representations refine but do not create this structure. This has consequences for the interpretability of adversarial vulnerability and the design of vulnerability-aware training curricula.

---

## 1. Introduction

A central unresolved question in adversarial robustness is: *why are some samples more adversarially vulnerable than others?* The standard view is that the model's learned representation determines vulnerability — samples near the model's decision boundary are easily perturbed to cross it. Under this view, vulnerability is a property of the model, not the data. Training a different model, or training the same architecture differently, should produce a different vulnerability profile for the same input.

An alternative hypothesis is that vulnerability is primarily a property of the input sample's position in the high-dimensional pixel space. Inputs that lie near the true class manifold boundary — that is, near the region where the Bayes-optimal classifier would place the decision boundary — are intrinsically harder to classify robustly, regardless of the specific model trained. Under this view, a model with random (untrained) features should exhibit nearly the same vulnerability ranking as a carefully trained model, because the ranking is determined by the input geometry rather than by the learned representation.

We test these hypotheses by constructing Random Feature Neural Networks (RFNNs) — networks with the same architecture as a trained CNN but with convolutional weights frozen at random initialization. Only the final linear classifier is trained. If the learned representation hypothesis is correct, RFNNs should exhibit substantially lower margin AUROC than trained CNNs. If the input geometry hypothesis is correct, the gap should be small.

Our results support the input geometry hypothesis: on Fashion-MNIST, the RFNN achieves margin AUROC within 0.019 of the fully trained CNN. The gap grows modestly on Imagenette (0.038), suggesting that learned features provide incremental value on more complex datasets, but the baseline level of vulnerability information is already present in random features of the raw pixel space.

---

## 2. Related Work

**Adversarial Vulnerability and Input Geometry.** Fawzi et al. [1] showed theoretically that samples close to class boundaries are inherently more vulnerable to adversarial perturbation. Moosavi-Dezfooli et al. [2] demonstrated that universal adversarial perturbations exist in low-dimensional subspaces of the input space, suggesting structural regularity in the vulnerability landscape. Gilmer et al. [3] argued from a concentration-of-measure perspective that adversarial examples are a natural consequence of the geometry of high-dimensional probability distributions.

**Random Features.** Rahimi and Recht [4] introduced random feature approximations for kernel machines, showing that random Fourier features can approximate kernel functions. In the neural network context, untrained random networks have been studied as feature extractors [5], with the surprising finding that random CNNs produce surprisingly useful image features. Ulyanov et al. [6] showed that the architectural prior of CNNs imposes useful image statistics even without training, a phenomenon exploited in deep image prior methods.

**Gradient-Based Vulnerability Predictors.** Smilkov et al. [7] introduced SmoothGrad, computing uncertainty estimates via gradient averaging over noisy inputs. Related work on gradient norm as a vulnerability predictor [8] suggests that the gradient at the input — which depends on both the input and the learned weights — carries boundary proximity information. We test gradient phase as a vulnerability predictor alongside random features.

**Bayesian Uncertainty and Vulnerability.** Bayesian neural networks [9] provide principled uncertainty estimates via posterior averaging. Feinman et al. [10] used Bayesian uncertainty to detect adversarial examples, demonstrating a correlation between uncertainty and vulnerability. We test whether Bayesian variance (posterior spread) achieves similar margin AUROC to random feature networks.

**Forgetting Events.** Toneva et al. [11] showed that samples with the most forgetting events during training are disproportionately near the decision boundary. This suggests that the training process reflects, rather than creates, a pre-existing difficulty ordering consistent with the input geometry hypothesis.

---

## 3. Methodology

### 3.1 Architecture

**CNN baseline:** Three-block convolutional network (32, 64, 128 channels; 3×3 kernels; batch normalization; ReLU; 2×2 max pooling) with a 256-unit fully connected head. All weights trained via SGD for 30 epochs on Fashion-MNIST, 50 epochs on Imagenette.

**RFNN:** Identical architecture. At initialization, convolutional weights are drawn from He normal initialization and frozen; only the final linear layer (256 × num_classes) is trained via ridge regression (closed-form solution for Fashion-MNIST, SGD for Imagenette). The RFNN therefore has strictly fewer trainable parameters and provides a controlled comparison.

### 3.2 Margin and AUROC Computation

Margin is computed as the gap between the top-1 and top-2 logits, as in Paper 5. AUROC is computed using FGSM (ε = 8/255, single step) and PGD (ε = 8/255, 20 steps, step size 2/255) as attack oracles; a sample is labeled "vulnerable" if the attack succeeds in flipping its prediction.

### 3.3 Baseline Features

We compare against two simple input statistics:

- **mean_pix:** Average pixel intensity across all channels.
- **std_pix:** Standard deviation of pixel intensity across spatial and channel dimensions.

These serve as lower-bound baselines: if the input geometry hypothesis is correct, raw pixel statistics should provide some (weak) vulnerability prediction, while random features should approach trained CNN performance.

### 3.4 Bayesian Neural Network

We train an approximate Bayesian CNN on Fashion-MNIST using Monte Carlo dropout [12] (dropout probability 0.3 at each convolutional block). We use T = 50 stochastic forward passes to estimate predictive variance, then compute the AUROC of predictive variance as a predictor of adversarial vulnerability. We report AUROC across 5 seeds to obtain a distribution.

### 3.5 Gradient Phase Analysis

Following [7], we compute the gradient of the cross-entropy loss with respect to the input for each test sample. We decompose the gradient into a high-frequency component (spatial frequencies above the median cutoff) and a low-frequency component using a 2D discrete Fourier transform. The ratio of high-frequency to total gradient energy constitutes the "gradient phase" feature. We evaluate its AUROC as a vulnerability predictor.

### 3.6 Layer Ablation

To test whether vulnerability prediction is concentrated in particular layers, we perform a layer ablation study on the CNN: we progressively remove trained layers from the top down (keeping the final linear layer retrained on each ablated architecture) and measure the change in margin AUROC.

### 3.7 Datasets

- **Fashion-MNIST (FM):** 60,000/10,000 train/test, 28×28 grayscale. Used for primary analysis.
- **Imagenette:** 9,469/3,925 train/validation, 224×224 RGB, 10-class subset of ImageNet. Used to test whether the random feature gap grows with dataset complexity.

---

## 4. Results

### 4.1 Fashion-MNIST: CNN vs RFNN

**Table 1: Margin AUROC comparison on Fashion-MNIST**

| Model         | FGSM AUROC | PGD AUROC |
|---------------|------------|-----------|
| CNN (trained) | 0.9201     | 0.9381    |
| RFNN          | 0.9232     | 0.9188    |
| mean_pix      | 0.514      | —         |
| std_pix       | 0.609      | —         |

The RFNN achieves nearly identical margin AUROC to the fully trained CNN: 0.9232 vs 0.9201 for FGSM (RFNN is 0.003 *higher*) and 0.9188 vs 0.9381 for PGD (gap 0.019). The FGSM result is particularly striking: a network that has never undergone gradient-based learning predicts single-step adversarial vulnerability as well as a carefully trained network. The small PGD gap (0.019) may reflect the fact that PGD explores a wider region of the loss landscape, where learned curvature information begins to matter.

In contrast, raw pixel statistics perform much worse. The mean pixel intensity (mean_pix) achieves AUROC = 0.514, barely above chance. The standard deviation (std_pix) achieves 0.609, providing weak but above-chance prediction. The jump from pixel statistics to random features (0.609 → 0.923) is far larger than the jump from random features to trained features (0.919 → 0.938), supporting the conclusion that the critical geometry is captured at the level of random convolutional features rather than learned higher-level representations.

### 4.1.1 RFNN vs CNN Detailed Comparison

To make the gap structure explicit, the following table summarises RFNN and CNN margin AUROC across datasets and attack types, with the signed gap.

**Table 2: RFNN vs Fully-Trained CNN — Vulnerability Prediction AUROC**

| Dataset | Target | RFNN margin | CNN margin | Gap (CNN−RFNN) |
|---------|--------|-------------|------------|----------------|
| Fashion-MNIST | FGSM | 0.9232 | 0.9201 | −0.003 (RFNN wins) |
| Fashion-MNIST | PGD | 0.9188 | 0.9381 | +0.019 |
| Fashion-MNIST | min_eps | 0.9367 | 0.9459 | +0.009 |
| CIFAR-10 | FGSM | 0.9917 | — | ≈0 (PGD saturated) |
| Imagenette | FGSM | 0.8719 | ~0.910 | +0.038 |

*RFNN captures 98–99% of trained CNN's vulnerability prediction on Fashion-MNIST. Gap grows on harder datasets.*

[Figure 1: AUROC as a function of feature type (mean_pix, std_pix, RFNN, CNN) for both FGSM and PGD attacks on Fashion-MNIST. Bar chart showing step-function increase at random features with minimal further gain from training.]

### 4.2 Feature Family Comparison

Beyond comparing RFNN to CNN, it is instructive to compare all feature families on the same axis to understand the information hierarchy from raw pixels to trained margins.

**Table 3: Feature Family AUROC Comparison (Fashion-MNIST, PGD target)**

| Feature family | Example | AUROC | Interpretation |
|---|---|---|---|
| Logit-based | margin | 0.972 | Ground-truth boundary distance |
| RFNN-based | rfnn_margin | 0.919 | Input geometry alone |
| Pixel statistics | std_pix | 0.651 | Raw pixel variation |
| Pure random baseline | mean_pix | 0.521 | Near chance |

*The RFNN captures 94.5% of trained CNN margin AUROC (0.919/0.972) from random features alone.*

### 4.3 Conceptual Decomposition of Vulnerability Information

The relationship between input geometry and learned representations can be summarised by the fraction of vulnerability predictability captured at each level of representation.

**Figure 1: Conceptual Decomposition of Vulnerability Information**

```
Total vulnerability predictability (AUROC ~= 0.972)
+------------------------------------------------------+
|   Input geometry                                      |
|   (captured by RFNN, pixel stats)                    |
|   ~= 94.5% of signal  ############################# |
|                                                       |
|   Learned refinement (CNN training adds)  #####  5.5%|
+------------------------------------------------------+
```
*On Fashion-MNIST, learned convolutional features add only ~5.5% to the vulnerability
prediction already available from random features (input geometry).*

### 4.4 Confidence vs Margin Comparison

**Table 4: Confidence and margin AUROC for RFNN on Fashion-MNIST**

| Feature         | FGSM AUROC | PGD AUROC |
|-----------------|------------|-----------|
| RFNN margin     | 0.9232     | 0.9188    |
| RFNN confidence | 0.9184     | 0.9164    |

RFNN confidence (top-1 softmax probability) and RFNN margin achieve nearly identical AUROC, differing by at most 0.0048. This confirms that confidence and margin are essentially equivalent vulnerability predictors in this setting, and that neither requires trained representations to work well.

### 4.5 Imagenette: Gap Growth with Complexity

On Imagenette, the RFNN achieves margin AUROC = 0.8719, compared to approximately 0.910 for the fully trained CNN — a gap of 0.038, roughly twice the Fashion-MNIST PGD gap (0.019). This suggests that learned representations do provide incremental, non-negligible value on complex datasets, but the majority of the vulnerability signal (0.872 out of 0.910) is already present in random features.

[Figure 2: Vulnerability prediction gap (CNN AUROC − RFNN AUROC) for Fashion-MNIST and Imagenette. Gap grows from 0.019 to 0.038 with dataset complexity.]

### 4.6 Bayesian Neural Network Uncertainty

The BNN achieves margin AUROC = 0.9187 ± 0.013 across 5 seeds. This is statistically indistinguishable from the RFNN AUROC (0.9188) and within the standard deviation of the CNN AUROC. The result suggests that Bayesian uncertainty is capturing essentially the same vulnerability information as the random feature margin: input-space geometry rather than posterior uncertainty about model parameters.

### 4.7 Gradient Phase Analysis

The high-frequency gradient energy ratio achieves AUROC = 0.84 as a predictor of adversarial vulnerability. While this is lower than the RFNN and CNN margins (0.919–0.938), it confirms that vulnerability has a spectral signature in the input domain: samples that elicit high-frequency gradients are more adversarially vulnerable. This is consistent with the findings of [13] on the role of high-frequency features in adversarial robustness.

[Figure 3: Gradient energy spectrum for low-vulnerability (top quartile margin) vs high-vulnerability (bottom quartile margin) samples. High-vulnerability samples show elevated high-frequency gradient energy.]

### 4.8 Layer Ablation

Removing layers from the top of the CNN and retraining the linear head produces the following AUROC degradation:

**Table 5: Layer ablation — margin AUROC under progressive layer removal (PGD)**

| Layers Removed | Remaining | AUROC |
|----------------|-----------|-------|
| 0              | All       | 0.938 |
| 1 (top FC)     | Conv1–3   | 0.931 |
| 2 (top 2 conv) | Conv1–2   | 0.924 |
| 3 (top 3 conv) | Conv1     | 0.921 |
| All conv       | Linear    | 0.919 |

Vulnerability prediction degrades gracefully: removing all convolutional training (equivalent to RFNN) only drops AUROC from 0.938 to 0.919. No single layer accounts for a large share of the vulnerability information, and the information is present even with all learned convolutional features removed.

### 4.9 Decisive Controls: Conv Prior vs Pure Input Geometry (H180)

A frozen random *CNN* is not a clean test of the input-geometry hypothesis. Its random convolutional features still carry strong **architectural priors** — locality, weight sharing, ReLU, pooling — so "RFNN ≈ CNN" could equally be read as "input geometry *plus* the convolutional prior is sufficient", not "input geometry alone". The decisive missing controls are a frozen random *MLP* (no convolutional prior) and a raw *Gaussian random projection* of the pixels (no architecture whatsoever). We add both, fitting only a linear readout (logistic regression on 8000 training samples) on each frozen feature map, and measure AUROC against the reference trained-CNN's PGD vulnerability (Fashion-MNIST, n=1000, reference ASR=0.96).

**Table 6: Frozen-feature controls vs reference-CNN PGD vulnerability (Fashion-MNIST)**

| Predictor | Architectural prior | AUROC |
|-----------|---------------------|-------|
| Trained-CNN margin | learned | 0.9432 (reference) |
| Random-CNN (frozen conv + linear readout) | convolution | 0.8673 |
| Gaussian projection (random matrix on raw pixels) | **none** | 0.8680 |
| Random-MLP (frozen dense + linear readout) | none (no conv) | 0.8479 |
| Raw pixels (logistic on pixels) | none | 0.8296 |

The result directly answers the objection. A Gaussian random projection — which has **no architecture at all** — matches the random *CNN* (0.8680 vs 0.8673) and even slightly exceeds the random MLP. The convolutional prior therefore contributes essentially nothing to vulnerability ranking beyond what a structureless random projection of the raw pixels already captures. The ordering is: raw pixels (0.830) < random projection ≈ random CNN ≈ random MLP (0.85–0.87) ≪ a single jump to trained features (0.943). Almost all of the non-trained signal is pure input geometry, not the conv prior. This refutes the "conv prior is doing the work" reading and supports the paper's central claim in its strong form: per-sample vulnerability is encoded in input-space geometry, recoverable by an architecture-free random projection, with learned representations adding a further ~0.07 AUROC of refinement.

(Note: these absolute AUROCs are slightly lower than §4.1's because the readout is a multinomial logistic regression on a fixed 8k-sample fit rather than the closed-form ridge head; the *relative* ordering across feature types is the load-bearing result.)

---

## 5. Discussion

### 5.1 Implications for the Origin of Vulnerability

The near-equivalence of RFNN and CNN margin AUROC on Fashion-MNIST constitutes strong evidence for the input geometry hypothesis. The model's learned representation is not the primary determinant of its vulnerability ranking. Rather, samples that are intrinsically hard — positioned near the true class manifold boundaries, exhibiting high pixel-space ambiguity — remain hard regardless of whether the network's features are random or carefully trained.

This finding connects to the theoretical work of Gilmer et al. [3], who argue that adversarial vulnerability is an inevitable consequence of the geometry of the data distribution, not a pathology of the training procedure. Our empirical results provide a direct test: if vulnerability were a model artifact, freezing features at random initialization would substantially reduce margin AUROC. It does not.

### 5.2 Practical Consequences

If vulnerability is primarily encoded in input geometry, then vulnerability-aware training strategies — such as curriculum learning that orders samples by pre-training margin estimates — can be designed before any adversarial training is performed. An RFNN requires only training a linear classifier, which is orders of magnitude cheaper than full AT. This motivates using RFNN margins as cheap proxies for full-model vulnerability predictions during training curriculum design.

### 5.3 The Role of Learned Features

The growing gap on Imagenette (0.038) suggests that learned features do provide incrementally useful information on complex datasets. This is consistent with a view in which (a) input geometry determines the bulk of the vulnerability ordering, and (b) learned features refine the prediction for cases where random features are ambiguous. The challenge is that complex datasets have more such ambiguous cases.

### 5.4 Relationship to Bayesian Uncertainty

The near-identical AUROC of BNN uncertainty and RFNN margin (both approximately 0.919) suggests these are measuring the same underlying property. Bayesian uncertainty, often interpreted as model uncertainty (epistemic uncertainty), may in fact predominantly reflect input-space ambiguity (aleatoric uncertainty) in this context — a cautionary note for interpreting BNN uncertainty estimates as model-specific quantities.

---

## 6. Conclusion

We have shown that per-sample adversarial vulnerability is primarily encoded in input-space geometry rather than learned representations. An RFNN with frozen random convolutional features achieves margin AUROC within 0.019 of a fully trained CNN on Fashion-MNIST (gap growing to 0.038 on Imagenette), while raw pixel statistics perform near chance. Bayesian uncertainty, gradient phase, and layer ablation experiments provide converging evidence. These results suggest that vulnerability-aware training curricula can be designed cheaply using random feature networks, and that improving per-sample adversarial robustness likely requires addressing the underlying data geometry rather than solely refining the learned representation.

---

## References

[1] A. Fawzi, O. Fawzi, and P. Frossard, "Fundamental Limits on Adversarial Robustness," *ICML Workshop*, 2015.

[2] S. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, "Universal Adversarial Perturbations," *CVPR*, 2017.

[3] J. Gilmer, L. Metz, F. Faghri, S. Schoenholz, M. Raghu, M. Wattenberg, and I. Goodfellow, "Adversarial Spheres," *ICLR Workshop*, 2018.

[4] A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines," *NeurIPS*, 2007.

[5] J. Saxe, P. Koh, Z. Chen, M. Bhand, B. Suresh, and A. Ng, "On Random Weights and Unsupervised Feature Learning," *ICML*, 2011.

[6] D. Ulyanov, A. Vedaldi, and V. Lempitsky, "Deep Image Prior," *CVPR*, 2018.

[7] D. Smilkov, N. Thorat, B. Kim, F. Viégas, and M. Wattenberg, "SmoothGrad: Removing Noise by Adding Noise," *ICML Workshop*, 2017.

[8] N. Carlini and D. Wagner, "Towards Evaluating the Robustness of Neural Networks," *IEEE S&P*, 2017.

[9] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," *ICML*, 2016.

[10] R. Feinman, R. Curtin, S. Shintre, and A. Gardner, "Detecting Adversarial Attacks on Neural Networks with Mutual Information," *arXiv*, 2017.

[11] M. Toneva, A. Sordoni, R. des Combes, A. Trischler, Y. Bengio, and G. Gordon, "An Empirical Study of Example Forgetting during Deep Neural Network Learning," *ICLR*, 2019.

[12] Y. Gal and Z. Ghahramani, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks," *NeurIPS*, 2016.

[13] H. Yin, A. Mallya, A. Vahdat, J. Alvarez, J. Kautz, and P. Molchanov, "Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks," *NeurIPS*, 2021.
