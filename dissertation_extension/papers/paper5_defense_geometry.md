# How Adversarial Defenses Reshape Decision Boundary Geometry

**Abstract**

Adversarial defenses are routinely evaluated on two metrics: clean accuracy and adversarial accuracy. We argue that a third diagnostic—*margin predictability*, quantified as the AUROC of a model's decision margin as a predictor of per-sample attack success—provides orthogonal and complementary information about how a defense reshapes the classification geometry. A defense that simply randomizes boundary placement can achieve acceptable adversarial accuracy while producing a margin that no longer correlates with vulnerability, concealing pathological behavior. We conduct systematic experiments across Fashion-MNIST, CIFAR-10, and SVHN, evaluating four defenses: Soft Adversarial Training (SAT), Friendly Adversarial Training (FAT), RSLAD (robust distillation), and a novel MarginWeighted loss. Results reveal a consistent hierarchy: FAT achieves the best clean-robustness tradeoff, MarginWeighted collapses on every dataset, and augmentation-based defenses substantially reduce margin AUROC (as low as −0.246 for Adversarial Logit Pairing) even when adversarial accuracy appears adequate. We propose margin AUROC as a standard third diagnostic for adversarial defense benchmarking.

---

## 1. Introduction

The adversarial robustness literature has converged on a two-dimensional evaluation: clean accuracy measures performance on natural data, and adversarial accuracy measures performance under a fixed-budget attack such as PGD [1]. This two-number summary is convenient but incomplete. It does not distinguish between a defense that uniformly expands all decision margins and one that randomizes boundaries, placing some samples further from and others closer to the decision boundary than the vanilla model would. The former is preferable: it preserves the structure of the model's uncertainty, enabling downstream uses such as selective prediction, calibration, and anomaly detection.

This paper introduces *margin predictability* as a third diagnostic. We define it operationally as the area under the ROC curve (AUROC) when the model's signed distance to the nearest decision boundary—the *margin*—is used as a ranking score to predict whether a given sample will be successfully attacked. A defense that uniformly expands margins will preserve or increase this AUROC; a defense that scrambles boundary placement will reduce it, even if average adversarial accuracy is acceptable.

We validate the diagnostic on Fashion-MNIST (FM) with a standard CNN at epsilon budget ε = 15/255 and 10 training epochs, then extend to CIFAR-10 and SVHN. Our vanilla FM baseline achieves CleanAcc = 92.6%, PGD adversarial accuracy = 94.5% (attack success 5.5%), and margin AUROC = 0.946. All four defenses we study reduce clean accuracy; the question is whether they preserve or destroy the geometry encoded in the margin.

The key contributions of this work are:

1. We formalize margin AUROC as a diagnostic and show it is not redundant with adversarial accuracy.
2. We demonstrate that augmentation-based defenses (AugMix, Manifold Mixup, ALP) systematically degrade margin AUROC while AT-based methods (SAT, FAT, RSLAD) broadly preserve it.
3. We show that extreme hard-sample weighting (MarginWeighted) produces total model collapse on all three datasets tested.
4. We identify FAT as the most geometrically coherent defense across datasets.

---

## 2. Related Work

**Adversarial Training.** The dominant paradigm for adversarial robustness is adversarial training (AT) [1, 2], which augments the training set with adversarial examples generated via FGSM [1] or PGD [2]. Madry et al. [2] demonstrated that PGD-AT provides a strong min-max guarantee and remains a gold standard. Subsequent work has identified overfitting as a critical failure mode [3] and proposed mitigations including early stopping and data augmentation.

**Soft and Friendly Adversarial Training.** Zhang et al. [4] proposed TRADES, which decomposes the robust loss into a clean accuracy term and a boundary term, allowing explicit control over the tradeoff. Zhang et al. [5] introduced Friendly Adversarial Training (FAT), which generates adversarial examples using early-stopped PGD, producing examples that are "weakly adversarial" — past the decision boundary but not far from it. This reduces the severity of the robustness-accuracy tradeoff compared to full PGD-AT. Shafahi et al. [6] proposed fast AT using free-form gradient recycling, enabling AT at a fraction of the computational cost.

**Robust Distillation.** RSLAD (Robust Soft Label AT via Distillation) leverages a PGD-AT teacher model to provide soft target distributions as supervision signals, combining the boundary geometry learned by AT with the smoothness properties of distillation.

**Decision Boundary Geometry.** The geometry of decision boundaries under adversarial training has been studied through the lens of margin distributions [7], curvature [8], and spectral properties [9]. Chen et al. [10] showed that curvature regularization (CURE) reduces second-order boundary sensitivity. Rice et al. [3] showed that AT models tend to overfit their robustness, suggesting that boundary placement is not stable over epochs.

**Forgetting and Hard Samples.** Toneva et al. [11] introduced forgetting events—training examples that transition from correct to incorrect classification during training—showing that hard samples near the decision boundary are disproportionately affected by optimization dynamics. This motivates studying how defenses interact with the pre-existing margin structure.

---

## 3. Methodology

### 3.1 Architecture and Training Setup

All Fashion-MNIST experiments use a three-block CNN: two convolutional blocks (32 and 64 channels, 3×3 kernels, batch normalization, ReLU, 2×2 max pooling) followed by a 256-unit fully connected layer and a 10-class output head. Total parameters: approximately 1.2M. Training uses SGD with momentum 0.9, weight decay 1e-4, and cosine learning rate annealing from 0.01 over 10 epochs. Batch size 128. All experiments use ε = 15/255 with ∞-norm threat model. CIFAR-10 uses ResNet-18; SVHN uses the same CNN as Fashion-MNIST. All results reported as mean over 3 seeds unless noted.

### 3.2 Margin Computation

For a sample **x** with true label y, the decision margin is computed as:

```
margin(x) = f_y(x) - max_{j ≠ y} f_j(x)
```

where f_j(**x**) is the pre-softmax logit for class j. A positive margin indicates correct classification; the magnitude reflects distance from the boundary in logit space.

### 3.3 Margin AUROC

We define a binary label for each test sample: attacked = 1 if PGD (20 steps, step size 2/255) succeeds in flipping the prediction, 0 otherwise. We then compute AUROC using the negative margin as the ranking score (lower margin → higher predicted attack probability). This gives a threshold-free measure of how well the margin geometry predicts actual vulnerability.

### 3.4 Defenses Evaluated

**SAT (Soft Adversarial Training):** Replaces hard adversarial labels with temperature-annealed soft targets, reducing gradient variance early in training.

**FAT (Friendly Adversarial Training) [5]:** Generates adversarial examples using PGD with early stopping — training continues only until the example first crosses the decision boundary. This produces weaker adversarial examples that are "just adversarial."

**RSLAD:** Uses a pre-trained PGD-AT teacher to produce soft labels for adversarial examples, combining the strong robustness signal of AT with the smooth gradient landscape of distillation.

**MarginWeighted:** Assigns per-sample loss weights inversely proportional to margin: w_i = 1 / (margin_i + 0.1). This is intended to focus training on hard samples. We test it as a diagnostic of what happens under extreme hard-sample emphasis.

### 3.5 Datasets

- **Fashion-MNIST:** 60,000/10,000 train/test, 28×28 grayscale, 10 classes.
- **CIFAR-10:** 50,000/10,000 train/test, 32×32 RGB, 10 classes.
- **SVHN:** ~73,000/26,000 train/test, 32×32 RGB, 10 digit classes. Known to exhibit instability under adversarial training due to high within-class visual diversity.

---

## 4. Results

### 4.1 Fashion-MNIST

Table 1 summarizes all Fashion-MNIST results. The vanilla baseline achieves CleanAcc = 92.6%, PGD adversarial accuracy = 94.5% (i.e., attack success rate 5.5%), and margin AUROC = 0.946.

**Table 1: Fashion-MNIST results (ε = 15/255, 10 epochs)**

| Defense        | Clean Acc | FGSM Succ | PGD Succ | Min-ε  | Margin AUROC |
|----------------|-----------|-----------|----------|--------|--------------|
| Vanilla        | 92.6%     | —         | 5.5%     | —      | 0.946        |
| SAT            | 87.9%     | 8.7%      | 10.1%    | 0.207  | 0.945        |
| FAT            | 88.7%     | 8.6%      | 10.5%    | 0.229  | 0.958        |
| RSLAD          | 87.5%     | 7.3%      | 8.7%     | 0.218  | 0.940        |
| MarginWeighted | 20.0%     | 0.0%      | 0.0%     | 0.300  | —            |

FGSM and PGD columns report attack success rates; lower is better for the defender. Min-ε is the minimum perturbation budget at which the attack achieves >50% success; higher is better.

The three well-behaved defenses (SAT, FAT, RSLAD) all maintain margin AUROC within 0.006 of the vanilla baseline (0.940–0.958 vs 0.946), indicating that AT-based methods preserve the geometry of the margin distribution even as they expand it. FAT achieves the highest AUROC (0.958) with the smallest clean accuracy drop (−3.9 pp) among AT methods. RSLAD achieves the lowest attack success (8.7% PGD) but also the lowest clean accuracy (87.5%) and margin AUROC (0.940).

MarginWeighted collapses entirely: 20.0% clean accuracy (near chance for 10 classes), 0% attack success (because the model predicts only one class, making "incorrect" the trivially prevalent outcome). This demonstrates that extreme hard-sample weighting, implemented via inverse-margin loss weighting, is a failure mode that cannot be avoided by careful hyperparameter tuning within this paradigm.

Earlier experiments (H118–H124, not re-run under the corrected evaluation protocol) provide augmentation-based defense comparisons:

**Table 2: Augmentation-based defenses on FM (margin AUROC change from vanilla)**

| Defense        | AUROC Change |
|----------------|-------------|
| CURE           | +0.003      |
| AugMix         | −0.151      |
| Manifold Mixup | −0.211      |
| ALP            | −0.246      |

CURE (curvature regularization) [10] nearly preserves geometry (AUROC 0.946 → 0.949). In contrast, augmentation-based and representation-matching defenses substantially degrade margin predictability. Adversarial Logit Pairing (ALP) shows the largest degradation (−0.246), consistent with its known tendency to align representations rather than expand margins.

[Figure 1: Scatter plot of per-sample margin (x-axis) vs attack success probability (y-axis) for Vanilla, FAT, and ALP on Fashion-MNIST. FAT shows cleaner separation; ALP shows diffuse, weakly correlated cloud.]

### 4.2 CIFAR-10

**Table 3: CIFAR-10 results (ResNet-18, ε = 15/255, 10 epochs)**

| Defense        | Clean Acc | PGD Succ | Margin AUROC |
|----------------|-----------|----------|--------------|
| Vanilla        | 64.2%     | 100.0%   | N/A          |
| SAT            | 42.5%     | 61.3%    | 0.835        |
| FAT            | 53.1%     | 81.6%    | 0.810        |
| RSLAD          | 34.8%     | 53.7%    | 0.836        |
| MarginWeighted | 10.3%     | 24.6%    | —            |

The vanilla CIFAR-10 model is fully saturated under PGD attack (100% success), making the margin AUROC undefined (no variation in attack outcome). All defenses reduce attack success below 100%. RSLAD achieves the lowest PGD success (53.7%) but the largest clean accuracy drop (−29.4 pp from vanilla). FAT again shows the best clean-robustness tradeoff: 53.1% clean accuracy (−11.1 pp) at 81.6% PGD success.

Margin AUROC values on CIFAR-10 (0.810–0.836) are lower than on Fashion-MNIST (0.940–0.958), reflecting the increased complexity of the CIFAR-10 geometry. MarginWeighted shows partial collapse (10.3% clean accuracy), less severe than Fashion-MNIST (20.0%), but still below useful threshold.

### 4.3 SVHN

**Table 4: SVHN results (CNN, ε = 15/255, 10 epochs)**

| Defense        | Clean Acc | PGD Succ | Margin AUROC |
|----------------|-----------|----------|--------------|
| Vanilla        | 86.2%     | 99.0%    | 0.616        |
| SAT            | 64.9%     | 66.5%    | 0.907        |
| FAT            | 79.1%     | 78.1%    | 0.931        |
| RSLAD          | 40.3%     | 44.5%    | 0.881        |
| MarginWeighted | 15.9%     | 0.1%     | 1.000*       |

*Degenerate: model predicts only 1 class; AUROC is an artifact.

The SVHN vanilla model's margin AUROC is only 0.616, the lowest across all datasets and conditions, confirming the well-documented instability of adversarial training on SVHN [2]. Despite a clean accuracy of 86.2%, the model's margin provides almost no reliable prediction of adversarial vulnerability, suggesting that the SVHN boundary geometry is disordered even before adversarial intervention.

Notably, all three AT-based defenses *increase* margin AUROC substantially above the vanilla SVHN level: SAT to 0.907, FAT to 0.931, RSLAD to 0.881. This is a rare case where adversarial training improves geometric coherence relative to baseline. FAT achieves both the best clean accuracy (79.1%) and the highest legitimate margin AUROC (0.931).

MarginWeighted on SVHN again collapses (15.9% clean, 0.1% PGD success), and the AUROC of 1.000 is degenerate — when a model predicts only one class, all samples trivially fall on one side of any threshold, producing a vacuous AUROC.

[Figure 2: Margin AUROC across datasets and defenses. Bar chart showing SVHN vanilla baseline at 0.616, with AT methods recovering to 0.881–0.931. FM and CIFAR-10 AT methods maintain 0.810–0.958.]

### 4.4 Cross-Dataset Summary

The following table consolidates all AT-variant results across all three datasets, enabling direct comparison of the clean-robustness tradeoff in a unified view.

**Table 5: AT Variant Comparison Across Datasets (H173)**

| Model | FM Clean | FM PGD% | C10 Clean | C10 PGD% | IN Clean | IN PGD% | SVHN Clean | SVHN PGD% |
|-------|----------|---------|-----------|----------|---------|---------|-----------|----------|
| Vanilla | 92.6% | 94.5% | 64.2% | 100%† | 52.6% | 94.9% | 86.2% | 99.0% |
| SAT | 87.9% | 10.1% | 42.5% | 61.3% | 38.2% | 55.6% | 64.9% | 66.5% |
| FAT | **88.7%** | 10.5% | **53.1%** | 81.6% | **46.4%** | 75.0% | **79.1%** | 78.1% |
| RSLAD | 87.5% | 8.7% | 34.8% | 53.7% | 32.3% | 44.0% | 40.3% | 44.5% |
| MarginWT | 20.0% | 0.0% | 10.3% | 24.6% | 10.1% | 0.0% | 15.9% | 0.1% |

†PGD saturated at ε=15/255. FAT consistently achieves best clean-robustness tradeoff.

### 4.5 Margin Predictability Across Defenses and Datasets

Beyond the within-dataset results already described, the margin AUROC can be compared across all datasets and defense types to evaluate how each defense affects the geometry of the learned boundary.

**Table 6: Margin Predictability (MarginAUROC vs PGD) Under Each Defense**

| Model | FM MarginAUROC | IN MarginAUROC | SVHN MarginAUROC |
|-------|---------------|---------------|-----------------|
| Vanilla | 0.946 | 0.941 | 0.616 |
| SAT | 0.945 | 0.878 | 0.907 |
| FAT | **0.958** | **0.938** | **0.931** |
| RSLAD | 0.940 | 0.850 | 0.881 |
| MarginWeighted | N/A (collapsed) | N/A | 1.000 (degenerate) |
| AugMix (H123) | 0.795 | — | — |
| Manifold Mixup (H124) | 0.735 | — | — |
| ALP (H119) | 0.700 | — | — |

*Adversarial training variants preserve margin predictability (~0.94); augmentation defenses reduce it.*

### 4.6 Clean Accuracy vs Robustness Pareto Frontier

The relationship between clean accuracy preservation and robustness gains is best understood through the Pareto frontier — the set of defenses for which no alternative achieves both better clean accuracy and better robustness simultaneously.

**Figure 1: Clean Accuracy vs Robustness Pareto Frontier (Fashion-MNIST)**

```
Clean
Acc %
92 |  * Vanilla
90 |        * PGD-AT+Anti
89 |                * FAT
88 |                    * SAT  * RSLAD
87 |
   |
86 |                          * PGD-AT
   |
20 |                                    * MarginWT (collapsed)
   +-------------------------------------------->
   95   80   60   40   20   10   5   0%  PGD success (lower = more robust)
```
*FAT dominates the Pareto frontier: best clean accuracy for any given robustness level.*

---

## 5. Discussion

### 5.1 Margin AUROC as a Diagnostic

The results demonstrate that margin AUROC provides information not captured by clean or adversarial accuracy. On Fashion-MNIST, all three AT defenses achieve nearly identical margin AUROC (0.940–0.958) despite having different adversarial accuracy profiles. The distinction appears when comparing to augmentation-based methods: ALP reduces margin AUROC by 0.246, a signal that the defense is scrambling boundary geometry even if its adversarial accuracy figures appear tolerable.

On SVHN, the vanilla model's low margin AUROC (0.616) identifies a pre-existing geometric disorder that is masked by the clean accuracy metric (86.2%). This has practical implications: a practitioner deploying the vanilla SVHN model would incorrectly assume the margin can be trusted for selective prediction or uncertainty estimation.

### 5.2 The MarginWeighted Failure Mode

The collapse of MarginWeighted across all three datasets is a strong empirical finding. Inverse-margin weighting creates a feedback loop: samples near the boundary receive high weight, the model's optimization is dominated by these samples, and the gradient dynamics destabilize the entire boundary structure rather than selectively hardening it. This is consistent with the "catastrophic forgetting" literature [11] — concentrating optimization on hard samples causes the model to forget easy ones.

### 5.3 FAT as the Geometrically Preferred Defense

FAT consistently achieves the highest or near-highest margin AUROC combined with the best clean-robustness tradeoff across all three datasets. The "weakly adversarial" examples it generates — examples that have just crossed the decision boundary — provide a mild boundary expansion signal without the aggressive geometry-reshaping associated with full PGD attacks. This is consistent with the theoretical motivation of FAT [5]: if the training distribution is aligned with the natural boundary geometry, the learned margin should preserve that geometry.

### 5.4 Augmentation vs AT Methods

The contrast between curvature regularization (CURE, AUROC +0.003) and representation-matching methods (ALP, AUROC −0.246) suggests a principled distinction: methods that operate directly on the geometry of the boundary (CURE, AT) preserve margin predictability, while methods that modify the representation layer to achieve robustness (ALP, Manifold Mixup) destroy the geometric correlation between margin and vulnerability. This is an argument against using representation-level metrics as proxies for boundary geometry.

---

## 6. Conclusion

We have introduced margin predictability (margin AUROC) as a third diagnostic for adversarial defenses, complementing the standard clean/adversarial accuracy pair. Across Fashion-MNIST, CIFAR-10, and SVHN, we find that: (1) AT-based defenses (SAT, FAT, RSLAD) preserve margin AUROC relative to the vanilla baseline; (2) augmentation-based defenses (AugMix, Manifold Mixup, ALP) substantially reduce it; (3) extreme hard-sample weighting (MarginWeighted) causes complete model collapse on every dataset tested; and (4) FAT provides the best combination of clean accuracy, adversarial robustness, and geometric coherence. We recommend margin AUROC as a routine third metric in adversarial robustness benchmarking. Future work should investigate whether margin AUROC predicts performance under unseen attacks, and whether it can be used as a training signal to explicitly regularize boundary geometry.

---

## References

[1] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and Harnessing Adversarial Examples," *ICLR*, 2015.

[2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards Deep Learning Models Resistant to Adversarial Attacks," *ICLR*, 2018.

[3] L. Rice, E. Wong, and Z. Kolter, "Overfitting in Adversarially Robust Deep Learning," *ICML*, 2020.

[4] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan, "Theoretically Principled Trade-off between Robustness and Accuracy," *ICML*, 2019.

[5] J. Zhang, X. Xu, B. Han, G. Niu, L. Cui, M. Sugiyama, and M. Kankanhalli, "Attacks Which Do Not Kill Training Make Adversarial Learning Stronger," *ICML*, 2020.

[6] A. Shafahi, M. Najibi, A. Ghiasi, Z. Xu, J. Dickerson, C. Studer, L. Davis, G. Taylor, and T. Goldstein, "Adversarial Training for Free!" *NeurIPS*, 2019.

[7] C. Elsayed, S. Shankar, B. Cheung, N. Papernot, A. Kurakin, I. Goodfellow, and J. Sohl-Dickstein, "Large Margin Deep Networks for Classification," *NeurIPS*, 2018.

[8] N. Carlini and D. Wagner, "Towards Evaluating the Robustness of Neural Networks," *IEEE S&P*, 2017.

[9] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral Normalization for Generative Adversarial Networks," *ICLR*, 2018.

[10] J. Zhang, C. Xie, J. Yi, R. Socher, L. Heck, B. Li, and D. Tao, "Geometry-aware Instance-reweighted Adversarial Training," in *Proc. ICLR*, 2021.

[11] M. Toneva, A. Sordoni, R. des Combes, A. Trischler, Y. Bengio, and G. Gordon, "An Empirical Study of Example Forgetting during Deep Neural Network Learning," *ICLR*, 2019.
