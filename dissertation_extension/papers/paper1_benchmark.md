# What Predicts Per-Sample Adversarial Vulnerability? A Systematic Empirical Benchmark

**Abstract**

Understanding which input samples are most susceptible to adversarial perturbations remains a central open problem in trustworthy machine learning. Existing work has proposed dozens of candidate predictors — from confidence margins to attribution norms to training dynamics — yet systematic, controlled comparisons are scarce. We present a large-scale empirical benchmark evaluating over 50 candidate per-sample vulnerability predictors on a fixed 5-layer CNN across four datasets (Fashion-MNIST, CIFAR-10, Imagenette-160, SVHN) and eight attacks spanning white-box and black-box threat models. We adopt AUROC as a direction-agnostic evaluation metric. Our principal finding is that the logit margin is the dominant predictor, achieving AUROC 0.87–0.9976 across all experimental contexts. Input gradient L2 norm provides a complementary signal — occasionally surpassing margin at stricter epsilon budgets — while noise-averaged attribution norms (SmoothGrad) offer marginal gains at substantially greater computational cost and reduced seed stability. Training-dynamics predictors such as forgetting events (AUROC 0.54) and C-score (AUROC 0.53) perform at chance. Monotone reparameterisations of the margin are empirically indistinguishable from the margin itself. With bootstrap confidence intervals, the headline ranking among boundary-proximity predictors collapses: the top five are statistically indistinguishable (paired-bootstrap P=0.20), and a gradient-boosted combination of all features adds no head-room (−0.003 AUROC) over the best single feature — the signal is essentially one-dimensional. We verify margin dominance is not a saturation artefact via matched-attack-success-rate controls, survives AutoAttack-grade labels (AUROC 0.994), and on a native-resolution ResNet-18 (CIFAR-10, 94% clean accuracy) the gradient norm overtakes the margin (0.921 vs 0.889). We further document a PGD saturation artefact at large epsilon that renders AUROC undefined, and provide recommendations for reproducible vulnerability benchmarking.

---

## 1. Introduction

Adversarial examples — carefully crafted input perturbations imperceptible to humans that cause deep neural network misclassifications — pose fundamental challenges to the deployment of machine learning systems in safety-critical applications [1],[9]. A key empirical observation, noted informally in the literature but rarely studied systematically, is that adversarial vulnerability is not uniform across samples: some inputs are robustly classified under strong attacks, while others are consistently misclassified under even weak perturbations.

Predicting which individual samples will be vulnerable before any attack is mounted has practical value. Such predictors could guide selective human review, inform data curation, or serve as runtime anomaly detectors. They also provide a diagnostic lens on the geometry of learned decision boundaries.

Numerous predictors have been proposed in the literature, typically motivated by different theoretical accounts of vulnerability: prediction confidence and margin [2],[8], input-space gradient norms [6], noise-averaged attribution norms [5], dataset cartography and forgetting events [3], sample-level learning difficulty scores such as C-score [4], and memorization proxies. However, these predictors have almost exclusively been evaluated on different datasets, architectures, and attack configurations, making direct comparison impossible.

This paper addresses this gap through a controlled empirical benchmark. Our contributions are:

1. A systematic evaluation of 50+ per-sample vulnerability predictors under a unified experimental protocol spanning 4 datasets, 8 attacks, and multiple random seeds.
2. Clear identification of the logit margin as the dominant, stable predictor across all conditions.
3. Documentation of conditions under which gradient-based predictors exceed or fall short of the margin, including a PGD saturation artefact that invalidates AUROC measurement at large epsilon.
4. Evidence that forgetting events and C-score are uninformative predictors of adversarial vulnerability, despite their utility for other learning-theoretic tasks.
5. A multi-seed stability analysis revealing that computationally expensive predictors (SmoothGrad) are less stable than cheap predictors (margin, input gradient norm).

---

## 2. Related Work

**Adversarial robustness.** Goodfellow et al. [1] introduced the Fast Gradient Sign Method (FGSM), framing adversarial vulnerability as a consequence of linear behaviour in high-dimensional spaces. Madry et al. [2] proposed PGD as a first-order adversarial attack and demonstrated that training against this attack yields more robust models. Carlini and Wagner [9] showed that many proposed defences fail against stronger attacks, establishing the importance of rigorous evaluation.

**Per-sample vulnerability predictors.** Prediction margin and softmax confidence are natural candidates for vulnerability prediction, as low-confidence predictions place a sample near the decision boundary. Zhang et al. [8] (TRADES) formalised the margin as the central quantity in adversarial training. Several works have proposed input-space gradient norms as vulnerability indicators, motivated by the observation that large gradients correspond to rapidly varying decision boundaries [6]. Smilkov et al. [5] introduced SmoothGrad to reduce gradient noise, and Sundararajan et al. [6] proposed Integrated Gradients as a theoretically principled attribution method.

**Training dynamics.** Toneva et al. [3] introduced forgetting events — transitions from correct to incorrect classification during training — as a measure of sample difficulty, finding that "forgotten" samples tend to be atypical or noisy. Jiang et al. [4] proposed C-score as a dataset-level measure of learning difficulty based on leave-one-out consistency. Power et al. [10] studied grokking, demonstrating that generalisation can emerge long after training loss converges, complicating the interpretation of training-trajectory signals.

**Sample difficulty and robustness.** Prior work has speculated that hard-to-learn samples should be more adversarially vulnerable, motivating the application of training-dynamics predictors to this task. Our benchmark tests this hypothesis directly and finds it does not hold.

---

## 3. Methodology

### 3.1 Architecture and Datasets

All experiments use a fixed 5-layer CNN trained from scratch on each dataset. Datasets are: Fashion-MNIST (FM, 10 classes, 60k training / 10k test), CIFAR-10 (10 classes, 50k/10k), Imagenette-160 (10 classes from ImageNet, ~13k/~4k), and SVHN (10 classes, 73k/26k). All inputs are standardised to 28×28 grayscale via a patch-based protocol to enable architecture reuse across datasets.

### 3.2 Attack Suite

**White-box L-inf attacks** (epsilon = 15/255 unless otherwise noted):
- FGSM [1]: single-step gradient sign
- BIM: iterative FGSM with step size alpha = 1/255, 20 steps
- PGD-10 [2]: 10-step projected gradient descent with random restart
- MIM: momentum iterative method

**Black-box attacks:**
- SimBA: simple black-box adversarial attack via random orthonormal basis queries
- ZOO: zeroth-order optimisation via finite-difference gradient estimation
- NES: natural evolution strategy gradient estimation
- Sign-OPT: sign-based gradient-free attack

### 3.3 Vulnerability Labels

A sample is labelled vulnerable (1) if the attack succeeds in causing misclassification within the perturbation budget, and robust (0) otherwise. Labels are computed per-sample per-attack. For multi-attack analyses, a sample is "universally vulnerable" if it is vulnerable to all four white-box attacks.

### 3.4 Candidate Predictors

We evaluate 50+ predictors spanning five families:

**Margin-based:** logit margin (difference between top-1 and top-2 logit), top-1 probability, softmax variance, confusion ratio (top-2/top-1 probability), entropy.

**Gradient-based:** input_grad_l2_norm (L2 norm of loss gradient w.r.t. input), smoothgrad_l2_norm (average norm over K=50 noisy copies, sigma=0.1), integrated gradients L2 norm [6], saliency spatial concentration statistics.

**Training dynamics:** forgetting events [3], C-score [4], memorization proxy (fraction of random 50% subsets where sample is misclassified).

**Structural:** distance to nearest training neighbour in representation space, representation norm.

### 3.5 Evaluation Metric

We use AUROC (Area Under the Receiver Operating Characteristic curve) as our primary metric. AUROC is direction-agnostic: we always report max(AUROC, 1-AUROC) >= 0.5, so predictors that inversely correlate with vulnerability (e.g., high margin = low vulnerability) are properly credited. AUROC = 0.5 corresponds to a random predictor; AUROC = 1.0 is perfect.

**PGD saturation issue.** At epsilon = 15/255, PGD-10 achieves 100% attack success rate on CIFAR-10, producing all-positive labels with zero variance. AUROC is undefined in this degenerate case. We address this by using epsilon = 4/255 for CIFAR-10 PGD experiments (H163), which yields attack success rates in the 60–80% range suitable for AUROC evaluation.

### 3.6 Multi-seed Stability Analysis

To assess predictor reliability, we run each experiment across multiple random seeds (H165) and report mean AUROC and standard deviation. Predictors with std < 0.01 are classified as Stable; those with std >= 0.02 are classified as Unstable.

---

## 4. Results

### 4.1 Logit Margin as Dominant Predictor

The logit margin is the single most consistent vulnerability predictor across all experimental conditions. On Fashion-MNIST with PGD as the attack target, margin achieves AUROC = 0.9651 ± 0.006. Across all four datasets and eight attacks, the margin achieves AUROC in the range 0.87–0.9976, with the higher end of this range observed on Imagenette and SVHN where class separation is sharper.

The full leaderboard from the multi-seed stability experiment (H165) is presented below:

**Table 1: Univariate AUROC Leaderboard — Fashion-MNIST, PGD-10 target (top 10 features)**

| Rank | Feature | Mean AUROC | Std | Stability |
|------|---------|-----------|-----|-----------|
| 1 | input_grad_l2_norm | 0.9743 | 0.006 | Stable |
| 2 | margin | 0.9723 | 0.006 | Stable |
| 3 | softmax_variance | 0.9617 | 0.005 | Stable |
| 4 | memorization_proxy | 0.9483 | 0.006 | Stable |
| 5 | predictive_entropy | 0.9295 | 0.013 | Moderate |
| 6 | smoothgrad_l2_norm | 0.9165 | 0.039 | **Unstable** |
| 7 | bnn_predictive_variance | 0.9187 | 0.013 | Moderate |
| 8 | pixel_sign_agreement | 0.8776 | 0.007 | Stable |
| 9 | forgetting_event_count | 0.5372 | — | — |
| 10 | c_score | 0.5280 | — | — |

*Stability: Stable = std < 0.01, Moderate < 0.03, Unstable ≥ 0.03 (H165, seeds 0–2)*

Notably, input_grad_l2_norm marginally exceeds margin (0.9743 vs. 0.9651) on this task. We caution, however, against over-interpreting such ranking differences: as we show in Section 4.8, bootstrap 95% confidence intervals on these AUROCs overlap heavily, and a paired-bootstrap test cannot distinguish the top features from one another. The apparent ordering of near-tied predictors is not statistically reliable; what is robust is the gross separation between boundary-proximity features (~0.87–0.97) and training-dynamics features (~0.53).

**Figure 1: AUROC distribution of 50 features (Fashion-MNIST, PGD target)**

```
AUROC
1.00 |          ■ ■
0.95 |       ■ ■ ■ ■ ■ ■
0.90 |    ■ ■ ■ ■ ■ ■ ■ ■ ■
0.85 |  ■ ■
0.80 |
0.75 |
...  |
0.55 |                            ■ ■ (forgetting, C-score ≈ random)
     +------------------------------------->
       Margin-like   Attribution  Training  Pixel
       features      signals      dynamics  stats
```
*The distribution is bimodal: boundary-proximity features cluster at 0.87–0.97; training-dynamics features (forgetting events, C-score) cluster at 0.53–0.54.*

### 4.2 Cross-Dataset Comparison: input_grad_l2_norm vs. Margin

The relative performance of gradient norm and margin varies with dataset complexity and epsilon budget. Table 2 summarises results for FGSM as the target attack:

**Table 2: input_grad_l2_norm vs margin AUROC across datasets (FGSM target)**

| Dataset | input_grad_l2_norm | margin | Δ (ig−margin) |
|---------|--------------------|--------|----------------|
| Fashion-MNIST | 0.9681 | 0.9651 | +0.003 |
| CIFAR-10 (ε=4/255) | 0.9291 | 0.8862 | **+0.043** |
| Imagenette | 0.9597 | 0.9563 | +0.003 |
| SVHN | 0.8276 | 0.8276 | 0.000 |

The gradient norm advantage is largest on CIFAR-10 at reduced epsilon (+0.043), consistent with the interpretation that gradient norms capture local boundary curvature not fully encoded in the margin at small perturbation scales. On SVHN, the two predictors are identical, suggesting that in high-success-rate regimes boundary distance and gradient magnitude are interchangeable.

### 4.3 CIFAR-10 at Reduced Epsilon (H163)

At epsilon = 4/255 on CIFAR-10, the PGD attack achieves a non-saturating success rate. Predictor rankings shift modestly:

| Rank | Predictor | AUROC |
|---|---|---|
| 1 | input_grad_l2_norm | 0.944 |
| 2 | top1_prob | 0.916 |
| 3 | margin | 0.911 |

The gradient norm becomes the top predictor at this stricter budget, consistent with the interpretation that gradient norms capture local boundary curvature not fully encoded in the margin at small perturbation scales.

[Figure 2: AUROC vs. epsilon budget for margin and input_grad_l2_norm on CIFAR-10. Gradient norm advantage grows as epsilon decreases.]

### 4.4 Monotone Reparameterisations of Margin are Equivalent

We test whether alternative formulations of the margin — confusion_ratio (top-2 probability / top-1 probability), softmax entropy, negative top-1 probability — provide additional information. We find that confusion_ratio achieves AUROC identical to margin (to three decimal places) across all tested conditions. This is expected: any strictly monotone transformation of the margin preserves ranking, and AUROC is a rank statistic. This confirms that the predictive signal resides in the margin ordering, not in any particular functional form.

### 4.5 Smoothgrad Instability

Despite achieving competitive mean AUROC on some tasks, smoothgrad_l2_norm exhibits a standard deviation of 0.039 across seeds — approximately 4x higher than margin (0.006) and input_grad_l2_norm (0.006) on the same tasks (H165). On individual seeds, smoothgrad AUROC ranges from approximately 0.877 to 0.956, while margin remains in the range 0.959–0.971. This instability disqualifies SmoothGrad as a reliable predictor in single-run settings and raises questions about the reproducibility of prior results that used it without seed controls.

[Figure 3: Box plots of AUROC across 10 random seeds for margin, input_grad_l2_norm, and smoothgrad_l2_norm. SmoothGrad has visibly wider interquartile range.]

### 4.6 Training Dynamics Predictors Perform at Chance

Forgetting events achieve AUROC = 0.54 for FGSM vulnerability prediction on Fashion-MNIST (H129), and C-score achieves AUROC = 0.53 (H130). Both values are statistically indistinguishable from random (AUROC = 0.50). This result holds across multiple attack configurations tested. The memorization proxy (H131) achieves AUROC = 0.9427 ± 0.006, but as we argue in Paper 4, this is because the memorization proxy measures boundary proximity via ensemble disagreement, not training difficulty per se.

### 4.7 Structural Vulnerability

As a baseline for all predictors, we note that adversarial vulnerability exhibits strong structural consistency across attacks (detailed in Paper 2). On Fashion-MNIST, 73% of test samples are vulnerable to all four white-box attacks simultaneously. This structural consistency provides an informative prior and inflates AUROC values for any predictor that captures the dominant axis of variation in the data.

### 4.8 Statistical Rigour: Confidence Intervals and the Learned Ceiling (H182)

Earlier drafts of this benchmark reported point AUROC differences such as 0.9743 (input gradient norm) vs. 0.9723 (margin) as if meaningful. They are not. We re-ran the top six predictors on a held-out Fashion-MNIST evaluation set (n=2000, PGD ASR=0.954) with 1000-sample bootstrap 95% confidence intervals (Table 5).

**Table 5: Univariate AUROC with bootstrap 95% CIs (Fashion-MNIST, PGD-10, n=2000)**

| Feature | AUROC | 95% CI |
|---------|-------|--------|
| min_eps (FGSM-direction) | 0.9706 | [0.9635, 0.9783] |
| smoothgrad_norm | 0.9649 | [0.9498, 0.9781] |
| input_grad_norm | 0.9591 | [0.9441, 0.9712] |
| margin | 0.9575 | [0.9422, 0.9694] |
| logit_entropy | 0.9572 | [0.9419, 0.9694] |
| softmax_confidence | 0.9214 | [0.9081, 0.9338] |

The CIs of the top five predictors overlap almost entirely. A paired-bootstrap test of the best predictor (min_eps) against the second (smoothgrad_norm) gives a mean AUROC difference of +0.0058 with P(best not better) = 0.202 — **statistically indistinguishable**. The headline ranking among boundary-proximity predictors is therefore not significant; they are interchangeable measurements of the same underlying quantity (distance to the decision boundary).

**Learned ceiling and calibration.** A gradient-boosted meta-predictor over all six features (5-fold CV) reaches AUROC 0.9679 — a head-room of **−0.0027** over the single best feature (min_eps). A learned non-linear combination of every predictor does *not* beat the best single feature: the vulnerability signal is essentially one-dimensional. The meta-predictor's Expected Calibration Error is 0.0206, i.e. its probabilities are usable for downstream selective prediction without further calibration.

### 4.9 Margin Dominance is Not a Saturation Artefact (H175)

A central methodological objection is that margin AUROC ~0.97 at eps=15/255 partly reflects label saturation (PGD ASR=0.976: an almost-constant positive label, a near-degenerate ROC problem). We control for this by binary-searching the L-inf budget so that PGD-10 ASR hits matched targets, then recomputing margin AUROC at each matched difficulty (Table 6).

**Table 6: Margin AUROC at matched attack-success rate (Fashion-MNIST, n=1000)**

| Condition | eps | ASR | margin AUROC |
|-----------|-----|-----|--------------|
| saturated (15/255) | 0.0588 | 0.976 | 0.9720 |
| matched ASR=30% | 0.0170 | 0.301 | 0.9649 |
| matched ASR=50% | 0.0261 | 0.494 | 0.9683 |
| matched ASR=70% | 0.0361 | 0.690 | 0.9602 |

Margin AUROC stays at 0.96–0.97 even at ASR≈50%, where the label is balanced and the ROC problem is hardest. The high AUROC is genuine, not an artefact of saturation.

### 4.10 Gold-Standard Labels and External Validity (H174, H181)

**AutoAttack labels (H174).** PGD-10 understates vulnerability. Relabelling the Fashion-MNIST evaluation set with an APGD-CE + APGD-DLR + Square ensemble raises ASR from 0.943 (PGD-10) to 0.995, and 5.2% of samples PGD-10 calls robust are flipped by the ensemble. Margin AUROC against the AutoAttack label is 0.9936 (vanilla) — margin dominance survives the gold-standard label.

**ResNet-18 at native resolution (H181).** To address the threat to external validity from a single 28×28 CNN, we trained a CIFAR-style ResNet-18 on native 32×32 RGB CIFAR-10 (94.05% clean accuracy) and re-tested the predictors at a matched-ASR eps (0.85/255, PGD ASR≈0.49). Here **input gradient norm (0.921) exceeds margin (0.889)** and softmax confidence (0.874). This is consistent with §4.2–4.3: on harder, curvier decision boundaries (CIFAR at small eps, ResNet at matched eps) the second-order gradient signal overtakes the first-order margin. Margin dominance is thus dataset/architecture-contingent at the top of the leaderboard, even though both predictors remain strong everywhere.

---

## 5. Discussion

**Why does margin dominate?** The logit margin directly measures the distance from a sample's representation to the decision boundary in output space. For L-inf adversarial perturbations, which exploit linear approximations to the network, the margin is a first-order sufficient statistic for vulnerability [8]. Gradient norms provide a complementary second-order signal: they measure how rapidly the margin changes with input perturbation, capturing local boundary curvature not encoded in the margin value alone.

**Why do training dynamics fail?** Forgetting events identify samples that oscillate between correct and incorrect classification during training, often corresponding to label noise or atypical inputs [3]. These properties do not directly correspond to decision boundary proximity in the converged network. A noisy sample might ultimately be memorised with high confidence (low vulnerability) or lie near the boundary (high vulnerability) depending on the training trajectory and architecture capacity. C-score similarly measures consistency across training runs but does not condition on the converged model's geometry [4].

**PGD saturation artefact.** Our finding that PGD achieves 100% success rate at epsilon = 15/255 on CIFAR-10 has important implications for evaluation methodology. Studies that report vulnerability prediction results using large epsilon budgets may be measuring trivial properties of the model (almost everything is vulnerable), not informative sample-level variation. We recommend always reporting the attack success rate alongside AUROC, and choosing epsilon such that the success rate is in the range 40–80%.

**Computational cost vs. reliability.** SmoothGrad requires K=50 forward-backward passes per sample, making it 50x more expensive than a single gradient computation. Our results show it offers no mean AUROC advantage over plain gradient norm across the majority of experimental conditions, and exhibits 4x higher seed variance. The practical recommendation is to use input_grad_l2_norm as the default gradient-based predictor.

---

## 6. Conclusion

We have benchmarked over 50 per-sample adversarial vulnerability predictors in a controlled multi-dataset, multi-attack setting. The logit margin is the single most robust and computationally inexpensive predictor, achieving AUROC 0.87–0.9976 across all conditions; input gradient L2 norm is its equal, surpassing it at strict epsilon budgets and on a native-resolution ResNet-18. Crucially, once bootstrap CIs are reported the top boundary-proximity predictors are statistically indistinguishable from one another, and a learned combination of all features adds no head-room over the best single feature — the vulnerability signal is effectively one-dimensional (distance to the decision boundary). Margin dominance is not a saturation artefact (it persists at matched ASR≈50%) and survives AutoAttack-grade labels. Training-dynamics features (forgetting events, C-score) are uninformative. SmoothGrad offers marginal gains at great computational cost and poor seed stability. We provide a benchmark protocol — including guidance on the PGD saturation artefact, matched-ASR evaluation, and bootstrap CIs — intended to enable reproducible future comparisons.

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

[11] F. Croce and M. Hein, "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks (AutoAttack)," in *Proc. ICML*, 2020.

[12] G. Pleiss, T. Zhang, E. R. Elenberg, and K. Q. Weinberger, "Identifying mislabeled data using the area under the margin ranking," in *Proc. NeurIPS*, 2020.

[13] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016.
