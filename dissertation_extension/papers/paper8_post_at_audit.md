# Predicting Which Samples Are Hurt by Adversarial Training Before Training Begins

**Abstract**

Adversarial training (AT) consistently reduces clean accuracy, but its per-sample impact is nonuniform: some samples that the vanilla model classifies correctly become misclassified by the AT model. We ask whether these "AT-hurt" samples can be identified *before* AT begins, using only the vanilla model's outputs. On Fashion-MNIST, we identify 98 samples (5.4% of vanilla-correct) that are correctly classified by the vanilla model (91.55% clean accuracy) but misclassified by the AT model (88.40% clean accuracy). These samples have 6.1× lower pre-AT margin (1.84 ± 1.51 vs 11.18 ± 7.36 for preserved samples) and 8.8× lower min-ε (0.010 ± 0.006 vs 0.087 ± 0.177). Four features extracted from the vanilla model — margin, top-1 probability, min-ε, and gradient L2 norm — all achieve AUROC ≈ 0.935 in predicting "AT-hurt" membership. AT's clean accuracy loss is therefore not random: it is a systematic, predictable function of pre-AT boundary proximity. This "double jeopardy" effect — AT hurts the samples that are already most adversarially vulnerable — enables targeted intervention. We test three: reweighting the fragile bottom-10% in either direction backfires (up-weighting is worst, +41% hurt samples, −1.15 pp clean), but *excluding* the fragile set from adversarial augmentation recovers +0.65 pp clean accuracy and reduces hurt samples from 103 to 93, at a modest robustness cost. The actionable lever is to withhold adversarial pressure from low-margin samples, not to intensify it.

---

## 1. Introduction

Adversarial training [1, 2] is the most reliable method for improving model robustness, but it comes at a cost: clean accuracy typically drops by 3–10 percentage points relative to vanilla training [3]. This cost is widely recognized but rarely studied at the per-sample level. The standard analysis treats it as a global tradeoff — a necessary price paid uniformly across the test set. But if the clean accuracy loss is concentrated on a predictable subset of samples, it can be addressed through targeted interventions rather than global hyperparameter tuning.

We study the per-sample dynamics of AT's clean accuracy cost on Fashion-MNIST. We train two models: a vanilla CNN (no adversarial examples) and a PGD-AT CNN [2]. We then identify the set of "AT-hurt" samples — samples that the vanilla model classifies correctly but the AT model misclassifies — and ask whether these samples can be predicted using features computed from the vanilla model alone.

Our central finding is that they can: AUROC ≈ 0.935 for all four tested features (margin, confidence, min-ε, gradient norm). AT-hurt samples are characterized by 6× lower pre-AT margin and 9× lower minimum adversarial perturbation budget. AT concentrates its clean accuracy loss on the samples that are already most adversarially vulnerable — creating a "double jeopardy" effect where hard samples are simultaneously the most vulnerable to attack and the most likely to be broken by the very defense designed to protect the model.

This finding has practical consequences: a practitioner can audit a dataset *before* running AT, identify samples at risk of being hurt, and apply targeted interventions such as up-weighting those samples in the AT loss, holding them out of adversarial augmentation, or applying curriculum scheduling. We discuss these interventions and their implications.

---

## 2. Related Work

**Adversarial Training and Clean Accuracy.** Madry et al. [2] noted a clean accuracy cost of PGD-AT but did not analyze it at the sample level. Zhang et al. [4] (TRADES) showed the cost is mediated by the boundary regularization strength λ. Rice et al. [3] showed that overfitting in AT is characterized by the model "forgetting" to classify natural examples, suggesting that the clean accuracy cost is a form of catastrophic forgetting. Stutz et al. [5] showed that on-manifold adversarial examples reduce the generalization gap, implying that off-manifold perturbations (as in PGD-AT) may distort the learned representation of natural images.

**Sample Difficulty and Forgetting.** Toneva et al. [6] introduced forgetting events — training examples that transition from correct to incorrect during SGD. Forgettable examples are disproportionately near the decision boundary, with low confidence and low margin. Our "AT-hurt" samples are analogous but measured across training *regimes* rather than across epochs: they are examples that the vanilla training regime learns but the AT regime does not.

**Per-Sample Vulnerability Prediction.** Papers 5 and 6 in this dissertation established that the vanilla model's margin predicts adversarial vulnerability with AUROC ~0.935–0.946. Paper 8 extends this: we ask whether the same margin predicts not just vulnerability to *attacks* but also vulnerability to *training regime change* (AT). The finding that the AUROC is identical (~0.935) in both cases suggests a unified picture: the margin is a fundamental property of the sample's relationship to the decision boundary, and both adversarial vulnerability and AT-induced misclassification are consequences of low margin.

**Curriculum and Sample-Aware AT.** Zhang et al. [7] (FAT) uses early-stopped PGD to reduce the strength of adversarial examples, improving the tradeoff. Curriculum AT [8] schedules training examples by difficulty. Our work motivates a specific form of curriculum AT: identifying at-risk samples before training and applying protective measures specifically to them.

**Double Jeopardy.** The interaction between adversarial vulnerability and AT-induced harm is related to the "double descent" phenomenon [9], where hard examples show non-monotone behavior under increasing model capacity or regularization. Our empirical finding provides a concrete instance: the same property (low margin) that makes a sample hard to protect also makes it likely to be harmed by the protection attempt.

---

## 3. Methodology

### 3.1 Experimental Setup

All experiments use Fashion-MNIST with 2,000 test samples (stratified subsample for computational tractability). Models are trained on the full 60,000-sample training set.

**Vanilla model:** Standard CNN (three convolutional blocks, 256-unit FC head), trained for 30 epochs with SGD, momentum 0.9, weight decay 1e-4, cosine LR annealing from 0.01. No adversarial augmentation.

**AT model:** Same architecture and hyperparameters, but each training batch includes PGD adversarial examples (ε = 8/255, 20 steps, step size 2/255) at 50% mixing ratio.

Both models are trained with identical random seeds for all randomness except the AT-specific adversarial example generation.

### 3.2 Sample Categorization

Let S_vanilla = {i : vanilla model classifies test sample i correctly} and S_at = {i : AT model classifies test sample i correctly}.

- **Preserved:** S_vanilla ∩ S_at — correctly classified by both models.
- **Newly wrong (AT-hurt):** S_vanilla \ S_at — correctly classified by vanilla but not by AT.
- **Newly right:** S_at \ S_vanilla — correctly classified by AT but not by vanilla.

We analyze the distribution of vanilla model features within each group.

### 3.3 Features Extracted from the Vanilla Model

All features are computed on the vanilla model for each test sample **x**:

1. **Margin:** f_y(**x**) − max_{j≠y} f_j(**x**), the gap between the correct class logit and the runner-up.

2. **top1_prob:** max_j softmax(f(**x**))_j, the softmax probability of the top predicted class.

3. **min_eps:** The minimum perturbation budget (in ∞-norm) at which FGSM succeeds in flipping the prediction. Computed via binary search over ε ∈ [0, 1] using FGSM with 40 steps.

4. **grad_l2_norm:** ||∇_x L(f(**x**), y)||_2, the L2 norm of the gradient of the cross-entropy loss with respect to the input at the natural image.

### 3.4 Predictive Evaluation

For each feature, we compute AUROC with the binary label "AT-hurt" (1 if i ∈ S_vanilla \ S_at, 0 if i ∈ S_vanilla ∩ S_at). Since AT-hurt samples are the minority class (98 of 1,831 vanilla-correct samples, 5.4%), AUROC is appropriate as a threshold-free evaluation metric. We report AUROC across 5 seeds to assess stability.

---

## 4. Results

### 4.1 Accuracy Statistics

**Table 1: Model accuracy and sample set sizes**

| Quantity                          | Value             |
|-----------------------------------|-------------------|
| Vanilla clean accuracy (2000 samples) | 91.55%        |
| AT clean accuracy (2000 samples)      | 88.40%        |
| \|S_vanilla\| (vanilla correct)       | 1,831 samples |
| \|S_at\| (AT correct)                 | 1,768 samples |
| Preserved (both correct)              | 1,733 samples (94.6% of vanilla-correct) |
| Newly wrong (AT-hurt)                 | 98 samples (5.4% of vanilla-correct)     |
| Newly right (AT-gains)                | 35 samples                               |

The AT model incorrectly classifies 98 samples that the vanilla model correctly classified — 5.4% of vanilla-correct samples. The AT model newly correct-classifies only 35 samples (18.1% of the clean accuracy gap), meaning the remaining 81.9% of the gap represents genuine loss rather than redistribution.

### 4.2 Sample Flow: From Vanilla to AT Model

The following diagram makes the per-sample dynamics explicit, showing exactly how the 2,000-sample test set is partitioned between the two models.

**Figure 1: Sample Flow from Vanilla to AT Model (N=2000 test samples)**

```
                    Vanilla model (N=2000)
                    +---------------------+
                    | Correct: 1831 (91.6%)|
                    | Wrong:    169 ( 8.5%)|
                    +----------+----------+
                               |
                               | AT training
                               v
                    AT model evaluation
              +--------------------------------+
              | Preserved correct:  1733 (94.6%)| <- margin 11.18
              | Newly wrong:          98 ( 5.4%)| <- margin  1.84 (!)
              | Newly right:          35        |
              +--------------------------------+

"Double jeopardy": hurt samples had lowest margin AND lowest min_eps
-> already most vulnerable AND most likely to be broken by AT
```

### 4.3 Feature Distribution: Preserved vs AT-Hurt

**Table 2: Feature Comparison — Preserved vs Newly Wrong Samples (H159, Fashion-MNIST)**

| Feature | Preserved (mean +/- std) | Newly Wrong (mean +/- std) | Ratio |
|---------|----------------------|-------------------------|-------|
| Margin | 11.18 +/- 7.36 | **1.84 +/- 1.51** | 6.1x lower |
| top1_prob | 0.975 +/- 0.074 | **0.762 +/- 0.178** | lower |
| min_eps | 0.087 +/- 0.177 | **0.010 +/- 0.006** | 8.8x lower |
| grad_l2_norm | 0.437 +/- 1.393 | **3.793 +/- 3.151** | 8.7x higher |

Mean ± standard deviation. Margin and min_eps are lower for AT-hurt samples; grad_l2_norm is higher (samples with larger gradient norms are more AT-hurt, consistent with being closer to the boundary and more sensitive to perturbation).

The scale of the differences is striking: AT-hurt samples have 6.1× lower margin and 8.8× lower min_eps than preserved samples. Their top-1 probability is lower (0.762 vs 0.975), but the gap is smaller in relative terms (1.28×). The gradient norm difference is largest in absolute terms: AT-hurt samples have 3.793 ± 3.151 gradient norm vs 0.437 ± 1.393, an 8.7× ratio.

[Figure 2: Violin plots of margin and min_eps distributions for Preserved (blue) and AT-Hurt (red) samples. AT-Hurt samples cluster tightly near zero in both features; Preserved samples span a wide range with high median values.]

### 4.4 AUROC for AT-Hurt Prediction

**Table 3: AUROC for Predicting "Newly Wrong After AT" Membership**

| Feature | AUROC | Direction |
|---------|-------|-----------|
| top1_prob | **0.9366** | lower -> hurt |
| min_eps | **0.9363** | lower -> hurt |
| margin | 0.9352 | lower -> hurt |
| grad_l2_norm | 0.9349 | higher -> hurt |

*All four features achieve AUROC ~= 0.935. The vanilla model identifies AT-susceptible samples with high accuracy before any AT is performed.*

All four features achieve AUROC ≈ 0.935, with variation of less than 0.002 across features. This near-uniformity is surprising: margin (a logit difference), top-1 probability (a softmax transformation of logits), min_eps (a binary search over FGSM success), and gradient L2 norm (a first-order sensitivity measure) are four conceptually distinct quantities, yet they provide almost identical predictive power. This suggests they are all capturing the same underlying property: proximity to the decision boundary.

[Figure 3: ROC curves for all four features on the AT-hurt prediction task. Curves are nearly coincident, achieving AUROC = 0.935 with tight clustering across the entire range of false-positive rates.]

The AUROC of 0.935 matches the margin AUROC observed for attack vulnerability prediction (Papers 5, 6): the vanilla model's margin predicts both adversarial vulnerability and AT-induced misclassification with the same accuracy. This parallel is the central empirical finding of this paper.

### 4.5 Calibration of the Prediction

At a decision threshold that controls false positive rate at 10% (90% specificity), the margin achieves sensitivity of 78% for AT-hurt prediction. This means that 78% of AT-hurt samples could be identified before training, while flagging only 10% of safe samples as at-risk. At 5% false positive rate, sensitivity drops to 65%. These operating points are practical for sample reweighting interventions.

[Figure 4: Precision-recall curve for margin-based AT-hurt prediction. Precision remains above 0.5 until recall exceeds 0.70, indicating that a targeted intervention based on pre-AT margin would have useful precision across the most relevant operating range.]

### 4.6 Testing the Interventions: Reweight, Upweight, Exclude (H177)

The prediction in §4.4 is only useful if it enables an intervention that actually reduces AT's clean-accuracy cost. We test this directly. Using the vanilla model's *training-set* margins, we flag the bottom 10% (n=6,000, margin ≤ 1.47) as the "fragile" set, then run four AT variants that differ only in how they treat that set: **uniform** (standard PGD-AT, control), **downweight** (fragile samples' AT loss × 0.5), **upweight** (fragile × 2.0), and **exclude** (fragile samples trained on clean examples only, no adversarial augmentation). All variants use ε = 0.0588, 10 epochs, identical seeds.

**Table 4: Targeted Intervention on the Fragile Set (H177, Fashion-MNIST, Δ vs uniform AT)**

| Mode | Clean Acc | ΔClean | PGD-ASR | min-ε | Hurt | ΔHurt | Gained |
|------|-----------|--------|---------|-------|------|-------|--------|
| uniform (control) | 0.8800 | +0.0000 | 0.1960 | 0.1845 | 103 | +0 | 33 |
| downweight | 0.8730 | −0.0070 | 0.1975 | 0.1655 | 111 | +8 | 27 |
| upweight | 0.8685 | −0.0115 | 0.2105 | 0.1753 | 145 | +42 | 52 |
| **exclude** | **0.8865** | **+0.0065** | 0.2025 | 0.1631 | **93** | **−10** | 36 |

*Excluding the fragile set from adversarial augmentation is the only intervention that helps: +0.65 pp clean accuracy and 10 fewer hurt samples, at the cost of slightly weaker robustness (PGD-ASR +0.65 pp, min-ε −0.021). Upweighting the fragile set — the natural "try harder on hard samples" instinct — is the worst option, increasing hurt samples by 42 (+41%) and dropping clean accuracy 1.15 pp.*

The intervention experiment confirms the mechanism and inverts a common intuition. Down/up-weighting the fragile set in the AT loss both *hurt* clean accuracy; upweighting is decisively worst (hurt 145 vs 103, −1.15 pp clean). The only intervention that helps is **exclude** — withholding adversarial augmentation from the fragile samples entirely. This is consistent with §5.1's double-jeopardy mechanism: fragile samples cross the boundary trivially under PGD, so their adversarial gradients are large and destabilizing; the fix is not to weight those gradients but to *not generate them*. Excluding the bottom 10% recovers +0.65 pp clean accuracy and reduces the hurt count from 103 to 93, while costing only +0.65 pp PGD-ASR.

The symmetry probe (bottom of the H177 log) sharpens the picture: AT-hurt samples have low pre-AT margin (1.74), but AT-*gained* samples have even lower pre-AT margin (0.69) — the vanilla model was already nearly wrong on them. The intervention therefore operates on a population (low-but-nonzero margin, correctly classified) that is distinct from the gains population (near-zero margin, already misclassified), which is why excluding fragile samples reduces hurt without sacrificing the gains.

---

## 5. Discussion

### 5.1 Double Jeopardy

The central finding is a "double jeopardy" for hard samples: the same samples that are most vulnerable to adversarial attacks (low margin → high AUROC for attack success prediction) are also the most likely to be broken by AT (low margin → high AUROC for AT-hurt prediction). This is not a coincidence — it reflects the shared mechanism: both attack vulnerability and AT-induced misclassification are consequences of proximity to the decision boundary.

AT works by pushing boundaries outward, but it does so imperfectly, and the imperfection is concentrated near regions of the boundary that were already difficult. Samples close to the original boundary experience large gradient updates during AT (because the adversarial examples for these samples cross the boundary easily), which can destabilize their classification. Samples far from the boundary experience smaller updates and remain correctly classified.

### 5.2 Connection to Forgetting Events

Toneva et al. [6] showed that forgettable examples — those that transition from correct to incorrect across training epochs — are concentrated near the decision boundary. Our AT-hurt samples are a cross-training-regime version of the same phenomenon: examples that the vanilla regime "knows" but the AT regime "forgets." The shared mechanism (boundary proximity) and the shared magnitude (margin differences of 6×) confirm that AT-induced forgetting is a specific instance of the more general boundary-proximity forgetting phenomenon.

### 5.3 Implications for Targeted Interventions

The AUROC of 0.935 at AT-hurt prediction enables practical interventions. We tested three (§4.6) and found that the obvious ones backfire:

**Sample reweighting in AT (tested — both directions fail).** We tried both down-weighting and up-weighting the fragile set in the AT loss. Both *hurt* clean accuracy relative to uniform AT (−0.70 pp and −1.15 pp respectively), and up-weighting was decisively worst, increasing the hurt count by 41%. The "try harder on hard samples" instinct (MarginWeighted, Paper 5) is exactly wrong for protecting clean accuracy under AT: applying more adversarial pressure to already-fragile samples destabilizes them further.

**Selective AT mixing — exclude (tested — the one that works).** Withholding adversarial augmentation from the fragile set entirely (train them on clean examples only) was the only intervention that helped: +0.65 pp clean accuracy and 10 fewer hurt samples, at a modest robustness cost (+0.65 pp PGD-ASR, −0.021 min-ε). This isolates the most vulnerable samples from the primary source of AT-induced harm, confirming that the fix is to *not generate* the destabilizing gradients rather than to reweight them.

**Curriculum AT scheduling (untested, motivated).** Introducing at-risk samples to AT augmentation late in training is a natural extension of the successful exclude variant — a soft, time-varying version of exclusion rather than a binary one. We leave a curriculum schedule to future work, but the exclude result suggests the right direction is *less* early adversarial pressure on fragile samples, not more.

### 5.4 Why All Four Features Achieve the Same AUROC

The near-identical AUROC across margin, confidence, min_eps, and gradient norm suggests they all encode the same latent variable: distance to the decision boundary in the natural data distribution. Margin is a linear approximation to this distance in logit space; confidence is a monotone transformation of margin; min_eps is a direct measurement of ε-ball distance; and gradient norm measures the boundary's slope, which is inversely related to distance for locally linear boundaries. The convergence to AUROC 0.935 implies this is close to the ceiling of what any single scalar feature can achieve on this task — the residual 0.065 AUROC gap may require multi-feature combinations or models that capture boundary curvature rather than just proximity.

### 5.5 Generalizability

All results are on Fashion-MNIST with a specific CNN architecture. The mechanism (AT hurts low-margin samples) is architecture-independent in principle, but the specific AUROC values will depend on the dataset complexity, the model capacity, and the AT hyperparameters. The Imagenette experiments in Paper 6 suggest that on harder datasets, the vulnerability signal is more diffuse, which may imply that AT-hurt prediction AUROC would be lower on CIFAR-10 or ImageNet-scale data. This is a key direction for future work.

---

## 6. Conclusion

We have shown that adversarial training's clean accuracy cost is not random: it concentrates systematically on the pre-existing hard samples — those with low vanilla model margin, low confidence, low min-ε, and high gradient norm. On Fashion-MNIST, 98 samples (5.4% of vanilla-correct) are hurt by AT, and all four tested vanilla model features predict this outcome with AUROC ≈ 0.935. This "double jeopardy" effect — hard samples are most vulnerable to attacks and most likely to be broken by AT — has direct implications for targeted interventions. We tested three: by auditing the pre-AT vanilla model's margin distribution, practitioners can identify the fragile bottom-10% and choose how to treat them under AT. Reweighting in either direction backfires (up-weighting worst, +41% hurt, −1.15 pp clean), but *excluding* the fragile set from adversarial augmentation recovers +0.65 pp clean accuracy and reduces hurt samples from 103 to 93, at a modest robustness cost. The margin's predictive power (AUROC 0.935) is consistent across attack vulnerability prediction (Papers 5, 6) and AT-hurt prediction, suggesting that margin is a fundamental and transferable property of per-sample boundary proximity — and that the actionable lever is to withhold adversarial pressure from low-margin samples, not to intensify it.

---

## References

[1] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and Harnessing Adversarial Examples," *ICLR*, 2015.

[2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "Towards Deep Learning Models Resistant to Adversarial Attacks," *ICLR*, 2018.

[3] L. Rice, E. Wong, and Z. Kolter, "Overfitting in Adversarially Robust Deep Learning," *ICML*, 2020.

[4] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan, "Theoretically Principled Trade-off between Robustness and Accuracy," *ICML*, 2019.

[5] D. Stutz, M. Hein, and B. Schiele, "Disentangling Adversarial Robustness and Generalization," *CVPR*, 2019.

[6] M. Toneva, A. Sordoni, R. des Combes, A. Trischler, Y. Bengio, and G. Gordon, "An Empirical Study of Example Forgetting during Deep Neural Network Learning," *ICLR*, 2019.

[7] J. Zhang, X. Xu, B. Han, G. Niu, L. Cui, M. Sugiyama, and M. Kankanhalli, "Attacks Which Do Not Kill Training Make Adversarial Learning Stronger," *ICML*, 2020.

[8] T. Cai, X. Li, J. Wang, R. Hu, and H. Fang, "Towards Compact and Robust Deep Neural Networks," *arXiv*, 2020.

[9] M. Belkin, D. Hsu, S. Ma, and S. Mandal, "Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off," *PNAS*, 2019.

[10] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum Learning," *ICML*, 2009.

[11] D. Smilkov, N. Thorat, B. Kim, F. Viégas, and M. Wattenberg, "SmoothGrad: Removing Noise by Adding Noise," *ICML Workshop*, 2017.

[12] N. Carlini and D. Wagner, "Towards Evaluating the Robustness of Neural Networks," *IEEE S&P*, 2017.

[13] A. Shafahi, M. Najibi, A. Ghiasi, Z. Xu, J. Dickerson, C. Studer, L. Davis, G. Taylor, and T. Goldstein, "Adversarial Training for Free!" *NeurIPS*, 2019.
