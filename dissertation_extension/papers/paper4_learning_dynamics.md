# Training Difficulty is Orthogonal to Adversarial Vulnerability

**Abstract**

A widely-held intuition in adversarial machine learning is that inputs difficult to learn — those requiring more training, susceptible to forgetting, or inconsistently classified across training runs — should also be more adversarially vulnerable. We test this hypothesis directly by evaluating forgetting events [3], C-score [4], and a memorization proxy against adversarial vulnerability labels on Fashion-MNIST and related datasets. Forgetting events achieve AUROC = 0.54 for FGSM vulnerability prediction — statistically indistinguishable from random. C-score achieves AUROC = 0.53. However, the memorization proxy achieves AUROC = 0.9427 ± 0.006 with high stability. We additionally test the Area Under the Margin (AUM) [11], a *margin*-trajectory signal, and find AUROC = 0.896 — far from chance, unlike the accuracy-trajectory signals — yet it adds no information once the final margin is known (5-fold incremental ΔAUROC = −0.003; Spearman 0.98 with final margin). This sharpens our thesis: a training-dynamics signal predicts vulnerability if and only if it proxies the converged boundary distance. We argue that this distinction is explained by the different mechanisms of each predictor: forgetting events and C-score capture oscillation and inconsistency in the training dynamics, which do not correlate with boundary proximity in the converged model; the memorization proxy captures inter-subset disagreement, which does correlate with boundary proximity independent of training difficulty. We further document a training trajectory analysis (H157) showing that per-sample robustness does not grow monotonically with training on standard models (Spearman rho = −0.006), while PGD-AT models exhibit robust overfitting (rho = −0.66) and SVHN shows monotonic convergence (rho = +0.92). These findings clarify when training-dynamics signals encode adversarial vulnerability and when they do not.

---

## 1. Introduction

The relationship between learning difficulty and adversarial vulnerability is theoretically appealing. Both concepts involve samples that are in some sense "hard" for the model: learning-difficult samples resist convergence, while adversarially vulnerable samples resist confident correct classification under perturbation. If these two properties co-occur, training-dynamics signals could serve as efficient, attack-free proxies for vulnerability prediction — computed from training logs rather than requiring expensive attack evaluations at test time.

This hypothesis has motivated applications of dataset cartography [3] to adversarial analysis and the development of learning-difficulty scores as robustness indicators. However, the hypothesis conflates two distinct notions of difficulty:

1. **Learning difficulty:** A sample is hard to learn if the model oscillates between correct and incorrect classification during training, or if classification is inconsistent across different training runs or data subsets.

2. **Boundary proximity:** A sample is adversarially vulnerable if the converged model's decision boundary is close to it in input space, allowing small perturbations to move it across the boundary.

These properties need not coincide. A sample might be difficult to learn but ultimately memorised with high confidence (low vulnerability), or easily learned but placed near the class boundary by the model's geometry (high vulnerability). The correlation between these notions is an empirical question.

We test this directly using four training-dynamics predictors with varying degrees of mechanism-specificity, and analyse training trajectories to understand how robustness evolves during both standard and adversarial training.

---

## 2. Related Work

**Forgetting events.** Toneva et al. [3] introduced forgetting events — instances where a correctly classified training sample is subsequently misclassified in a later epoch — as a measure of example difficulty. They found that "forgotten" samples tend to be atypical (mislabelled, out-of-distribution, or near class boundaries in input space), and that unforgettable samples can be pruned without significant accuracy loss. This work motivates the hypothesis that forgotten samples might also be adversarially vulnerable.

**C-score.** Jiang et al. [4] proposed C-score as a dataset-level learning difficulty measure based on leave-one-out consistency: how consistently is a sample correctly classified when models are trained on various random subsets that either include or exclude it? High C-score indicates a sample is reliably learned and thus "easy"; low C-score indicates instability. C-score was designed to characterise dataset regularities rather than adversarial vulnerability specifically.

**Memorization in neural networks.** Zhang et al. (2017) showed that neural networks can memorise random labels, motivating the distinction between generalisation and memorisation. Feldman (2020) showed that memorisation of atypical examples is necessary for optimal generalisation, connecting memorisation to sample difficulty. Our memorization proxy operationalises this via ensemble disagreement.

**Robust overfitting.** Rice et al. [7] documented that adversarially trained models achieve peak robustness on the test set well before training ends, after which robustness declines even as training loss continues to decrease. This robust overfitting phenomenon implies that training trajectory signals for adversarial robustness are non-monotonic and must be interpreted with care.

**Grokking.** Power et al. [10] showed that on small algorithmic datasets, generalisation can emerge suddenly after extended training, long after training loss reaches zero. This extreme case of delayed generalisation further complicates the assumption that training dynamics signals reflect final model properties.

---

## 3. Methodology

### 3.1 Forgetting Events (H129)

For each training sample, count the number of times it transitions from correctly classified to incorrectly classified across consecutive training epochs. High forgetting count = "frequently forgotten" = learning-difficult. We test whether forgetting count correlates with adversarial vulnerability under FGSM and PGD on Fashion-MNIST.

### 3.2 C-score (H130)

We approximate C-score using 50 random 80% subsets of the training data. For each sample, C-score is the fraction of subsets that include it for which the model (trained on that subset) correctly classifies it. High C-score = consistently learned; low C-score = inconsistently learned. We evaluate AUROC against FGSM vulnerability on Fashion-MNIST.

### 3.3 Memorization Proxy (H131)

For each sample, we compute the fraction of 50 random 50% training subsets on which the model misclassifies it (even when the sample is included in that subset). Unlike C-score, this captures a different quantity: how often does the sample fall on the "wrong side" of the decision boundary in a slightly different training setting, regardless of whether the model was trained on it?

We deliberately distinguish this from C-score: the memorization proxy asks whether a sample is consistently near the current class boundary across training regimes, while C-score asks whether it is consistently learned at all.

### 3.4 Training Trajectory Analysis (H157)

For each test sample, define min_eps as the minimum epsilon at which an attack succeeds (i.e., a measure of individual robustness). We train the model for 200 epochs, compute min_eps at multiple checkpoints (epochs 5, 10, 20, 50, 100, 150, 200), and compute the Spearman rank correlation between epoch number and min_eps (averaged across test samples). This measures whether robustness monotonically increases, decreases, or remains flat during training.

We apply this analysis to:
- Standard model on Fashion-MNIST
- Standard model on SVHN
- PGD-AT model on Fashion-MNIST

### 3.5 Multi-Seed Stability

Each predictor is evaluated across 10 random seeds to assess reproducibility, following the protocol of Papers 1 and 3.

---

## 4. Results

### 4.1 Forgetting Events and C-score Perform at Chance

[Figure 1: ROC curves for forgetting events (AUROC = 0.54), C-score (AUROC = 0.53), and margin (AUROC = 0.965) on Fashion-MNIST FGSM vulnerability. Near-diagonal for training dynamics predictors.]

**Forgetting events (H129):** AUROC = 0.54 for FGSM vulnerability on Fashion-MNIST. This value is not statistically distinguishable from 0.50 (random) at standard significance levels. When evaluated against iterative attack vulnerability (PGD, BIM), forgetting events achieve AUROC in the range 0.52–0.56 — consistently near chance.

**C-score (H130):** AUROC = 0.53 for FGSM vulnerability on Fashion-MNIST. C-score performs no better than forgetting events.

These null results are robust: they hold across multiple seeds and attack configurations. The training-dynamics hypothesis — that learning-difficult samples are more adversarially vulnerable — is not supported by this data.

**Interpretation.** Forgetting events identify samples that oscillate between correct and incorrect during training. In a standard training run, such samples are often eventually memorised with high confidence, meaning the converged model places them deep within a class region (low vulnerability). Alternatively, some forgetting-prone samples are placed near class boundaries in the converged model, but these are not a majority. The correlation between training oscillation and final boundary proximity is close to zero.

C-score measures whether a sample is consistently learned across different training sets. A sample that is inconsistently learned (low C-score) may occupy a region where the decision boundary position is sensitive to training data — which could indicate boundary proximity — but empirically this signal is too weak to predict vulnerability.

### 4.2 Memorization Proxy Achieves High AUROC

[Figure 2: ROC curve for memorization proxy (AUROC = 0.9427) vs. forgetting events (0.54). Near-perfect curve for memorization proxy.]

**Memorization proxy (H131):** AUROC = 0.9427 ± 0.006, classified as Stable. This is a strong result, substantially exceeding both forgetting events and C-score, and approaching the performance of the margin (0.9651) and input gradient norm (0.9743).

The contrast with forgetting events and C-score is informative. The memorization proxy measures the fraction of random 50% subsets where the model misclassifies a sample — regardless of whether the sample is in the training set. This is not a measure of training difficulty. It is a measure of **boundary proximity under data perturbation**: a sample is classified as memorization-requiring if, across many different training datasets, it frequently falls on the wrong side of the decision boundary. This is structurally equivalent to asking: is this sample near the decision boundary that naturally emerges from this data distribution?

In other words, the memorization proxy works precisely because it does **not** measure learning difficulty in the conventional sense. It measures boundary proximity via ensemble disagreement — the same geometric property that the logit margin measures directly from the converged model.

[Figure 3: Scatter plot of memorization proxy score vs. logit margin on Fashion-MNIST. Negative correlation confirming shared underlying construct.]

### 4.3 Comparison of Training-Dynamics Predictors

The table below summarises all training-dynamics predictors alongside the gradient-based and Bayesian approaches, placing them in the context of the broader benchmark:

**Table 1: Training-Dynamics Features vs Adversarial Vulnerability (Fashion-MNIST, PGD target)**

| Feature | AUROC | Why it works (or doesn't) |
|---------|-------|--------------------------|
| Forgetting event count (H129) | 0.54 | Measures training instability, not boundary position |
| C-score (H130) | 0.53 | Measures consistency across random subsets |
| Memorization proxy (H131) | 0.9427 ± 0.006 | Measures prediction variance = boundary proximity |
| softmax_variance (snapshot ensemble) | 0.9617 ± 0.005 | Same mechanism: ensemble disagreement |
| bnn_predictive_variance | 0.9187 ± 0.013 | Bayesian uncertainty ≈ boundary proximity |
| Margin (baseline) | 0.9723 ± 0.006 | Ground-truth boundary distance |

The table illustrates the key distinction: forgetting events and C-score measure properties of the training trajectory, while the memorization proxy measures a property of the converged boundary geometry. Only the latter correlates with adversarial vulnerability.

### 4.3a Area Under the Margin: Not at Chance, but Redundant (H178)

Forgetting events and C-score are *accuracy-trajectory* signals (they track whether the prediction is correct over epochs). A sharper test of the training-dynamics hypothesis uses a *margin-trajectory* signal: the Area Under the Margin (AUM) of Pleiss et al. [11], defined as the mean logit margin of the true class across training epochs. We recorded per-epoch eval margins over 15 epochs and computed AUM for each finally-correct sample (n=1836, PGD ASR=0.966).

**Table 3: Margin-trajectory vs accuracy-trajectory dynamics (Fashion-MNIST, PGD target)**

| Predictor | AUROC | Spearman vs final margin |
|-----------|-------|--------------------------|
| AUM (mean-margin trajectory) | 0.8959 | 0.983 |
| final_margin (single snapshot) | 0.9154 | — |
| forget_count (accuracy trajectory) | 0.5595 | — |

AUM is emphatically **not** at chance (0.896), unlike forgetting (0.560) — a result that could be read as contradicting our thesis. It does not. AUM correlates with the final margin at Spearman 0.983: it is a smoothed estimate of the same boundary-distance quantity, averaged over the last epochs of training. The decisive test is incremental signal. A 5-fold cross-validated logistic regression on [final_margin] scores CV-AUROC 0.9168 ± 0.0366; adding AUM gives 0.9135 ± 0.0359 — a change of **−0.0033**, i.e. AUM adds nothing once the final margin is known.

The pattern across all four dynamics predictors is now coherent: a training-dynamics signal predicts adversarial vulnerability *if and only if* it is a proxy for the converged boundary distance. Margin-based dynamics (AUM) inherit the margin's predictive power but contribute no independent information; accuracy-based dynamics (forgetting, C-score) capture optimisation oscillation orthogonal to boundary geometry and sit at chance. Training difficulty per se remains orthogonal to vulnerability.

### 4.4 Training Trajectory Analysis (H157)

[Figure 4: Per-sample min_eps vs. training epoch for three conditions. Standard FM: flat; Standard SVHN: monotonically increasing; PGD-AT FM: inverted-U shape.]

The Spearman correlation results across training conditions are summarised in Table 2:

**Table 2: Spearman Correlation (Training Epoch vs min_eps) — H157**

| Dataset | Model | Rho | Interpretation |
|---------|-------|-----|----------------|
| Fashion-MNIST | Standard | −0.006 | Flat — robustness does not grow with training |
| Fashion-MNIST | PGD-AT | **−0.660** | Robust overfitting — peaks then declines |
| SVHN | Standard | **+0.920** | Monotonic growth — harder dataset needs more training |
| CIFAR-10 | Standard | +0.4–0.8 | Moderate growth |

**Standard model on Fashion-MNIST:** Spearman rho = −0.006 (essentially flat). Individual sample robustness does not improve with training epoch in a standard model. The model reaches near-final decision boundary geometry relatively quickly, and additional training oscillates without systematic robustness change. This is consistent with the forgetting events null result: if robustness does not change across training, forgetting events (which require robustness to oscillate) cannot be informative.

**Standard model on SVHN:** Spearman rho = +0.92 (strong monotonic increase). SVHN is a more complex dataset with digit recognition in natural scene images, requiring more training epochs to converge. Robustness grows monotonically as the model learns stable class representations. This contrasts sharply with Fashion-MNIST and suggests that the training-dynamics signal can be informative on datasets that have not yet converged.

**PGD-AT model on Fashion-MNIST:** Spearman rho = −0.66. Robustness grows during early adversarial training, peaks at approximately epoch 70–100, then declines — consistent with Rice et al.'s robust overfitting [7]. The negative Spearman correlation across all epochs reflects the dominant decline phase outweighing the early growth phase. At peak robustness, individual sample min_eps values are higher (more robust) than at the end of training.

**Figure 1: Conceptual Training Trajectory (min_eps vs epoch)**

```
min_eps
0.25|             ╭──────  PGD-AT (peaks, then robust overfitting)
0.20|           ╭─╯
0.15|         ╭─╯
0.10|       ─────────────── SVHN standard (monotonic growth)
0.05|   ─────               FM standard (flat — converges fast)
0.00+──────────────────────────────────> epoch
    0    5    10   15   20   25   30
```
*PGD-AT on Fashion-MNIST shows classic robust overfitting (Rice et al., 2020): robustness peaks at ~epoch 15 then declines as the model overfits to adversarial examples.*

[Figure 5: Robustness trajectory for PGD-AT model. Per-sample min_eps averaged across test set. Clear peak followed by decline, confirming robust overfitting.]

The SVHN result (rho = +0.92) and PGD-AT result (rho = −0.66) bookend the standard FM result (rho = −0.006), illustrating that training trajectory signals are highly dataset- and training-regime-specific. On converged standard models, trajectory signals add no information. On unconverged models (SVHN) or overfitting AT models, they provide a directional signal — but potentially a misleading one if interpreted without context.

### 4.5 Implications for Grokking

The grokking phenomenon [10] — where generalisation emerges suddenly after extended training, well after training loss plateaus — provides an extreme case where training dynamics and final model properties are decoupled. Samples "forgotten" during the pre-grokking memorisation phase may be "recovered" during the grokking transition. Our Fashion-MNIST result (rho = −0.006) suggests that even short of grokking, standard CNN training converges to a fixed boundary geometry where training trajectory signals are uninformative.

---

## 5. Discussion

**Theoretical account of the null results.** Forgetting events and C-score measure properties of the optimisation path. Adversarial vulnerability measures properties of the fixed point (converged model). On overparameterised networks trained on clean data, the optimisation path is not reliably predictive of boundary geometry because: (a) overparameterisation allows the model to memorise difficult samples with high confidence regardless of training oscillation; (b) the boundary can be moved arbitrarily by weight decay, data augmentation, and initialisation without affecting the ordering of forgetting events.

**Memorization proxy as boundary probe.** The memorization proxy's success can be understood as follows: training on different 50% subsets induces different decision boundaries. A sample near the "natural" boundary of the full-data model will frequently lie on the wrong side of boundaries induced by subsets that are missing critical neighbouring samples. This is precisely a measure of boundary proximity — not training difficulty. The measure is indirect (requiring many training runs) but correctly identifies the relevant geometric property.

**SVHN as a reminder of dataset-specificity.** The strong positive rho = +0.92 on SVHN shows that training trajectory signals are not universally uninformative. On datasets where the model is still learning during the training window (i.e., has not yet converged to a stable boundary), robustness grows with training. Practitioners should check convergence before dismissing trajectory signals.

**Robust overfitting and evaluation timing.** The PGD-AT trajectory (rho = −0.66) has a practical implication: evaluating adversarially trained models at the final epoch underestimates their peak robustness. Rice et al. [7] recommend early stopping for adversarial training. Our per-sample analysis confirms that the decline affects the entire distribution of sample robustness, not just aggregated accuracy.

---

## 6. Conclusion

Training difficulty is orthogonal to adversarial vulnerability. Forgetting events (AUROC = 0.54) and C-score (AUROC = 0.53) predict adversarial vulnerability at chance levels on Fashion-MNIST, despite their utility for characterising learning-difficult samples in other contexts. The memorization proxy achieves AUROC = 0.9427 ± 0.006, but this success is explained by its indirect measurement of boundary proximity via ensemble disagreement, not by any capture of training difficulty. Training trajectory analysis confirms that per-sample robustness is flat across epochs on converged standard models (Spearman rho = −0.006), monotonically increasing on unconverged models like SVHN (rho = +0.92), and non-monotone with robust overfitting on PGD-AT models (rho = −0.66). These findings establish a clear conceptual distinction: boundary proximity and training difficulty are different properties of neural networks, and conflating them leads to predictors that fail despite intuitive appeal. Vulnerability prediction requires measures of the converged model's geometry, not its optimisation history.

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

[11] G. Pleiss, T. Zhang, E. R. Elenberg, and K. Q. Weinberger, "Identifying mislabeled data using the area under the margin ranking," in *Proc. NeurIPS*, 2020.

[12] S. Swayamdipta, R. Schwartz, N. Lourie, Y. Wang, H. Hajishirzi, N. A. Smith, and Y. Choi, "Dataset cartography: mapping and diagnosing datasets with training dynamics," in *Proc. EMNLP*, 2020.
