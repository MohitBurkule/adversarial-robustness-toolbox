# Toward Human-Like Robustness: How Models Could Learn Semantic Feature Spaces

---

## 1. Why Ensembles Don't Solve Adversarial Vulnerability

A natural first intuition is that ensemble methods should confer robustness by averaging out the idiosyncratic failure modes of individual models. Empirically, the campaign data tell a different story. The gradient-boosted machine (GBM) ensemble tested in the benchmark campaign achieves an adversarial success rate (ASR) of **0.445** — nearly half of adversarial examples transfer and succeed — despite the ensemble containing hundreds of trees trained with diverse subsampling.

The theoretical reason is straightforward: ensembles increase **boundary complexity**, not boundary smoothness. Each additional tree in a boosted ensemble adds more piecewise-constant regions and more sharp boundaries in input space. The decision surface becomes more intricate, but intricacy is not the same as robustness. A smooth, gently-curving boundary is hard to exploit because small perturbations in any direction stay on the same side; a jagged boundary with many corners and tight corridors is easy to exploit because an attacker can find a corner and push through it.

Transferability within boosted ensembles is further enabled by the correlations baked into sequential boosting. Each successive tree is trained to correct the residuals of the previous trees, which means all trees share a common inductive bias toward the same boundary regions. An adversarial example crafted against one stage of the ensemble is already partially adapted to later stages. This is not a failure of any specific implementation; it is a structural property of the boosting objective.

More fundamentally, the **robustness-accuracy tradeoff** is not an artefact of architecture but of data geometry. In any region of input space where the data distribution places two classes in close proximity, a decision boundary must pass through that region. Models that achieve high clean accuracy must learn to discriminate these proximate classes accurately, which requires placing a boundary in a high-density region. That boundary is, by geometric necessity, close to many clean examples and therefore easily reachable by a small perturbation. Ensembles that preserve accuracy do not move this boundary; they only redraw it in greater detail.

---

## 2. Four Mechanisms Humans Use That Models Lack

Human judgement is adversarially robust in practice. A human loan officer shown a near-identical pair of applications will recognize them as equivalent. A radiologist shown two near-identical scans will give the same diagnosis. This robustness does not arise from the human deploying a more complex classifier; it arises from four qualitatively different cognitive mechanisms that current models lack.

### 2.1 Semantic Features, Not Pixel Features

Human perception operates on semantic features — objects, relations, categories — that are invariant to a wide range of nuisance dimensions. A face is recognizable under different lighting, at different resolutions, from different angles, because the human visual system has learned a representation that is invariant to these transformations. Models trained on pixel arrays or raw tabular values have no such invariance; every dimension of the input is potentially discriminative, including dimensions that carry no semantic content.

This is not merely a training-data problem. A model trained on enough augmented data can learn some invariances, but it does not know *which* invariances are semantically meaningful and which are nuisance. The human has prior knowledge — developed through embodied experience and causal understanding — that certain dimensions encode identity and others encode circumstance.

### 2.2 Abstention at Boundaries

Humans do not always produce a confident binary output. When presented with an ambiguous case, a human expert will say "I'm not sure — I need more information." This epistemic humility is not a failure mode; it is a safety mechanism. The adversarial vulnerability of a model is partly a consequence of its obligation to always output a label. If a model could say "this example lies near a decision boundary; I abstain," then adversarial examples — which are by construction close to boundaries — would be detected rather than misclassified.

Current models are trained under cross-entropy loss, which penalizes abstention implicitly: the softmax output always sums to one, so every input is assigned to some class with some probability. The model has no structural mechanism for expressing genuine uncertainty about class membership.

### 2.3 Causal Priors

Human reasoning distinguishes causes from correlates. A loan officer knows that income *causes* repayment capacity; credit score is a noisy proxy that measures a downstream consequence of past financial behaviour. If an applicant has a high income but a slightly degraded credit score due to a reporting error, the officer recognizes that the causal signal (income) dominates the proxy (score). The model sees only the joint distribution over input features and does not know which features are causes and which are effects.

This distinction matters for adversarial robustness because adversarial perturbations typically move along directions that change correlates (proxy features) while leaving causes unchanged. A model that correctly understood the causal structure would be unperturbed by such moves; the causal evidence has not changed.

### 2.4 Perceptual Resolution Limits and the Just-Noticeable-Difference Threshold

Human perception has a minimum discriminable difference — the just-noticeable difference (JND) — below which two stimuli are perceptually identical. Two patches of colour differing by one quantum of luminance are not distinguishable; two sounds differing by one sample in amplitude are not distinguishable. This resolution limit is not a bug; it is an implicit commitment to treating imperceptibly different inputs as the same input.

Models have no such threshold. Every change in a floating-point input, no matter how small, can in principle be detected and acted upon. This means a model can and will form different beliefs about inputs that a human would treat as identical. The adversary exploits exactly this gap.

---

## 3. Four Approaches to Model the Human-Like Feature Space

The gap between human and model robustness is real, but it is not necessarily permanent. Four research directions offer principled paths toward representations that share the robustness-conferring properties of human cognition.

### 3.1 Causal Representation Learning

The framework of **Invariant Risk Minimization** (IRM; Arjovsky et al., 2019) and its successors, articulated comprehensively by Schölkopf (2021), trains models to learn representations whose relationship to the target is invariant across environments. The core intuition is that spurious correlations — the proxy features that are not causes — will vary across environments, while causal features will not. By requiring that the same classifier head performs well in all environments simultaneously, IRM forces the representation toward the causal factors.

The limitation is significant: IRM requires that multiple environments with different spurious correlations be available at training time, and it benefits substantially from access to a causal graph that identifies which features to treat as potentially spurious. In most deployed settings, neither is available. Nevertheless, IRM-style objectives represent the most principled current approach to causal feature learning, and their robustness properties in tabular domains remain underexplored.

### 3.2 Multi-View Invariance and Semantic Augmentations

Standard adversarial training generates perturbations within a global $L_\infty$ ball, treating all input dimensions as equally perturbable by the same budget. This is unrealistic. In a loan application, a £50 change in monthly payments and a £50,000 change in annual income are both "budget-1" perturbations in some normalized space, but they are not semantically equivalent.

A more principled approach applies **domain-appropriate perturbation budgets per feature**: large budgets for features known to be noisy or proxy-like, small budgets for features that are direct causal indicators, and zero budget for features that are administratively fixed (e.g., date of birth). This is a form of multi-view invariance training, where the "views" are semantically meaningful perturbation directions rather than arbitrary pixel noise. The resulting model learns to be robust along the dimensions that matter semantically and discriminative along the dimensions that carry genuine signal.

### 3.3 Disentangled Representations via β-VAE

The **β-Variational Autoencoder** (Higgins et al., 2017) extends the standard VAE by increasing the weight on the KL divergence term, which encourages the latent space to disentangle independent factors of variation. A well-disentangled β-VAE separates semantic factors (class-relevant content) from nuisance factors (style, noise, instrument variation) into distinct dimensions of the latent space.

A classifier operating in a disentangled latent space is potentially more robust because adversarial perturbations that move along nuisance dimensions do not affect the semantic dimensions that determine the prediction. The challenge is that disentanglement without supervision is an ill-posed problem — the model cannot know which factors are semantic and which are nuisance without some form of inductive bias or label information. Recent work combining β-VAE with semi-supervised objectives or auxiliary losses shows promise, particularly in structured domains with interpretable factor structure.

### 3.4 Foundation Models as Semantic Proxies

The most practically tractable path currently available uses **foundation models** (CLIP, DINO, and their successors) as semantic proxies. These models are pre-trained on large, diverse, multimodal datasets using objectives — contrastive learning, masked prediction — that do not directly optimize for class boundary placement. The representations they learn are therefore less shaped by the specific spurious correlations present in any single downstream dataset.

A classifier trained in CLIP feature space, rather than in raw pixel space, is substantially harder to attack at the pixel level because the mapping from pixels to CLIP features is many-to-one in exactly the right way: it discards the nuisance variation (pixel-level texture, exact colour values) while preserving the semantic variation (object identity, scene category). The adversary must find perturbations that move the CLIP representation across a decision boundary, which requires semantic changes, not merely pixel-level changes.

This approach does not eliminate adversarial vulnerability — CLIP-space classifiers have their own decision boundaries, and those boundaries can be attacked in CLIP space. But it raises the effective cost of a successful attack and naturally aligns the model's decision surface with the human-meaningful variation in the data.

---

## 4. The Core Tension

All four approaches above are attempts to resolve a single underlying tension: **hard boundaries in dense data regions are inherently vulnerable**.

A decision boundary must exist wherever two classes co-occur in the data distribution. In a region where both classes are well-represented, the boundary passes through a high-density region. Every clean example in that region is close to the boundary by definition, so a small perturbation can cross it. Ensembles, regularization, and data augmentation do not resolve this tension; they only redistribute the boundary within the dense region.

Humans avoid this problem — in practice, if not in theory — because semantic features naturally separate data more cleanly than raw features. The semantic space in which a human represents a loan application or a clinical image is one in which the "same case under different nuisance conditions" maps to a single point, and "genuinely different cases" maps to well-separated points. The decision boundary in semantic space therefore lies in a sparse region of semantic space, even if the corresponding region of input space is dense.

The lesson is not that humans are using a more complex classifier. The lesson is that humans are classifying in a *different space*, and that space happens to have more favourable geometric properties for robust classification.

---

## 5. The Fundamental Theorem

A 2026 result (arXiv:2604.21395, April 2026) makes this tension precise at the theorem level. The result proves that **any model minimizing supervised cross-entropy retains sensitivity along label-correlated directions in input space**. Specifically: if a feature direction $v$ in input space is correlated with the label (i.e., moving along $v$ increases the probability of the correct class), then the gradient of the cross-entropy loss with respect to the input has a non-negligible component along $v$. This component is exploitable by an adversary.

The implication is that cross-entropy-trained models cannot, even in principle, become robust to perturbations along label-correlated directions without sacrificing accuracy. The robustness-accuracy tradeoff is not an empirical regularity that better training might overcome; it is a mathematical consequence of the objective.

Two principled escapes are identified:

1. **Change the loss** — objectives like RLHF that incorporate human feedback can learn boundaries aligned with human-level discriminability rather than label correlations in the training data. The boundary moves to where humans would place it, which is semantically grounded and therefore less exploitable.

2. **Change the feature space** — foundation models trained at scale on multimodal data with contrastive or masked objectives learn representations in which the label-correlated directions are genuinely semantic. The theorem still applies in this new space, but the adversary now needs to make semantically meaningful changes to exploit it.

Neither escape is free: RLHF requires large-scale human feedback, and foundation models require large-scale pre-training. But both are available and in active deployment. The theorem establishes that they are not merely engineering improvements; they are the only structural paths out of the vulnerability.

---

## 6. A Concrete Example: The Loan Dataset (H187)

Consider an applicant with an annual income of £116,000. By any reasonable standard, this individual has substantial repayment capacity. The model denies the application because the debt-to-income (DTI) ratio changed by 0.041 — corresponding to approximately £50 per month in additional outgoings.

A human loan officer reviewing this case would not treat a £50/month change as decision-relevant for an applicant earning £116,000/year. The change is within the noise of reported financial data: self-reported monthly outgoings have measurement error on the order of £100–200 for most applicants. The officer's implicit JND for DTI at this income level is larger than 0.041. The perturbation is subthreshold.

The model has no such threshold. The DTI feature enters the model as a floating-point value; a change of 0.041 is as real as a change of 4.1. The model's decision boundary, drawn to maximize discriminative accuracy on the training distribution, happens to pass through the region near this applicant's feature vector. The perturbation crosses the boundary.

This example is not exceptional; it is the norm for models operating on high-dimensional tabular data near decision boundaries. It illustrates that adversarial vulnerability in the loan domain is not primarily a failure of model complexity or training data size. It is a failure of the model to represent the income-to-repayment causal structure in a way that makes the DTI perturbation semantically irrelevant at high income levels.

A causally grounded model — one that understood that income directly constrains repayment capacity and that DTI is a derived ratio with measurement variance — would assign near-zero weight to a 0.041 DTI change for a £116k earner. The adversarial example would not exist in the causal feature space.

---

## 7. Open Hypothesis: Conformal Abstention as a Human-Uncertainty Analog (H214)

The abstention mechanism described in Section 2.2 is not only a conceptual argument; it is a testable empirical hypothesis. **Conformal prediction** provides a principled framework for abstention: a conformal predictor outputs a prediction set (possibly empty, possibly multi-class) calibrated to achieve a specified coverage guarantee on the clean distribution.

The hypothesis (H214, proposed but not yet implemented) is that **the abstention rate of a conformal predictor should be measurably higher for adversarial examples than for clean examples**. This follows from the boundary-proximity argument: adversarial examples are constructed to be close to decision boundaries, where the model's confidence is lowest. A conformal predictor calibrated on clean examples will have wider prediction sets (or empty prediction sets, triggering abstention) precisely in the regions where adversarial examples are concentrated.

If confirmed, this would have two implications. First, it would provide a detection mechanism for adversarial examples without requiring access to an adversarial training distribution. Second, it would validate the human-analogy argument: the conformal abstention rate, calibrated on clean data, would serve as a proxy for the human JND threshold — the model's learned acknowledgement that some inputs are simply too ambiguous to classify reliably.

The hypothesis is straightforwardly testable against any of the tabular or vision datasets in the existing campaign. It requires only a conformal calibration step on a held-out clean set, followed by measuring abstention rates separately on clean test examples and on adversarially perturbed examples. The expected finding is a statistically significant difference, with adversarial examples triggering abstention at a substantially higher rate.

---

## References

- Arjovsky et al. (2019). Invariant Risk Minimization. arXiv:1907.02893.
- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR 2017.
- Schölkopf et al. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*, 109(5), 612–634.
- Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
- Caron et al. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). ICCV 2021.
- arXiv:2604.21395 (April 2026). [Theorem on cross-entropy sensitivity along label-correlated directions.]
