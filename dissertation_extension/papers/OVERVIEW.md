# Dissertation Extension: Overview and Research Narrative

**Title:** Per-Sample Adversarial Vulnerability: Geometry, Signals, and Interventions  
**Dataset:** Fashion-MNIST (28×28 grayscale, 10 classes, 10 000 test samples)  
**Framework:** PyTorch; ART (Adversarial Robustness Toolbox) for attack execution  
**Campaign:** 522 hypothesis runs, 0 errors

---

## 1. Central Research Question

> *Can we predict, for a specific input sample, whether it will be successfully attacked — and if so, what signals, training dynamics, and interventions govern that vulnerability?*

This dissertation extension answers yes: the **logit margin** (softmax probability gap between the top and second class) is a reliable, calibrated per-sample vulnerability predictor across models, attacks, architectures, and training regimes. The eight papers below establish this claim, characterise its limitations, and explore interventions that exploit the vulnerability signal.

---

## 2. Research Themes

| Theme | Core Claim | Papers |
|-------|-----------|--------|
| **A — Geometry** | Margin predicts per-sample vulnerability; geometry is primary | 1, 2, 5, 8 |
| **B — Redundancy of signals** | Attribution, learning dynamics, and random features all encode the same margin signal | 3, 4, 6 |
| **C — Interventions** | Training interventions (AT, confirmatory, unlearning, relabelling) shift the vulnerability geometry | 2, 7, 8 |
| **D — Null results** | Human-illusion labelling and dynamic AUM trajectories add no independent signal | 4, H184 |

---

## 3. Paper Summaries and Key Results

### Paper 1: Benchmark — Margin as Per-Sample Vulnerability Predictor
**File:** `paper1_benchmark.md`  
**Hypothesis:** H150–H155

Establishes the benchmark AUROC for predicting FGSM and PGD attack success from vanilla model features (margin, confidence, gradient norm, min-ε). Best deployable predictor: **margin AUROC = 0.9575**, gradient norm AUROC = 0.9591 (statistically tied). min-ε achieves 0.9934 but is a near-oracle (leaks the label; not deployable). Resolution sensitivity caveat: 28×28 protocol; native-resolution check recommended for future work.

**Key numbers:**

| Feature | FGSM AUROC | PGD AUROC |
|---------|-----------|-----------|
| Margin | 0.957 | 0.958 |
| Confidence | 0.956 | 0.957 |
| Grad norm | 0.959 | 0.959 |
| Min-ε | 0.993 | 0.993 |

---

### Paper 2: Structural Vulnerability — Cross-Model, Cross-Attack Transferability
**File:** `paper2_structural_vulnerability.md`  
**Hypothesis:** H156, H175–H181

Vulnerability is not model-specific: high-margin samples of one model are high-margin in others, and their adversarial vulnerability transfers. kNN, GBM, logistic regression, and shallow feedforward networks — all trained on Fashion-MNIST features — show AUROC 0.77–0.99 for vulnerability prediction, confirming that vulnerability is a **data-geometry property**, not an artefact of gradient-descent training (Theme B extension). Cross-attack ASR: FGSM 0.345, PGD 0.494, BIM 0.505, MIM 0.473.

**Key insight:** Transferability of vulnerability across architectures is consistent with Tramèr (2017) and Demontis (2019): adversarial subspaces are shared because they reflect the data manifold geometry, not the model's parametrisation.

---

### Paper 3: Attribution Signals — Can Saliency Maps Predict Vulnerability?
**File:** `paper3_attribution_signals.md`  
**Hypothesis:** H160–H165

Gradient-based attribution signals (gradient magnitude, integrated gradients, GradCAM activations) collapse uniformly under adversarial perturbation: all norms shrink by ~22×. This is a **rank-order destruction** effect (floor effect), not merely a scale change — AUROC is scale-invariant, so the collapse damages discriminative ranking. Smoothed gradients (σ=0.2) partially recover the collapse direction: AUROC drops from 0.78 to 0.586 but signal survives. Attribution maps are redundant with margin: they add no independent predictive content once margin is known.

**Caveat:** AT-model evaluation is class-imbalance-confounded (ASR ~10% → AUROC is minority-dominated). Collapse direction is robust to this confound.

---

### Paper 4: Learning Dynamics — AUM, Forgetting, C-Score
**File:** `paper4_learning_dynamics.md`  
**Hypothesis:** H166–H172

Training-time dynamics signals (Area Under the Margin trajectory, forgetting count, C-score) are **orthogonal to attack vulnerability**. AUM at eval time correlates near-perfectly with final margin (Spearman 0.983) — it is redundant, not independent. Forgetting count and C-score add no signal beyond AUM. The null result is robust: the Spearman between eval-AUM and final margin explains the apparent AUM predictive power entirely. **Note:** eval-set AUM ≠ Pleiss et al. training-set AUM; the two constructs are conceptually different.

---

### Paper 5: Defense Geometry — Does Adversarial Training Protect Hard Samples?
**File:** `paper5_defense_geometry.md`  
**Hypothesis:** H173

Adversarial training (PGD-AT) reduces attack success from 94.5% to 5.5% on average — but this benefit is **not uniform**. Pre-AT low-margin (hard) samples retain disproportionately high vulnerability after AT. Adversarial logit pairing (ALP) degrades margin AUROC from 0.960 to 0.714 (ΔAUROC = −0.246) — it disrupts the geometric signal that makes per-sample prediction possible. Vanilla AT preserves the margin-AUROC correlation; ALP does not.

---

### Paper 6: Random Features — Is the Margin Signal in the Data or the Trained Model?
**File:** `paper6_random_features.md`  
**Hypothesis:** H182

Random convolutional projections (untrained CNNs with random weights) applied to Fashion-MNIST images, followed by a supervised logistic-regression readout, achieve **AUROC 0.938–0.972** for vulnerability prediction — nearly matching the trained-CNN benchmark. This confirms that the vulnerability signal is present in **raw data geometry** before any training, not just in learned representations. Limitations: the readout is supervised (label-free baseline not tested); the three AUROC values are evaluated at different eps/steps (not inconsistencies — different experimental conditions).

---

### Paper 7: Confirmatory Training — Pushing Samples Deeper into Their Class Interior
**File:** `paper7_confirmatory_training.md`  
**Hypothesis:** H158

Confirmatory examples (gradient-descent perturbations toward the class interior, opposite direction to adversarial examples) increase margin AUROC to **0.996** but *worsen* robustness: PGD attack success rises from 2.9% (vanilla) to 97.1% (Anti-Adv Aug alone). The mechanism: confirmatory training sharpens boundary curvature without expanding the ε-ball. When combined with PGD-AT, +2.89 pp clean accuracy is recovered at a cost of +5.5 pp attack success. **Dissociation:** high margin AUROC is necessary but not sufficient for robustness — boundary curvature matters equally.

**Key Table (Table 1):**

| Condition | PGD ASR | Clean Acc |
|-----------|---------|-----------|
| Vanilla | 97.1% | 91.2% |
| Anti-Adv Aug | 99.4% | 89.3% |
| PGD-AT | 2.9% | 86.7% |
| PGD-AT + Anti | 8.4% | 89.6% |

---

### Paper 8: Post-AT Audit — Predicting Which Samples AT Will Hurt
**File:** `paper8_post_at_audit.md`  
**Hypothesis:** H174

AT's clean accuracy cost concentrates on pre-existing hard samples: 98 of 1800 vanilla-correct samples (5.4%) are hurt by AT. All four vanilla model features predict this outcome with **AUROC ≈ 0.935**. Three interventions tested: (1) exclude fragile bottom-10% from adversarial augmentation → +0.65 pp clean acc (within ~0.8 SE, not conclusive); (2) up-weight worst → −1.15 pp clean acc, +41% hurt; (3) up-weight best → marginally worse. **Double-jeopardy:** hard samples are most vulnerable to attacks AND most likely to be broken by AT — a systematic inequity in robustness.

---

## 4. Cross-Paper Dependency Map

```
Paper 1 (Benchmark)
  ├── Paper 2 (Cross-model): confirms geometry > model
  ├── Paper 5 (Defense): asks if AT changes geometry
  └── Paper 8 (AT audit): asks which samples AT hurts
       └── uses Paper 5's AT model as baseline

Paper 3 (Attribution)
  └── Paper 4 (Dynamics): both test alternative signals; both find redundancy with margin

Paper 6 (Random features)
  └── strengthens Paper 2's "data geometry" claim (pre-training signal)

Paper 7 (Confirmatory)
  └── motivates Paper 8's intervention design (training pressure direction matters)
```

---

## 5. Hypothesis-to-Paper Mapping

| Hypothesis | Paper | Theme | Key Result |
|-----------|-------|-------|-----------|
| H150–H155 | 1 | A | Margin AUROC 0.957 benchmark |
| H156 | 2 | A,C | Cross-model transferability confirmed |
| H158 | 7 | C | Confirmatory training worsens robustness (AUROC ceiling 0.996) |
| H160–H165 | 3 | B | Attribution signals redundant; collapse=rank destruction |
| H166–H172 | 4 | B,D | AUM/forgetting null; AUM≡final margin |
| H173 | 5 | A,C | ALP disrupts margin AUROC; AT non-uniform protection |
| H174 | 8 | A,C | AT-hurt predictable AUROC 0.935; double-jeopardy |
| H175–H181 | 2 | A,B | Non-GD learners AUROC 0.77–0.99; Theme B extension |
| H182 | 6 | B | Random features AUROC 0.938–0.972; signal pre-training |
| H183 | — | C | Unlearning: 54% adversarial recovery; attractor 0.081→0.001 |
| H184 | — | D | Illusion labels: constant-prediction model (degeneracy, not masking) |

---

## 6. New Hypotheses (Post-Campaign)

### H183: Machine Unlearning and Adversarial Residual Knowledge
**File:** `hypotheses/h183_unlearning_adversarial_holes.py`  
**Result file:** `results/fashion_mnist/h183_unlearning_adversarial_holes_output.txt`

After finetune_retain class-unlearning on Fashion-MNIST, the "forgotten" class (T-shirt/top) drops clean accuracy from 0.839 to 0.368. However, targeted PGD (eps=0.05) recovers 53.6% of these samples to their true label — indicating **residual knowledge** in the model weights. The adversarial attractor rate (fraction of other-class adversarials landing in the forgotten class) drops from 0.081 to 0.001 after unlearning, confirming partial but incomplete erasure.

**Implication:** Machine unlearning via fine-tuning does not fully erase adversarial pathways; certified unlearning or weight-space regularisation may be required.

### H184: Illusion Robustness — Gradient Masking or Label Degeneracy?
**File:** `hypotheses/h184_illusion_robustness_masking.py`  
**Result file:** `results/fashion_mnist/h184_illusion_robustness_masking_output.txt`

Human-illusion labelling schemes produce models with PGD ASR = 0 — previously interpreted as robustness. H184 diagnoses this as **label degeneracy**: the human-illusion labels are near-constant, producing a constant-prediction model. Input gradient norm = 0.00, black-box ASR = 0, transfer ASR = 0 — all consistent with a model that ignores the input, not one with genuine robustness. Physical-scheme models show PGD=1.0, black-box=0.007 (sharp narrow boundary) — genuine vulnerability, not masking.

---

## 7. Key Recurring Findings

1. **Margin is king.** Across 8 papers, 522 experiments, and 11 hypotheses, the logit margin remains the most reliable per-sample vulnerability predictor. No tested alternative (attribution signals, learning dynamics, random features, post-AT audit features) surpasses it while adding independent information.

2. **Vulnerability is data-geometric.** Non-gradient-descent learners (kNN, GBM, logistic regression) and random untrained projections achieve AUROC 0.77–0.99. The signal is in the data manifold, not the training procedure.

3. **Interventions must respect boundary geometry.** Both confirmatory training (Paper 7) and ALP (Paper 5) demonstrate that improving one geometric property (margin AUROC, internal consistency) does not guarantee improved robustness if boundary curvature is simultaneously worsened.

4. **Double jeopardy is systematic.** Hard samples (low margin) are both most vulnerable to adversarial attacks and most likely to be harmed by adversarial training. Any robustness intervention that treats samples uniformly exacerbates this inequity.

5. **Null results are informative.** AUM/forgetting (Paper 4), illusion labelling (H184), and multi-task attribution (Paper 3) all produce clean nulls: they demonstrate that certain plausible signals are redundant with or weaker than margin, sharpening the margin claim.

---

## 8. Prior Art and Positioning

| This work | Prior art |
|-----------|-----------|
| Per-sample margin AUROC benchmark | Margin Consistency [2406.18451] — margin as vulnerability score |
| Vulnerability scales with input complexity | Simon-Gabriel et al. — vulnerability ∝ input dimension |
| Cross-model transferability | Tramèr 2017, Demontis 2019 — adversarial subspaces |
| Margin-adaptive AT exclusion | DyART [2302.03015] — soft margin-adaptive boundary control |
| AT non-uniform cost concentration | Rice et al. 2020 — AT overfitting; Stutz et al. 2019 — robustness vs generalisation |

---

## 9. Dataset and Experimental Protocol

- **Dataset:** Fashion-MNIST, 60k train / 10k test, 10 classes, 28×28 grayscale
- **Architecture:** Standard CNN (2 conv + 2 fc layers) unless otherwise noted
- **Attacks:** FGSM (eps=0.1), PGD-10 (eps=0.1, step=0.01), AutoAttack ensemble (Papers 1–2), black-box random search L∞ (K=50)
- **Adversarial training:** Madry PGD-AT, 10 steps, eps=0.1
- **Evaluation:** AUROC (rank-based, scale-invariant), Spearman ρ (monotone correlation), ASR (attack success rate)
- **Statistical note:** Most results are single-seed; Poisson SE or binomial SE reported where computed; multi-seed ablation is the primary future work item across all papers

---

## 10. File Index

```
dissertation_extension/
├── papers/
│   ├── OVERVIEW.md                          ← this file
│   ├── REVIEWER_CRITIQUE.md                 ← adversarial reviewer comments + resolutions
│   ├── paper1_benchmark.md
│   ├── paper2_structural_vulnerability.md
│   ├── paper3_attribution_signals.md
│   ├── paper4_learning_dynamics.md
│   ├── paper5_defense_geometry.md
│   ├── paper6_random_features.md
│   ├── paper7_confirmatory_training.md
│   └── paper8_post_at_audit.md
├── hypotheses/
│   ├── h183_unlearning_adversarial_holes.py
│   └── h184_illusion_robustness_masking.py
└── results/fashion_mnist/
    ├── h183_unlearning_adversarial_holes_output.txt
    └── h184_illusion_robustness_masking_output.txt
```
