# Research Paper Analysis — Burkule 2026 Extension Study

Full analysis of how many distinct publishable papers the H107–H162 data supports,
what each paper would argue, how strong the evidence is, what flaws exist,
and which new hypotheses are needed to close each gap.

*Written: 2026-05-24*

---

## Overview: How Many Papers?

**6 papers are clearly supportable from existing data. 2 more are emergent from the new H157–H162 hypotheses (results pending).**
Below each is graded on evidence strength across the three completed datasets
(Fashion-MNIST = FM, CIFAR-10 = C10, Imagenette = IN).

---

## Paper 1 — The Main Benchmark Paper
**"What Predicts Per-Sample Adversarial Vulnerability? A Systematic Empirical Benchmark of 50 Predictors"**

### What it argues

A single deep neural network trained on image classification has some samples that
are trivially adversarially flippable and others that are surprisingly robust.
*Why* differs between samples. This paper systematically evaluates 50 candidate
per-sample predictors — from logit margin and attribution norms to training dynamics
and architectural signals — across four attack types (FGSM, BIM, PGD, MIM),
three datasets, and two architectures (CNN + ViT). The key contributions are:

1. **The logit margin is the dominant univariate predictor** (AUROC 0.87–0.9976 across
   all contexts). It serves as the benchmark baseline no other feature consistently beats.
2. **A ranked taxonomy of 28 predictors** from effective (smoothgrad 0.97, ensemble
   entropy 0.95) to useless (forgetting events 0.54, C-score 0.53).
3. **Monotone margin reparameterisations are equivalent** (confusion ratio AUROC exactly
   equals margin AUROC — H150). All logit-gap re-encodings carry identical information.
4. **Soft ensemble signals dominate hard vote signals** (softmax variance AUROC 0.92–0.97
   vs hard vote agreement 0.53 — H132, H141). Calibrated probabilities must be retained.
5. **Adversarial vulnerability is a structural, attack-invariant property** (H148/H156).
   74.7–94% of samples are universally vulnerable to all 4 attacks simultaneously.

### Evidence strength

| Finding | FM | C10 | IN | Verdict |
|---|---|---|---|---|
| Margin dominance | ✅ 0.94–0.99 | ⚠️ PGD saturated, min_eps 0.88+ | ✅ 0.90–0.99 | Robust |
| SmoothGrad ≈ margin | ✅ 0.97 | ✅ 0.93 FGSM | ✅ 0.98 | Robust — actually STRONGER on harder datasets |
| Forgetting/C-score ≈ random | ✅ 0.53–0.54 | ❓ Not compared | ❓ Not compared | FM only — needs cross-dataset confirmation |
| Structural vulnerability 73–94% | ✅ 74.7% | ✅ 94.0% | ✅ 93.3% | Very robust — replicates strongly |
| Soft > hard ensemble | ✅ 0.97 vs 0.53 | ⚠️ PGD saturated | ❓ partial | Partially confirmed |

### Critical flaws

**Flaw 1 — CIFAR-10 PGD saturation (major)**
On CIFAR-10 with ε=15/255, PGD achieves 100% attack success on 15 of 18 model
variants. When the target variable has zero variance (everyone flipped), AUROC is
undefined (outputs 0.5 trivially). This means almost all CIFAR-10 PGD-based AUROC
results are **methodologically void**. Only FGSM and min_eps targets are valid for
CIFAR-10 as currently set up. The entire CIFAR-10 section of the paper is weakened.

**Flaw 2 — Single random seed**
Every hypothesis script uses `seed=0`. AUROC scores from a single training run have
no confidence intervals. Any finding based on AUROC differences < 0.05 could be seed
artefact.

**Flaw 3 — No multivariate analysis**
All 50 features are evaluated *univariately*. The diagnostic_test (Finding 4, README)
showed that `S(x)` adds Δ AUROC = 0.0000–0.0008 beyond margin in a multivariate model.
The same collinearity may apply to most other features — smoothgrad, gradient L2 norm,
and confusion ratio are all monotone functions of boundary proximity and likely add
zero independent information beyond the margin. Without a proper multivariate AUROC
or conditional independence test, the "feature A is a good predictor" claim is
conflated with "feature A is a good proxy for margin."

**Flaw 4 — Only L-inf attacks, 10 training epochs**
The entire study uses L-inf gradient attacks. L2 attacks (C&W) and semantic attacks
might reveal different vulnerability structures. 10 training epochs may leave models
under-trained relative to their full capacity.

**Flaw 5 — H135 ensemble > margin fails cross-dataset**
On CIFAR-10, ensemble predictive entropy achieves only 0.64 AUROC vs PGD (saturated,
meaningless). On Imagenette, margin AUROC (0.98) > ensemble entropy AUROC (0.96) —
the direction reverses. The "ensemble beats margin" claim is Fashion-MNIST specific.

### New hypotheses needed to close gaps

- **H163: CIFAR-10 re-run with ε=4/255** — re-run all CIFAR-10 hypothesis scripts
  with reduced epsilon to avoid 100% PGD saturation. This fixes Flaw 1 and unlocks
  valid cross-dataset PGD AUROC comparisons.
- **H164: Multivariate conditional AUROC (partial dependence test)** — for the top-10
  features, compute Δ AUROC of each feature after conditioning on margin. Determines
  which features have genuine independent predictive value vs are margin proxies.
- **H165: Multi-seed stability (3 seeds × top-10 features)** — repeat top-10 feature
  evaluations with seeds 0, 1, 2 on Fashion-MNIST and report mean ± std AUROC.

---

## Paper 2 — The Structural Vulnerability Paper
**"Adversarial Vulnerability is Structural and Attack-Invariant: Evidence from a Multi-Attack Overlap Audit"**

### What it argues

The adversarial robustness community often treats vulnerability as attack-specific
(a sample that is vulnerable to FGSM may be robust to PGD, etc.). This paper shows
that is wrong: vulnerability is a *structural property* of the sample's position
relative to the decision boundary, not an artefact of any particular attack.

1. **73–94% of samples are universally vulnerable** to all four attacks simultaneously
   (H148). Only 1–2% are immune to all attacks. The "attack-specific vulnerability"
   model is empirically false.
2. **BIM, PGD, and MIM identify near-identical vulnerable sets** (Jaccard ≥ 0.99 on FM,
   confirmed on C10 and IN). These three attacks are functionally interchangeable
   vulnerability detectors.
3. **FGSM identifies a different, slightly smaller vulnerable set** (FGSM ∩ PGD Jaccard
   ~0.74 on FM). FGSM vulnerability is a strict subset of iterative vulnerability.
4. **The margin predicts "universally vulnerable" membership with AUROC 0.70–0.91**
   across datasets — the same feature that predicts single-attack vulnerability predicts
   universal cross-attack vulnerability.

### Evidence strength

| Claim | FM | C10 | IN |
|---|---|---|---|
| 73–94% universally vulnerable | ✅ 74.7% | ✅ 94.0% | ✅ 93.3% |
| BIM/PGD/MIM near-identical sets | ✅ Jaccard ≥ 0.99 | ❓ (saturated, trivially 100%) | ❓ need Jaccard |
| Margin predicts universal vuln | ✅ 0.90 | ✅ 0.70 (FGSM) | ✅ 0.91 |

This is the **strongest paper in the set**. The 94% finding on CIFAR-10 and Imagenette
is actually more striking than the Fashion-MNIST result.

### Critical flaws

**Flaw 1 — Only gradient-based attacks**
All four attacks (FGSM, BIM, PGD, MIM) use gradient information from the same model.
They may share structural bias. A fairer test requires including a *black-box* attack
(e.g. SimBA from H113, ZOO from H114) in the overlap analysis. If black-box attack
vulnerable sets also overlap at 90%+ with PGD vulnerable sets, the structural claim
is much stronger.

**Flaw 2 — H156 orthogonality claim is dataset-specific**
H156 finds that FGSM and iterative attacks have orthogonal feature importances (r=−0.034)
on Fashion-MNIST. But on Imagenette the correlation is +0.90, and on CIFAR-10 all
correlations are NaN (a script error from 100% attack success causing constant target
vectors). The "FGSM exploits fundamentally different geometry" claim is not established
cross-dataset and must be qualified or removed.

**Flaw 3 — 1000 sample evaluation**
The 74.7% / 94% / 93.3% figures come from 1000 test samples each. With only 1.0–7.3%
of samples in the partial-vulnerability regime, the sample counts for those cells are
very small (10–73 samples). These proportions have high variance.

### New hypotheses needed

- **H166: Multi-attack overlap including black-box attacks** — re-run H148 with 6 attacks
  (FGSM, BIM, PGD, MIM, SimBA, ZOO). Test whether black-box attack vulnerable sets
  overlap as strongly as white-box sets.
- **H167: Fix H156 on CIFAR-10** — the NaN correlations come from target variable being
  constant (PGD 100% success). Re-run H156 with ε=4/255 on CIFAR-10 to get valid
  feature importance correlations and test the orthogonality claim cross-dataset.

---

## Paper 3 — The Attribution Signals Paper
**"Noise-Averaged Attribution Norms Encode Adversarial Boundary Proximity as Well as the Logit Margin"**

### What it argues

Adversarial vulnerability prediction has relied on model-internal signals (logit margin,
softmax entropy). This paper shows that *attribution methods* — specifically SmoothGrad —
independently encode boundary proximity with accuracy matching the logit margin, without
requiring knowledge of the logits.

1. **SmoothGrad L2 norm achieves 0.97–0.98 AUROC** (FM, IN), **exceeding** the logit
   margin on CIFAR-10 FGSM (0.93 vs 0.75). The margin's lower AUROC on CIFAR-10 FGSM
   is because CIFAR-10 is harder (lower clean accuracy), and the margin becomes a noisier
   signal; but SmoothGrad remains stable.
2. **The gradient L2 norm (plain, unnormalized) also matches margin** (H151: 0.9408 AUROC
   on FM, slightly *exceeding* margin 0.9366). This is the simplest possible attribution
   signal — just one forward-backward pass.
3. **Integrated Gradients (H127) are weaker** (0.76 AUROC) because path integration from
   baseline captures semantic attribution, not boundary proximity oscillation.
4. **Saliency spatial structure** (H149) provides moderate prediction (0.78–0.85):
   peripherally concentrated saliency → more vulnerable. This gives geometric intuition
   for what vulnerability "looks like" in attribution space.

### Evidence strength

| Signal | FM | C10 FGSM | IN FGSM | IN PGD |
|---|---|---|---|---|
| SmoothGrad L2 | 0.9704 | 0.9252 | 0.9847 | 0.9975 |
| Margin | 0.9710 | 0.7481 | 0.9597 | 0.9920 |
| SmoothGrad advantage | ≈0 | **+0.18** | **+0.025** | **+0.006** |

**On harder datasets, SmoothGrad consistently beats the margin.** This is the headline finding.

### Critical flaws

**Flaw 1 — Computation cost not justified**
SmoothGrad requires 50+ forward-backward passes per sample (adding Gaussian noise each
time). The paper's practical claim is undercut if the clean gradient L2 norm (one pass,
H151) achieves nearly identical AUROC (0.94 vs 0.97). The paper needs to compare
information per compute unit, not just absolute AUROC.

**Flaw 2 — No AT model evaluation**
All attribution experiments are on vanilla (non-adversarially trained) models. It is
unknown whether SmoothGrad retains its AUROC advantage on AT models, where the decision
boundary geometry is qualitatively different (expanded margin). H147 partial data shows
AT model margin AUROC *increases* — does SmoothGrad track that?

**Flaw 3 — No cross-attack consistency check for attribution signals**
The multivariate finding (H156 CIFAR-10 NaN) means we don't know if SmoothGrad remains
a strong predictor for iterative attacks vs single-step FGSM on CIFAR-10.

### New hypotheses needed

- **H168: SmoothGrad vs gradient L2 norm head-to-head with compute budget** — explicitly
  compare smoothgrad (k=50 samples), smoothgrad (k=5), and plain gradient L2 at identical
  compute cost. If plain grad L2 ≈ smoothgrad, the expensive attribution is unnecessary.
- **H169: Attribution signals on AT models** — repeat H126 (SmoothGrad) and H151
  (gradient FFT) on FGSM-AT and PGD-AT trained models to see if attribution-vulnerability
  correlation is preserved.

---

## Paper 4 — The Learning Dynamics Paper
**"Training Difficulty is Orthogonal to Adversarial Vulnerability: A Negative Result with a Nuanced Exception"**

### What it argues

Two popular approaches to understanding hard-to-learn examples — forgetting events
(Toneva et al. ICLR 2019) and C-score (Jiang et al. 2021) — measure how difficult it
is to *learn* a sample during training. This paper shows these are orthogonal to
adversarial vulnerability at test time. The result matters because both signals have
been proposed as proxies for per-sample "quality" or "atypicality", and it is a natural
hypothesis that atypical (hard-to-learn) samples are also adversarially fragile.

1. **Forgetting event count (H129): AUROC 0.54** — effectively coin-flip prediction of
   PGD vulnerability.
2. **C-score (H130): AUROC 0.53** — same conclusion.
3. **Nuanced exception — memorization proxy (H131): AUROC 0.93** — at first this seems
   contradictory. The resolution: the memorization proxy measures *prediction variance
   across independent random-subset ensembles*, which is a boundary proximity signal,
   not a learning difficulty signal. High variance = model disagrees across subsets =
   sample is near the decision boundary.
4. The paper argues: *what matters is the boundary geometry at test time, not the learning
   trajectory during training*. The model can struggle to learn a sample for reasons
   (mislabelled neighbours, label ambiguity) that have nothing to do with whether the
   final model places that sample close to a decision boundary.

### Evidence strength

All three hypotheses (H129, H130, H131) were run on Fashion-MNIST only. No cross-dataset
replication exists.

### Critical flaws

**Flaw 1 — Single dataset, no replication (major)**
The entire paper rests on Fashion-MNIST results. The claim "forgetting events are
uninformative" must replicate on at least CIFAR-10 and Imagenette. It is plausible that
on harder datasets (where the model genuinely struggles), forgetting events correlate
more strongly with final boundary proximity.

**Flaw 2 — Implementation of forgetting events may be too coarse**
H129 counts total forgetting events across 10 epochs. Baldock et al. (NeurIPS 2021)
use a more careful "learning epoch" measure. The paper needs to compare the full
Dataset Cartography characterisation (confidence, variability, correctness) not just
forgetting count.

**Flaw 3 — The memorization proxy explanation needs testing**
The claim that memorization proxy works *because* it measures boundary proximity (not
learning difficulty) is asserted but not tested. It needs a conditional AUROC: what is
the AUROC of memorization proxy *after conditioning on margin*? If it drops to ~0.5,
the explanation is confirmed.

### New hypotheses needed

- **H170: Forgetting events on CIFAR-10 and Imagenette** — re-run H129 and H130 on
  harder datasets. If forgetting events remain near-random, the claim is strong.
- **H171: Full Dataset Cartography vs vulnerability** — compute the full three-feature
  cartography space (confidence, variability, correctness) and evaluate all three against
  vulnerability. This replaces H129's simple count with the proper Swayamdipta (2020)
  implementation.
- **H172: Memorization proxy conditional on margin** — conditional AUROC test to confirm
  that memorization proxy adds no information beyond margin (i.e., high-memorisation =
  boundary-proximate by definition).

---

## Paper 5 — The Defense Geometry Paper
**"How Adversarial Defenses Reshape Decision Boundary Geometry: Margin Predictability as a Diagnostic"**

### What it argues

Adversarial defenses are typically evaluated by: (1) clean accuracy, and (2) adversarial
accuracy. This paper proposes a third diagnostic: **how much does a defense reduce the
margin's predictive power?** The intuition: if a defense scatters the decision boundary
in a way that breaks the margin's informativeness (low-margin samples are no longer the
vulnerable ones), then the defense is geometrically non-trivial.

1. **CURE (curvature regularization, H118) preserves margin predictability** (AUROC change
   ≈ 0 on FM/IN) while meaningfully reducing PGD success (93% → 39.5% on FM). Signature:
   uniform margin expansion without reordering.
2. **PGD-AT (adversarial training, H147 partial) increases margin predictability** (+0.025)
   while dramatically expanding mean min-ε (0.031 → 0.209). Adversarial training uniformly
   pushes all samples away from the boundary.
3. **Augmentation-based defenses (AugMix −0.15, Manifold Mixup −0.21, ALP −0.25) reduce
   margin predictability** while providing only moderate robustness. These defenses scramble
   boundary geometry without uniformly expanding it.
4. **Proposed taxonomy**: defenses that *uniformly expand* boundaries (CURE, PGD-AT) preserve
   the margin's predictive ordering. Defenses that *distort* boundary geometry (augmentation,
   logit-pairing) reduce it. This is a new way to characterise what a defense actually does
   to the decision boundary.

### Evidence strength

| Defense | FM PGD success | FM margin AUROC Δ | C10 | IN |
|---|---|---|---|---|
| CURE | 93→39.5% | **+0.003** | ⚠️ C10 PGD saturated | ≈0 (−0.001) |
| PGD-AT | 95→~0.7% | **+0.025** (partial) | ❓ bugs | ❓ bugs |
| AugMix | 94→70% | **−0.151** | ❓ needs rerun | ❓ |
| Manifold Mixup | 94→88% | **−0.211** | ❓ | ❓ |
| ALP | 95→70% | **−0.246** | ❓ | ❓ |

**The FM findings are the cleanest; cross-dataset evidence is sparse** because
H118–H124 were not extensively compared across datasets.

### Critical flaws

**Flaw 1 — H144–H147 were buggy; patched but partial data**
Four key hypotheses (SAT, FAT, adversarial distillation, margin-weighted AT) all failed
due to a PGD return-type bug. They were patched but not fully re-run. The partial data
(H147: vanilla model before crash) suggests PGD-AT is the strongest defense, but the
margin-weighted variant comparison is incomplete.

**Flaw 2 — Only Fashion-MNIST for defense comparison**
The defense comparison (H118–H124) was not re-run on CIFAR-10 or Imagenette in a way
that gives valid PGD AUROC results (CIFAR-10 saturated). The margin-predictability
diagnostic is unverified on harder datasets.

**Flaw 3 — 10 training epochs is too short for AT**
Adversarial training typically requires 100+ epochs to converge properly (Rice et al.
2020 show overfitting in AT). The H147 partial data shows the model after only 10 epochs.
The margin-predictability change may be different at convergence.

**Flaw 4 — No Pareto analysis**
The paper makes stronger sense if it shows a Pareto curve: clean accuracy vs robustness
vs margin AUROC for all 7 defenses simultaneously. Currently the three metrics are
compared pairwise.

### New hypotheses needed

- **H173: Re-run H144–H147 on Fashion-MNIST (fixed)** — the patched scripts should be
  re-run to get full results for SAT, FAT, adversarial distillation, and margin-weighted AT.
- **H174: Defense comparison on Imagenette with valid ε** — re-run H118–H124 on Imagenette
  (where PGD does not saturate) to get valid cross-dataset margin-predictability changes.
- **H175: Extended training (50 epochs) for CURE and PGD-AT** — repeat H118 and H147 with
  50 epochs to see if the margin-predictability pattern is a training-epoch artefact.

---

## Paper 6 — The Random Features Paper
**"Per-Sample Adversarial Vulnerability is Encoded in Input Geometry: Evidence from Random Feature Networks"**

### What it argues

Deep networks trained on image classification achieve high predictive AUROC for adversarial
vulnerability (via the logit margin). A natural explanation is that the network *learns*
to encode vulnerability information during training. This paper provides evidence against
that explanation: a Random Feature Neural Network (RFNN) — where the convolutional layers
are *fixed random* and only the final linear layer is trained — achieves essentially the
same vulnerability prediction AUROC as the fully trained network.

1. **RFNN margin ≈ CNN margin on Fashion-MNIST** (0.94 vs 0.95, H142). Vulnerability is
   predictable without any learned convolutional features.
2. **RFNN is weaker on harder datasets** (IN: 0.87 vs 0.91, FGSM target). This means the
   learned features *do* add incremental value on complex datasets — but the random features
   capture the majority of vulnerability information from input geometry alone.
3. **Interpretation**: adversarial vulnerability is primarily determined by the *input-space
   geometry* of the data — where clean samples sit relative to class boundaries in raw pixel
   space — rather than by the deep learned representation. The CNN's training refines this
   geometry but does not fundamentally create it.

### Evidence strength

| Dataset | RFNN PGD AUROC | CNN PGD AUROC | Gap |
|---|---|---|---|
| Fashion-MNIST | 0.9367 | 0.9459 | 0.009 |
| CIFAR-10 | 0.9917 | (saturated) | N/A |
| Imagenette | 0.8719 | ~0.91 | 0.038 |

**The finding is strong on Fashion-MNIST but weakens on Imagenette.** The gap between
RFNN and trained CNN grows with dataset complexity.

### Critical flaws

**Flaw 1 — Only min_eps target on FM; only FGSM target on C10/IN**
The AUROC comparisons use different target variables across datasets, making direct
comparison hard. The C10 result (0.99) is on PGD which saturates, making it trivially
high for any predictor.

**Flaw 2 — One RFNN architecture (same as CNN but with frozen layers)**
The RFNN uses the same architecture as the trained CNN but with random weights. A true
architecture-agnostic test would use a different random projection (e.g. a random linear
map from pixels directly, no convolutions).

**Flaw 3 — No theoretical explanation**
The paper shows empirically that RFNN ≈ CNN for vulnerability, but doesn't explain *why*.
The "input geometry" interpretation is a narrative; a theoretical model (e.g. showing
that random convolutional features preserve the margin-relevant Lipschitz constant) would
strengthen the paper significantly.

**Flaw 4 — No comparison on AT models**
Does the RFNN still match the trained CNN margin after adversarial training? Adversarial
training explicitly modifies the model to push boundaries away from data points. If RFNN
still matches the AT model's margin, the "input geometry" argument is very strong. If it
doesn't, AT changes the geometry the CNN sees but not the input geometry.

### New hypotheses needed

- **H176: RFNN on AT model** — compare RFNN margin vs AT-CNN margin AUROC. If RFNN fails
  to predict AT model vulnerability, adversarial training creates a new learned geometry
  distinct from input geometry.
- **H177: Raw linear RFNN (no convolutions)** — replace the CNN-shaped RFNN with a single
  random linear map from 784 pixels to 128 dimensions, then a learned softmax. Tests
  whether the convolutional structure of the RFNN is essential.

---

## Paper 7 (Pending) — The Anti-Adversarial Training Paper
**"Confirmatory Examples as Training Augmentation: The Mirror Image of Adversarial Training"**

*Requires H158 results. Planned.*

### What it would argue

Adversarial training (AT) is known to reduce clean accuracy (robustness-accuracy tradeoff).
The proposed mirror experiment: generate *anti-adversarial* (confirmatory) examples by
applying gradient *descent* on the cross-entropy loss (reverse of FGSM), pushing samples
deeper into the correct-class interior. Train on a mixture of clean + confirmatory examples.

Expected findings (from H158, not yet run):
1. Anti-adversarial augmentation alone does not significantly improve clean accuracy but
   may slightly reduce adversarial vulnerability (by increasing average margin).
2. PGD-AT + anti-adversarial augmentation recovers part of the clean accuracy lost by
   AT alone — confirmatory examples counteract the excessive margin compression from AT.
3. Training *only* on confirmatory examples produces a model with very high clean accuracy
   but near-zero robustness (the opposite of AT).

### Known gap to close before paper

- H158 not yet run. Results needed.
- Closest prior work (FAT, Zhang et al. ICML 2020) uses "least adversarial" examples
  (still cross-boundary). The confirmatory framing (fully interior) must be carefully
  distinguished.

---

## Paper 8 (Pending) — The Post-AT Audit Paper
**"Predicting Which Samples Are Hurt by Adversarial Training Before Training Begins"**

*Requires H159 results. Planned.*

### What it would argue

Adversarial training necessarily reduces clean accuracy. The clean accuracy drop is not
uniform across samples — some samples that vanilla models classified correctly are
misclassified by AT models. This paper asks: can we predict *before AT* which samples
will be hurt, using only the vanilla model?

Expected findings:
1. "Newly wrong" samples (correct in vanilla, wrong after AT) have significantly lower
   vanilla margin than "preserved" samples.
2. "Newly wrong" samples also have lower min_eps in the vanilla model — they are the
   same samples that are most adversarially vulnerable in the first place.
3. This creates a predictive loop: the samples AT is meant to protect (low-margin,
   adversarially vulnerable) are also the samples AT accidentally breaks on clean inputs.

This would extend the theoretical explanation from ICLR 2023 ("excessive margin AT")
into a concrete per-sample predictive audit.

### Known gap to close before paper

- H159 not yet run. Results needed.
- H147 partial data gives a preview: PGD-AT expands mean min-ε by 6.7× but drops clean
  accuracy to 88.3%. The 11.7% accuracy loss comes from somewhere — H159 identifies where.

---

## Summary Table: Papers, Status, and Gap Hypotheses Needed

| # | Paper title (short) | Status | Key datasets needed | Gap hypotheses to add |
|---|---|---|---|---|
| 1 | Main benchmark (50 predictors) | Near-complete | CIFAR-10 with lower ε, multi-seed | H163, H164, H165 |
| 2 | Structural / attack-invariant vulnerability | Strong, near-complete | Black-box confirmation, H156 fix | H166, H167 |
| 3 | Attribution norms encode boundary proximity | Strong | AT model evaluation, compute tradeoff | H168, H169 |
| 4 | Learning difficulty ⊥ adversarial vulnerability | FM only | Cross-dataset replication, cartography | H170, H171, H172 |
| 5 | Defenses and margin predictability | Partial (bugs) | Re-run H144–H147, 50-epoch runs | H173, H174, H175 |
| 6 | Random features = input geometry | Moderate | RFNN on AT model, linear projection | H176, H177 |
| 7 | Anti-adversarial (confirmatory) training | Pending H158 | H158 results | — |
| 8 | Post-AT misclassification audit | Pending H159 | H159 results | — |

---

## Most Critical Single Gap: CIFAR-10 ε Saturation (H163)

Almost all papers above have a "CIFAR-10 evidence is weak" note because PGD achieves
100% success at ε=15/255 on CIFAR-10 (15 of 18 model variants). This makes all
CIFAR-10 PGD AUROC results meaningless (target variance = 0). Re-running with ε=4/255
would unlock valid PGD AUROC for CIFAR-10 and either confirm or challenge every
Fashion-MNIST finding in a proper second dataset.

**Recommended next run**: implement H163 as a variant of patch_dataset that passes
`EPS_OVERRIDE=4/255` and re-runs H107–H156 for CIFAR-10.

---

## The H156 FGSM Orthogonality Problem

The "FGSM and iterative attacks exploit orthogonal geometric properties" claim (r=−0.034,
Paper 2) is one of the most striking results. But:
- On CIFAR-10: NaN (all target vectors constant → feature importances degenerate)
- On Imagenette: r=+0.90 (high positive correlation — the opposite!)

On Imagenette, FGSM and PGD feature importances are nearly identical. This means the
orthogonality claim may be a Fashion-MNIST artefact. The reason is likely that on
Fashion-MNIST, PGD achieves 97%+ success while FGSM achieves only 70% — FGSM has a
different "error regime" (only the easiest samples flip) whereas PGD flips almost
everything. On Imagenette, both attacks have similar ~90%+ success, so they target
the same samples.

**This is a significant flaw in Paper 2** and requires H167 before publication.

---

*Gap hypotheses H163–H177 are documented here; implementations to follow.*
