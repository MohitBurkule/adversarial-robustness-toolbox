# Campaign Gap Map: H173 - H413

Scope: Fashion-MNIST, SmallCNN (pure-torch, no ART). All findings here are
read off `RESULTS_SUMMARY.md`, `papers/OVERVIEW.md`, the script docstrings
of `hypotheses/h173_*.py - hypotheses/h433_*.py`, and selected
`results/fashion_mnist/h*_output.txt` files (notably H391).

Default protocol throughout the campaign:
N_TRAIN = 6000, EPOCHS = 10, eps = 0.1, PGD-10, single GPU (RTX 4090),
single seed unless otherwise stated.

---

## Section 1 - Findings so far (clusters)

### C1. AT variants and AT-augmenting losses (SUPPORTED but mostly replications)
- H173 (SAT/FAT/RSLAD/Margin-weighted), H303 ALP, H304/H346 TRADES,
  H316/H351 AWP, H324/H358 VAT, H342/H326 AT+grad-penalty,
  H348 SupCon-AT, H363 Reverse-KL TRADES, H367 Adv-Mixup, H370 ELLE+AT.
- Finding: standard PGD/FGSM-AT is the only consistently effective lever
  (PGD ASR ~0.32-0.34 vs ~0.92 baseline). TRADES, ALP, Reverse-KL TRADES,
  adv-mixup match it; AWP gives a small additive gain (-0.02 pp);
  everything ELSE (ELLE, SupCon, VAT, Fisher-Rao H350, FGSM-AT+ALP) either
  ties AT or collapses. **Evidence:** RESULTS_SUMMARY entries for H304,
  H324, H342, H348, H363, H367; H348 supcon collapsed clean to ~0.077.

### C2. Implicit / gradient-penalty defences (overwhelmingly NEGATIVE)
- H281, H285-H288, H292-H299, H305-H319, H321-H322, H325-H326,
  H329-H342, H349, H352-H355, H361, H369, H372, H385-H386, H389-H390,
  H398-H402.
- Finding: ~30 variants of input-gradient L1/L2, Hessian-trace,
  curvature, double-backprop, spectral, contrastive-gradient, etc. give
  marginal or zero PGD reduction at the campaign's 6k-sample/10-epoch
  scale. Only **H323 Jacobian-Frobenius** (PGD 0.68), **H344 Jacobian
  spectral** (PGD 0.68), **H365 confidence-weighted grad-penalty**
  (PGD 0.65), and **H372 per-layer Jacobian** (PGD 0.76) cross the
  "meaningful gain" line, and even those don't stack with AT (H342).
- **Evidence:** H288 (7-condition comparison) - "input-grad penalty is
  best (PGD 0.514) but does not stack with AT"; H342 "gp adds nothing on
  top of AT"; H391 masking battery confirms input-grad penalty model is
  not robust, jacobian-penalty model is **genuinely** robust (transfer
  0.059, 10x-restart 0.309).

### C3. Optimizer / weight-space tricks (NEGATIVE except for one surprise)
- H273 SAM (worse), H274 SWA (FGSM-only), H276 spectral-norm (worse),
  H281 grad-norm penalty (null), H286/H292 batch-grad-variance (small),
  H300/H328 grad-clipping (masking or null), H301 Lion (masking),
  H302 grad-centralization (worse), H309 batch-size (under-training
  confound), H314 natural-gradient (masking), H371/H375/H377/H380
  weight-trajectory budget, H374/H378 layer-LR mimic, H376 anti-SAM
  (POSITIVE, PGD 0.79), H379 trajectory straightness (null).
- Finding: **H376 anti-SAM** (perturb TOWARD sharper minima) is the only
  positive optimizer surprise (PGD 0.79 vs 0.92). H377 weight-step
  budget also works (PGD 0.73) but kills clean acc. The campaign tried
  ~20 ways to "mimic AT's optimizer behaviour" without doing AT - all
  failed (H374-H382). **Evidence:** H374, H378, H380, H382.

### C4. Noise-augmentation defences (PARTIAL, mostly FGSM-only)
- H262-H272 (sigma sweep, curriculum, per-sample, hidden-noise
  position, learned noise, input vs weight noise), H327 randomised
  smoothing training, H357 consistency-RS, H384 noise-copy multiplicity,
  H397 EOT-PGD audit.
- Finding: noise-aug helps FGSM ASR by ~10-20pp, helps PGD by ~3pp.
  H327 (sigma=0.3) cuts PGD to 0.67. H271 learned-noise+AT cuts PGD to
  0.77. **H397 confirms most noise-aug "robustness" is not masking** -
  EOT-PGD does not blow up the white-box result at test time because
  the deployed model is deterministic at eval. But the "noise alone is
  enough" framing is NOT supported: AT remains necessary for low PGD.

### C5. Architecture / inductive-bias defences (MIXED, mostly NULL)
- H219 SSL (worse), H276 spectral-norm (worse), H290 hidden-layer AT
  (SUPPORTED, block0 AT cuts PGD to 0.42), H291 layer profile,
  H356 Lipschitz-stochastic-depth (PGD 0.80), H366 SiLU vs ReLU (null),
  H372 per-layer Jacobian (SUPPORTED), H381 activation-stats-matching
  (PGD 0.72), H407 i-RevNet invertible, H408 iResNet Lipschitz,
  H410 invertible+jacobian, H411 VOneNet/Gabor, H412 recurrent variable-T,
  H414 capsule, H415 group-equivariant, H416 scattering transform,
  H417 multi-scale pyramid, H418 skip-only, H419 attention/ViT,
  H420 dilated-conv, H422 hypernetwork, H423 sparse MoE.
- Finding: architecture rarely buys robustness for free. Block-0 AT
  transfers to input-space robustness (H290); per-layer Jacobian
  helps but is slow (H372); shape-biasing (H220) gives small FGSM
  improvement. Mostly the inductive biases just match baseline.
  **Evidence summary in RESULTS_SUMMARY H276, H290, H291, H372, H381;
  arch-only entries H414-H423 mostly tested as ablations against AT.**

### C6. Detection / vulnerability prediction (mostly REDUNDANT with margin)
- H186-H191, H193-H199, H201-H208, H213-H215, H237-H240, H253-H260,
  H280, H282-H284.
- Finding: **logit margin is the universal champion** (AUROC ~0.95).
  Confidence, gradient norm, integrated-gradients, feature-norm, BNN
  variance, step-to-flip, AUM, self-influence (H201), randomised-smoothing
  radius (H268), ensemble disagreement (H189), perturbation L2 (H259),
  symmetric-KL between std and AT models (H243) all tie or trail margin.
  Cross-architecture (H208) and cross-seed (H258) vulnerability is highly
  shared (Jaccard 0.956), confirming Tramer/Ilyas "shared subspace"
  picture. **Evidence:** RESULTS_SUMMARY H190, H201, H208, H243, H258.

### C7. Per-sample / data-geometry probes (SUPPORTED conceptually, narrow)
- H192 (AIGN coreset), H218 (per-sample overfitting),
  H235 (memorisation), H236 (witness samples), H238 (forgetting),
  H241 (label-noise migration), H244 (catastrophic interference),
  H246 (robust islands), H247 (jointly-robust samples), H252 (per-class
  asymmetry).
- Finding: vulnerability is concentrated in a stable subset of low-margin
  samples (H246: 87% persistence), per-class FGSM ASR varies strongly
  (Shirt=1.0 vs Trouser=0.30, H252), but PGD ASR saturates at 1.0 across
  all classes making per-class signal mostly invisible at eps=0.1.

### C8. Threat-model probing (THIN)
- H197 L2-vs-Linf (correlated 0.80), H222 input-dim Simon-Gabriel
  (NOT SUPPORTED), H224 subject-vs-bg targeting, H226 spatial
  quantisation, H227 JPEG training, H254 cross-epsilon ranking,
  H260 targeted-vs-untargeted.
- Finding: campaign barely leaves Linf eps=0.1. The few L2 experiments
  (H197, H283 smooth-FGSM) show training norm dictates eval norm.
  Spatial/StAdv, ReColorAdv, patch, common corruptions, distribution
  shift, decision-based and query-based attacks are essentially absent.

### C9. Adaptive attacks / masking audits (PRESENT but minimal)
- H183 unlearning holes, H184 illusion-masking, H300 grad-clip masking,
  H301 Lion masking, H306 mixup masking, H314 natural-grad masking,
  H325 confidence-reg masking, H391 5-signal masking battery.
- Finding: H391 is the only systematic adaptive audit. It confirms
  Jacobian and PGD-AT models are GENUINE; input-grad-penalty model is
  not robust at all. No audit covers stochastic / TTA / detection
  defences.

### C10. Distillation / data-side (NEGATIVE)
- H209 random-input distillation, H210/H212 trajectory replay,
  H231 trajectory transfer, H277/H278 dataset distillation.
- Finding: robustness does not transfer through random inputs; weight
  trajectories cannot be replayed off a new init; 10-image dataset
  distillation fails. Negative across the board.

### C11. Biological / illusion / unlearning (MOSTLY NEGATIVE / INCONCLUSIVE)
- H183 unlearning, H184 illusion masking, H220 shape-bias, H221
  temporal averaging, H229 test-time input opt (SUPPORTED but trivial),
  H248 sleep consolidation, H411 VOneNet, H413 perceptual-AT.

---

## Section 2 - Methodological issues in H173-H413

### M1. Single-seed runs everywhere
Almost every entry reports one seed (SEED=0). H258 only used 5 seeds to
*establish* cross-seed Jaccard; nothing else reports cross-seed CIs.
Many "PARTIAL" verdicts (e.g. H286, H311, H321, H340, H362) hover in
the ±0.02 PGD band, well inside plausible single-seed noise.

### M2. N_TRAIN = 6000 (10% of Fashion-MNIST)
Sub-scale training means (a) every defence operates in an
under-training regime - H309 already noted batch-size results are
confounded with under-training; (b) gradient-penalty signals may not
saturate (H331 epoch-by-epoch shows no separation up to ep 10); (c)
TRADES/ALP appear "as good as AT" because PGD-AT itself only reaches
ASR 0.33 in this budget vs ~0.45 at full scale on CIFAR-10. Conclusions
that a defence "ties AT" are unsafe.

### M3. Eval threat model collapsed to Linf eps=0.1 PGD-10
- No AutoAttack (Croce & Hein 2020). H38 exists but is one isolated
  test, not applied across H173-H413 defences.
- No CW-L2 systematic eval beyond H247.
- No L0/L1 attack (SparseFool, JSMA, Pixle).
- No spatial/StAdv (Xiao 2018), ReColorAdv (Laidlaw 2019).
- No patch attacks.
- No common corruptions (Hendrycks F-MNIST-C analogue).
- No query/decision-based (HopSkipJump exists at H43 but not used as
  primary eval; SimBA only at H113).
- 10-step PGD is *too weak* for any defence that gets near AT-level.
  H391 ran 50-step and confirmed plateau for genuinely-robust models;
  most other entries did not.

### M4. Missing adaptive attacks for every "winner"
H323 (Jacobian-Frob), H344 (Jacobian spectral), H365 (conf-weighted GP),
H372 (per-layer Jacobian), H376 (anti-SAM), H377 (step budget),
H381 (activation-stats matched) - all crossed the "interesting"
threshold but **only H391 actually adapt-attacked four of them**. The
H411 VOneNet, H412 recurrent-T, H407 i-RevNet defences need EOT-PGD,
attack-at-many-T, and inverse-network probes respectively - the scripts
acknowledge this but full adaptive coverage is not in
`RESULTS_SUMMARY.md`.

### M5. Transfer-attack controls absent
H199/H230/H332 are the only transfer-attack experiments. No defence
in C2 / C3 / C5 is systematically tested with transfer attacks from
the standard baseline (H391 is the exception). When the white-box
attack fails, transfer is the *first* sanity check.

### M6. Mean-only reporting (no per-class, no worst-case)
H252 establishes that FGSM ASR varies 0.30-1.00 across classes, but
mean-PGD-ASR is the headline metric everywhere. No defence reports
per-class ASR, worst-class ASR, or Q1/Q4 margin-bucketed ASR. A
defence that lowers mean PGD ASR by 0.05 might be entirely concentrated
in already-robust high-margin samples (Tsipras-style "robust-feature"
trade-off).

### M7. No compute-equalised comparisons
H372 explicitly notes "9x slower"; H287 notes "3-4x compute". Other
costly methods (H323 Jacobian-Frob, H344 spectral) are reported at
nominal lambda without compute-matching to the AT baseline. The AT
baseline (10 PGD steps every batch) is itself ~6x standard training.
"AT-tied" claims need ablation at matched FLOPs.

### M8. No external-dataset transfer
Almost everything is Fashion-MNIST. H105 mentions cross-dataset
transfer, H181 has a CIFAR-10 ResNet18 run, but none of the H300-H413
defences are re-tested on KMNIST or CIFAR-10 to check whether
findings are dataset-artefacts. Fashion-MNIST is famously easy and
class-overlapping (Shirt/Coat/Pullover).

### M9. PGD ASR saturation at eps=0.1
Many entries (H241 label-noise, H244 interference, H247 jointly-robust,
H249 combined predictor, H260 targeted-vs-untargeted) report
"INCONCLUSIVE - degenerate labels" because PGD ASR sits at 1.000 for
nearly every sample. The eps choice destroys signal. eps sweep
(H254) was tried but rank preservation breaks down precisely because
eps=0.1 is the saturating point.

### M10. No certified-robustness evaluation
H117 IBP exists; H204 randomised smoothing exists; H268 cert-vs-empirical;
H327 randomised-smoothing-training. None feed into the campaign's defence
ladder. There is no certified accuracy reported for any "winner". The
field's standard (Cohen et al. 2019 RS) is missing from the eval.

### M11. Logit-margin / -confidence shortcut never re-checked
Almost all "vulnerability predictor" claims (~30 hypotheses) use logit
margin as the gold benchmark, but the calibration of the gold benchmark
itself isn't re-checked under shifts (BN at eval, T scaling, AT-models
where margin AUROC is known to fall from 0.96 to 0.71, per Paper 5).

### M12. EOT / stochastic-defence audits incomplete
H397 audits noise-aug under EOT-PGD, but H134 BNN, H132 snapshot
ensemble, H362 ensemble-diversity, H411 VOneNet stochastic neurons,
H423 sparse-MoE - none received an EOT pass. Athalye 2018 is half-
heeded.

---

## Section 3 - Gaps in the hypothesis space

### G1. Threat models barely touched
- **L0 / sparse pixel attacks** (Pixle 2020, JSMA 2016, SparseFool).
  Only H41 covers sparse-L0; no defence is tested against it.
- **L1 attacks** (Pointwise, EAD - Chen 2018). Absent post-H50.
- **Spatial / geometric** (StAdv Xiao 2018, AffineTransfer). H42 spatial
  attack exists; no defence tested against it.
- **Patch / physical** (Brown 2017, AdvPatch). Entirely absent.
- **ReColorAdv / semantic** (Laidlaw 2019). Absent.
- **Common corruptions** (Hendrycks 2019 ImageNet-C). Fashion-MNIST-C
  analogue is constructable but not used.
- **Distribution shift** robustness (KMNIST-as-shift). H105 only.
- **Query-based / decision-based** (HopSkipJump H43, SimBA H113,
  BoundaryAttack absent, RayS absent). None applied to defended models.
- **Targeted vs untargeted** under non-saturating eps. H260 tried but
  collapsed.

### G2. Defence families barely touched
- **Certified defences:** IBP (Gowal 2018) only at H117 as a probe;
  CROWN-IBP, randomised smoothing (Cohen 2019) only as eval not training.
  No SmoothAdv (Salman 2019), no MACER (Zhai 2020).
- **Diffusion purification:** DiffPure (Nie 2022), GDMP (Wang 2022) -
  zero coverage. A tiny pure-torch DDPM on Fashion-MNIST is feasible.
- **Test-time adaptation:** TENT (Wang 2021), MEMO (Zhang 2022),
  CoTTA (Wang 2022). H229 ("test-time input opt") is in the
  neighbourhood but does not test these algorithms.
- **MoE / routing-based defence with adaptive attack:** H211, H423 exist
  but no adaptive routing-aware attack tested.
- **Retrieval-augmented classification:** k-NN final layer (Papernot
  2018 "Deep k-NN"). Absent.
- **MART** (Wang 2020) - explicit hypothesis exists (H45) but is not
  in the H173-H413 range. Re-test at modern scale.
- **GAIRAT** (Zhang 2021 geometry-aware AT). Absent.
- **AWP-LBGAT / SCORE** (Pang 2022). Absent.
- **HAT** (Rade & Moosavi-Dezfooli 2021 "Helper-based AT"). Absent.

### G3. Loss / training objectives missing
- **TRADES variants** beyond H304/H346/H347/H363: no Margin-TRADES,
  no Focal-TRADES, no Class-balanced TRADES.
- **MART** (mis-class weighting). Absent in the H173+ range.
- **GAIRAT** (geometry-aware reweighting).
- **Contrastive AT** beyond H348 (which collapsed). InfoNCE-AT,
  CLAW (Yu 2022).
- **Label-noise AT:** what happens when you AT *under* label noise?
- **Mixup variants:** Manifold-Mixup-AT, Cutmix-AT, AugMix-AT (H433
  exists but ablation incomplete).
- **Distillation-based AT:** ARD (Goldblum 2020), RSLAD (Zi 2021).
  H173 partially covers RSLAD but no later replication.

### G4. Architecture axes not explored
- **Depth-width scaling laws for robustness** (Wu, Xia 2021).
  H217 tried 4 settings; need a proper scaling curve.
- **Normalization layer choice** under AT: BN vs LN vs GN vs none.
  None tested at H173+.
- **Activation choice** beyond H366 (SiLU). GELU, Mish, Swish, soft-relu,
  smooth-relu under AT.
- **Attention pattern** under AT - axial, sliding-window, sparse;
  H419 ViT-tiny only.
- **Frequency-domain backbones** (FFT-conv). Absent.
- **Capsule + AT** (H414 alone, not combined with AT).

### G5. Optimization side not covered
- **SAM variants** (ASAM, F-SAM, mSAM, GSAM Zhuang 2022). Only base
  SAM tested (H273).
- **EMA / Polyak averaging** under AT. H274 SWA only.
- **Lookahead optimiser**. Absent.
- **Second-order**: K-FAC, Shampoo. H314 natural-grad failed but no
  diagonal/block-diagonal preconditioner tested under AT.

### G6. Theoretical probes
- **Margin distribution** (per-sample, not aggregate) under AT vs
  standard. The campaign reports mean margin everywhere; full margin
  CDF, tail behaviour, kurtosis would matter for worst-case.
- **Loss-landscape sharpness** (top-k Hessian eigenvalues). H102 / H388
  exist; no per-defence comparison.
- **NTK signature** (Jacot 2018) of robust vs non-robust models. Absent.
- **Double descent for robust acc** (Nakkiran 2020). Absent.
- **Sample complexity of AT** (Schmidt 2018) - the canonical
  "AT needs more data" claim never directly tested in this campaign.
- **Information bottleneck** under AT (Tishby 1999, Saxe 2018).

### G7. Per-sample structure / learning dynamics
- **Memorisation vs generalisation split** (Feldman & Zhang 2020).
  H131 covers C-score; not crossed with AT.
- **Learning-order effects** (Toneva 2019 forgetting). H238 covers
  it; not crossed with AT.
- **Influence functions on adversarial training** (Koh & Liang 2017).
  H201 is on a standard model only.
- **Easy/hard sample partition under AT** (Maini 2022 "How does
  adversarial training affect dataset?").

### G8. Diagnostic / mechanistic
- **Circuit-level**: what neurons / channels are robust vs not?
  (Bau 2017 network dissection, Olah 2020 Zoom-in).
- **Feature-level**: robust vs non-robust features (Ilyas 2019). H214
  score-network angle is adjacent but no clean partition.
- **Linear-mode connectivity for robust models** (Frankle 2020).
  H275 mode-connectivity tried, no robustness-loss landscape
  characterisation.
- **Lottery-ticket robustness** (Frankle 2019, Diffenderfer 2021).
  Absent.

### G9. Eval-protocol gaps
- **AutoAttack ensemble** (APGD-CE + APGD-DLR + FAB + Square) as the
  *default* eval for any "robust" model. Currently absent.
- **RobustBench-style protocol** (Croce 2021). Absent.
- **Compute-equalised comparison** vs PGD-AT. Always missing.
- **Per-class worst-case** reporting. Always missing.
- **Adaptive attacker design** as a first-class output (Tramer 2020
  "On Adaptive Attacks to Adversarial Example Defenses").

### G10. Recent (2023-2025) directions absent
- **Diffusion AT** (Wang 2023 "Better Diffusion Models Further Improve
  Adversarial Training").
- **LLM/prompt-based defences** (not applicable here, but VLM-style
  prompt-tuning of CNN features is).
- **Robust SSL** (Kim 2023 "RobustCLR").
- **Lightweight robust pretraining** (Singh 2023).
- **NeurIPS 2024 / ICML 2024 AT scaling laws.**

---

## Section 4 - Papers to draw from (40)

Cite-key format `[author-year]`. All claims one-line.

**Classic AT**
1. `madry-2018` Madry et al., "Towards Deep Learning Models Resistant to
   Adversarial Attacks", ICLR 2018. — PGD-AT is the canonical reference;
   formalised inner-max / outer-min objective.
2. `zhang-2019-trades` Zhang et al., "Theoretically Principled Trade-off
   between Robustness and Accuracy", ICML 2019. — TRADES objective. (H304)
3. `wang-2020-mart` Wang et al., "Improving Adversarial Robustness
   Requires Revisiting Misclassified Examples", ICLR 2020. — MART, weights
   adv loss by misclassification probability. (gap G3)
4. `wu-2020-awp` Wu et al., "Adversarial Weight Perturbation Helps Robust
   Generalization", NeurIPS 2020. — Flatness in weight space. (H316/H351)
5. `zhang-2021-gairat` Zhang et al., "Geometry-aware Instance-reweighted
   Adversarial Training", ICLR 2021. — Per-sample weighting by attack
   geometric distance. (gap G2)
6. `pang-2022-score` Pang et al., "Robustness and Accuracy Could Be
   Reconcilable by (Proper) Definition", ICML 2022. — SCORE method.
7. `rade-2022-hat` Rade & Moosavi-Dezfooli, "Helper-based AT", ICLR 2022.
   — Robustness-accuracy via helper class. (gap G2)
8. `goldblum-2020-ard` Goldblum et al., "Adversarially Robust
   Distillation", AAAI 2020. — Robust knowledge distillation.

**Certified / smoothing**
9. `cohen-2019-rs` Cohen, Rosenfeld & Kolter, "Certified Adversarial
   Robustness via Randomized Smoothing", ICML 2019. — L2 certification.
   (H204, H268)
10. `salman-2019-smoothadv` Salman et al., "Provably Robust Deep Learning
    via Adversarially Trained Smoothed Classifiers", NeurIPS 2019.
11. `gowal-2018-ibp` Gowal et al., "On the Effectiveness of Interval
    Bound Propagation for Training Verifiably Robust Models", arXiv 2018.
    — IBP. (H117)
12. `zhai-2020-macer` Zhai et al., "MACER: Attack-free and Scalable
    Robust Training via Maximizing Certified Radius", ICLR 2020.

**Purification / TTA**
13. `nie-2022-diffpure` Nie et al., "Diffusion Models for Adversarial
    Purification", ICML 2022. — DiffPure. (gap G2)
14. `wang-2022-gdmp` Wang et al., "Guided Diffusion Model for Adversarial
    Purification", arXiv 2022.
15. `wang-2021-tent` Wang et al., "TENT: Fully Test-time Adaptation by
    Entropy Minimization", ICLR 2021. — Test-time BN/entropy. (gap G2)
16. `zhang-2022-memo` Zhang et al., "MEMO: Test Time Robustness via
    Adaptation and Augmentation", NeurIPS 2022.

**Threat-model expansion**
17. `laidlaw-2020-pat` Laidlaw, Singla, Feizi, "Perceptual Adversarial
    Robustness", NeurIPS 2021. — LPIPS-bounded AT. (H413)
18. `laidlaw-2019-recoloradv` Laidlaw & Feizi, "Functional Adversarial
    Attacks", NeurIPS 2019. — ReColorAdv. (gap G1)
19. `xiao-2018-stadv` Xiao et al., "Spatially Transformed Adversarial
    Examples", ICLR 2018. — StAdv. (gap G1)
20. `chen-2018-ead` Chen et al., "EAD: Elastic-net Attacks", AAAI 2018.
    — L1 attack. (gap G1)
21. `croce-2020-autoattack` Croce & Hein, "Reliable Evaluation of
    Adversarial Robustness with an Ensemble of Diverse Parameter-free
    Attacks", ICML 2020. — AutoAttack (APGD+FAB+Square). (gap G9)
22. `tramer-2020-adaptive` Tramer et al., "On Adaptive Attacks to
    Adversarial Example Defenses", NeurIPS 2020. — Adaptive-attack
    methodology. (M4)
23. `athalye-2018-obfuscated` Athalye, Carlini, Wagner, "Obfuscated
    Gradients Give a False Sense of Security", ICML 2018. — EOT-PGD,
    BPDA. (H391, H397)

**Datasets / benchmarks**
24. `hendrycks-2019-imagenetc` Hendrycks & Dietterich, "Benchmarking
    Neural Network Robustness to Common Corruptions and Perturbations",
    ICLR 2019. — ImageNet-C / CIFAR-10-C. (gap G1)
25. `croce-2021-robustbench` Croce et al., "RobustBench: a Standardized
    Adversarial Robustness Benchmark", NeurIPS 2021 (datasets track).
    — Reference eval protocol. (gap G9)

**Theory**
26. `schmidt-2018-sample` Schmidt et al., "Adversarially Robust
    Generalization Requires More Data", NeurIPS 2018. — Sample
    complexity. (gap G6)
27. `tsipras-2019-tradeoff` Tsipras et al., "Robustness May Be at Odds
    with Accuracy", ICLR 2019. — Robust/accurate tradeoff. (M6 lens)
28. `ilyas-2019-features` Ilyas et al., "Adversarial Examples Are Not
    Bugs, They Are Features", NeurIPS 2019. — Robust vs non-robust
    features. (gap G8)
29. `bubeck-2021-isoperimetric` Bubeck & Sellke, "A Universal Law of
    Robustness via Isoperimetry", NeurIPS 2021. — Overparam needed for
    robust interpolation. (gap G6)
30. `simon-gabriel-2019-curse` Simon-Gabriel et al., "First-Order
    Adversarial Vulnerability of Neural Networks and Input Dimension",
    ICML 2019. — sqrt(n) scaling. (H222)
31. `jacot-2018-ntk` Jacot, Gabriel, Hongler, "Neural Tangent Kernel:
    Convergence and Generalization in Neural Networks", NeurIPS 2018.
    — NTK theory. (gap G6)

**Biological / inductive bias**
32. `dapello-2020-vonenet` Dapello et al., "Simulating a Primary Visual
    Cortex at the Front of CNNs Improves Robustness to Image
    Perturbations", NeurIPS 2020. — VOneNet. (H411)
33. `bruna-2013-scattering` Bruna & Mallat, "Invariant Scattering
    Convolution Networks", IEEE TPAMI 2013. (H416)
34. `cohen-2016-gcnn` Cohen & Welling, "Group Equivariant Convolutional
    Networks", ICML 2016. (H415)

**Invertibility / Lipschitz**
35. `jacobsen-2018-irevnet` Jacobsen et al., "i-RevNet: Deep Invertible
    Networks", ICLR 2018. (H407)
36. `behrmann-2019-iresnet` Behrmann et al., "Invertible Residual
    Networks", ICML 2019. (H408)
37. `anil-2019-soc` Anil, Lucas, Grosse, "Sorting Out Lipschitz Function
    Approximation", ICML 2019. — Orthonormal Lipschitz layers.
    (adjacent to H276/H344)

**Recent (2023-2025)**
38. `wang-2023-betterdm` Wang et al., "Better Diffusion Models Further
    Improve Adversarial Training", ICML 2023. — Diffusion-generated AT
    data. (gap G10)
39. `singh-2023-revisit` Singh et al., "Revisiting Adversarial Training
    for ImageNet: Architectures, Training and Generalization across
    Threat Models", NeurIPS 2023. — Modern AT scaling. (gap G4/G10)
40. `kireev-2022-effectiveness` Kireev, Andriushchenko, Flammarion,
    "On the effectiveness of adversarial training against common
    corruptions", UAI 2022. — Joint Linf-AT + corruption robustness.
    (gap G1)

---

## Section 5 - 80 candidate hypothesis seeds (h434-h513)

Format: `h###: TITLE — gap filled — paper anchor — key knob`.
Constraints: all feasible at N_train=6000, 10 epochs, single RTX 4090.
All claims falsifiable. Non-overlapping with H173-H413.

### Defences (30 seeds: h434-h463)

- h434: **MART loss** — G3 (missed standard AT variant) — `wang-2020-mart` — weight lambda on misclassified samples in {0.5,1,2,5}.
- h435: **GAIRAT geometry-aware AT** — G3 — `zhang-2021-gairat` — kappa (geometric reweight strength) in {0.5,1.0,2.0}.
- h436: **HAT helper-class AT** — G2 — `rade-2022-hat` — helper-margin parameter.
- h437: **ARD robust distillation** — G3 — `goldblum-2020-ard` — temperature x teacher (vanilla AT vs TRADES teacher).
- h438: **SCORE method** — G2 — `pang-2022-score` — alpha trade-off knob.
- h439: **Margin-TRADES** — G3 — extends `zhang-2019-trades` — margin-based KL surrogate.
- h440: **Class-balanced TRADES** — G3, M6 — fix worst-class PGD ASR — per-class beta sweep.
- h441: **Focal-TRADES** — G3 — focal weighting inside TRADES KL term — gamma in {0,1,2}.
- h442: **Diffusion-AT (tiny DDPM)** — G2, G10 — `nie-2022-diffpure` + `wang-2023-betterdm` — purification timestep t in {25,50,100}.
- h443: **TENT test-time adaptation** — G2 — `wang-2021-tent` — adapt BN+entropy on adversarial batch.
- h444: **MEMO test-time augmentation** — G2 — `zhang-2022-memo` — number of TTA samples in {4,8,16}.
- h445: **Deep k-NN classifier head** — G2 — `papernot-2018-knn` — k in {1,5,25,100}.
- h446: **SmoothAdv certified training** — G2 — `salman-2019-smoothadv` — sigma in {0.12, 0.25, 0.5}.
- h447: **MACER attack-free certified training** — G2 — `zhai-2020-macer` — lambda for certified radius term.
- h448: **CROWN-IBP training** — G2 — extends `gowal-2018-ibp` — IBP eps schedule.
- h449: **Soft-Lipschitz Orthonormal layer** — G4 — `anil-2019-soc` — replace convs with SOC blocks.
- h450: **Robust EMA / Polyak under AT** — G5 — extends `wu-2020-awp` — EMA decay sweep.
- h451: **ASAM (adaptive SAM)** — G5 — `kwon-2021-asam` — rho schedule.
- h452: **AT with diffusion-generated data** — G10 — `wang-2023-betterdm` — synthetic-to-real ratio.
- h453: **AugMix-AT joint** — G3 — `hendrycks-2020-augmix` + AT — mix-weight x adv-weight grid.
- h454: **CutMix-AT joint** — G3 — improves H432 — patch-prob in {0.25,0.5,0.75}.
- h455: **Manifold-Mixup-AT** — G3 — extends H124/H270 — hidden-layer mix point.
- h456: **InfoNCE contrastive-AT (fixed SupCon)** — G3 — fix H348 with explicit linear probe — temperature.
- h457: **Label-noise + AT crossing** — G3, G7 — symmetric noise rate x AT epsilon — what does AT do under 20%/40% label noise.
- h458: **GroupNorm vs BatchNorm under AT** — G4 — none/BN/GN/LN x AT — robust ASR.
- h459: **Activation sweep under AT** (ReLU/GELU/SiLU/Mish/Softplus) — G4 — choice of activation.
- h460: **Depth-width scaling under AT** — G4 — `singh-2023-revisit` — full 4x4 grid of (depth, width) at constant params.
- h461: **Sparse-MoE with adaptive routing-aware attack** — G2 — extends H423 — k in {1,2}, adaptive PGD that attacks gate.
- h462: **Lottery-ticket robustness** — G8 — `frankle-2019` — IMP at 30%, 50%, 70% sparsity then AT.
- h463: **Curriculum-noise + AT mix** — G3 — H264 extension under AT.

### Attack / evaluation (20 seeds: h464-h483)

- h464: **AutoAttack ladder on top-10 campaign winners** — M3, G9 — `croce-2020-autoattack` — APGD-CE+DLR+FAB+Square.
- h465: **EOT-PGD on every stochastic defence** — M12 — `athalye-2018-obfuscated` — K in {1,10,20,40}.
- h466: **Adaptive attack on H372 per-layer Jacobian** — M4 — `tramer-2020-adaptive` — design loss using per-layer norms.
- h467: **Adaptive attack on H407 i-RevNet** — M4 — invert decoder, attack in feature space.
- h468: **Adaptive attack on H411 VOneNet** — M4 — attack after EOT(K=20) and through Gabor inverse.
- h469: **Adaptive attack on H412 variable-T recurrent** — M4 — PGD averaged across T in {1..8}.
- h470: **L1 attack EAD** — G1 — `chen-2018-ead` — beta, alpha sweep.
- h471: **L0 attack JSMA / Pixle** — G1 — max changed pixels in {10, 50, 200}.
- h472: **Spatial / StAdv attack** — G1 — `xiao-2018-stadv` — flow-budget sweep.
- h473: **ReColorAdv attack** — G1 — `laidlaw-2019-recoloradv` — colour-grid resolution.
- h474: **Patch attack (16x16)** — G1 — `brown-2017-advpatch` — patch size, location.
- h475: **Fashion-MNIST-C corruptions** — G1 — `hendrycks-2019-imagenetc` — synthesise 15 corruptions x 5 severities.
- h476: **Decision-based BoundaryAttack** — G1, M3 — Brendel 2018 — query budget {500, 5000}.
- h477: **RayS decision-based attack** — G1 — Chen & Gu 2020 — query budget.
- h478: **HopSkipJump on top-10 winners** — G1 — extends H43.
- h479: **Targeted CW-L2 at non-saturating eps** — M9 — eps that gives ASR 0.5 on baseline.
- h480: **Per-class & worst-class ASR reporting** — M6 — re-evaluate top-10 winners with per-class CDF.
- h481: **Transfer-attack matrix** — M5 — source in {std, FGSM-AT, PGD-AT, TRADES} x target in 10 defences.
- h482: **PGD-50 + 5 restarts on top-10 winners** — M3 — does H304 TRADES survive stronger attack.
- h483: **Cross-dataset transfer** — M8 — re-test top-10 on KMNIST and MNIST.

### Diagnostic / mechanistic (15 seeds: h484-h498)

- h484: **Robust-channel identification** — G8 — `bau-2017` — ablate top-K channels per-defence, measure PGD drop.
- h485: **Robust-feature vs non-robust-feature partition** — G8 — `ilyas-2019-features` — train on PGD-AT-frozen features.
- h486: **Margin-distribution CDF per defence** — M6, G6 — full CDF (not mean), report 5th/25th/median/95th.
- h487: **Per-class ASR x per-class margin scatter** — M6 — does AT shift the curve or just translate it.
- h488: **Loss-landscape sharpness via top-5 Hessian eigs** — G6 — H102 extension — per-defence comparison.
- h489: **NTK signature at init vs after AT** — G6 — `jacot-2018-ntk` — kernel rank, top-eigenvector overlap with PGD direction.
- h490: **Linear-mode connectivity between AT seeds** — G8 — `frankle-2020` — barrier height under PGD-loss.
- h491: **Adversarial example basin overlap** (Jaccard of advs across seeds at fixed eps).
- h492: **Forgetting-event x AT crossing** — G7 — does AT increase or decrease forgetting events?
- h493: **C-score x AT crossing** — G7 — `feldman-2020` — do high-memorisation samples become more robust under AT?
- h494: **Influence functions on PGD-AT model** — G7 — H201 extension to AT model.
- h495: **Per-layer adversarial sensitivity profile** — G8 — extends H289/H291 — gradient norm by depth, before vs after AT.
- h496: **Activation-stat matching reach** — extends H381 — what is the *minimum* statistic set that recovers H381's effect.
- h497: **Feature-space adversarial distance** — G8 — measure delta in penultimate layer vs input space.
- h498: **Robust subnetwork via IMP pruning** — G8 — does an AT lottery ticket exist at 50% sparsity.

### Theory probes (15 seeds: h499-h513)

- h499: **Schmidt sample-complexity curve** — G6 — `schmidt-2018-sample` — sweep N_train in {1k, 3k, 6k, 12k, 30k, 60k} for PGD-AT, fit eps_robust vs N.
- h500: **Tsipras tradeoff Pareto** — G6 — `tsipras-2019-tradeoff` — sweep eps_train in {0, 0.02, 0.05, 0.10, 0.15, 0.20}.
- h501: **Bubeck-Sellke overparam** — G6 — `bubeck-2021-isoperimetric` — fix N_train, sweep width in 8/16/32/64/128, observe robust-acc plateau.
- h502: **Double-descent for robust accuracy** — G6 — extends `nakkiran-2020` — width sweep with AT.
- h503: **Per-sample certified radius vs margin** — G6, M10 — extends H204/H268 — Spearman across AT vs std.
- h504: **Robust-vs-non-robust feature decomposition** — G6, G8 — `ilyas-2019-features` — measure transfer of features alone.
- h505: **Margin CDF tail behaviour under AT** — G6 — does AT thin or fatten the low-margin tail.
- h506: **Eps-vs-margin scaling law** — G6 — for each defence, fit eps_50 = a * margin_q.
- h507: **Sample-efficiency of TRADES vs AT** — G6 — extends h499 separately for TRADES.
- h508: **NTK alignment with adversarial direction** — G6 — does AT rotate the NTK top-eigenvector toward PGD direction.
- h509: **Information-bottleneck under AT** — G6 — `tishby-1999`, `saxe-2018` — MI(x; h) for standard vs AT.
- h510: **Spectrum of Jacobian under AT vs std** — G6 — full SVD, not just top sigma.
- h511: **AT and double-descent of clean loss** — G6 — train-loss curve under varying width with AT.
- h512: **Margin-AUROC stability under threat-model shift** — M11 — recompute AUROC under L1, L2, Linf, StAdv.
- h513: **Cross-attack vulnerability rank stability** — extends H197 — Spearman over (FGSM, PGD, CW-L2, AutoAttack, StAdv) per sample.

---

End of gap map.
