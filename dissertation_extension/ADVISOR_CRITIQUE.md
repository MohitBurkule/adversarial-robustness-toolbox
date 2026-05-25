# Advisor Critique — Dissertation Extension (8 Papers)

**Reviewer role:** Senior academic advisor / peer reviewer for an MSc dissertation extension on per-sample adversarial vulnerability, attribution-based predictors, and defence geometry.

**Reviewing posture:** Critical and honest. The body of work as a whole is empirically careful and tells a coherent story, but it leans heavily on one small architecture and one or two small datasets, and several individual papers overclaim. I have flagged everything I would push back on in a real review.

---

## Paper 1 — Systematic Empirical Benchmark of Per-Sample Vulnerability Predictors

**Verdict:** Moderate. Realistic target: a strong empirical workshop (e.g., NeurIPS ML Safety / ICLR RobustML / SaTML), or a TMLR submission. Not a top-tier main-conference paper as currently scoped.

**Strengths**
- The premise — a unified protocol across 50+ predictors, 4 datasets, 8 attacks — is genuinely useful; the field does need such a comparison.
- Honest documentation of the PGD saturation pathology (AUROC undefined at high ASR) is a real methodological contribution and rarely surfaced in prior work.
- Multi-seed stability classification (Stable / Moderate / Unstable) is unusually disciplined for this literature.
- The unification of "monotone reparameterisations are equivalent under AUROC" is correct and worth stating in print.

**Major weaknesses (must fix before submission)**
- **Single architecture.** A 5-layer CNN with 28×28 grayscale inputs *for every dataset*, including CIFAR-10 and Imagenette downsampled and greyscaled, is an extraordinary restriction. The claim of generality is not supported. Without at least one ResNet-18/WideResNet result the headline "50+ predictors benchmarked" is misleading. The paper currently cannot be cited as a benchmark in the conventional sense.
- **AUROC ceiling artefact.** Margin AUROC ~ 0.97 on FM at ε=15/255 partly reflects that ~97% of samples are attacked successfully, leaving a small ROC discrimination problem against a near-constant label. Section 4.7 acknowledges this in one sentence but never controls for it. The benchmark needs results at *matched* attack-success rates (40–60%) across datasets, not constant ε.
- **The "50+" claim is rhetorical.** The leaderboard shows ~10 predictors; the other 40 are not enumerated, not tabulated, not in the appendix as presented. Either list them properly or drop the figure.
- **No statistical tests.** Differences like "0.9743 vs 0.9723" (margin vs gradient norm) are reported as if meaningful but no DeLong test or bootstrap CI is presented. With std=0.006 across only 3 seeds (H165, "seeds 0–2"), these AUROCs are statistically indistinguishable.

**Minor weaknesses**
- "Patch-based 28×28 grayscale standardisation" of CIFAR-10 / Imagenette is reductive and probably destroys most of the relevant signal — reviewers will object.
- Black-box attack subsection promises 4 attacks but no black-box results appear in §4.
- Figure 1 is ASCII; the abstract claims AUROC 0.87–0.9976 but the leaderboard only shows FM/PGD — the full grid is missing.

**Specific claims to fix / hedge**
- *"the logit margin is the dominant predictor, achieving AUROC 0.87–0.9976 across all experimental contexts"* — at this architecture, at these inputs. Hedge.
- *"systematic evaluation of 50+ per-sample vulnerability predictors"* — show the full list or reduce the number.
- *"Training-dynamics predictors such as forgetting events (AUROC 0.54) and C-score (AUROC 0.53) perform at chance"* — this is shown only for FM/FGSM. The abstract overgeneralises; the body shows the result on essentially one configuration.
- *"PGD saturation artefact that renders AUROC undefined"* — fine, but the way it is propagated through the paper (CIFAR-10 dropped to ε=4/255 only for one experiment) is inconsistent.

**Missing baselines / experiments**
- ResNet-18 (or any modern architecture) at native resolution for at least one dataset.
- A learned baseline: gradient-boosted predictor on top of all features, to establish the AUROC ceiling.
- DeLong / bootstrapped AUROC confidence intervals.
- Calibration metrics (ECE) alongside AUROC, since downstream uses (selective prediction) need calibrated scores.
- A held-out cross-attack generalisation experiment (predictor fit on attack A, evaluated on attack B).

---

## Paper 2 — Adversarial Vulnerability is Structural and Attack-Invariant

**Verdict:** Moderate-to-Strong (one of the better papers in the bundle). Target: workshop or TMLR; with stronger evidence, possibly ICLR.

**Strengths**
- Clear, falsifiable thesis with a clean empirical test (Jaccard overlap, count distribution).
- The resolution of the "FGSM orthogonality" paradox via differential ASR is genuinely clarifying.
- Cross-model transferability result (85.2% on SVHN) is a useful auxiliary finding.
- The framing — defence design implications if vulnerability is structural — is appropriately scoped.

**Major weaknesses**
- **The conclusion is partly tautological.** When two attacks both achieve ASR ≈ 95% on the same fixed model and test set, the Jaccard *must* be high purely by pigeonhole. The paper does not present the null distribution: what Jaccard would we expect from two attacks with ASR 95% choosing *uniformly at random* across the test set? Without this null, "0.993" is not interpretable.
- **Only L∞ white-box attacks tested.** The thesis is "attack-invariant" but no L2, no decision-based (Boundary, HopSkipJump), no AutoAttack, no transfer-based attacks are shown. Black-box attacks are mentioned in Paper 1 but not used here. The "structural" claim cannot be made without at least one truly different threat model.
- **Universal AUROC on CIFAR-10 of 0.700 is buried.** The paper highlights 0.902 (FM) but on CIFAR-10 the universal-vulnerability AUROC drops to 0.700 — that is much weaker evidence for the thesis, and the explanation ("label saturation") deserves its own subsection, not a parenthetical.
- **No adversarially trained model.** Structural attack-invariance might be a property only of vanilla models. AT models reshape boundaries; the attack-invariance hypothesis should be tested there too.

**Minor weaknesses**
- SHAP correlation r in Table 4 — only 2,000 samples per dataset, no significance test reported.
- "FGSM as a proxy" subsection is speculative — Jaccard 0.74 means 26% disagreement; on CIFAR-10 you'd lose a lot.
- The 5-model transferability matrix is on one dataset only; SVHN is also the dataset where margin AUROC is worst (Paper 5), which complicates interpretation.

**Specific claims to fix / hedge**
- *"Adversarial vulnerability is a property of samples relative to decision boundaries, not of attack algorithms"* — strong, philosophically loaded claim from L∞-only evidence. Hedge to "within the L∞ family, …".
- *"73% of test samples are vulnerable to all four white-box attacks simultaneously"* — true but needs the chance baseline alongside.

**Missing baselines / experiments**
- Null Jaccard distribution under random attack assignment matched to ASR.
- At least one L2 attack (DeepFool, C&W) and one decision-based attack.
- AutoAttack as the gold-standard ensemble — its overlap with PGD/BIM/MIM individually would be the cleanest test.
- Repeat the analysis on a PGD-AT model.

---

## Paper 3 — Noise-Averaged Attribution Norms Encode Adversarial Boundary Proximity

**Verdict:** Moderate. Strongest single contribution in the bundle (the SmoothGrad-collapses-under-AT finding is genuinely novel and surprising). Target: ICLR / NeurIPS workshop on interpretability/safety, or AISTATS short paper. Could be a main-conference paper with more rigour.

**Strengths**
- The SmoothGrad-collapses-under-AT result (H169) is the single most interesting finding across all 8 papers; it directly refutes the conventional wisdom that smoothed attributions are universally better.
- Clean mechanistic explanation tied to the geometry of the AT loss landscape (§4.3).
- Honest cost-benefit framing: plain gradient at 1× compute is competitive with SmoothGrad at 50× compute.
- The Δ=−0.416 on Imagenette/PGD-AT is a striking number that will stick in readers' minds.

**Major weaknesses**
- **The mechanistic explanation is hand-waved.** §4.3 argues that PGD-AT flattens the loss landscape *inside* the SmoothGrad noise neighbourhood. This is plausible but not directly measured. The paper needs an actual measurement of gradient norm distributions inside the SmoothGrad ball, before and after AT.
- **K=50, σ=0.1 are hand-picked.** No sensitivity analysis. The whole "SmoothGrad collapse" claim could shift dramatically at σ=0.05 or σ=0.25. This is the single most important ablation missing.
- **Integrated Gradients badly evaluated.** A single AUROC=0.76 is reported across "all tested conditions" with no table. IG is famously sensitive to baseline choice (zero baseline is one of the worst); no exploration of black/blurred/mean baselines.
- **Multi-seed numbers contradict Paper 1.** Paper 1 Table 1 reports input_grad std=0.006; this paper reports input_grad std=0.006 but also classes its SmoothGrad std as 0.039–0.046 "across datasets," with only 3 seeds (H165). Stability classification on n=3 is itself unstable.

**Minor weaknesses**
- The "saliency spatial concentration" predictor (H149) at AUROC 0.78–0.85 is mentioned once and never tabulated by dataset.
- Some AUROCs differ from Paper 1 (margin on FM/PGD given as 0.9651 there, 0.9366 here). Reconcile.
- Imagenette PGD AUROC = 0.9975 is suspicious — at what ASR? If saturated, this is the same artefact Paper 1 warned against.

**Specific claims to fix / hedge**
- *"SmoothGrad's advantage entirely collapses on adversarially trained models"* — true for K=50, σ=0.1, this architecture. Hedge.
- *"plain input gradient norm as the default attribution-based predictor"* — yes, but caveat: for L∞ vulnerability prediction. For other purposes (explanations to humans) SmoothGrad still has merits.
- *"Integrated Gradients achieves only AUROC = 0.76"* — needs a baseline-ablation table before this is publishable.

**Missing baselines / experiments**
- σ sweep for SmoothGrad: AUROC vs σ ∈ {0.025, 0.05, 0.1, 0.2}.
- K sweep: AUROC vs K ∈ {5, 10, 25, 50, 100}.
- IG with multiple baselines (zero, black, blurred, Gaussian, mean image).
- Direct measurement of ‖∇L‖ inside the SmoothGrad ball, for vanilla vs AT.
- A modern attribution method (Grad-CAM, DeepLIFT, or Layer-IG) as control.

---

## Paper 4 — Training Difficulty is Orthogonal to Adversarial Vulnerability

**Verdict:** Moderate. Workshop paper. The negative result is real and worth publishing, but the framing is more interesting than the evidence supports.

**Strengths**
- Negative results on forgetting events / C-score are useful — the field has casually assumed these *should* correlate with adversarial vulnerability.
- The conceptual unbundling — "boundary proximity ≠ learning difficulty" — is well articulated.
- The training-trajectory analysis (H157) with three regimes (FM standard flat, SVHN standard monotone, FM PGD-AT robust-overfitting) is a nice contrast.

**Major weaknesses**
- **The "memorization proxy" semantics are confused.** As defined (fraction of 50% subsets where the sample is misclassified, *even when included in the subset*), this is essentially a bagged-ensemble disagreement measure, and the paper correctly notes it's a boundary-proximity probe. But then it stops being a meaningful test of "training difficulty" — the result is essentially that ensemble disagreement predicts vulnerability, which is unsurprising and already in the literature (Feinman et al.). The paper should either rename the predictor or own the fact that this is not a training-dynamics result.
- **Single approximation of C-score.** True C-score requires hundreds of full retrainings; this paper uses 50 80%-subsets. No comparison to published C-scores on a shared dataset. The negative result could be "C-score approximation is too coarse," not "C-score is uninformative."
- **Forgetting events are computed on training samples; vulnerability is on test samples.** This mismatch is not addressed. The paper measures forgetting on training inputs and then asks whether *test* samples have forgetting counts (which is undefined for test samples unless extrapolated). The mechanism by which a test-sample forgetting count is even computed needs to be explicit.
- **Only Fashion-MNIST.** Multi-dataset generalisation of a negative claim ("training dynamics don't predict vulnerability") requires multi-dataset evidence. The SVHN trajectory result helps the trajectory subsection but does not rescue the forgetting / C-score claims.

**Minor weaknesses**
- "AUROC = 0.54, not significantly different from 0.50" — no actual significance test reported.
- The grokking discussion (§4.5) is decorative and not load-bearing.
- Spearman ρ = +0.92 for SVHN is presented without checkpoint-count or sample-count details; with few checkpoints, ρ can be very noisy.

**Specific claims to fix / hedge**
- *"Training difficulty is orthogonal to adversarial vulnerability"* — title-level overclaim. Evidence is "two specific approximate training-dynamics measures, on FM/FGSM, are uninformative." Hedge to "Standard training-trajectory predictors fail to capture …".
- *"These findings establish a clear conceptual distinction"* — the conceptual distinction is fine; the *empirical* establishment is overstated.

**Missing baselines / experiments**
- Replicate on at least one more dataset for forgetting/C-score.
- Compare to AUM (Area Under the Margin, Pleiss et al. 2020) — a closely-related training-dynamics measure that may *not* perform at chance.
- Joint feature analysis: do forgetting events add *incremental* signal on top of margin? Univariate AUROC near chance is consistent with non-zero conditional information.

---

## Paper 5 — How Adversarial Defences Reshape Decision Boundary Geometry

**Verdict:** Moderate. Workshop / TMLR. The "margin AUROC as third diagnostic" idea is promising but the evidence has serious issues.

**Strengths**
- Margin AUROC is a sensible, cheap, and underused diagnostic; proposing it as a standard is a contribution.
- The MarginWeighted collapse is a useful negative result and a fair illustration of why naive hard-sample emphasis fails.
- The Augmentation-vs-AT contrast (ALP −0.246 vs CURE +0.003) tells a clean story.

**Major weaknesses**
- **10 epochs of training is far too few.** Fashion-MNIST CNNs are not converged at 10 epochs; CIFAR-10 ResNet-18 at 10 epochs is dramatically undertrained (note 64.2% clean accuracy — published vanilla ResNet-18 on CIFAR-10 reaches 93–95%). The CIFAR-10 results are essentially meaningless as evidence about defence geometry; they describe undertrained models.
- **PGD evaluation is the same PGD used to train.** No AutoAttack, no transfer attack, no PGD-100. The "FAT achieves best clean-robustness tradeoff" claim is therefore against a weak attack only — exactly the failure mode Carlini & Wagner warn about.
- **Augmentation-based defence numbers are from earlier experiments (H118–H124) "not re-run under the corrected evaluation protocol."** This is a serious flag — comparing defences across protocols is a known confounder, and it is explicitly admitted. These numbers should not be in Table 2 as if they are comparable.
- **MarginWeighted degenerate AUROC=1.000 on SVHN.** This entry should not appear in the AUROC column at all — it is misleading even with the asterisk. Either omit or move to a "failure modes" subsection.

**Minor weaknesses**
- Vanilla FM "PGD success 5.5%" is implausibly low at ε=15/255 unless the model has somehow learned robustness without AT. Either confirm or audit the attack code.
- The ASCII Pareto frontier figure is unnecessary; a real plot belongs in a real paper.
- Table 5 introduces "Imagenette" but the body never reports Imagenette experimental setup (architecture, epochs).

**Specific claims to fix / hedge**
- *"FAT consistently achieves best clean-robustness tradeoff"* — against PGD only, at 10 epochs. This will not survive AutoAttack.
- *"margin AUROC provides information not captured by clean or adversarial accuracy"* — partly true, but it is highly correlated with the *PGD attack success rate label* used to define it. The orthogonality claim needs a partial-correlation analysis.
- *"MarginWeighted ... cannot be avoided by careful hyperparameter tuning"* — too strong; one weighting scheme was tried.

**Missing baselines / experiments**
- TRADES, MART, and at least one recent SOTA AT variant (e.g., AWP — actually mentioned in the worktree's commit log but not in this paper).
- AutoAttack evaluation. Without it, the paper will be rejected from any S&P/ML venue.
- Train to convergence (≥50 epochs CIFAR, ≥100 CIFAR ResNet).
- Margin AUROC under unseen attacks (the paper itself flags this as future work — it should be in the paper).

---

## Paper 6 — Per-Sample Vulnerability Encoded in Input Geometry (Random Feature Networks)

**Verdict:** Moderate. Workshop or TMLR. The thesis is provocative and the experimental design is clean, but the conclusion is broader than the evidence.

**Strengths**
- The RFNN comparison is an elegant experimental design — exactly the right control for "does training matter?"
- The convergence between RFNN, BNN, and gradient phase to similar AUROC values is genuinely interesting.
- Layer ablation is a thoughtful additional probe.

**Major weaknesses**
- **The "input geometry vs learned representation" dichotomy is false as stated.** A randomly initialised CNN's frozen features still encode strong architectural priors (convolution, locality, ReLU rectification, pooling). The result that RFNN ≈ CNN does *not* show "input geometry alone" — it shows "input geometry + CNN architectural prior" is sufficient. Compare to a frozen *MLP* random feature network to make the input-geometry claim. As written, this is a category error in the conclusion.
- **The "94.5% of signal" attribution is a numerical sleight of hand.** Dividing 0.919/0.972 gives 0.945, but AUROC is not an additive quantity — you cannot decompose it like this. Random features at AUROC 0.5 also satisfy "57%" of margin AUROC by this arithmetic. This is the kind of claim a careful reader will not let slide.
- **The Imagenette result undermines the headline.** A gap of 0.038 doubling from 0.019 on FM extrapolates ominously to ImageNet. The paper acknowledges this but does not adjust the abstract.
- **BNN with MC dropout p=0.3 is a weak Bayesian baseline.** Modern BNN baselines (SWAG, deep ensembles, last-layer Laplace) would likely produce different AUROC and might reveal that "epistemic uncertainty ≈ vulnerability" is dropout-specific.

**Minor weaknesses**
- "RFNN trained via closed-form ridge regression for FM, SGD for Imagenette" — different training procedures for the same predictor confound the gap result.
- Pixel statistics baseline is too weak to be informative; consider PCA / wavelet baseline.
- "Gradient phase analysis AUROC = 0.84" but the construction is described in one paragraph without enough detail to reproduce.

**Specific claims to fix / hedge**
- *"adversarial vulnerability is primarily encoded in the input-space geometry"* — should be "in the input + architectural-prior geometry."
- *"the RFNN captures 94.5% of trained CNN's vulnerability prediction"* — drop the percentage; it is not a percentage.
- *"a model with random (untrained) features should exhibit nearly the same vulnerability ranking"* — only true if "random features" preserves the architectural prior.

**Missing baselines / experiments**
- Random MLP / random fully-connected feature network as the "no architectural prior" control.
- Random projections (Gaussian / Rademacher matrices) on raw pixels.
- Train RFNN with SGD on FM as well, to remove the closed-form/SGD confound.
- Larger architectures (ResNet) — does the gap stay small?
- A genuine deep ensemble for the BNN comparison.

---

## Paper 7 — Confirmatory Examples as Training Augmentation

**Verdict:** Moderate. Workshop. The story is cute and self-aware, but two of the headline numbers are suspect, which would kill the paper as written.

**Strengths**
- The "mirror image of AT" framing is pedagogically valuable.
- Honest acknowledgement that the high margin AUROC (0.996) does *not* translate to robustness — this dissociation is the paper's most interesting contribution.
- The PGD-AT + Anti combination as a method for recovering clean accuracy is a concrete, testable proposal.

**Major weaknesses**
- **The headline number is internally inconsistent.** The abstract says "PGD attack success rises from 2.9% to 0.6%" — that is a *decrease*, not a rise. The body then explains this is "a PGD artifact" where PGD "wraps around." This explanation is hand-wavy and not supported by any diagnostic. Two possibilities: (a) the PGD code has a bug at this regime, or (b) the model's loss landscape genuinely traps PGD. Either way, the paper cannot publish a headline claim of "robustness worsens" while the PGD success number drops. Use only Min-ε and FGSM-success consistently, or run a stronger attack (C&W, AutoAttack) to resolve.
- **Vanilla PGD success at 97.1% (Table 2) vs 2.9% (Table 1) for the same vanilla configuration.** Table 1 and Table 2 contradict each other on the same row. This must be a labelling error but as written the paper is internally inconsistent.
- **Only FM, only one architecture, only one ε.** The "boundary curvature sharpening" hypothesis is plausible but never directly measured — no curvature/Hessian statistic is presented despite multiple citations to CURE.
- **+2.89 pp clean accuracy recovery from PGD-AT + Anti** is the most interesting practical finding but it is reported on a single seed (no std). Given the sensitivity of AT to seed, this is not yet a credible effect size.

**Minor weaknesses**
- The "Anti-Adv Aug" naming conflicts loosely with "anti-adversarial" methods in the literature (e.g., Alfarra et al.) — clarify.
- The conceptual ASCII margin-distribution figure is decorative.
- Comparison to FAT is conceptual, not experimental — no FAT row in Table 1.

**Specific claims to fix / hedge**
- *"PGD attack success rises from 2.9% to 0.6%"* — fix the wording; the number went down, the *effect* (the paper claims) is worse robustness as measured by Min-ε.
- *"the highest margin AUROC ever observed in our experiments"* — true but tautological given the construction.
- *"PGD-AT + Anti recovers clean accuracy"* — needs multi-seed CI.

**Missing baselines / experiments**
- C&W or AutoAttack to resolve the PGD anomaly.
- Direct curvature measurement (Hessian trace / largest eigenvalue) before and after Anti-Adv Aug.
- FAT row in Table 1 for direct comparison.
- CIFAR-10 replication (the paper acknowledges this is needed).
- Multi-seed (≥5) for PGD-AT + Anti's headline +2.89 pp claim.

---

## Paper 8 — Predicting Which Samples Are Hurt by Adversarial Training (Post-AT Audit)

**Verdict:** Moderate-to-Strong (alongside Paper 2 and Paper 3, this is the most publishable). Target: workshop or TMLR; the per-sample AT-cost framing is genuinely fresh.

**Strengths**
- The "double jeopardy" framing is sharp and gives the paper a memorable identity.
- The convergence of margin / top-1 / min-ε / grad-L2 to AUROC ≈ 0.935 is a clean, honest result that demonstrates these features are measuring the same latent variable.
- The proposed interventions (down-weighting, selective AT mixing, curriculum) are concrete and testable.
- The conceptual link to Toneva et al.'s forgetting events is well drawn.

**Major weaknesses**
- **The interventions are proposed but not tested.** This is the obvious next experiment, and without it, the paper is descriptive-only. Even a single ablation showing that down-weighting low-margin samples in AT reduces the AT-hurt set by N% would transform this paper.
- **98 hurt samples is a small N.** AUROC variance on n=98 positives, n=1733 negatives is non-trivial — please report a bootstrap CI (likely ±0.02–0.03), which would make the "all four features at 0.935" claim more nuanced.
- **Single dataset (FM), single architecture, single AT recipe.** The "AT-hurt prediction is 0.935" headline needs at least CIFAR-10 PGD-AT to be credible. Otherwise the appropriate framing is a case study.
- **The 35 "newly right" samples are barely analysed.** Symmetry would suggest these have high pre-AT margin or some other distinguishing feature; the paper does not look. This is a missed opportunity.

**Minor weaknesses**
- The flow diagram (Figure 1) is helpful but the numbers (1831 → 1733 + 98) imply ~94.6% preservation, which is fine, but the gain (35) should be in the diagram too.
- min_eps via binary search "over ε ∈ [0,1]" — this is enormous for a 28×28 input and likely has step-size issues at small ε. Document.
- "Same AUROC ≈ 0.935 as attack vulnerability prediction" — this parallel is highlighted, but Paper 5 reports vanilla margin AUROC of 0.946 on FM. Small inconsistency; reconcile.

**Specific claims to fix / hedge**
- *"AT's clean accuracy loss is therefore not random: it is a systematic, predictable function of pre-AT boundary proximity"* — strong, but accurate for FM. Hedge to FM until replicated.
- *"The vanilla model identifies AT-susceptible samples with high accuracy before any AT is performed"* — true but: only 5.4% of vanilla-correct are hurt, so at AUROC 0.935 the precision at useful operating points is still modest. The PR curve discussion partially addresses this but the abstract overstates.

**Missing baselines / experiments**
- A targeted intervention experiment: re-train PGD-AT with the top-decile low-margin samples (a) up-weighted (b) down-weighted (c) excluded from adversarial augmentation. Compare clean accuracy and robustness.
- Replicate on CIFAR-10 PGD-AT (the paper itself flags this).
- Analyse the 35 "newly right" samples — what predicts gain?
- Comparison to TRADES, FAT, MART — does the AT-hurt set change with the AT variant?

---

# Meta-Review

## Strongest papers (publishable with focused revision)
1. **Paper 3 (Attribution / SmoothGrad collapse)** — most novel single finding, clean mechanistic story, narrowly missing rigour around σ/K sensitivity.
2. **Paper 8 (Post-AT audit)** — sharpest framing ("double jeopardy"), clearest practical implications, only one experiment away from being a strong workshop paper.
3. **Paper 2 (Structural attack-invariance)** — clean thesis and result, needs harder attacks (AutoAttack, decision-based) to survive S&P-grade scrutiny.

## Should be merged
- **Papers 1, 2, 4** could be consolidated into a single benchmark-style paper: "What predicts per-sample vulnerability and what doesn't, across attacks and training-dynamics." The current separation forces each paper to repeat the same architecture/dataset/protocol description and dilutes the central, actually-strong result (margin dominates; training-dynamics measures don't; attack-invariance explains the unified ranking). One consolidated paper at TMLR / NeurIPS Datasets&Benchmarks would land harder than three workshop submissions.
- **Papers 5 and 7** are both about how defences reshape margins — Paper 7's confirmatory-example findings naturally extend Paper 5's "margin AUROC as third diagnostic" framing. Merge into "Margin geometry under adversarial and confirmatory training."

## Should be demoted to appendix / dropped
- **Paper 6 (Random Feature Networks)** as a standalone paper has a conceptual flaw at its centre (the architectural prior is not "input geometry"). Demote to a section/appendix inside the consolidated benchmark paper, framed as a control experiment showing that learned features add ≤4% AUROC beyond random features, *without* the over-strong "input geometry hypothesis" framing.

## Single most important missing experiment (across the whole body of work)
**AutoAttack evaluation, end-to-end, on every result that claims robustness or vulnerability.** The body of work currently rests on PGD-10 / PGD-20 with the same hyperparameters used in training. This is exactly the configuration Carlini & Wagner, Croce & Hein (AutoAttack), and every subsequent robustness benchmark warn against. Without AutoAttack (or at minimum APGD-CE + APGD-DLR + Square), the headline claims of all four defence-related papers (5, 7, 8 and the relevant subsections of 1, 2) are vulnerable to a single reviewer comment. Conversely, *if* the margin-AUROC story survives AutoAttack, the whole bundle becomes substantially stronger.

A close second: at least one **modern architecture** (ResNet-18 / WideResNet-28-10) trained to convergence on at least CIFAR-10 native. Every paper currently bottoms out on the same 5-layer CNN at 10–30 epochs and a 28×28 grayscale input pipeline; this is the single greatest threat to external validity across the entire dissertation extension.

## Overall recommendation
The empirical work is more disciplined than typical MSc-level output (multi-seed stability classes, honest documentation of the PGD saturation artefact, careful negative results). The framing is consistently too strong for the evidence base, which is dominated by one architecture on Fashion-MNIST. Consolidate to 3–4 papers, run AutoAttack everywhere, add one modern-architecture/native-resolution result, and the strongest two consolidated papers become credible workshop or TMLR submissions. As-is, the bundle reads as eight related notes rather than three publishable contributions.
