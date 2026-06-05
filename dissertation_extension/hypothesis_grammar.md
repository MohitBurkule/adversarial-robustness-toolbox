# Unified Hypothesis Grammar
## Version 2 — Full Coverage, Omissions Allowed
## 2026-06-05

Every adversarial robustness hypothesis maps onto this single structure.
Parts marked `?` are optional (omit when not applicable).

---

## Template

```
H[ID]:
  INTERVENTION?:  CHANGE_TARGET → CHANGE_TYPE ( CHANGE_SUBJECT )
  SIGNAL:         SIGNAL_TYPE  from  SOURCE
  CLAIM:          SIGNAL  RELATE  OUTCOME
  DIRECTION?:     improves | degrades | no_change | non_monotonic | migrates | unknown
  GRANULARITY:    per_sample | per_class | per_model | per_epoch | per_layer | per_pixel
  CONDITIONS:     ARCHITECTURE × TRAINING × ATTACK × EPS × DATASET
  COMPARISON?:    vs BASELINE_ID
```

Omission rules:
- `INTERVENTION = none` → omit entire block (pure observation study, H01–H60 style)
- `DIRECTION = unknown` → omit (we measure, not predict — exploratory)
- `COMPARISON` → omit if RELATE already specifies the comparison target
- `SOURCE` can be inferred from CHANGE_SUBJECT when INTERVENTION is present

---

## Slot Dictionaries

### CHANGE_TARGET
```
data        — what training/test data looks like
model       — architecture, weights, components
training    — how optimisation runs
inference   — what happens at test time
attack      — how the adversary operates
none        — pure observation, no experimental manipulation
```

### CHANGE_TYPE
```
add         — introduce new component/data
remove      — delete component/data/samples
replace     — substitute one thing for another
scale_up    — increase continuous variable
scale_down  — decrease continuous variable
combine     — merge K things together
split       — separate into parts
constrain   — restrict to subset
randomise   — introduce stochasticity
transfer    — apply something from a different context
freeze      — lock component (no update)
unfreeze    — allow previously frozen component to update
alternate   — switch between two modes during training
```

### CHANGE_SUBJECT
```
# Data subjects
augmentation            — standard augmentation policy
labels                  — training labels (random, noisy, flipped)
class_balance           — proportion of each class in training
sample_difficulty       — subsample by margin quartile
pixel_bits              — bit depth of pixel values
compression             — lossy/lossless compression codec
noise_level             — additive noise σ at train/test
temporal_frames         — stack of frames (video simulation)
spurious_correlation    — shortcut features in training data
padding                 — border pixels around image
lsb_bits                — least significant bits of each pixel
patch_pixels            — local spatial patch content

# Model subjects
architecture_width      — number of filters/units per layer
architecture_depth      — number of layers
expert_count            — number of MoE experts
layer                   — specific layer (first, last, all)
component               — specific submodule (attention, BN, etc.)
weight_values           — raw parameter tensors
weight_displacement     — net delta W_final - W_init
bottleneck_size         — dimensionality of bottleneck layer
codebook_size           — VQ codebook number of entries

# Training subjects
loss_function           — cross-entropy, TRADES, NT-Xent, etc.
optimiser               — SGD, Adam, etc.
lr_schedule             — cosine, cyclic, wake-sleep alternation
batch_composition       — which samples go in each batch
curriculum_order        — easy→hard or hard→easy sequencing
epochs                  — number of training steps
regulariser             — dropout, weight decay, spectral norm
single_class_data       — one class at a time (catastrophic interference)

# Inference subjects
input_pixels            — raw pixel values at test time
smoothing_copies        — K noisy copies for majority vote
purification_step       — denoiser/diffusion applied pre-classifier
quantisation_level      — bit-depth applied at inference
padding_noise           — random border pixels at inference

# Attack subjects
attack_type             — FGSM, PGD, CW, random, Langevin
epsilon_value           — perturbation budget
step_count              — number of PGD iterations
direction_source        — gradient, random unit vector, trajectory
pixel_subset            — restrict perturbation to K pixels
adversarial_trajectory  — sequence of gradient-sign steps from another image
```

### SIGNAL_TYPE
```
# Input-space signals
gradient_norm           — ||∇_x L||
gradient_direction      — sign(∇_x L) as direction vector
gradient_sign_agreement — cosine similarity of gradients across models
saliency_entropy        — spatial entropy of saliency map
saliency_spatial_moment — centre-of-mass / eccentricity of saliency
subject_energy_ratio    — ||δ * subject_mask|| / ||δ|| (perturbation on salient pixels)
pca_tail_energy         — fraction of δ in low-variance PCA subspace
symmetry_gap            — |margin(x) - margin(transform(x))|
pixel_lsb_noise         — bit-flip randomisation level
patch_variance_map      — local 4x4 patch variance

# Model-space signals
logit_margin            — max_logit - second_max_logit
softmax_gap             — top_prob - second_prob
logit_entropy           — H(softmax(logits))
feature_norm            — ||penultimate_layer(x)||
mahalanobis_distance    — Mahalanobis in feature space
lid                     — local intrinsic dimensionality
dead_neuron_count       — ReLU outputs == 0 per sample per layer
weight_displacement_norm  — ||W_final - W_init||
weight_displacement_direction — unit vector of net training delta
interference_magnitude  — margin change after single-class fine-tuning

# Generative/auxiliary model signals
score_magnitude         — ||s_θ(x)|| from score network
score_direction_cosine  — cos(−score, FGSM_direction)
reconstruction_mse      — ||G(∇_x) - x||² from inversion network
reconstruction_ncc      — normalised cross-correlation
certified_radius        — σ·Φ⁻¹(p_A) from randomised smoothing
smoothed_margin         — majority-vote margin over K noisy copies

# Training dynamics signals
per_epoch_margin        — margin at checkpoint epoch k
forgetting_count        — number of correct→incorrect flips during training
overfitting_delta       — margin_peak - margin_final
auroc_at_epoch_k        — margin AUROC at checkpoint k
forgetting_horizon      — first epoch of stable correct classification
robust_island_membership — top-10% margin at final epoch (binary)

# Attack signals
min_eps_to_flip         — smallest ε achieving 50% ASR (sweep)
steps_to_flip           — first PGD step where prediction flips
asr_of_transferred_trajectory — ASR when applying source image's attack steps to target
transfer_gradient_cosine — cosine sim between source-epoch and final-epoch gradients

# Composite/set signals
jointly_robust_set      — {x : all attacks fail(x)}
vulnerable_set_overlap  — Jaccard(vulnerable_before, vulnerable_after)
ensemble_disagreement   — prediction mismatch across K models
dual_model_disagreement — clean_model vs AT_model prediction mismatch
rank_order_correlation  — Spearman ρ between two vulnerability vectors
```

### SOURCE
```
# Direct — no auxiliary model needed
input_pixels
input_gradient
input_jacobian
model_logits
model_features          — penultimate layer activations
model_weights
conv_feature_maps       — intermediate conv outputs (for GradCAM)

# Auxiliary trained models
score_network           — denoising score matching network
inversion_network       — gradient → image reconstruction network
teacher_model           — pretrained AT model for distillation
denoising_autoencoder   — CNN purifier trained on clean/adv pairs
diffusion_model         — DDPM-based purifier
ssl_encoder             — SimCLR / rotation-prediction pretrained encoder
vq_bottleneck           — vector quantisation codebook

# Process / trajectory
weight_trajectory       — sequence of weight snapshots W_0...W_T
gradient_trajectory     — sequence of ∇_x at each PGD step
checkpoint_sequence     — saved model at epochs [k1, k2, ...]
adversarial_trajectory  — PGD step directions from a source image
training_loss_curve     — per-sample loss at each epoch

# Transforms applied to input
jpeg_transform          — PIL JPEG encode/decode at quality q
bit_depth_transform     — round(x*(2^b-1))/(2^b-1)
lsb_flip_transform      — XOR bottom-K bits with random mask
noise_augmentation      — additive Gaussian / uniform noise
anti_adversarial_transform — x - ε·sign(∇_x L) [move away from boundary]
temporal_sequence       — stack of [x+noise, x, x+noise]
padded_input            — image padded with B rows/cols of random noise
pca_projection          — project to top-K PCA components and reconstruct
variable_precision_image — per-patch adaptive bit depth
```

### RELATE
```
predict           — SIGNAL ranks/predicts OUTCOME (AUROC / Spearman ρ)
correlate_with    — Pearson / Spearman ρ between two continuous signals
align_with        — cosine similarity / directional match
transfer_to       — SIGNAL applied in new context → produces OUTCOME
improve           — intervention → OUTCOME increases (↑ good)
reduce            — intervention → OUTCOME decreases (↓ bad)
recover           — SIGNAL restores OUTCOME after perturbation
exceed_baseline   — SIGNAL beats known BASELINE by threshold
compare_to        — neutral comparison (direction is the finding)
analyse           — exploratory: describe structure of SIGNAL set
migrate           — OUTCOME changes location/membership, not magnitude
preserve          — rank order / membership maintained despite other change
```

### OUTCOME
```
# Attack success
fgsm_success            — binary 0/1 per sample
pgd_success             — binary 0/1 per sample
cw_success              — C&W L2 attack success
autoattack_success      — AutoAttack (strongest baseline)
random_fgsm_success     — random-direction ε-perturbation success

# Vulnerability ranking
margin_rank             — ordinal rank of logit margin
auroc                   — AUROC of signal predicting attack success
spearman_rho            — Spearman ρ between signal and margin/success
cross_architecture_auroc — AUROC tested on different architecture

# Model performance
clean_accuracy
robust_accuracy         — accuracy under attack
certified_accuracy      — randomised smoothing certified acc at radius r
generalisation_gap      — train_acc - test_acc

# Geometric / structural outcomes
functional_classifier   — does displaced model actually classify? (H210)
decision_boundary_distance — geometric distance to nearest boundary
manifold_distance       — distance to data manifold
pca_subspace_alignment  — how much δ aligns with low-variance directions
pareto_frontier         — clean_acc vs robust_acc curve

# Temporal / set outcomes
margin_trajectory       — sequence of margins across training epochs
robust_island_persistence — does top-10% margin set persist across epochs?
vulnerable_set_jaccard  — overlap of vulnerable sets across conditions
rank_preservation       — Spearman ρ between vulnerability ranks of two models
cross_seed_overlap      — Jaccard of robust/vulnerable sets across random seeds
recovery_rate           — fraction of adversarials correctly reclassified after purification

# Training dynamics outcomes
robust_overfitting_epoch — epoch where robust accuracy peaks then declines
forgetting_count        — training-time oscillation count
peak_to_final_delta     — margin_peak - margin_at_final_epoch
```

### GRANULARITY
```
per_sample    — one measurement per test image
per_class     — aggregate over all samples of same class
per_model     — one number per trained model
per_epoch     — one measurement per training checkpoint
per_layer     — one measurement per network layer
per_pixel     — one measurement per pixel location
per_patch     — one measurement per spatial patch
```

### CONDITIONS slots
```
ARCHITECTURE: [cnn_small, cnn_wide, cnn_deep, resnet18, vit_tiny, moe_k4,
               capsnet, mlp, ssl_encoder+linear_head]

TRAINING:     [standard, pgd_at, trades, augmix, jpeg_q30, jpeg_q50,
               lsb_flip_k2, bit_depth_4, anti_adversarial, distillation,
               unlearning, score_matching, simclr, rotation_ssl,
               wake_sleep_alternation, constrained_no_memorise]

ATTACK:       [fgsm, pgd_10, pgd_20, cw_l2, deepfool, uap, random_fgsm,
               score_guided_langevin, trajectory_transfer, spatial_attack]

EPS:          [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]

DATASET:      [fashion_mnist, cifar10, cifar100, svhn]
```

---

## Constraint Graph (valid SIGNAL → SOURCE pairings)

```
score_magnitude         requires  SOURCE = score_network
reconstruction_mse      requires  SOURCE = inversion_network
reconstruction_ncc      requires  SOURCE = inversion_network
certified_radius        requires  SOURCE = noise_augmentation (randomised smoothing)
smoothed_margin         requires  SOURCE = noise_augmentation
weight_displacement_*   requires  SOURCE = weight_trajectory
gradient_trajectory     requires  SOURCE = gradient_trajectory OR adversarial_trajectory
per_epoch_margin        requires  SOURCE = checkpoint_sequence
forgetting_count        requires  SOURCE = training_loss_curve OR checkpoint_sequence
dead_neuron_count       requires  SOURCE = conv_feature_maps (ReLU hooks)
subject_energy_ratio    requires  SOURCE = conv_feature_maps (GradCAM) AND input_gradient
pca_tail_energy         requires  SOURCE = pca_projection AND input_gradient
symmetry_gap            requires  SOURCE = input_pixels (transform applied inline)
interference_magnitude  requires  SOURCE = model_weights + single_class_finetuning
dual_model_disagreement requires  SOURCE = two trained models (clean + AT)
ensemble_disagreement   requires  SOURCE = K trained models
asr_transferred_traj    requires  SOURCE = adversarial_trajectory (from source image)
```

---

## Example Mappings (H01–H249)

| ID | INTERVENTION | SIGNAL | SOURCE | RELATE | OUTCOME |
|----|-------------|--------|--------|--------|---------|
| H01 | none | min_eps_deepfool | model_logits | predict | pgd_success |
| H12 | none | saliency_entropy | input_gradient | predict | fgsm_success |
| H46 | inference.add(bit_depth_transform) | — | bit_depth_transform | reduce | fgsm_success |
| H117 | training.replace(loss=IBP) | certified_radius | noise_augmentation | exceed_baseline | pgd_at_certified |
| H158 | data.replace(anti_adversarial_transform) | — | anti_adversarial_transform | improve | clean_accuracy |
| H183 | training.remove(class_k_samples via unlearning) | pgd_success | model_weights | compare_to | pre_unlearn_pgd_success |
| H206 | none | neighbourhood_density | input_pixels (kNN) | predict | fgsm_success |
| H210 | model.transfer(weight_displacement) | clean_accuracy | weight_trajectory | compare_to | random_displacement_baseline |
| H213 | model.add(inversion_network) | reconstruction_mse | inversion_network | predict | pgd_success |
| H214 | model.add(score_network via DSM) | score_magnitude | score_network | predict | pgd_success |
| H220 | training.add(random_patch_mask) | pgd_success | — | compare_to | baseline_pgd |
| H222 | data.scale_down(pca_projection, n_dims) | min_eps_to_flip | pca_projection | compare_to | full_dim_baseline |
| H227 | data.replace(jpeg_transform, q=30) | pgd_success | jpeg_transform | compare_to | q=100_baseline |
| H229 | inference.add(gradient_descent_on_input, T steps) | logit_margin | model_logits | improve | original_margin |
| H231 | attack.transfer(adversarial_trajectory from X to X') | asr_transferred | adversarial_trajectory | compare_to | from_scratch_pgd |
| H-A | data.remove(high_influence_samples) | vulnerable_set_jaccard | training_loss_curve | migrate | original_vulnerable_set |
| H-D | none | forgetting_count | checkpoint_sequence | predict | pgd_success |
| H-E | none | dead_neuron_count | conv_feature_maps | predict | fgsm_vs_pgd_discrepancy |
| H-G | none | pca_tail_energy | pca_projection+gradient | predict | fgsm_success |
| H-I | model.add(vq_bottleneck, codebook_size=C) | pareto_frontier | model_logits | analyse | clean_acc vs robust_acc |
| H-L | none | margin_trajectory | checkpoint_sequence | analyse | robust_island_persistence |

---

## Generator Logic (pseudocode)

```python
for intervention in INTERVENTIONS:
    for signal in SIGNALS:
        for relate in RELATES:
            for outcome in OUTCOMES:
                for granularity in GRANULARITIES:
                    for conditions in CONDITIONS_PRODUCT:
                        h = Hypothesis(intervention, signal, relate, outcome,
                                       granularity, conditions)
                        if constraint_graph.valid(h):
                            if not already_in_corpus(h):  # check H01-H249
                                score = novelty(h) + feasibility(h) + info_density(h)
                                ranked_queue.push(h, score)
```

Estimated space: ~10 × 50 × 11 × 40 × 6 × (9×14×10×6×4) = millions of combinations
After constraint filtering: ~50,000 valid
After deduplication against H01–H249: ~49,700 novel
After feasibility filter (Fashion-MNIST runnable in <10 min): ~8,000 runnable

This is the full generative hypothesis space.
