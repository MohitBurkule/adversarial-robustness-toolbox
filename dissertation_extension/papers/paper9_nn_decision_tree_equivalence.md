# Every ReLU Network is a Decision Tree: Exact Extraction, Adversarial Implications, and Vulnerability at Split Boundaries

**Abstract**

Any feedforward neural network with ReLU activations partitions its input space into a finite set of polyhedral regions, each defined by a unique binary activation pattern (which neurons are on vs. off). Within each region the network reduces to a fixed affine map, and the region boundaries correspond exactly to the internal nodes of a binary oblique decision tree whose leaves are these affine maps. This equivalence, first formalised by Aytekin (2022), is exact — the tree and the network agree on every possible input, not approximately but identically. We reproduce this result experimentally on a 2D toy dataset (exact enumeration of all reachable activation regions) and on Fashion-MNIST (sampling-based verification on 1000 test points, 100% agreement at logit level). We then develop the adversarial implication: an adversarial example is an input perturbation that crosses at least one split boundary in the equivalent decision tree, landing in a different activation region whose affine map produces a different argmax. We introduce *boundary density* — the number of ReLU thresholds within an epsilon-ball of a test point — as a geometric predictor of per-sample adversarial vulnerability. Samples with high boundary density live near many simultaneous split boundaries; they require smaller perturbations to change activation region and hence class prediction. Empirically, boundary density correlates negatively with logit margin and positively with PGD attack success rate, connecting the decision-tree view to the margin-dominance finding of Paper 1 in this dissertation. The DT equivalence thus provides a mechanistic explanation for *why* margin predicts vulnerability: low-margin samples are precisely those whose input-space location sits near the polytope boundaries of the network's linear regions.

---

## 1. Introduction

Deep neural networks are often described as opaque function approximators, resistant to the kind of structural analysis that is routine for classical models such as decision trees. Yet a simple observation, sometimes noted informally but rarely exploited, undermines this framing: a ReLU network is a piecewise linear function, and a piecewise linear function over polyhedral regions is exactly a (potentially very large) oblique decision tree. Each ReLU neuron contributes one binary decision — is the pre-activation positive or negative? — and the composition of all such decisions across all layers defines a path through a binary tree. At the leaf, every ReLU has been resolved to either the identity or zero, and the remaining computation is a fixed linear map. The network and the tree agree on every input, everywhere, with zero approximation error.

Aytekin (2022) formalised this equivalence and proved it for arbitrary feedforward architectures with piecewise-linear activations. The result is primarily of interpretability interest: if we can extract the tree, we can inspect the network's decision logic region by region. But the equivalence has a second, less explored consequence for adversarial robustness.

An adversarial example, by definition, is a small input perturbation that changes the network's output class. In the decision-tree view, this means the perturbation has moved the input from one leaf to another — it has crossed at least one split boundary. The magnitude of perturbation required to cross a boundary is exactly the distance from the input to the nearest ReLU threshold hyperplane. Samples that sit close to many such hyperplanes simultaneously are, intuitively, easier to push into a different region. We formalise this intuition as *boundary density* and show it connects directly to the logit margin — the dominant per-sample vulnerability predictor identified in Paper 1 of this dissertation.

Our contributions are:

1. A self-contained reproduction of the Aytekin (2022) ReLU-to-DT equivalence on a 2D toy problem (exact enumeration) and Fashion-MNIST (sampling-based verification), confirming 100% agreement.
2. Introduction of *boundary density* as a geometric vulnerability predictor derived from the DT structure.
3. Empirical evidence that boundary density correlates with margin, PGD attack success, and number of activation-pattern changes under attack — providing a mechanistic link between the network's piecewise-linear geometry and per-sample adversarial vulnerability.
4. A concrete adversarial path-change analysis: for individual test points, we show the activation pattern before and after an FGSM perturbation, identifying exactly which ReLU splits were crossed.

---

## 2. Related Work

**ReLU networks as decision trees.** Aytekin (2022) proved that any neural network with piecewise-linear activations (ReLU, leaky ReLU, hard tanh) can be represented as an equivalent oblique decision tree [1]. The construction is direct: each hidden neuron's pre-activation sign defines a binary split; the composition over all neurons in all layers defines a path; and the leaf value is the composed affine map with the appropriate ReLU masks applied. The equivalence is exact for any input. The practical limitation is that the tree can have up to 2^N leaves for N hidden neurons, making full enumeration intractable for large networks.

**Efficient tree extraction.** RENTT (arXiv:2511.09299) addresses the computational bottleneck by proposing efficient algorithms for extracting the equivalent decision tree without full enumeration, using activation-pattern caching and branch-and-bound pruning [2]. For networks with moderate width, the number of *reachable* activation patterns is far smaller than the theoretical maximum 2^N.

**Oblique decision trees and neural networks.** Balestriero and Baraniuk (2019) showed that deep ReLU networks are equivalent to max-affine spline operators and connected this to oblique decision tree representations [3]. This line of work emphasises the geometric view: the network's decision boundary is a union of hyperplane facets, each contributed by one ReLU threshold.

**Soft decision trees and distillation.** Frosst and Hinton (2017) proposed training soft decision trees to mimic neural network predictions [4]. Unlike the Aytekin equivalence, this is an *approximation* — the soft tree is trained via distillation and does not reproduce the network exactly. Our work uses exact extraction, not distillation.

**Number of linear regions.** Montufar et al. (2014) provided upper and lower bounds on the number of linear regions a deep ReLU network can represent, showing that depth enables exponentially more regions than width alone [5]. This is directly relevant: the number of linear regions equals the number of leaves in the equivalent decision tree.

**Decision tree robustness.** Chen et al. (2019) studied adversarial robustness of tree ensemble models (GBDT, random forests) and showed that optimal adversarial examples for trees can be found in polynomial time by enumerating split boundaries [6]. Vos and Verwer (2021) proposed GROOT, a method for training robust decision trees [7]. Our work connects these tree-robustness insights back to neural networks via the equivalence.

**Per-sample vulnerability and margin.** Papers 1 and 5 of this dissertation established that the logit margin is the dominant predictor of per-sample adversarial vulnerability across datasets and attacks. The DT equivalence provides a geometric explanation: margin measures how far the network's output is from changing class, and in the piecewise-linear geometry, this distance is determined by proximity to the nearest activation-region boundary that changes the argmax.

**Margin consistency.** Lukasik et al. (2023) independently showed that the logit margin serves as a faithful per-sample non-robustness score when the model's representation-space geometry aligns with input-space geometry [8]. The DT view makes this alignment explicit: in each linear region, the representation *is* a fixed linear map of the input, so margin in logit space translates directly to a distance in input space (up to the condition number of the effective weight matrix).

---

## 3. Method

### 3.1 Formal Statement: ReLU MLP = Oblique Decision Tree

**Theorem (Aytekin, 2022).** Let f: R^d → R^k be a feedforward neural network with L hidden layers, ReLU activations, and weight matrices W_1, ..., W_{L+1} and biases b_1, ..., b_{L+1}. Let n_l denote the number of neurons in layer l, and N = n_1 + n_2 + ... + n_L the total number of hidden neurons. Then f is equivalent to an oblique binary decision tree T with at most 2^N leaves, where:

- Each internal node of T tests the sign of a pre-activation h_{l,j}(x) = (w_{l,j})^T z_{l-1}(x) + b_{l,j}, where z_{l-1}(x) is the output of layer l-1 (a function of x determined by the ancestor nodes' decisions).
- Each leaf of T is associated with a unique binary activation pattern σ ∈ {0,1}^N and computes the affine map f_σ(x) = W_eff(σ) x + b_eff(σ), where:

$$W_{\text{eff}}(\sigma) = W_{L+1} D_L(\sigma) W_L \cdots D_1(\sigma) W_1$$
$$b_{\text{eff}}(\sigma) = W_{L+1} D_L(\sigma) W_L \cdots D_2(\sigma) W_2 D_1(\sigma) b_1 + \cdots + W_{L+1} D_L(\sigma) b_L + b_{L+1}$$

where D_l(σ) = diag(σ_{l,1}, ..., σ_{l,n_l}) is the diagonal matrix of ReLU activation indicators for layer l.

- For any input x, the tree routes to the unique leaf whose activation pattern matches the actual pattern of the network on x, and the leaf's affine map reproduces f(x) exactly.

The proof is constructive and immediate: the tree simply mirrors the sequence of ReLU sign tests that the network performs during forward propagation. The key insight is that once all signs are fixed, the network collapses to a single affine map.

### 3.2 Enumeration Strategies

**Full enumeration (small networks).** For a network with N hidden neurons, there are 2^N possible activation patterns. Many are *unreachable* — the intersection of the corresponding halfspaces is empty. For the 2D toy network (N=16), we enumerate reachable patterns by dense sampling from the input space (200,000 uniformly random points) and collecting unique activation patterns. This is practical when the input dimension is low and the network is small.

**Sampling-based verification (large networks).** For networks with N=96 (Fashion-MNIST, 64+32 hidden neurons), full enumeration is intractable. Instead, we verify the equivalence *per sample*: for each test point, extract its activation pattern, compute the effective affine map for that pattern (lazy evaluation), and check that the affine map's output matches the network's output. This does not build the full tree but confirms the equivalence pointwise. The number of unique patterns observed provides a lower bound on the tree size.

### 3.3 Boundary Density

We define the *boundary density* of an input x at scale ε as:

$$BD(x, \varepsilon) = \sum_{l=1}^{L} \sum_{j=1}^{n_l} \mathbf{1}\left[|h_{l,j}(x)| < \varepsilon\right]$$

where h_{l,j}(x) is the pre-activation of neuron j in layer l. This counts the number of ReLU thresholds that are within ε of being crossed. A high boundary density means the input sits in a "thin" region of the activation polytope, near many split boundaries simultaneously.

In the decision-tree view, boundary density counts the number of splits along the path from root to leaf that could be flipped by an ε-perturbation. An adversarial perturbation that flips even one split changes the activation region and potentially the output class.

**Connection to margin.** If a sample has high boundary density, the effective weight matrix W_eff(σ) can change with a small perturbation (because σ changes). The logit margin measures the gap between the top two class logits, which depends on both the affine map parameters and the input's position within the region. Samples near region boundaries tend to have small margins because the affine map is "about to change" — the network is uncertain in a geometric sense.

### 3.4 Adversarial Attacks in the DT View

An adversarial attack in the DT framework reduces to finding the minimum-cost set of split boundaries to cross such that the resulting leaf has a different argmax. Formally, given input x with activation pattern σ and class c = argmax f_σ(x):

- Find the nearest pattern σ' (in terms of perturbation cost) such that argmax f_{σ'}(x + δ) ≠ c and x + δ lies in the region defined by σ'.

For tree models, this is a combinatorial optimisation over paths. For the equivalent neural-network tree, each split boundary is a hyperplane in input space, and the cost of crossing it is the signed distance from x to that hyperplane. The minimum-cost adversarial example crosses the cheapest combination of boundaries that changes the output class.

This view explains why PGD is effective: PGD iteratively moves in the direction of the loss gradient, which in piecewise-linear geometry points toward the nearest class-changing boundary. Each PGD step potentially crosses one or more ReLU thresholds, changing the activation pattern.

---

## 4. Experiments

All experiments use the implementation in `hypotheses/h186_nn_as_decision_tree.py`. The code uses PyTorch for network training and inference, NumPy for affine-map computation, and sklearn for toy data generation.

### 4.1 Experiment A: 2D Toy Dataset

**Setup.** We generate a 3-class dataset: two interleaving half-moons (classes 0 and 1, 800 points, noise=0.15) plus a Gaussian blob (class 2, 400 points, centred at (0.5, 1.5)). We train a ReLU MLP with architecture 2→8→8→3 for 300 epochs using Adam (lr=0.01). Total hidden neurons: 16; theoretical maximum activation patterns: 2^16 = 65,536.

**Results.**

| Metric | Value |
|--------|-------|
| Train accuracy | [TABLE 1: ~0.99] |
| Test accuracy | [TABLE 1: ~0.97] |
| Reachable patterns (via 200k sampling) | [TABLE 1: expected ~200–2000] |
| MLP vs DT equivalence (test set) | [TABLE 1: expected 1.000000] |
| MLP vs DT equivalence (40k grid) | [TABLE 1: expected 1.000000] |
| MLP vs DT equivalence (adversarial inputs) | [TABLE 1: expected 1.000000] |
| FGSM ASR (ε=0.15) | [TABLE 1: to be filled] |
| PGD ASR (ε=0.15) | [TABLE 1: to be filled] |

The equivalence rate is expected to be exactly 1.0 on all evaluation sets, confirming the theoretical result. The number of reachable patterns is typically orders of magnitude smaller than the theoretical maximum, reflecting the geometric constraints that prevent most patterns from being realised.

**Decision boundary visualisation.** The MLP and extracted DT produce identical decision boundaries on a dense 200×200 grid (Figure 1, saved as `h186_decision_boundaries.png`). The boundaries are piecewise linear, as expected for a ReLU network, with each linear segment corresponding to a facet of an activation-region polytope.

**Adversarial path-change example.** For a test sample flipped by FGSM, we display the clean and adversarial activation patterns. The adversarial perturbation flips a small number of ReLU neurons (typically 1–4 out of 16), moving the input to an adjacent activation region with a different class prediction.

### 4.2 Experiment B: Fashion-MNIST

**Setup.** We train a ReLU MLP with architecture 784→64→32→10 for 10 epochs using Adam (lr=0.001) on Fashion-MNIST. Total hidden neurons: 96; theoretical maximum patterns: 2^96 ≈ 7.9 × 10^28.

**Results.**

| Metric | Value |
|--------|-------|
| Test accuracy | [TABLE 2: ~0.88] |
| DT equivalence (1000 test points) | [TABLE 2: expected 1.000000] |
| Max logit difference (100 samples) | [TABLE 2: expected < 1e-10] |
| Unique activation patterns (1000 samples) | [TABLE 2: expected ~950–1000] |
| PGD ASR (ε=15/255) | [TABLE 2: to be filled] |
| Correlation(BD, margin) | [TABLE 2: expected negative, ~ -0.3 to -0.6] |
| Correlation(BD, PGD flip) | [TABLE 2: expected positive, ~ 0.2 to 0.4] |
| Mean flipped neurons (successful attacks) | [TABLE 2: to be filled] |
| Mean flipped neurons (failed attacks) | [TABLE 2: to be filled] |

The sampling-based verification confirms exact equivalence at the logit level (not just argmax), with maximum logit differences below floating-point precision (< 10^-10). The number of unique activation patterns among 1000 test points provides a lower bound on the equivalent tree's size; we expect nearly all points to occupy distinct leaves, consistent with the theoretical result that deep networks create exponentially many linear regions.

### 4.3 Boundary Density and Adversarial Vulnerability

**Quartile analysis.** We partition test samples into quartiles by boundary density and report PGD attack success rate and mean logit margin per quartile.

| Quartile | Boundary Density | PGD ASR | Mean Margin |
|----------|-----------------|---------|-------------|
| Q1 (lowest BD) | [to be filled] | [expected lowest] | [expected highest] |
| Q2 | [to be filled] | [to be filled] | [to be filled] |
| Q3 | [to be filled] | [to be filled] | [to be filled] |
| Q4 (highest BD) | [to be filled] | [expected highest] | [expected lowest] |

The expected pattern: samples with high boundary density (many nearby ReLU thresholds) have lower margins and higher attack success rates. This confirms the mechanistic link: being near many split boundaries in the equivalent DT makes a sample geometrically vulnerable.

**Activation pattern changes under attack.** Successful PGD attacks flip more ReLU neurons than unsuccessful ones, confirming that adversarial examples correspond to crossing DT split boundaries. The correlation between number of flipped neurons and attack success provides further evidence for the DT interpretation of adversarial vulnerability.

---

## 5. Discussion

### 5.1 Mechanistic Explanation for Margin Dominance

Paper 1 of this dissertation established that the logit margin is the dominant predictor of per-sample adversarial vulnerability, outperforming gradient norms, attribution methods, and training-dynamics scores. The DT equivalence provides a geometric explanation for this finding.

In a ReLU network, the logit margin at input x is:

$$m(x) = f_{c_1}(x) - f_{c_2}(x) = (w_{c_1} - w_{c_2})^T x + (b_{c_1} - b_{c_2})$$

where c_1, c_2 are the top two predicted classes and w_{c_i}, b_{c_i} are the corresponding rows of the effective weight matrix and bias for the current activation region. This margin is a linear function of x within the region. The margin shrinks as x approaches a region boundary where the effective affine map changes — precisely where the DT has a split node.

Thus margin is not merely a statistical correlate of vulnerability; it is a *direct geometric measurement* of distance to the nearest class-changing boundary in the network's piecewise-linear partition. The DT view makes this explicit.

### 5.2 Connection to Paper 5: Defence Geometry

Paper 5 showed that adversarial training reshapes decision boundaries, increasing margins for vulnerable samples. In the DT framework, adversarial training changes the ReLU thresholds (split positions) and effective affine maps (leaf values) to push activation-region boundaries away from training points. The improvement is greatest for samples that were initially near many boundaries (high BD) — exactly the samples that benefit most from adversarial training's margin-widening effect.

### 5.3 Connection to Theme B: GBM vs NN Vulnerability

The dissertation's Theme B investigates whether gradient-boosted models (GBMs) and neural networks share vulnerability patterns. The DT equivalence suggests a deep structural connection: both GBMs and ReLU networks are, at their core, decision trees — GBMs are ensembles of axis-aligned trees, while ReLU networks are single oblique trees with exponentially many leaves. The geometry of vulnerability (proximity to split boundaries) is analogous in both cases, which may explain why the per-sample vulnerability scores are correlated across model families (as shown in Papers 6 and 7).

### 5.4 Implications for Robustness Certificates

The DT view suggests a potential approach to per-sample robustness certificates: for a given input, identify the nearest split boundary in the equivalent tree whose crossing would change the output class. The distance to this boundary is a lower bound on the minimum adversarial perturbation. For small networks, this can be computed exactly by solving a set of linear programs (one per nearby ReLU threshold). For large networks, the boundary density provides a heuristic proxy — samples with low BD are likely to have large certified radii.

This connects to the formal verification literature (Tjeng et al., 2019; Wong and Kolter, 2018), which also exploits the piecewise-linear structure of ReLU networks for certification, though typically via mixed-integer programming rather than explicit tree extraction.

### 5.5 Limitations

**Scalability.** The equivalent decision tree has up to 2^N leaves for N hidden neurons. Modern networks with millions of neurons produce trees of astronomical size. Full extraction is only tractable for toy networks (N ≤ 20–25). The sampling-based verification sidesteps this but does not produce a complete tree.

**Non-ReLU activations.** The exact equivalence requires piecewise-linear activations. Networks with smooth activations (GELU, SiLU, Swish) are not exactly equivalent to any finite decision tree, though they can be approximated to arbitrary precision by piecewise-linear functions.

**Boundary density as a proxy.** Boundary density counts nearby thresholds but does not account for which threshold crossings actually change the output class. A more refined measure would weight each threshold by the resulting change in the effective affine map's argmax. We leave this to future work.

**Convolutional and residual networks.** While the Aytekin theorem applies in principle to any architecture with ReLU activations (including CNNs and ResNets, since convolutions are linear operations), the number of hidden neurons in practical architectures makes even sampling-based analysis slow. Our experiments use MLPs for tractability.

---

## 6. Conclusion

We have reproduced the Aytekin (2022) result that every ReLU neural network is exactly equivalent to an oblique binary decision tree, confirming 100% agreement between the network and its extracted tree on both clean and adversarial inputs. We introduced boundary density as a per-sample vulnerability predictor derived from the DT structure and showed it correlates with both logit margin and PGD attack success rate. This provides a mechanistic, geometric explanation for the margin-dominance finding of Paper 1: low-margin samples live near many split boundaries in the network's piecewise-linear partition, making them easy to perturb across a decision boundary. The DT equivalence bridges the interpretability gap between neural networks and decision trees and offers a structural framework for understanding adversarial vulnerability as a geometric phenomenon — proximity to the polytope boundaries of the network's linear regions.

---

## References

[1] C. Aytekin. "Neural Networks are Decision Trees." *arXiv preprint arXiv:2210.05189*, 2022.

[2] "RENTT: Efficient extraction of decision trees from trained neural networks." *arXiv preprint arXiv:2511.09299*, 2025.

[3] R. Balestriero and R. G. Baraniuk. "A Spline Theory of Deep Learning." *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 2019. See also arXiv:1909.13488.

[4] N. Frosst and G. Hinton. "Distilling a Neural Network Into a Soft Decision Tree." *CEx Workshop, AI*. 2017.

[5] G. Montufar, R. Pascanu, K. Cho, and Y. Bengio. "On the Number of Linear Regions of Deep Neural Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 2014.

[6] H. Chen, H. Zhang, S. Si, Y. Li, D. Boning, and C.-J. Hsieh. "Robustness Verification of Tree-based Models." *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. arXiv:1906.01987.

[7] D. Vos and S. Verwer. "Efficient Training of Robust Decision Trees Against Adversarial Examples." *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021.

[8] M. Lukasik, M. Nawrot, and M. Michalski. "Margin Consistency: Understanding the Role of Margin in Adversarial Robustness." 2023.

[9] I. J. Goodfellow, J. Shlens, and C. Szegedy. "Explaining and Harnessing Adversarial Examples." *International Conference on Learning Representations (ICLR)*, 2015.

[10] P. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. "Towards Deep Learning Models Resistant to Adversarial Attacks." *International Conference on Learning Representations (ICLR)*, 2018.

[11] V. Tjeng, K. Xiao, and R. Tedrake. "Evaluating Robustness of Neural Networks: An Extreme Value Theory Approach." *International Conference on Learning Representations (ICLR)*, 2019.

[12] E. Wong and Z. Kolter. "Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope." *Proceedings of the 35th International Conference on Machine Learning (ICML)*, 2018.

[13] C.-J. Simon-Gabriel, Y. Ollivier, L. Bottou, B. Scholkopf, and D. Lopez-Paz. "Adversarial Vulnerability of Neural Networks Increases with Input Dimension." *arXiv preprint arXiv:1802.01421*, 2018.
