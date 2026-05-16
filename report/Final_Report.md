# Final_Report
# Exploring Residual Connection Variants in Modern CNN Architectures

## CMPE 257, Machine Learning
### Spring 2026
### San José State University

- Emily Moberly
- Maximilian Garcia
- Mu Chen
- Napoleon Salazar

---

# 2. Abstract

The depth-induced degradation problem documents that increased depth alone does not guarantee higher accuracy in plain convolutional networks, motivating the residual learning paradigm of He et al. (2015). Four routing strategies were compared in this paper, namely a plain ablation with no shortcut, a Baseline residual with an identity skip, a Scaled variant with a learnable per-channel scalar on the residual branch, and a Gated variant with an input-dependent sigmoid gate on the residual branch. All four variants were trained on CIFAR-10 at depths {4, 32, 50} for 200 epochs with SGD and momentum, a StepLR schedule, and Batch Normalization as the only normalization-style regularizer. The Plain network degraded sharply with depth, with training error rising from 0.13% at depth 4 to 20.84% at depth 50, while the three residual variants reached essentially 100% training accuracy and above 93% validation accuracy at depths 32 and 50. Within the residual variants, Scaled was statistically indistinguishable from Baseline at every depth, and only the Gated variant outperformed Baseline by margins clearing single-seed noise, with a 0.46 percentage point advantage on validation and a 0.62 percentage point advantage on test at depth 50.
# 3. Introduction

**TODO:** Mu
### 3.1 Motivation and the Depth Paradox

- Reuse and expand the opening paragraph from the proposal's §1 ("The architectural evolution of convolutional neural networks…").
- Cite Goodfellow Ch. 5 (capacity / representation) and Bishop §1.1 (hypothesis space) when stating that deeper ≠ strictly better.
- Frame the **degradation problem** explicitly as distinct from overfitting: training error itself rises with depth in plain nets. Cite He et al. §1, Fig. 1.

### 3.2 Residual Learning as the Established Fix

- Reuse the proposal's §1 paragraph that introduces $H(x)$, $F(x) := H(x) - x$, $y = F(x) + x$.
- Add textbook grounding: cite **Goodfellow §8.2.5** for the optimization-difficulty framing and **Bishop §5.3** for backpropagation through the additive shortcut.
- One sentence on why the identity shortcut helps gradient flow: $\partial \mathcal{L} / \partial x = \partial \mathcal{L} / \partial y \cdot (1 + \partial F / \partial x)$ — the "+1" guarantees a direct gradient path.

### 3.3 Contributions of This Work

Bulleted list. Each bullet = one concrete deliverable:

1. A controlled, identical-protocol comparison of four residual routing strategies (Plain / Baseline / Scaled / Gated) at multiple depths.
2. An empirical reproduction of the degradation problem at our implementation's depth scale.
3. Layer-wise L2 gradient-norm diagnostics that visualize *how* each routing variant preserves (or fails to preserve) learning signal.
4. An open-source reference implementation released at `github.com/muchensthughs/CMPE257_ResNet`.

### 3.4 Roadmap

One short paragraph mapping sections to questions ("§4 states the questions, §5 reviews prior work, §6 describes the four solutions, §7–§8 detail experiments, §9 reports results, §10–§12 discuss conclusions, limitations, and next steps").

---

# 4. Problem Statement

Section 3 (Introduction) introduced residual learning as the established fix to the depth-induced degradation problem and identified the four routing variants studied here. The problem this paper takes up is whether the *form* of the residual shortcut matters in practice, and if so, by how much. That question decomposes into two parts. **RQ1 (Reproduction)** asks how well each residual variant solves the degradation problem compared to a plain (no-skip) network of the same depth, when trained under an identical protocol. **RQ2 (Comparison)** asks whether the more complex Scaled and Gated variants outperform the simpler Baseline residual, or whether the simple identity shortcut proves sufficient on this benchmark. RQ1 is a reproduction question whose positive answer is corroborative rather than novel; RQ2 is the comparative question, since the literature offers competing intuitions about whether richer shortcut-routing mechanisms repay their parameter overhead at the depths considered here.

### 4.1 Success Criteria

Two predictions are defined in advance to discipline the interpretation of results.

1. **Plain degrades with depth.** Validation accuracy of the Plain network is non-increasing as depth grows from 4 to 50, with a measurable collapse at the deepest setting.
2. **Residual variants recover.** Validation accuracy of the Baseline, Scaled, and Gated variants is non-decreasing across the same depth range, up to single-seed noise on the order of $\pm 0.2$–$0.3$ percentage points.

---


# 5. Related Work

**TODO:** Max
**Target length:** 2 – 3 pages

Reuse the proposal's §3 structure (it already has seven well-organized subsections). Expand each subsection by ~30–50%. Integrate textbook citations as listed below.

### 5.1 The Vanishing Gradient Problem
- Reuse proposal §3.1 wording (Hochreiter [4]).
- Add **Goodfellow §8.2.5** for the modern reframing of the problem.
- Add the formal inequality $\|\partial \mathcal{L}/\partial x_\text{early}\| \leq (\max \sigma)^L \cdot \|\partial \mathcal{L}/\partial x_\text{late}\|$.

### 5.2 Depth as a Representational Lever — VGG
- Reuse proposal §3.2.
- Add **Bishop §5.1** on universal approximation: a single hidden layer is enough *in theory* but exponentially wide; depth is the practical lever.

### 5.3 Batch Normalization
- Reuse proposal §3.3.
- Cite **Goodfellow §8.7.1**.
- **Critical sentence to add (used later in §6):** He et al. demonstrated that BN alone does **not** fix the degradation problem — the 34-layer plain net trained *with* BN still underperforms the 18-layer plain net. This rules out vanishing gradients as the sole cause and motivates a *structural* fix.

### 5.4 Highway Networks — The Conceptual Precursor
- Reuse proposal §3.4.
- Explicitly frame Highway as the parent of our **Gated** variant.
- Note (per He et al. §2): Highway gates *can* close, blocking the shortcut — a property our Gated variant inherits and that the Baseline avoids by design.

### 5.5 Attention as Adaptive Routing
- Reuse proposal §3.5.
- Add one sentence linking attention's "soft selection" mechanism to the Gated variant's sigmoid gate.

### 5.6 Transformers and Data-Dependent Routing
- Reuse proposal §3.6.
- Add a sentence noting transformers also use residual + LayerNorm — residual connectivity transcends CNNs.

### 5.7 MobileNetV2 — Residuals at the Efficiency Frontier
- Reuse proposal §3.7.
- One sentence on inverted residuals as evidence that the residual paradigm survives even under aggressive parameter budgets.

### 5.8 Bias, Variance, and Regularization Context (NEW subsection)
**TODO:** Maximilian Garcia
- Brief paragraph grounding the project in classical generalization theory. Cite **Bishop §3.2** and **Murphy §6.4** for the decomposition $\mathbb{E}[(y - \hat f)^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$.
- Use this to justify our **deliberate omission of dropout**: BatchNorm + data augmentation already control variance; adding dropout on top would conflate the routing-variable ablation we are trying to isolate. (This is the same rationale He et al. give in §3.4 of the original ResNet paper.)

---

# 6. Solution: Four Residual Routing Variants

**TODO:** Mu

### 6.1 Shared Block Structure (Reused Across All Four Variants)

Every block in every variant, at every depth, is constructed from the same convolutional pathway $F(x)$. This pathway is defined once in `models/blocks/base.py` and inherited by all four variant subclasses, so that any observed difference between Plain, Baseline, Scaled, and Gated is attributable to the shortcut-routing function alone.

The shared pathway consists of two $3 \times 3$ convolutional layers, each with 64 output channels, stride 1, and padding 1, so the spatial dimensions of the input are preserved through the block. Each convolution is followed by Batch Normalization, a ReLU activation is applied between the two BN-Conv stages, and a final ReLU is applied to the output of the block after the shortcut has been combined with $F(x)$. The convolution layers carry no bias term, since biases are omitted from the convolutional layers as they are naturally absorbed by the subsequent BatchNorm parameters. Concretely :

$$
F(x) \;=\; \text{BN}\!\big(\text{Conv}_{3\times3,\,64}\big(\text{ReLU}\!\big(\text{BN}(\text{Conv}_{3\times3,\,64}(x))\big)\big)\big), \qquad y \;=\; \text{ReLU}\!\big(s(F(x),\,x)\big),
$$

where $s(\cdot,\cdot)$ is the variant-specific shortcut function defined in Section 6.2 (Plain Network) through Section 6.5 (Gated Residual). Figure 1 illustrates the shared pathway together with the four variant routings.

![Shared block structure across all four variants](Shared_Conceptual_Block.png)

**Shared block structure** (`Shared_Conceptual_Block.png`). The convolutional pathway $F(x)$ is identical across Plain, Baseline, Scaled, and Gated; only the shortcut routing $s(\cdot,\cdot)$ differs.

No dropout is applied at any point inside the block, neither between the two convolutions nor after the shortcut combination. The only normalization-style regularizers active in this architecture are Batch Normalization on each $F(x)$ pathway (Goodfellow §8.7.1) and the random-crop and horizontal-flip data augmentation described in Section 8 (Experiment Set-up and Data Set Details).

### 6.2 Variant 1 — Plain Network (Ablation)

**TODO:** Mu
- Reuse proposal §4 verbatim.
- Equation: $y = F(x)$
- One paragraph on **expected** behavior (degradation), one paragraph on what observing degradation **proves** vs. what it would mean if absent.

### 6.3 Variant 2 — Baseline Residual

**TODO:** Mu
- Reuse proposal §5.1 verbatim.
- Equation: $y = x + F(x)$
- Explicitly link to He et al. Eqn. (1).
- Note: identity shortcut, no extra parameters, no extra FLOPs — emphasize this as the "control" that any more complex variant must beat to justify its overhead.

### 6.4 Variant 3 — Scaled Residual

**TODO:** Max
- Reuse proposal §5.2.
- Equation: $y = x + \alpha \cdot F(x)$, $\alpha$ learnable scalar, initialized to $1.0$.
- **Expand** with: this variant anticipates the **ReZero** (Bachlechner et al., 2020) and **LayerScale** (Touvron et al., 2021) mechanisms used in modern ViTs and ConvNeXt. (Add these references to §13.)
- Diagnostic angle: record the learned $\alpha$ value per block per epoch — this is a free interpretability signal worth analyzing in Results.

### 6.5 Variant 4 — Gated Residual

The Gated Residual variant replaces the static identity shortcut of the Baseline
with a learned, input-dependent gate that modulates how much of the convolutional
pathway $F(x)$ is added back to the shortcut. Each Gated block computes

$$
y \;=\; x \;+\; g(x) \,\odot\, F(x), \qquad g(x) \;=\; \sigma\!\big(W x + b\big),
$$

where $\odot$ is element-wise multiplication, $\sigma$ is the logistic sigmoid,
and $W \in \mathbb{R}^{C \times C}$ is implemented as a $1\!\times\!1$ convolution
applied to the block input (so the gate has shape $[B, C, H, W]$, matching $F(x)$
per channel and per spatial location). Specifically, We initialize $W = 0$ and $b = 3$,
so that $g(x) \approx \sigma(3) \approx 0.95$ at the start of training — i.e.,
so the block opens in a Highway-style configuration
that is numerically close to the Baseline residual and lets the gate *learn* to
close. The implementation lives in
`models/blocks/gated.py`.

![Gated Residual](gated_residual.png)

#### 6.5.1 Gradient signal through the gate

In the original Highway Networks (Srivastava et al., 2015), the shortcut path itself is multiplied by a learned carry gate. However, gating the skip connection introduces a critical risk. If the network drives the gate toward zero, the residual connection is disrupted. This effectively reverts the model back into a plain sequential network, reviving the gradient problems previously described. He et al. (2015) explicitly flagged this vulnerability, demonstrating that gated shortcuts underperform plain identity skips because they cannot guarantee an unobstructed gradient path to earlier layers.To resolve this, our Gated variant avoids this failure mode by shifting where the gate is applied. Instead of gating the shortcut, our architecture applies the gate exclusively to the residual branch ($F(x)$), leaving the identity shortcut ($x$) completely unhindered. Because the shortcut remains purely additive, the gradient flow is mathematically preserved. Differentiating this output with respect to the block input yields
$$
\frac{\partial y}{\partial x} \;=\; I \;+\; g(x)\,\frac{\partial F}{\partial x}
\;+\; F(x)\,\frac{\partial g}{\partial x},
$$
As shown, the identity matrix $I$ ensures that even if the gate $g(x)$ saturates and attenuates the residual updates, a clean gradient path always remains open to propagate backward to earlier layers.


### 6.6 Why These Four Variants Form a Clean Ablation

The empirical comparisons reported in Section 9 (Results) rest on the methodological commitment that the four variants studied here differ along exactly one axis. This subsection makes that axis explicit. Across Plain, Baseline, Scaled, and Gated, the convolutional pathway $F(x)$, the block depth, the filter width, the BatchNorm placement, the optimizer, the learning-rate schedule, the random seed, and the augmentation pipeline are all held identical, as detailed in Section 6.1 (Shared Block Structure) and Section 8 (Experiment Set-up and Data Set Details). The only quantity that varies is how $F(x)$ is recombined with the block input $x$ on its way to the block output $y$.

The four variants form a strict capability ordering of that recombination function.

- **Plain.** $y = F(x)$. No shortcut; the residual branch is the only output.
- **Baseline.** $y = x + F(x)$. Identity shortcut, full-strength residual, no learnable routing parameters.
- **Scaled.** $y = x + \alpha \odot F(x)$, $\alpha \in \mathbb{R}^{C}$. Identity shortcut plus a learnable per-channel scalar on the residual.
- **Gated.** $y = x + g(x) \odot F(x)$, $g(x) = \sigma(Wx + b) \in (0,1)^{C \times H \times W}$. Identity shortcut plus a learnable input-dependent per-channel, per-spatial gate on the residual.

Each residual variant nests the previous one inside a richer parameter family. Scaled reproduces Baseline exactly when $\alpha = \mathbf{1}$, and Gated reproduces a constant-$\alpha$ Scaled block when its weight matrix $W$ is zero. The initial values are chosen so this nesting also holds numerically at step 0, with $\alpha$ initialized to $\mathbf{1}$ in Scaled and the gate bias $b$ initialized to $3$ in Gated to give $g(x) \approx 0.95$ on the first batch, so all three residual variants begin training with outputs within roughly five percent of Baseline. Any later divergence therefore reflects what the additional routing parameters have learned, not differences in starting state.

Because $F(x)$ is held fixed across the four variants, any observed difference in validation accuracy or gradient flow can be attributed to the shortcut-routing function alone. The Plain ablation isolates the contribution of having any shortcut at all, and the Baseline → Scaled → Gated progression isolates the contribution of giving that shortcut increasingly expressive learnable modulation of the residual branch. The remainder of the paper exploits this attribution.

---

# 7. List of Experiments

This section enumerates the training runs whose results are reported in Section 9 (Results). The experimental grid spans the four routing variants defined in Section 6 (Solution) across three depths, all trained under the protocol specified in Section 8 (Experiment Set-up and Data Set Details).

### 7.1 Experimental Grid

Each row corresponds to one training run. The 12-cell grid is the basis of every comparison reported in Section 9 (Results).

| Run ID | Variant | Depth (layers) | Seed | Epochs | Config file |
|--------|---------|----------------|------|--------|-------------|
| `d4_no_residual` | Plain | 4 | 42 | 200 | `configs/plain_4.yaml` |
| `d4_baseline` | Baseline | 4 | 42 | 200 | `configs/baseline_4.yaml` |
| `d4_scaled` | Scaled | 4 | 42 | 200 | `configs/scaled_4.yaml` |
| `d4_gated` | Gated | 4 | 42 | 200 | `configs/gated_4.yaml` |
| `d32_no_residual` | Plain | 32 | 42 | 200 | `configs/plain_32.yaml` |
| `d32_baseline` | Baseline | 32 | 42 | 200 | `configs/baseline_32.yaml` |
| `d32_scaled` | Scaled | 32 | 42 | 200 | `configs/scaled_32.yaml` |
| `d32_gated` | Gated | 32 | 42 | 200 | `configs/gated_32.yaml` |
| `d50_no_residual` | Plain | 50 | 42 | 200 | `configs/plain_50.yaml` |
| `d50_baseline` | Baseline | 50 | 42 | 200 | `configs/baseline_50.yaml` |
| `d50_scaled` | Scaled | 50 | 42 | 200 | `configs/scaled_50.yaml` |
| `d50_gated` | Gated | 50 | 42 | 200 | `configs/gated_50.yaml` |

All runs use SGD with momentum 0.9, weight decay $10^{-4}$, initial learning rate 0.1, StepLR decay at epochs 100 and 150 with $\gamma = 0.1$, batch size 128, and a random-crop with 4-pixel padding plus horizontal-flip augmentation pipeline. The complete protocol is given in Section 8.3 (Training Protocol).

### 7.2 What Each Run Tests

- **Depth-4 runs.** Sanity floor; all four variants are expected to converge to similar accuracy, and the degradation problem should not yet be visible at this depth.
- **Depth-32 runs.** The depth at which the resolution of the degradation problem is expected to become noticeable, with the residual variants pulling measurably above Plain.
- **Depth-50 runs.** The deepest setting in this study, and the regime where any validation-accuracy advantage of the richer Scaled and Gated variants over the simple Baseline residual is expected to become measurable. This regime is closest to the CIFAR-10 depths of He et al. (2015).

---

# 8. Experiment Set-up and Data Set Details

### 8.1 Dataset: CIFAR-10

**TODO:** Mu

- Brief paragraph: 60k 32×32 color images, 10 classes, 50k train / 10k test (cite Krizhevsky 2009 [6]).
- Note we follow He et al. §4.2's CIFAR-10 protocol (4-pixel pad + 32×32 random crop + horizontal flip + per-channel normalization).
- Confirm: no test-time augmentation; single 32×32 center evaluation, matching He et al.

### 8.2 Macro-Architecture

**TODO:** Mu

- Reuse proposal §6.2 verbatim.
- Pipeline: Input Conv → N residual blocks (depth-dependent) → Global Average Pool → Linear(10).
- Filter width fixed at **64** across all blocks (per proposal — this is a deliberate "flat" design that isolates the routing variable, *not* the channel-doubling design of He et al.).

### 8.3 Training Protocol

**TODO:** Mu

Reuse proposal §6.2.4 (locked optimizer settings):

- Optimizer: SGD
- Initial LR: 0.1
- Momentum: 0.9
- Weight decay: $1 \times 10^{-4}$
- LR schedule: StepLR — 0.1 → 0.01 at epoch 100, → 0.001 at epoch 150
- Epochs: 200
- Batch size: (fill in from `configs/base.yaml`)
- **No dropout** (reiterate; cite §6.1 rationale)

Add a one-paragraph theoretical note citing **Goodfellow §8.3 (momentum), §8.5 (adaptive methods — explain why SGD+momentum was chosen over Adam), §8.3.1 (LR schedules)**. This is the section where the prof-flagged "Optimization" content earns credit.

### 8.4 Measurement Protocol

**TODO:** Mu

Reuse proposal §6.2.5. Bullets:

- Training loss + accuracy per epoch
- Validation loss + accuracy per epoch
- L2 norm of gradients at each Conv2d layer per epoch (mean across batch)
- Per-layer norm logged separately (required for the "Gradient Norm vs. Layer Depth" plots)

### 8.5 Divergence from the Original Proposal

The measurement protocol described in Section 8.4 (Measurement Protocol) was applied over a depth ladder that differs from the one originally proposed, and the change is documented here for transparency. The original proposal specified depths {4, 8}, while the implementation reported in Section 7 (List of Experiments) instead uses depths {4, 32, 50}. Depth 8 was found insufficient to demonstrate the degradation problem reliably, so it was replaced by depths 32 and 50, with the 4-layer floor preserved as a sanity check. The reason that replacement was necessary is methodological rather than logistical.

At depth 8, with the flat 64-filter block defined in Section 6.1 (Shared Block Structure) and Batch Normalization on every convolution, the gap between Plain and the residual variants is expected to lie well inside single-seed noise, which would prevent any of the success criteria in Section 4.1 (Success Criteria) from being adjudicated. Depths 32 and 50 place the experiment in the regime where the He et al. (2015) degradation phenomenon is reliably reproducible on CIFAR-10, and the residual variants pull above Plain by margins that exceed single-seed noise. With the final depth grid fixed, the implementation details that govern reproducibility follow in Section 8.6 (Implementation Details and Reproducibility).

### 8.6 Implementation Details and Reproducibility

**TODO:** Mu Chen
- Framework: PyTorch (specify version from `requirements.txt` or `venv`).
- Hardware: (fill in — GPU model, number of GPUs).
- Random seeds: fixed for all runs (specify value).
- Wall-clock training time per run (optional but valuable for the Limitations discussion).

### 8.7 Code and Repositories

**TODO:** Mu
- Primary repo: `https://github.com/muchensthughs/CMPE257_ResNet`
- Original prototyping notebook (Colab): `https://colab.research.google.com/drive/1nMPHSxqM1fVwguqIo2TaDhbgI0GPdYUU`
- Open-source code utilized:
  - PyTorch (BSD-3) — model, optimizer, dataloaders
  - torchvision (BSD-3) — CIFAR-10 dataset and transforms
  - (List any other dependencies from `requirements.txt`)
- One-sentence statement that all training scripts, configs, and result CSVs/plots used in this report are committed to the repo for reproducibility.

---

# 9. Results

Following the experimental protocol defined in Section 8 (Experiment Set-up and Data Set Details), the outcomes of the twelve runs in the experimental grid are reported here and evaluated against the two research questions of Section 4 (Problem Statement). Section 9.1 (Master Comparison Table) presents the aggregate validation and test-set numbers across the full grid. The subsequent subsections decompose those numbers per variant in Sections 9.2 through 9.5, examine the underlying gradient flow in Section 9.6 (Gradient Flow Analysis), compare all four variants directly at depth 50 in Section 9.7 (Cross-Variant Comparison), and report the parameter count and wall-clock cost of each variant in Section 9.8 (Computational Overhead).

### 9.1 Master Comparison Table

The headline accuracies for all twelve runs in the experimental grid are reported below.

| Variant  | Depth 4 (val / train) | Depth 32 (val / train) | Depth 50 (val / train) |
|----------|-----------------------|------------------------|------------------------|
| Plain    | 90.44 / 99.87         | 90.42 / 99.77          | 76.54 / 79.16          |
| Baseline | 90.38 / 99.86         | 93.44 / 100.00         | 93.16 / 100.00         |
| Scaled   | 90.20 / 99.88         | 93.76 / 99.99          | 93.30 / 100.00         |
| Gated    | 90.44 / 99.90         | 93.34 / 99.99          | 93.62 / 99.98          |
The chart above represents the **best top-1 validation accuracy and final-epoch training accuracy** (percentages) for the twelve runs of the experimental grid. Each validation value is the maximum over 200 epochs, and each training value is the accuracy at epoch 200. The headline finding is the depth-50 Plain row validation accuracy collapses from 90.42% at depth 32 to 76.54% at depth 50, equivalent to a rise in validation error from 9.58% to 23.46%, and training accuracy collapses from 99.77% to 79.16% in lockstep. The simultaneous failure on both training and validation confirms that the degradation problem is reproduced as a training failure . In contrast, all three residual variants instead climb above 93% validation accuracy at depths 32 and 50, equivalent to validation errors between 6.24% and 6.84%, while holding their training accuracy at essentially 100%, which is the signature of the residual fix described by He et al. (2015).

Within the three residual variants, the central question is whether the more complex Scaled and Gated mechanisms improve on the simple identity skip of Baseline. At depths 4 and 32 every gap between Baseline, Scaled, and Gated is at most 0.32 percentage points of validation accuracy and falls inside the single-seed noise band of roughly ±0.2 to 0.3 percentage points, so none of those cells supports a defensible variant ranking. At depth 50, Gated reaches 93.62% validation accuracy against Baseline's 93.16%, a gap of 0.46 percentage points that is the only within-residual margin in the grid to clear that noise band. Scaled at depth 50 lands at 93.30%, only 0.14 percentage points above Baseline and well within noise. The depth-50 Gated cell is therefore the single point at which a more complex shortcut routing meaningfully outperforms the simple identity skip.

The training and validation error trends are visualized in the four charts that follow. Each metric is shown both across all four variants and across the residual variants only.

![Training error at epoch 200 by variant and depth](training_error_all_variants.png)
**Training error at epoch 200 by variant and depth.** Plain's training error rises from 0.13% at depth 4 to 20.84% at depth 50, while the three residual variants remain at or below 0.23% across all three depths.

![Training error at epoch 200 by depth, residual variants only](training_residual_variants.png)
**Training error at epoch 200 by depth, residual variants only.** With Plain removed and the vertical axis tightened, Baseline, Scaled, and Gated all converge to essentially zero training error at depths 32 and 50, confirming none of them suffers any residual optimization failure.

![Best-epoch validation error by variant and depth](validation_error_all_variants.png)
**Best-epoch validation error by variant and depth.** Plain rises from 9.56% at depth 4 to 23.46% at depth 50, while the three residual variants drop from around 9.6% at depth 4 to between 6.24% and 6.84% at depths 32 and 50.

![Best-epoch validation error by depth, residual variants only](validation_error_residual_variants.png)
**Best-epoch validation error by depth, residual variants only.** With the axis zoomed to the 6.24% to 9.80% range, the small within-residual differences become readable. Scaled is lowest at depth 32 (6.24%), and Gated is lowest at depth 50 (6.38%).

The validation findings reported above are corroborated against the held-out 10,000-image CIFAR-10 test set, using the best-validation checkpoint of each run.

| Variant  | Depth 4 | Depth 32 | Depth 50 |
|----------|---------|----------|----------|
| Plain    | 89.37   | 89.52    | 75.16    |
| Baseline | 89.59   | 93.02    | 92.20    |
| Scaled   | 89.64   | 93.06    | 92.44    |
| Gated    | 89.28   | 92.62    | 92.82    |
**Test-set top-1 accuracy** (percentages) on the held-out 10,000-image CIFAR-10 test set, evaluated using the best-validation checkpoint of each run.

The test-set numbers confirm the validation picture. Plain at depth 50 reaches only 75.16% on test, a drop of more than 14 percentage points from its depth-32 test accuracy of 89.52%, and the three residual variants instead remain above 92% test accuracy at depths 32 and 50. At depth 50 Gated reaches 92.82% on test against Baseline's 92.20%, a margin of 0.62 percentage points that is slightly larger than the corresponding validation margin of 0.46 percentage points and that confirms the depth-50 Gated cell as the only within-residual gap to clear single-seed noise on both metrics.

The test accuracy of each variant at each depth is visualized below.

![Test accuracy on CIFAR-10 by variant and depth, all variants](testing_accuracy_all_variants.png)
**Test accuracy (Acc@1) on CIFAR-10 by variant and depth.** Plain's depth-50 bar drops to 75.16%, while the three residual variants remain above 92% at depths 32 and 50. Gated's depth-50 bar (92.82%) edges Baseline (92.20%) by 0.62 percentage points, the only within-residual cell to clear single-seed noise.
### 9.2 Reproducing the Degradation Problem (Plain Network)

**TODO:** Mu
- Plot: validation accuracy vs. epoch for `d4_no_residual`, `d32_no_residual`, `d50_no_residual` on one axis.
- Plot: **training error** vs. epoch for the same — this is the critical figure for proving degradation (not overfitting).
- Reuse proposal §4.1 / §4.2 wording for the expected vs. observed framing.
- Explicitly state whether the prediction held.

### 9.3 Baseline Residual

**TODO:** Mu
- Same set of plots as §9.2 but for the Baseline variant.
- One paragraph contrasting against Plain — is the degradation gone? By how much does d50 outperform d50 Plain?

### 9.4 Scaled Residual

**TODO:** Maximilian Garcia
- Validation/training curves at all three depths.
- **Bonus plot:** learned $\alpha$ value per residual block, per depth — this is the variant's free interpretability output. Does $\alpha$ stay near 1.0? Does it drift down (suggesting full-strength residuals are too aggressive)? Does it vary across block depth?
- Discussion: did Scaled meaningfully outperform Baseline? If yes, at which depth? If no, what does that say about LayerScale-style mechanisms on small datasets?

### 9.5 Gated Residual

This subsection reports the Gated variant's training and validation behavior at
all three depths, the dynamics of the learned gate $g(x)$ over training, and a
head-to-head comparison against the Baseline residual.

**Baseline network vs Gated network vs Plain Network.** Top 100 epochs

![Training](training_validation_errors.png)

**Baseline network vs Gated network.** Top 100 epochs validation error

![Validation](validation_100_epochs.png)

**Headline numbers.** Best top-1 validation accuracy from `runs/d{4,32,50}_gated/metrics_epoch.csv`:

| Depth | Gated (best val) | Baseline (best val) | Plain (best val) | Δ(Gated − Baseline) |
|-------|------------------|---------------------|------------------|---------------------|
| 4     | 90.44 % (ep 112) | 90.38 % (ep 108)    | 90.44 % (ep 118) | **+0.06**           |
| 32    | 93.34 % (ep 106) | 93.44 % (ep 180)    | 90.42 % (ep 125) | **−0.10**           |
| 50    | 93.62 % (ep 143) | 93.16 % (ep 119)    | 76.54 % (ep 197) | **+0.46**           |

Based on these metrics, the following observations can be made:

1. **Gated solves the degradation problem:** At depth 50, the Plain network
   collapses to 76.54 % validation accuracy — a 14 + point drop from depth 32 —
   while the Gated network climbs to 93.62 %, its best result across all three
   depths. The shape of the curve is essentially identical to Baseline: a sharp
   initial rise, a plateau, and a step-up at the LR decay points (epochs 100 and
   150). Therefore, the protected identity skip is sufficient to keep
   gradient flow healthy.
2. **The cost is real but small:** Each Gated block adds $C^2 + C = 4{,}160$
   parameters and one $1\!\times\!1$ convolution per forward pass. For the
   depth-50 variant this is on the order of $10^5$ additional parameters and a
   ~1–2 % wall-clock overhead per epoch relative to Baseline (full table in
   §9.8).
3. **Training curves:** Across all three depths the Gated training accuracy
reaches the same near-100 % asymptote as Baseline within the first 50 epochs
and then sits flat; the validation curve hits its plateau shortly after the
first LR decay (epoch 100) and improves only marginally after the second
decay (epoch 150). The Gated best-validation epoch is **d4 at epoch 112**,
**d32 at epoch 106**, and **d50 at epoch 143** — all three sit between the
two LR decays (epochs 100 and 150), the post-first-decay window where
Baseline also peaks.
4. **Validation Curves:** Gated equals or beats Baseline
on best-epoch validation accuracy at every depth: **+0.06 pp at d4**,
**−0.10 pp at d32**, and **+0.46 pp at d50**. The d4 and d32 gaps fall
inside the ±0.2–0.3 pp noise band and should be
read as ties; the d50 +0.46 pp is the only Gated-vs-Baseline gap that
clears that noise floor and is therefore the cell where Gated's advantage is
most defensible. Averaged across the three depths, Gated improves best-epoch
validation accuracy by +0.14 pp over Baseline (92.47 % vs. 92.33 %) for an
additional ~0.5 % parameter count and one $1\!\times\!1$ convolution per
forward pass (§9.8). This is a small but consistent edge — not strong enough
on to recommend Gated over Baseline as a default, but strong
enough that the depth-50 result deserves the multi-seed follow-up evaluation.

### 9.6 Gradient Flow Analysis

**TODO:** Mu
- Plot: L2 gradient norm vs. layer index for all four variants at depth 50, at epochs 1, 50, 150.
- Expectation (per proposal §4.3): Plain shows steep decay toward early layers; residual variants stay flatter.
- Cite **Goodfellow §8.2.5** when interpreting the curves.

### 9.7 Cross-Variant Comparison

The per-variant analyses in the preceding subsections are consolidated here into a direct comparison at depth 50, where the four variants separate most clearly. The two research questions of Section 4 (Problem Statement) are addressed in turn against the depth-50 validation and test numbers.

#### Does each residual variant solve the degradation problem?

The margin over Plain at depth 50 is reported in the table below for both validation and test.

| Variant  | Solves degradation? | Margin over Plain at depth 50 (val / test) |
|----------|---------------------|--------------------------------------------|
| Baseline | Yes                 | +16.62 pp / +17.04 pp                      |
| Scaled   | Yes                 | +16.76 pp / +17.28 pp                      |
| Gated    | Yes                 | +17.08 pp / +17.66 pp                      |

All three residual variants clear the Plain network by more than 16 percentage points of validation accuracy and more than 17 percentage points of test accuracy at depth 50. Every margin is more than fifty times the single-seed noise band of ±0.2 to 0.3 percentage points, so the answer to RQ1 is unambiguous on this implementation.

#### Do the more complex Scaled and Gated variants outperform Baseline?

Gated provided a small but consistent improvement over Baseline on both validation and test at depth 50. Validation accuracy reached 93.62% against Baseline's 93.16%, and test accuracy reached 92.82% against Baseline's 92.20%, gaps of 0.46 and 0.62 percentage points that both clear the single-seed noise band of ±0.2 to 0.3 percentage points. Scaled's results were marginal against Baseline and are examined in Section 9.4 (Scaled Residual). Gated is therefore the only routing mechanism in the experimental grid that meaningfully outperforms the simple identity shortcut at the depths considered.

### 9.8 Computational Overhead

**TODO:** Maximilian
- Small table: parameter count and wall-clock training time per variant per depth.
- Argue (or refute) the cost/benefit case for Scaled and Gated.

---

# 10. Conclusions

The depth-induced degradation problem was reproduced in the Plain network, with training error rising from 0.13% at depth 4 to 20.84% at depth 50 and validation accuracy collapsing from 90.44% to 76.54% in parallel. The simultaneous failure on both training and validation matches the optimization-failure framing of He et al. (2015) and rules out overfitting as the cause.

In contrast, all three residual variants eliminated the degradation problem at every depth tested, with each reaching essentially 100% training accuracy and clearing the depth-50 Plain network by more than sixteen percentage points of validation accuracy. The fix is attributable to the additive identity path through the block, which preserves a direct gradient route to earlier layers (Goodfellow §8.2.5).

Among the residual variants, the simple identity shortcut of the Baseline variant remained competitive with the more elaborate Scaled and Gated alternatives. Scaled tied Baseline at every depth within single-seed noise on both validation and test. Gated produced a small but consistent improvement only at depth 50, with margins of 0.46 percentage points on validation and 0.62 percentage points on test that both clear the noise band.

In summary, the reproduction question of whether each residual variant solves the degradation problem is answered in the affirmative for all three. For the comparison question of whether the more complex Scaled and Gated variants outperform the simpler Baseline residual, Scaled was statistically indistinguishable from Baseline at every depth, and Gated outperformed Baseline by a small but consistent margin at depth 50.

---

# 11. Limitations

The conclusions reported in Section 10 (Conclusions) rest on a deliberately narrow experimental scope. The limitations of that scope are listed below, both to qualify the strength of the present claims and to delimit which questions are left open for future work.

- **Single dataset.** The study used only CIFAR-10. The variant ranking has not been validated on CIFAR-100, ImageNet, or any non-vision modality.
- **Single seed per run.** Each run was trained with a single random seed (42). Without multiple seeds, accuracy gaps smaller than the ±0.2 to 0.3 percentage point single-seed noise band cannot be distinguished from seed variance, which limits the strength of within-residual claims.
- **No dropout ablation.** Dropout was deliberately omitted from all runs to keep the routing-variable ablation clean (Section 6.1 (Shared Block Structure)). Whether adding dropout would change the within-residual ranking is not measured here.
- **No state-of-the-art training tricks.** Cutout, mixup, label smoothing, stochastic depth, and test-time augmentation are all absent. The accuracy numbers reported here should not be compared against modern CIFAR-10 leaderboards.

Some of these limitations point toward a concrete follow-up that would tighten or extend the present findings. Those follow-ups are catalogued in Section 12 (Future Work).

---

# 12. Future Work

The four-variant ablation in this paper is limited in scope. A single
dataset, a single seed per cell, a fixed flat-64 macro-architecture, and three
depths are not enough to paint the full picture. Each of the following directions removes one of those restrictions and
turns these limitations into prospective follow-up studies.

- **Multi-seed runs with confidence intervals.** The single most defensible
  next step. Rerunning all experiments with at least five seeds and reporting mean $\pm$ standard
  error would let us state whether the depth-50 Gated advantage is real or an
  artifact of the single seed used here.

- **Cross-dataset validation.** Repeat the full ablation on CIFAR-100 and
  Tiny-ImageNet, holding the architecture and training protocol fixed. CIFAR-100
  shares CIFAR-10's image distribution but is harder per-class; Tiny-ImageNet
  introduces 64$\times$64 inputs and 200 classes. If the variant ranking is
  consistent across all three datasets, then there's a stronger generalization
  claim.

- **Extension to extreme depths (110 / 1202 layers):** Our depth ladder (4 / 32 / 50) sits below the original Resnet 2015 depth regime and may be
  systematically *under-resolving* the Scaled and Gated advantages. Repeating
  at depths 110 and 1202 would either replicate He et al.'s headline finding
  (with our four-variant lens) or
  suggest that BatchNorm plus the flat-64 macro-architecture has its own
  upper-depth ceiling distinct from the residual mechanism. Either outcome is
  publishable.

---

# 13. References

**TODO:** Mu + all members (verify their cited works)
**Target length:** 1 – 1.5 pages

Start from the proposal's bibliography (Refs [1]–[10]). **Additions required:**

- He et al. 2015 (already present as [3]) — verify CVPR 2016 citation is preserved.
- Krizhevsky 2009 (already present as [6]) — CIFAR-10 tech report.
- Hochreiter 1998 (already present as [4]).
- Goodfellow, Bengio, Courville — *Deep Learning*, MIT Press 2016. Cite specifically Ch. 5 (capacity), Ch. 6 (MLPs/backprop), §7.12 (dropout), §8.2.5 (vanishing/exploding gradients), §8.3 (momentum), §8.5 (adaptive methods), §8.7.1 (BatchNorm).
- Bishop — *Pattern Recognition and Machine Learning*, Springer 2006. Cite §1.1 (hypothesis space), §3.2 (bias-variance), §5.1 (universal approximation), §5.3 (backprop).
- Murphy — *Machine Learning: A Probabilistic Perspective*, MIT Press 2012. Cite §6.4 (bias-variance), Ch. 11 (clustering — peripheral, may drop).
- Bachlechner et al. 2020 — *ReZero is All You Need* (for Scaled variant lineage).
- Touvron et al. 2021 — *LayerScale / CaiT* (for Scaled variant lineage).
- Huang et al. 2016 — *Stochastic Depth* (only if used in Future Work).
- Loshchilov & Hutter 2017 — *Cosine annealing / SGDR* (optional, if mentioned in §8.3).

Use a consistent style (`alpha` or numeric — match the proposal's existing style).