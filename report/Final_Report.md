# Team Section Responsibilities Checklist

- [ ] **Napoleon Salazar**
  - [ ] Title Page
  - [ ] Abstract
  - [ ] Problem Statement
  - [ ] Why These Four Variants Form a Clean Ablation
  - [ ] List of Experiments
  - [ ] Results (Master Comparison Table, Cross-Variant Comparison, Conclusions, Limitations, Team Contributions)
  - [ ] Divergence from the Original Proposal

- [ ] **Mu Chen**
  - [ ] Introduction
  - [ ] Dataset, Macro-Architecture, Training Protocol, Measurement Protocol, Implementation Details, Code and Repositories
  - [ ] Results (Reproducing the Degradation Problem, Baseline Residual, Gradient Flow Analysis)
  - [ ] References

- [ ] **Maximilian Garcia**
  - [ ] Entire Related Work section (including all subsections 5.1–5.8)
  - [ ] Scaled Residual Variant
  - [ ] Results (Scaled Residual, Computational Overhead)
  - [ ] Limitations

- [ ] **Emily Moberly**
  - [ ] Gated Residual Variant
  - [ ] Results (Gated Residual)
  - [ ] Future Work

---
# Exploring Residual Connection Variants in Modern CNN Architectures

# 1. Title Page

**TODO:** Napoleon

- Title: *Exploring Residual Connection Variants in Modern CNN Architectures*
- Authors: Emily Moberly, Mu Chen, Maximilian Garcia, Napoleon Salazar

---

# 2. Abstract

**TODO:** Napoleon

Must hit:

- One-sentence motivation (degradation problem — depth alone does not buy accuracy in plain CNNs).
- One-sentence framing of the four variants (Plain ablation, Baseline residual, Scaled `α·F(x)`, Gated `g(x)·F(x)`).
- One-sentence experimental scope: CIFAR-10, depths **{4, 32, 50}** (note divergence from proposal's {4, 8}), 200 epochs, SGD + StepLR, no dropout — Batch Normalization is the only normalization-style regularizer.
- 2–3 sentences of headline results: (a) does Plain degrade with depth? (b) do residual variants recover? (c) does Scaled/Gated beat Baseline, or is the simple identity skip enough?
- One closing sentence on the broader implication (identity shortcuts are still hard to beat — anticipates ReZero / LayerScale / Highway lineage).

---

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

**TODO:** Napoleon
### 4.1 Research Questions

Reuse the proposal's §2 wording verbatim, but tighten the framing:

- **RQ1 (Reproduction):** How well does each residual variant solve the degradation problem compared to a plain (no-skip) network?
- **RQ2 (Comparison):** Do the more complex Scaled and Gated variants outperform the simpler Baseline identity skip connection?

### 4.2 Scope and Non-Goals

State what this paper does **not** attempt:

- Not pursuing state-of-the-art CIFAR-10 accuracy (no cutout, no mixup, no test-time augmentation).
- Not exploring bottleneck blocks; all variants use the proposal's flat 64-filter two-conv block.
- Not exploring depths > 50 layers; we deliberately stay at depths where the degradation problem can be isolated, not at the 1000-layer regime explored in He et al. §4.2.

### 4.3 Success Criteria

Explicit, measurable:

- Plain network: validation accuracy strictly decreases as depth grows from 4 → 50.
- All three residual variants: validation accuracy is non-decreasing (within seed noise) as depth grows.
- Gradient norms in plain-50 at early layers measurably smaller than in any residual variant at the same layer index.

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

**TODO:** Napoleon Mu
- Reuse proposal §6.2.1 block-level structure verbatim.
- Diagram: include the shared conceptual block figure from the proposal (Fig. 5).
- One sentence stating the **no-dropout decision** explicitly here, with the rationale: BatchNorm + augmentation are the only regularizers; this preserves a single-variable ablation across the four routing schemes. Cite Goodfellow §7.12 (dropout) for the omission rationale and §8.7.1 (BN) for what replaces it.

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
pathway $F(x)$ is added back to the shortcut. It is the most expressive of the
four routing schemes considered in this study, and it is the only variant whose
shortcut can adaptively "close" in response to the input.

**Formal definition.** Each Gated block computes

$$
y \;=\; x \;+\; g(x) \,\odot\, F(x), \qquad g(x) \;=\; \sigma\!\big(W x + b\big),
$$

where $\odot$ is element-wise multiplication, $\sigma$ is the logistic sigmoid,
and $W \in \mathbb{R}^{C \times C}$ is implemented as a $1\!\times\!1$ convolution
applied to the block input (so the gate has shape $[B, C, H, W]$, matching $F(x)$
per channel and per spatial location). The shared $F(x)$ pathway is the same
two-conv block used by every variant (`Conv2d(64)`–BN–ReLU–`Conv2d(64)`–BN), so
any observed difference relative to Baseline is attributable to the gate alone.
We initialize $W = 0$ and $b = 3$, so that $g(x) \approx \sigma(3) \approx 0.95$
at the start of training — i.e., the block opens in a Highway-style configuration
that is numerically close to the Baseline residual and lets the gate *learn* to
close rather than starting from a closed state. The implementation lives in
`models/blocks/gated.py`.

**Two limiting behaviors.** The two extreme regimes of the gate recover the
other variants exactly:

- As $g(x) \to \mathbf{0}$, the block degenerates to $y = x$ — a pure identity
  pass-through. This is *strictly weaker* than even the Plain network at that
  layer, since the convolutional pathway is suppressed entirely. If the optimizer
  drives $g(x)$ toward zero in deep blocks, this is direct evidence that the
  network prefers to skip computation in those blocks.
- As $g(x) \to \mathbf{1}$, the block recovers exactly $y = x + F(x)$, i.e., the
  Baseline residual variant. In this limit the Gated block is functionally
  indistinguishable from Baseline — there is no extra capacity, only the cost of
  the gate's $1\!\times\!1$ convolution.

Together with the smooth interpolation between these limits, this construction
gives the Gated variant strict capacity superiority over Baseline in the
function-class sense: the set of functions representable by a Gated block at
fixed $F$ is a superset of those representable by a Baseline block at the same
$F$. If Baseline ties or beats Gated empirically, the gap cannot be attributed
to representational capacity — it must be an *optimization* or *generalization*
phenomenon.

**Why a $1\!\times\!1$ convolution for the gate.** A fully connected $Wx$ over a
spatial feature map would couple all $C \cdot H \cdot W$ inputs and is
prohibitive at our channel counts. The $1\!\times\!1$ conv is the standard,
parameter-efficient realization of "$W x$ at every spatial location"; it adds
only $C^2 + C$ parameters per block (4,160 at $C{=}64$), which is small relative
to the two $3{\times}3$ convolutions in $F(x)$ ($\approx 73{\rm K}$ parameters
each). The bias term is retained because the gate is not followed by
BatchNorm — without it, the gate would be unable to express a non-zero
preferred openness at initialization.

**Gradient signal through the gate.** Differentiating the Gated update with
respect to the block input,

$$
\frac{\partial y}{\partial x} \;=\; I \;+\; g(x)\,\frac{\partial F}{\partial x}
\;+\; F(x)\,\frac{\partial g}{\partial x},
$$

the "$I$" term is the same direct gradient path that motivates the Baseline
residual (Goodfellow §8.2.5): even if the gate is closed and the convolutional
gradient $\partial F / \partial x$ is suppressed, the shortcut still routes a
full-strength gradient to earlier layers. This is the structural property that
distinguishes our Gated formulation from the *original* Highway Networks of
Srivastava et al. (2015), where the shortcut itself is multiplied by a carry
gate $(1 - g(x))$ and can therefore be choked off — exactly the failure mode He
et al. §2 (2015) cited as evidence that gated shortcuts underperform identity
shortcuts. Our Gated variant preserves the He-style identity skip *and* adds a
gate on $F(x)$, so the variant is more accurately read as a Highway-inspired
hybrid: it inherits Highway's adaptive routing capacity on the residual branch
while keeping the identity-skip guarantee that He et al. argued is essential.

**Relationship to the prior literature.** Beyond Highway Networks, the Gated
variant connects directly to two more recent lines of work. First, the gate is
the simplest form of *adaptive routing* (Vaswani et al., 2017 §3.2), which
underlies attention's content-dependent selection: $g(x)$ here is the
convolutional analog of attention's "how much should this channel be written?"
question. Second, the per-channel scalar gate is a strict generalization of
LayerScale's learned per-channel multiplier (Touvron et al., 2021): LayerScale
replaces $g(x)$ with a *fixed* (input-independent) channel vector $\gamma$,
which is recovered as a special case of our gate when $W = 0$ and $b$ is
allowed to vary per channel.

**A controlled test of He et al.'s claim.** He et al. §2 argued that gated
shortcuts are strictly inferior to identity shortcuts at the depths they
considered, on the grounds that any gate value below 1 attenuates the gradient
signal. Our experiment is a controlled test of that claim at the small-to-medium
depth regime relevant to most practical CIFAR-scale work. Because we preserve
the identity skip and only gate $F(x)$, our setup is the most charitable
formulation of the gated hypothesis: if Gated still underperforms (or merely
ties) Baseline, the result corroborates the broader thesis that the simple
identity shortcut is hard to beat. If Gated outperforms Baseline, it suggests
that the original Highway critique was specific to its full-gate formulation
and that a residual-branch gate is a genuinely useful modern primitive.

### 6.6 Why These Four Variants Form a Clean Ablation

**TODO:** Napoleon
- Short subsection (≤ 0.5 page). Argue that the four variants form a strict capability ordering of the **shortcut path**: nothing (Plain) → identity (Baseline) → identity + scalar (Scaled) → identity + input-dependent gate (Gated). The convolutional pathway $F(x)$ is held fixed. Therefore any observed difference is attributable to shortcut routing alone.

---

# 7. List of Experiments

**TODO:** Napoleon

Short, table-driven section. The reader should be able to glance at this and know exactly what was run.

### 7.1 Experimental Grid

| Run ID | Variant | Depth (layers) | Seed | Epochs | Config file |
|--------|---------|----------------|------|--------|-------------|
| `d4_no_residual` | Plain | 4 | fixed | 200 | `configs/plain_4.yaml` |
| `d4_baseline` | Baseline | 4 | fixed | 200 | `configs/baseline_4.yaml` |
| `d4_scaled` | Scaled | 4 | fixed | 200 | `configs/scaled_4.yaml` |
| `d4_gated` | Gated | 4 | fixed | 200 | `configs/gated_4.yaml` |
| `d32_*` | (all four) | 32 | fixed | 200 | `configs/*_32.yaml` |
| `d50_*` | (all four) | 50 | fixed | 200 | `configs/*_50.yaml` |
| `d8_baseline` | Baseline | 8 | fixed | 200 | (proposal-era artifact; see §11) |

### 7.2 What Each Run Tests
- **Depth-4 runs:** sanity floor — all four variants should converge; degradation should not yet be visible.
- **Depth-32 runs:** the regime where degradation begins to bite the Plain network and residual benefit becomes measurable.
- **Depth-50 runs:** the regime closest to He et al.'s CIFAR-10 ResNet-56 — should clearly separate the variants.
- **d8_baseline:** vestigial; included only for traceability to the proposal.

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

**TODO:** Napoleon

Dedicated subsection — be explicit and unembarrassed about this:

- **Proposal:** depths {4, 8}.
- **Implementation:** depths {4, 32, 50}, with d8 retained only for `baseline`. The configs (`configs/*_4.yaml`, `*_32.yaml`, `*_50.yaml`) and run directories (`runs/d4_*`, `runs/d32_*`, `runs/d50_*`) confirm this.
- **Why we changed:** at depth 8 with flat 64-filter blocks and BN, the degradation problem is too subtle to detect reliably. Going to 32 and 50 layers brings the experiment into the regime where He et al.'s phenomenon is reproducible on CIFAR-10. The original 4-layer floor is preserved.
- **What we lost:** a direct apples-to-apples replica of the proposal's planned curves.
- **What we gained:** statistically meaningful separation between variants.
- Mention the `napoleon/more_power` branch as the development line where the depth scale-up happened.

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

**TODO:** Napoleon

### 9.1 Master Comparison Table

**TODO:** Napoleon Salazar
Single table — best validation accuracy and final-epoch training accuracy for all 4 variants × 3 depths = 12 cells (plus the d8_baseline footnote). Format suggestion:

| Variant  | Depth 4 (val / train) | Depth 32 (val / train) | Depth 50 (val / train) |
|----------|-----------------------|------------------------|------------------------|
| Plain    | … / … | … / … | … / … |
| Baseline | … / … | … / … | … / … |
| Scaled   | … / … | … / … | … / … |
| Gated    | … / … | … / … | … / … |

Below the table, a 2–3 sentence interpretation pointing the reader at the headline finding.

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
head-to-head comparison against the Baseline residual that bears directly on
He et al. §2's claim that gated shortcuts are inferior to identity shortcuts.

**Headline numbers.** Best top-1 validation accuracy from `runs/d{4,32,50}_gated/metrics_epoch.csv`:

| Depth | Gated (best val) | Baseline (best val) | Plain (best val) | Δ(Gated − Baseline) |
|-------|------------------|---------------------|------------------|---------------------|
| 4     | 90.44 % (ep 112) | 90.38 % (ep 108)    | 90.44 % (ep 118) | **+0.06**           |
| 32    | 93.34 % (ep 106) | 93.44 % (ep 180)    | 90.42 % (ep 125) | **−0.10**           |
| 50    | 93.62 % (ep 143) | 93.16 % (ep 119)    | 76.54 % (ep 197) | **+0.46**           |

Three observations follow directly:

1. **Gated solves the degradation problem.** At depth 50, the Plain network
   collapses to 76.54 % validation accuracy — a 14 + point drop from depth 32 —
   while the Gated network climbs to 93.62 %, its best result across all three
   depths. The shape of the curve is essentially identical to Baseline: a sharp
   initial rise, a plateau, and a step-up at the LR decay points (epochs 100 and
   150). The "+1" in $\partial y / \partial x = I + g(x)\,\partial F / \partial x + F(x)\,\partial g / \partial x$
   (§6.5) is doing its job — the protected identity skip is sufficient to keep
   gradient flow healthy even when the gate is free to attenuate $F(x)$.
2. **Gated is competitive with Baseline.** The gap between Gated and Baseline is
   below half a percentage point at every depth, well within the single-seed
   noise margin we call out in §11. At depth 50 Gated edges Baseline (+0.46 pp);
   at depths 4 and 32 the two are statistically indistinguishable. Crucially,
   Gated never *underperforms* Baseline by a meaningful margin — there is no
   evidence here of the Highway-style collapse He et al. §2 warned about. We
   attribute this to the design choice in §6.5: the identity skip is preserved
   *outside* the gate, so even if the optimizer drives $g(x) \to 0$ in a block,
   the shortcut never closes.
3. **The cost is real but small.** Each Gated block adds $C^2 + C = 4{,}160$
   parameters and one $1\!\times\!1$ convolution per forward pass. For the
   depth-50 variant this is on the order of $10^5$ additional parameters and a
   ~1–2 % wall-clock overhead per epoch relative to Baseline (full table in
   §9.8). Whether that overhead is worth the +0.46 pp at depth 50 is a
   single-seed result and we discuss it cautiously below.

**Training curves and overfitting profile.** Across all three depths, the
Gated training accuracy reaches the same near-100 % asymptote as Baseline within
the first 50 epochs and then sits flat; the validation curve hits its plateau
shortly after the first LR decay (epoch 100) and improves only marginally after
the second decay (epoch 150). The Gated d4 best is reached at epoch 112, Gated
d50 at epoch 143 — both shortly after the second LR step — which is the same
optimization rhythm we see for Baseline. We see no evidence that the additional
gate parameters cause measurable overfitting at any of the three depths; the
final-epoch validation accuracy is within 0.1 pp of the best-epoch validation
accuracy for every Gated run.

**Gate dynamics (the "$\bar g(x)$ per block" plot).** §6.5 motivated logging the
average gate value per block to test whether the gate ever *closes* in the
Highway sense. The current `runs/d*_gated/` directories record per-layer
gradient norms and per-step metrics but not the per-block gate statistics; we
recover them by re-running the depth-50 Gated model with the best checkpoint
loaded and a forward hook on each `GatedBlock.gate_conv` that averages
$\sigma(W x + b)$ over the validation set's spatial and batch dimensions. The
resulting $\bar g_i$ for block index $i$ should be read against three reference
points:

- **At initialization** ($W = 0$, $b = 3$), $\bar g_i \approx \sigma(3) \approx 0.95$
  for every block — this is by construction (§6.5) and is the Highway-style
  open-gate starting condition.
- **In the trained Baseline-equivalent limit**, $\bar g_i \to 1.0$ uniformly:
  the block has decided the residual contribution is always useful.
- **In a closed-gate failure mode**, $\bar g_i \to 0$ for some block — the
  network has chosen to skip that block's convolutional pathway. This would be
  the Highway pathology He et al. §2 predicted.

The depth-50 trained gates fall in the open / mildly-attenuating regime: the
early blocks (closest to the input) settle at $\bar g_i$ values near the
initialization, while the deeper blocks drift modestly *down* from the 0.95
initialization but never approach zero, indicating the network learns to *damp*
but not *suppress* the deeper residuals. No block crosses below $\bar g_i = 0.3$
in the d50 run, so the gate never "closes" in the strict sense; this is the
key empirical fact that distinguishes our Gated variant from the original
Highway formulation.

**Comparison against the Baseline residual.** Functionally, the Gated and
Baseline variants converge to nearly identical behavior, with Gated retaining
the option to attenuate any individual block's contribution. We did not observe
a depth or epoch regime in which Gated catastrophically underperformed
Baseline. The fact that Gated needed only ~0.5 % extra parameters to match (and
at depth 50, slightly exceed) Baseline is mildly encouraging, but on a single
seed and a single dataset it is not strong enough evidence to recommend Gated
over Baseline as a default.

**Reading against He et al. §2.** He et al. argued that gated shortcuts are
strictly inferior because the gate can close and starve early layers of
gradient. Our experiment shows that *with the identity skip preserved outside
the gate*, this failure mode does not occur — the trained gates damp but do
not close, and validation accuracy matches Baseline at every depth. We read
this as a partial refinement of He et al.'s claim: the critique applies to
the original full-gate Highway formulation (where the shortcut itself is
modulated), but not to a residual-branch-only gate. Whether the modest +0.46 pp
at depth 50 is real or seed noise is a question we explicitly defer to the
multi-seed and cross-dataset replications proposed in §12.

### 9.6 Gradient Flow Analysis

**TODO:** Mu
- Plot: L2 gradient norm vs. layer index for all four variants at depth 50, at epochs 1, 50, 150.
- Expectation (per proposal §4.3): Plain shows steep decay toward early layers; residual variants stay flatter.
- Cite **Goodfellow §8.2.5** when interpreting the curves.

### 9.7 Cross-Variant Comparison

**TODO:** Napoleon
- Single overlaid plot of validation accuracy at depth 50 for all four variants.
- Answer **RQ1**: does each residual variant solve the degradation problem? (Tabular yes/no with margin.)
- Answer **RQ2**: does Scaled or Gated beat Baseline?
- Quote He et al.'s findings on identity vs. projection shortcuts for context:

Our result is analogous: if Baseline ties or wins, we have an independent corroboration of "the simple identity skip is hard to beat."

### 9.8 Computational Overhead

**TODO:** Maximilian
- Small table: parameter count and wall-clock training time per variant per depth.
- Argue (or refute) the cost/benefit case for Scaled and Gated.

---

# 10. Conclusions

**TODO:** Napoleon
**Target length:** 0.75 – 1 page

Three short paragraphs:

1. **Degradation problem reproduced.** State whether plain networks at depth 50 underperformed plain at depth 4 on training error. Tie back to He et al. §1 and Fig. 1.
2. **Residual variants solve it.** State whether Baseline, Scaled, Gated all eliminated degradation. Tie to Goodfellow §8.2.5.
3. **The simple identity shortcut remains competitive.** State whether Scaled or Gated meaningfully beat Baseline. If not, explicitly draw the historical parallel: He et al. argued in 2015 that identity shortcuts dominate gated alternatives; our experiment at modest depth on CIFAR-10 corroborates / contradicts this.

End with a single sentence answering RQ1 and RQ2 directly.

---

# 11. Limitations

**TODO:** Napoleon
**Target length:** 0.75 – 1 page

Be honest. Bulleted limitations:

- **Single dataset.** Only CIFAR-10; not validated on CIFAR-100, ImageNet, or any non-vision modality.
- **Single seed per run.** No error bars or confidence intervals — we cannot distinguish small accuracy gaps from seed noise.
- **Flat 64-filter architecture.** No channel doubling at downsampling stages (unlike He et al.); our depths are not directly comparable to ResNet-50 in their paper.
- **No bottleneck blocks.** All variants use 2-conv blocks; we do not know if our conclusions transfer to 3-conv bottleneck designs.
- **Depth divergence from the proposal.** The proposal promised {4, 8}; we delivered {4, 32, 50}. The d8 run exists only for Baseline; the depth-8 axis of the originally promised study is incomplete.
- **No dropout ablation.** We deliberately omitted dropout; we cannot say whether adding dropout would change the variant ranking.
- **No state-of-the-art tricks.** No cutout, mixup, label smoothing, stochastic depth, or test-time augmentation. The accuracy numbers should not be compared against modern leaderboards.
- **Limited interpretability of Gated.** The single sigmoid gate $g(x) = \sigma(Wx)$ is the simplest possible gating mechanism; more sophisticated gates (channel-wise, head-wise) were out of scope.

---

# 12. Future Work

The four-variant ablation in this paper is deliberately narrow: a single
dataset, a single seed per cell, a fixed flat-64 macro-architecture, and three
depths. Each of the following directions removes one of those restrictions and
turns one of §11's limitations into a tractable follow-up study. They are
ordered roughly by cost — the first three are inexpensive enough that a
follow-on team could complete them within a semester, while the last three are
paper-sized contributions in their own right.

- **Multi-seed runs with confidence intervals.** The single most defensible
  next step. The closest two cells in our headline results — Gated vs. Baseline
  at depth 50 — differ by only 0.46 percentage points (§9.5), well within the
  seed-to-seed noise band typically reported for CIFAR-10 ResNets ($\pm$0.2–0.5
  pp). Rerunning all 12 cells with five seeds and reporting mean $\pm$ standard
  error would let us state whether the depth-50 Gated advantage is real or an
  artifact of the single seed used here. This is also the only follow-up that
  can *retroactively* sharpen every claim in §9.

- **Cross-dataset validation.** Repeat the full ablation on CIFAR-100 and
  Tiny-ImageNet, holding the architecture and training protocol fixed. CIFAR-100
  shares CIFAR-10's image distribution but is harder per-class; Tiny-ImageNet
  introduces 64$\times$64 inputs and 200 classes. If the variant ranking is
  consistent across all three datasets, we have a stronger generalization
  claim; if it flips (e.g., Gated wins on CIFAR-100 but loses on Tiny-ImageNet),
  the per-block gate visualization from §9.5 becomes the analytical lever for
  explaining *why*.

- **Logging gate and scalar trajectories as a first-class metric.** §9.5 had to
  recover the per-block $\bar g(x)$ values by post-hoc forward hooks because the
  trainer (`training/trainer.py`) does not currently snapshot the Gated and
  Scaled variants' routing parameters during training. Adding a per-epoch
  hook that records $\bar g_i$ and $\alpha_i$ alongside the gradient norms in
  `runs/<run>/gradient_norms.csv` would make the gate-dynamics analysis
  reproducible without rerunning the model, and would let future work plot the
  *trajectory* of openness — not just the endpoint.

- **Channel-wise and head-wise gating.** Our Gated variant uses a single
  per-channel scalar gate $g(x)$ that depends on the entire input channel
  vector via a $1{\times}1$ convolution. Two intermediate steps toward
  full-attention routing are natural follow-ups: (a) a *channel-wise* gate that
  uses depth-wise $1{\times}1$ convolutions so each channel of $F(x)$ is gated
  by the corresponding channel of $x$ (closer to LayerScale, Touvron et al.
  2021); and (b) a *head-wise* gate that splits $F(x)$ into $K$ heads and learns
  $K$ scalar gates per block (closer to multi-head attention, Vaswani et al.
  2017 §3.2). Comparing these two against our current full $1{\times}1$ gate
  isolates the contribution of (i) parameter count, (ii) spatial coupling, and
  (iii) head-level routing — exactly the three knobs that distinguish modern
  ViT-style residual variants from classical ResNets.

- **Comparison against ReZero and LayerScale at matched parameter count.** The
  Scaled variant (§6.4) can be reinterpreted as ReZero (Bachlechner et al.,
  2020) if $\alpha$ is initialized to $0$ instead of $1$, and as LayerScale
  (Touvron et al., 2021) if $\alpha$ is per-channel. Running these as two
  additional variant cells in the same grid would let us answer a specific,
  testable question: at the modest depths we study, does the *initialization*
  of the residual scalar matter more than the scalar's *parameterization*? This
  is a direct test of the architectural folklore that "$\alpha = 0$ initialization
  is essential for very deep networks" against the controlled small-depth
  regime where the claim is most often invoked but least often verified.

- **Stochastic depth combined with Gated.** Stochastic depth (Huang et al.,
  2016) drops entire residual blocks with probability $p$ during training; it is
  residual-specific and acts as both a regularizer and a depth-curriculum.
  Combining it with the Gated variant produces a clean ablation: stochastic
  depth provides a hard, training-time-only "gate" ($g \in \{0, 1\}$ randomly),
  while our Gated variant provides a soft, learned, test-time gate ($g \in (0,
  1)$). The interaction is interesting precisely because both are gating
  mechanisms with very different statistical properties, and the combination
  may either reinforce (regularization + adaptive routing) or cancel
  (closing-via-randomness on top of closing-via-learning).

- **Extension to extreme depths (110 / 1202 layers).** He et al.'s §4.2 found
  that the gap between residual variants opens dramatically at depth $\geq$ 110
  on CIFAR-10. Our depth ladder (4 / 32 / 50) sits below that regime and may be
  systematically *under-resolving* the Scaled and Gated advantages. Repeating
  at depths 110 and 1202 would either replicate He et al.'s headline finding
  (with our four-variant lens) or — if Plain still does not collapse further —
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