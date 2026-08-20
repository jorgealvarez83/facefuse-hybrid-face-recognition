# FaceFuse: quality-adaptive fusion of deep and handcrafted face descriptors

Technical report · Ahmed Sayed

> This is the method write-up behind the implementation. Measured results are in
> the [README](../README.md#results); design rationale is in
> [architecture.md](architecture.md).

## 1. Problem

A face embedding network trained with an angular-margin loss is excellent on
clean, frontal, well-lit faces and degrades on everything else — motion blur,
defocus, harsh or dim lighting, low-resolution capture. The failure is not
graceful: the network still returns a unit-norm 512-d vector, and that vector
still has a nearest neighbour in the gallery. It fails confidently.

Classical texture descriptors have the opposite profile. Local Binary Patterns
compare each pixel to its neighbours, so any monotonic change in illumination
leaves the code untouched. Histograms of Oriented Gradients encode edge
direction, which survives moderate contrast loss. Neither carries anything like
the identity-discriminative power of a learned embedding, but both degrade
predictably.

The two are complementary. The question this project asks is how to decide, per
face and at inference time, which one to believe.

## 2. Approach

Score-level fusion with a weight derived from a per-crop sharpness estimate.

### 2.1 Similarity

Both branches are compared to the gallery by cosine similarity:

$$
S(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}
$$

### 2.2 Local Binary Patterns

For a centre pixel $g_c$ with eight neighbours $g_p$:

$$
\mathrm{LBP} = \sum_{p=0}^{7} s(g_p - g_c)\, 2^{p},
\qquad
s(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}
$$

The image of codes is reduced to a 256-bin histogram, L1-normalised. Because
every code depends only on *comparisons*, a global gain or offset on the image
leaves the histogram unchanged — the property tested in
`tests/test_handcrafted.py::test_lbp_is_invariant_to_uniform_brightness_gain`.

### 2.3 Histogram of Oriented Gradients

$$
G_x = I(x{+}1, y) - I(x{-}1, y), \qquad G_y = I(x, y{+}1) - I(x, y{-}1)
$$

$$
\theta = \arctan\!\left(\frac{G_y}{G_x}\right), \qquad
\lVert G \rVert = \sqrt{G_x^2 + G_y^2}
$$

Orientations are binned over a 64×128 window with 16×16 blocks, 8×8 cells and
9 orientation bins, giving 3780 dimensions. Concatenated with the LBP
histogram, the handcrafted descriptor is 4036-d.

### 2.4 Quality estimate

Sharpness is the variance of the Laplacian response over the enhanced crop:

$$
Q = \operatorname{Var}\!\big(\nabla^2 I\big)
$$

One convolution. Blur removes high-frequency content, which collapses the
variance.

### 2.5 Fusion

$$
w_{\text{deep}} = \min\!\Big(1,\ \max\!\big(0,\ \tfrac{Q}{\text{quality\_scale}}\big)\Big),
\qquad w_{\text{hand}} = 1 - w_{\text{deep}}
$$

$$
S_{\text{fused}} = w_{\text{deep}} \cdot S_{\text{deep}} + w_{\text{hand}} \cdot S_{\text{hand}}
$$

The predicted identity is the gallery entry maximising $S_{\text{fused}}$,
subject to $S_{\text{fused}} \geq \tau$; otherwise `Unknown`.

Fusion happens on the *scalars*, not the vectors. Concatenating a 512-d
L2-normalised embedding with a 4036-d descriptor whose HOG part is unnormalised
would let the longer block dominate the cosine — the fusion weight would be an
accident of dimensionality rather than a decision.

## 3. Pipeline

| Stage | Implementation |
| --- | --- |
| Detection | YOLO11-L fine-tuned on WIDER FACE, single class `face`, confidence ≥ 0.55 |
| Cropping | detected box expanded by 20% in each direction |
| Enhancement | bilateral filter (`d`, σ_color, σ_space), edge-preserving |
| Deep branch | letterbox to 320×320, InsightFace `buffalo_l` → 512-d |
| Handcrafted branch | LBP (256) ⊕ HOG (3780) → 4036-d |
| Quality | Laplacian variance of the enhanced crop |
| Matching | cosine against every stored gallery sample, fused, thresholded |

The 20% margin is not cosmetic. InsightFace runs its own detection and
five-point alignment inside `app.get()`; handed a tight box it frequently finds
no face at all and returns an empty list. The margin restores the context that
alignment stage expects.

## 4. Hyper-parameter search

Five parameters are tuned jointly: the match threshold $\tau$, `quality_scale`,
and the three bilateral-filter parameters. They interact — a stronger filter
raises apparent sharpness, which raises $w_{\text{deep}}$, which changes the
score distribution the threshold cuts.

Top-1 accuracy is piecewise constant, so there is no gradient. The search uses a
real-valued genetic algorithm:

- **Initialisation** — uniform over the bounded box.
- **Selection** — truncation; the fitter half survives.
- **Crossover** — arithmetic mean of two randomly drawn parents.
- **Mutation** — Gaussian, σ = 10% of each parameter's range, applied with
  probability 0.15, clipped back into bounds.
- **Elitism** — the best individual seen is tracked across all generations.

Bounds: $\tau \in [0.3, 0.7]$, `quality_scale` $\in [100, 2000]$,
$d \in [3, 9]$, $\sigma_{\text{color}}, \sigma_{\text{space}} \in [10, 100]$.

The optimiser is seedable, and `tests/test_optimizer_and_metrics.py` verifies it
converges on a known quadratic optimum, respects bounds, and reproduces exactly
for a fixed seed.

## 5. Evaluation protocol

Gallery and probe use the same folder-per-identity layout; the folder name is
the ground-truth label. Metrics are macro-averaged accuracy, precision, recall
and F1 from scikit-learn, plus a confusion matrix and mean per-image latency.

**A caveat that limits every tuned number reported by this project**: the
genetic search maximises accuracy on the probe set and the benchmark then
reports accuracy on that same probe set. The tuned figures are therefore
optimistic by construction. `--no-optimize` produces the untuned comparison,
which is what the README reports. A three-way gallery/validation/test split is
the correct fix and has not been implemented.

## 6. Results summary

The detector reaches mAP@50 = 0.757 and mAP@50–95 = 0.417 on the WIDER FACE
validation split after 40 epochs.

On a 10-identity, 32-probe-image evaluation set with default parameters, the
hybrid recogniser and the deep-only baseline both score 0.969 top-1 — **the
fusion changed no prediction**. The probe images are sharp press photographs, so
$Q \gg$ `quality_scale` and $w_{\text{deep}}$ saturates at 1.0, which makes the
hybrid path mathematically identical to the baseline. The mechanism is
implemented and unit-tested; on this data it is inactive. Establishing whether it
helps requires a probe set with real quality degradation.

## 7. Limitations

Closed-set matching against a fixed threshold; exhaustive linear gallery scan;
alignment degradation past roughly 45° of yaw; no anti-spoofing; and a probe set
far too small for the percentages to be stable. See the README's *Limitations*
section.

## 8. References

1. Deng, J., Guo, J., Xue, N., Zafeiriou, S. *ArcFace: Additive Angular Margin
   Loss for Deep Face Recognition.* CVPR, 2019.
2. Ojala, T., Pietikäinen, M., Mäenpää, T. *Multiresolution Gray-Scale and
   Rotation Invariant Texture Classification with Local Binary Patterns.*
   IEEE TPAMI 24(7), 2002.
3. Dalal, N., Triggs, B. *Histograms of Oriented Gradients for Human Detection.*
   CVPR, 2005.
4. Yang, S., Luo, P., Loy, C. C., Tang, X. *WIDER FACE: A Face Detection
   Benchmark.* CVPR, 2016.
5. Tomasi, C., Manduchi, R. *Bilateral Filtering for Gray and Color Images.*
   ICCV, 1998.
6. Pertuz, S., Puig, D., Garcia, M. A. *Analysis of Focus Measure Operators for
   Shape-from-Focus.* Pattern Recognition 46(5), 2013.
7. Jocher, G., et al. *Ultralytics YOLO.* https://github.com/ultralytics/ultralytics
