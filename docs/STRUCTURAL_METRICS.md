# Structural evaluation diagnostics

These pixel-domain quantities are evaluation diagnostics, not training losses. Predictions are binarized at 0.50 inside the field of view (FOV).

Let `G` be the binary manual vessel mask, `P` the binary prediction, and `F` the FOV. A morphological skeleton of `G ∩ F` is formed by iterative erosion/opening with a 3×3 cross. The Euclidean distance transform of `G` supplies local skeleton radius.

- **Thin FN:** skeleton pixels with radius at most `rho` are dilated with an elliptical kernel of radius `ceil(rho)` and clipped to `G`. Thin FN is the number of pixels in this thin-region mask that are absent from `P` inside `F`.
- **Far FP:** false-positive pixels in `P ∩ F` whose Euclidean distance from the nearest `G` pixel is strictly greater than the configured distance.
- **Branches:** junction skeleton pixels (8-neighbour count at least 3) are removed; 8-connected components are candidate branches. Components shorter than the minimum length are excluded.
- **Tolerant branch coverage:** proportion of each branch skeleton covered by `P` after elliptical dilation by the branch tolerance. Tolerance zero uses `P` directly.
- **Recovered / partial / missed:** coverage at least 0.80 / from 0.20 to below 0.80 / below 0.20.
- **Length-weighted coverage:** `sum(branch_length × coverage) / sum(branch_length)` over eligible branches. Dataset aggregation pools covered and total branch length.
- **Long-branch coverage:** the same pooled ratio restricted to branches meeting the long-branch threshold.

The primary configuration (`D0_PRIMARY`) is `rho=2 px`, tolerance `1 px`, minimum branch length `5 px`, long-branch threshold `20 px`, and Far-FP distance strictly `>5 px`.

Sensitivity analysis uses 11 prespecified one-factor-at-a-time configurations: D0; `rho` 1/3; tolerance 0/2; minimum branch length 3/7; long-branch threshold 15/25; and Far-FP distance strictly >3/>7. No Cartesian search or result-based setting selection is used.

Implementation: `src/structural_metrics.py`. Configuration list: `configs/structural_sensitivity.yml`.
