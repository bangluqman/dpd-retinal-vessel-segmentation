# Controlled comparison models

LMFR-Net and Spatial-AttVesNet were reimplemented under the same DRIVE preprocessing, split, optimization, checkpoint-selection, threshold, and evaluation pipeline used for Standard U-Net and DPD. The purpose was a controlled architectural comparison, not an exact reproduction of the original papers.

## LMFR-Net Controlled

The local model uses base width 16, ICB-style depthwise/pointwise and ordinary-convolution paths, LMAFF dilation pairs `(9,7)`, `(7,5)`, `(5,3)`, spatial and SE attention, learned down/up-sampling, two decoder streams, and fixed 0.5/0.3/0.2 fusion. Green-CLAHE input is repeated into three channels. Training and evaluation follow the common pipeline used in this study rather than the original optimization recipe.

Original paper: W. Zhang, S. Qu, and Y. Feng, “LMFR-Net: lightweight multi-scale feature refinement network for retinal vessel segmentation,” *Pattern Analysis and Applications*, 28(2), article 44, 2025. DOI: `10.1007/s10044-025-01424-x`.

Original repository: https://github.com/MCloud31/LMFR-Net

## Spatial-AttVesNet Controlled

The local model uses base width 64, encoder widths 64/128/256/256, a 512→256 bottleneck, four max-pooling levels, attention-gated skip connections, nearest-neighbor upsampling, and Conv-ReLU-BN ordering. Green-CLAHE input is repeated into three channels and DRIVE images use deterministic pad/crop where required. Training and evaluation again follow the common study pipeline.

Original paper: S. Lamti et al., “Spatial-AttVesNet: Compact U-Net with Attention for Retinal Vascular Image Segmentation,” *International Journal of Intelligent Engineering and Systems*, 19(2), 195–214, 2026. DOI: `10.22266/ijies2026.0228.13`.

Original repository: https://github.com/SanaeLamti/Spatial-AttVesNet-A-Low-Parameter-Vascular-Segmentation-Using-Attention-Mechanisms

The source code of these two local reimplementations is not included in this repository. Their numeric comparison results are available in `results/controlled_baseline_per_seed.csv` and `results/controlled_baseline_summary.csv`.
