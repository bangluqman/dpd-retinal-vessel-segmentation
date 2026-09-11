# DPD U-Net for Thin Retinal Vessel Segmentation

This repository contains the code and numeric results for the paper **“Early Dual-Path Detail Preservation for Thin Retinal Vessel Segmentation with U-Net.”**

DPD-2 makes one small change to a standard U-Net: the usual max-pooling path is kept, while a space-to-depth residual path is added at the first two encoder reductions. The purpose is to preserve fine vessel information without redesigning the whole network.

On DRIVE, the most consistent effects were higher sensitivity and fewer thin-vessel false negatives. Other structural measures were more dependent on the training seed. STARE and CHASE_DB1 were evaluated with dataset-specific out-of-fold training and showed smaller, mixed changes.

DPD-2 adds about **0.95% trainable parameters** and **2.70% convolutional GMAC** over the Standard U-Net.

## What is included

- `src/` — models, DRIVE preprocessing, training, evaluation, and structural metrics
- `analysis/` — sensitivity, bootstrap, observer, parameter, and GMAC analyses
- `configs/` — model and experiment settings
- `splits/` — dataset splits and statistical units
- `results/` — per-seed, per-image/per-subject, and summary results
- `docs/` — short methodological notes
- `provenance/` — checksums for study artifacts that are not redistributed
- `tests/` — dataset-free consistency checks

LMFR-Net and Spatial-AttVesNet were used only as controlled same-pipeline comparisons. Their local source code is not redistributed; the implementation notes and original references are in `docs/CONTROLLED_BASELINES.md`.

## Data

The repository does not include DRIVE, STARE, or CHASE_DB1. Please obtain the datasets from their original providers and follow their terms of use.

The DRIVE loader expects:

```text
DRIVE/
├── training/{images,1st_manual,mask}/
└── test/{images,1st_manual,2nd_manual,mask}/
```

Dataset references: [DRIVE](https://drive.grand-challenge.org/), [STARE](https://cecas.clemson.edu/~ahoover/stare/), and CHASE_DB1 (Fraz et al., 2012; DOI: `10.1109/TBME.2012.2205687`).

## Setup

```bash
conda env create -f environment.yml
conda activate ijies-dpd
```

The reported experiments used Python 3.10.20, PyTorch 2.12.0+cu126, and an NVIDIA RTX 4080 SUPER. The exact GPU model is not required.

## Reproduce the main DRIVE experiment

Set `DRIVE_ROOT` to your local DRIVE directory.

Train Standard U-Net and DPD-2:

```bash
python src/train.py --data_root "$DRIVE_ROOT" --model unet_standard --seed 41 --output_dir runs/standard_seed41
python src/train.py --data_root "$DRIVE_ROOT" --model unet_dpd_v3 --seed 41 --output_dir runs/dpd2_seed41
```

Repeat with seeds 42 and 43. The fixed split and training settings are in `splits/drive.json` and `configs/drive_training.yml`.

Evaluate a trained DPD-2 checkpoint:

```bash
python src/evaluate.py --data_root "$DRIVE_ROOT" --model unet_dpd_v3 --checkpoint checkpoint_best.pt --observer 1 --output_dir evaluation/dpd2_seed41
```

Check model size and GMAC:

```bash
python analysis/model_audit.py --output audit_models.csv --layers audit_layers.csv
```

## Reproduce the additional analyses

Structural sensitivity:

```bash
python analysis/structural_sensitivity.py --manifest prediction_manifest.csv --output structural_sensitivity.csv
```

Case and seed uncertainty:

```bash
python analysis/uncertainty.py \
  --drive results/uncertainty_inputs/drive_per_image.csv \
  --stare results/uncertainty_inputs/stare_per_image.csv \
  --chase results/uncertainty_inputs/chase_per_eye.csv \
  --output_dir uncertainty_reproduction
```

DRIVE second-observer analysis:

```bash
python analysis/second_observer.py --input results/drive_second_observer_per_image.csv --output observer2_bootstrap.csv
```

The structural definitions are described in `docs/STRUCTURAL_METRICS.md`, and the bootstrap procedure in `docs/UNCERTAINTY_ANALYSIS.md`.

## A few notes

The study used three training seeds (41, 42, and 43), so the seed bootstrap should not be interpreted as the full distribution of training randomness. The structural measures are pixel-based research diagnostics, not clinically validated endpoints. STARE and CHASE_DB1 results are dataset-specific OOF evaluations, not zero-shot transfer experiments.

## Citation and license

Citation metadata are in `CITATION.cff`. The code is released under the MIT License.
