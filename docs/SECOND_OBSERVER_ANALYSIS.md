# DRIVE second-observer analysis

DRIVE provides two manual vessel annotations for its 20 test images. The first annotation is used for the main analysis, while the second is used to check whether the main conclusions depend strongly on one annotation set.

The same frozen Standard U-Net and DPD-2 predictions from seeds 41, 42, and 43 are evaluated against both annotations. The field of view, threshold (0.50), and structural settings are unchanged. No observer-specific tuning or postprocessing is used.

For Observer 2, the case bootstrap averages the three seeds within each image and uses 10,000 percentile bootstrap replicates (`numpy.random.default_rng(20260818)`). A second bootstrap resamples both the three observed seeds and the 20 images, also for 10,000 replicates (`numpy.random.default_rng(20265840)`).

Raw manual masks are not redistributed. The numeric results are included in `results/`.
