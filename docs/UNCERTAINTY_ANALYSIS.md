# Uncertainty analysis

Effects are paired Standard-versus-DPD differences, oriented so that positive values favor DPD. Higher-is-better metrics use `DPD − Standard`; Thin FN, Far FP, and missed branches use `Standard − DPD`.

## Case/unit bootstrap

The three training runs are first averaged within each independent evaluation unit. Paired unit effects are then resampled with replacement for 10,000 replicates. The 2.5th and 97.5th percentiles form the 95% interval. Reproduction uses `numpy.random.default_rng(20260818)`.

This interval describes variation across the evaluated cases after averaging the three runs.

## Seed-and-unit bootstrap

For each of 10,000 replicates, the three observed seed labels and the independent evaluation units are resampled with replacement. The paired effect is then averaged over the sampled seed × unit combinations. Reproduction uses `numpy.random.default_rng(20265840)`.

This gives a sensitivity analysis that includes variation across both evaluated cases and the three observed training runs. With only three seeds, it should not be interpreted as the full distribution of optimization randomness.

## Statistical units

- **DRIVE:** 20 test images.
- **STARE:** 20 OOF images; long-branch coverage uses 17 evaluable pairs because three images have no eligible long branches for either model.
- **CHASE_DB1:** 14 subjects. Left/right-eye count outcomes are summed within subject, while non-count metrics are averaged. The 28 eyes are not treated as independent bootstrap units.
