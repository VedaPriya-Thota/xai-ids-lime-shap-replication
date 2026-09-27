# Phase 3B — LIME and SHAP Explanations

This is a sampled XAI reconstruction, not an all-instance explanation.

**This is a strict reconstructed ADFA-LD baseline, not the authors' exact 52,656-instance dataset.**

Phase 3A model and preprocessing artifacts were loaded frozen and were not retrained or refit.

## Selection
20 frozen test windows: 5 TP, 5 FN, 5 TN, 5 FP, selected with seed 42.

## LIME
- LimeTabularExplainer-compatible implementation: `vendored-compat-0.2.0.1`
- Training data: full TRAIN transformed matrix (61,553 × 150)
- class_names: `['normal', 'attack']`
- num_features: 10
- num_samples: 5000
- random_state: 42
- discretize_continuous: false

## SHAP
- KernelExplainer only
- Background: 100 TRAIN rows, 50 normal + 50 attack
- seed: 42
- nsamples: 512
- l1_reg: `num_features(150)`
- explained output: attack probability

## Limitations
The exact author-generated dataset was not recovered. LIME/SHAP are computed for only 20 sampled test instances. The external `lime` package was unavailable in the offline runtime, so the repository uses a vendored `LimeTabularExplainer` compatibility implementation exposing the requested API; this is a documented implementation deviation and should not be described as the exact external package runtime.

No perturbation analysis, retraining, balancing, preprocessing refit, or raw-data modification was performed.
