# Phase 3A — Strict Reconstructed ADFA-LD MLP Baseline

**This is a strict reconstructed ADFA-LD baseline, not the authors' exact 52,656-instance dataset.**

Paper: Gaspar, Silva, and Silva (2024), DOI 10.1109/ACCESS.2024.3368377.

## Pipeline
- Frozen 30-call, stride-30, non-overlapping windows; incomplete tails dropped.
- 2-gram tokens use the exact form `callA_callB` (e.g. `1_2`).
- TF-IDF fitted on TRAIN only.
- chi-square SelectKBest fitted on TRAIN only, k=150.
- Validation/test are transformed only after fitting.

## MLP
- Paper-reported: Keras/TensorFlow model; two hidden dense layers with ReLU; sigmoid output; binary prediction.
- Independent defaults: 64 and 32 hidden units, Adam, learning rate 0.001, batch size 256, max 100 iterations with early stopping on an internal 10% subset of TRAIN only, seed 42.
- No class weights or resampling.

## Results

### Validation
- Accuracy: 0.931014
- Balanced accuracy: 0.816806
- Precision (normal/attack): 0.951831 / 0.763685
- Recall (normal/attack): 0.970038 / 0.663573
- F1 (normal/attack): 0.960848 / 0.710118
- Macro F1: 0.835483
- ROC-AUC: 0.948639
- PR-AUC / average precision: 0.807822
- Confusion matrix [normal, attack]: [[11461, 354], [580, 1144]]

### Test
- Accuracy: 0.937391
- Balanced accuracy: 0.850313
- Precision (normal/attack): 0.964654 / 0.734481
- Recall (normal/attack): 0.964336 / 0.736291
- F1 (normal/attack): 0.964495 / 0.735385
- Macro F1: 0.849940
- ROC-AUC: 0.959989
- PR-AUC / average precision: 0.739275
- Confusion matrix [normal, attack]: [[11681, 432], [428, 1195]]

## Leakage and reproducibility
- Trace-disjoint manifest assignments were verified.
- TF-IDF and chi-square were fit on training windows only.
- Validation/test labels were not used for fitting or model selection.
- Test metrics were computed after the model configuration was finalized.
- Raw data and Phase 2.5/2.6 artifacts were not modified.

## Scope boundary
LIME and SHAP have **not** been run in Phase 3A.
