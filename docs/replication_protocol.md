# Replication protocol

## 1. Claims policy
- This is an **independent replication**, not an exact reproduction. We will not call any result an
  exact reproduction unless the authors' data, architecture, splits, and package versions are obtained.
- Every undocumented decision is listed in `ASSUMPTIONS.md` with an ID (A1, A2, ...).
- Synthetic data is used **only** to test code. No synthetic number is ever reported as an ADFA-LD result.

## 2. Fixed decisions
| Item | Decision |
|---|---|
| Task | Binary: normal (0) vs attack (1) |
| Data | Raw ADFA-LD in `data/raw/ADFA-LD/` (never committed) |
| Windows | 30 calls, stride 30, non-overlapping, incomplete tails dropped |
| Split | Trace-disjoint stratified, seed 42, 15% validation, 15% test |
| Features | TF-IDF bigrams, chi-squared to 150, fit on training windows only |
| Model | Keras MLP 150 -> 128 (ReLU) -> 64 (ReLU) -> 1 (sigmoid) |
| Seed | 42 for Python, NumPy, TensorFlow, split, LIME, SHAP, random baselines |
| XAI | LIME (`LimeTabularExplainer`) and SHAP (`KernelExplainer`), top-10 features |
| Metrics | Accuracy, precision, recall, F1, specificity, confusion matrix on the held-out test set |

## 3. Pipeline and leakage rules
1. Load raw traces, label them, build windows, keep metadata (source path, trace ID, window index, label, window ID).
2. **Split windows first** (70/15/15 stratified, seed 42; see A7 for the window-vs-trace question).
3. Fit TF-IDF and chi-squared selection on **training windows only**; transform validation/test with the fitted objects.
4. Train with early stopping on validation loss. The test set is never used for tuning.
5. Evaluate once on the test set.
6. Draw LIME/SHAP background data from training data; draw explained instances only from the test set.

## 4. Explained instances (deterministic)
In test-set order (ordered by window ID): first TP, first TN, and, if they exist, first FN and first FP.
For each: instance ID, seed, true label, predicted label, predicted attack probability, top-10 features
(name, signed weight, direction), CSV and PNG. These are *comparable* examples, not the paper's exact rows.

## 5. Perturbation experiments
"Removing" a bigram means setting its TF-IDF column to 0 for that instance, without re-normalizing the row (A19).
Ranking is by absolute weight (A18). For each perturbed instance we save original and new attack probability,
probability change, original and new label, and whether the label flipped.

- **Stage A (mandatory):** for the explained instances, perturb top-1 and top-3 features from LIME and SHAP separately,
  plus a random baseline (k features sampled from non-top-ranked columns, seed 42). One row per instance, explainer, k, strategy.
- **Stage B (mandatory):** remove the top-10 features per explainer for available TP, TN, FP, FN instances.
  Report flip rate and mean probability change per group and explainer, plus all individual rows.
- **Stage C (stretch, only after A and B work):** approximate the paper's class-exclusive replacement and its random variant.
  Documented as an approximation because TF-IDF features do not map exactly back to raw sequence edits.

Limitation to state in the report: zeroing a feature that is already 0 in an instance changes nothing, so the number of
"no-op" features will be reported.

## 6. Comparison with the paper
Compare rates (accuracy, precision, recall, specificity, F1) computed from the paper's text counts
(TP 24,038, TN 25,311, FP 475, FN 2,832) with our test-set metrics. Do not compare raw counts unless normalized.
State the differences: no official data, no author split, unknown architecture/hyperparameters,
held-out evaluation vs full-dataset counts, and the Figure 9 label ambiguity.

## 7. Phases
1. Scaffold (this phase)
2. Data pipeline
3. MLP
4. LIME and SHAP
5. Perturbation
6. Results and reporting

## 8. Reproducibility checklist
- [ ] Python version and OS recorded (`docs/environment.md`)
- [ ] `pip freeze` saved as `requirements-lock.txt` after a real install
- [ ] Seed 42 set in every component
- [ ] Dataset source and download date recorded (`data/README.md`)
- [ ] Preprocessing artifacts saved (vectorizer, selector, feature names, label map, split indices)
- [ ] All commands runnable from the project root
