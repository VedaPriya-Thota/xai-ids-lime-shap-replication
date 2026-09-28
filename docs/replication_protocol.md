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
| Model | scikit-learn `MLPClassifier`, 150 -> 64 (ReLU) -> 32 (ReLU) -> 1 (logistic/sigmoid) |
| Seed | 42 for Python, NumPy, scikit-learn `random_state`, split, LIME-compatible surrogate, SHAP, random baselines |
| XAI | LIME: vendored LIME-compatible local-surrogate reconstruction (the official `lime==0.2.0.1` package could not be installed in the offline execution environment; see `docs/lime_official_package_verification.md`); SHAP: actual `shap.KernelExplainer`; top-10 features for both |
| Metrics | Accuracy, precision, recall, F1, specificity, confusion matrix on the held-out test set |

## 3. Pipeline and leakage rules
1. Load raw traces, label them, build windows, keep metadata (source path, trace ID, window index, label, window ID).
2. **Split windows first** (70/15/15 trace-disjoint stratified, seed 42; see A7 for the window-vs-trace question).
3. Fit TF-IDF and chi-squared selection on **training windows only**; transform validation/test with the fitted objects.
4. Train with scikit-learn `MLPClassifier`'s internal early stopping (`early_stopping=True`, `validation_fraction=0.1` of the training split, `n_iter_no_change=10`). The test set is never used for tuning.
5. Evaluate once on the test set.
6. Draw LIME/SHAP background data from training data only; draw explained instances only from the test set.

## 4. Explained instances (deterministic)
Exactly 20 deterministic test windows: 5 TP, 5 FN, 5 TN, 5 FP. Within each outcome group, 5 rows are selected by
seeded random sampling (`random_state=42`) and then sorted by window index — not simply the first occurrence of
each type. For each: instance ID, seed, true label, predicted label, predicted attack probability, top-10 features
(name, signed weight/value, direction). These are *comparable* examples, not the paper's exact rows.

## 5. Perturbation experiments (Phase 3C)
Phase 3C is an **additional reconstruction analysis**, not a reproduction of a specific paper experiment. The
paper's own perturbation protocol intervenes in feature space with class-exclusive/random feature replacement;
this reconstruction instead edits the **raw 30-call syscall sequence** directly:

1. Take a frozen, explained test instance and its explanation method's top-10 ranked 2-gram features.
2. Parse each feature as an adjacent syscall pair (e.g. `221_197` = syscall 221 immediately followed by syscall 197).
3. Find exact adjacent occurrences of that pair in the instance's raw 30-call sequence.
4. Process target features in explanation rank order; each raw position may be edited at most once.
5. Replace the **second** syscall of an actionable occurrence with a deterministic neutral-replacement syscall
   (ID 3 — the most frequent syscall observed in training-only *normal* raw windows). Non-actionable features
   (no matching adjacent occurrence) are recorded, not silently skipped.
6. Preserve the sequence at exactly 30 calls; re-tokenize into 2-grams; re-run through the frozen, transform-only
   TF-IDF → chi-square selector → MLP pipeline (no refitting, no retraining, no rebalancing).
7. Compare the original and new predicted attack probability, original and new label, and whether the label flipped.
8. A **matched random baseline** applies the same number of raw-position edits to an independent copy of the
   original sequence, using the same neutral-replacement policy, with a deterministic seed derived from base seed
   42 + method + instance ID.

No perturbation-vs-guided difference is treated as statistically significant (no significance test is run), and
results are reported as descriptive model-output sensitivity under this intervention policy — not as causal
syscall importance, and not as evidence that LIME or SHAP is generally superior to the other.

## 6. Comparison with the paper
Compare rates (accuracy, precision, recall, specificity, F1) computed from the paper's text counts
(TP 24,038, TN 25,311, FP 475, FN 2,832) with our test-set metrics — see `docs/phase1_result_comparison.md` for the
full table. Do not compare raw counts unless normalized: the paper's counts cover its full 52,656-instance dataset,
while ours cover a 13,736-window held-out test split only. This comparison is a required replication deliverable
that assesses how closely the reconstructed methodology reproduces the paper's reported behavior; it is not a
controlled head-to-head performance comparison, and matching or differing accuracy is not treated as evidence of
exact replication either way. State the differences: no official data, no author split, unknown paper
architecture/hyperparameters beyond what Table 1 and the confusion-matrix discussion report, held-out evaluation
vs. full-dataset counts.

## 7. Phases
This project's phases, as tracked throughout the repository (`results/experiments/`, `reports/`, `AI_USAGE.md`):

1. Scaffold and dataset provenance investigation
2. Phase 2.5 — dataset reconciliation/audit (`reports/phase2_5_dataset_audit.md`/`.json`, `docs/paper_dataset_forensics.md`)
3. Phase 3A — MLP reconstruction and evaluation (`results/experiments/phase3a_strict_reconstructed_mlp/`)
4. Phase 3B — LIME and SHAP explanations (`results/experiments/phase3b_xai_explanations/`)
5. Phase 3C — explanation-guided perturbation evaluation (`results/experiments/phase3c_perturbation_evaluation/`)
6. Phase 1 finalization — documentation and reproducibility cleanup (this file, `README.md`, `docs/phase1_result_comparison.md`, `reports/phase1_replication_summary.md`, `AI_USAGE.md`)

## 8. Reproducibility status
- [x] Seed 42 set in every component (split, TF-IDF/chi-square fitting, MLP training, LIME-compatible surrogate, SHAP, perturbation selection).
- [x] Dataset source recorded (`data/README.md`); download date not recorded in the available provenance (stated explicitly rather than invented).
- [x] Preprocessing artifacts saved (vectorizer, selector, feature names, model, split assignment) under `results/experiments/phase3a_strict_reconstructed_mlp/`.
- [x] One documented Python environment (`requirements.txt`) for the whole canonical workflow (dataset → Phase 3A → Phase 3B → Phase 3C); no second virtual environment or manual package-version workaround is used.
- [x] Phase 3B/3C accept explicit `--phase3a-dir`/`--phase3b-dir`/`--experiment-dir` overrides, so a freshly regenerated Phase 3A/3B artifact set (produced under this same environment) can feed the next phase directly, instead of requiring a historical serialized artifact that may predate the documented environment's package versions. Omitting these flags preserves the original default behavior against the historical frozen directories.
- [x] All commands runnable from the project root (see `README.md`, "Reproducibility").
- [x] Test suite: 67 passed (`python -m pytest tests/ -q`).
