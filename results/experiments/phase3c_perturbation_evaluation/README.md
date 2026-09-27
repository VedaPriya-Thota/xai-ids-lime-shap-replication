# Phase 3C — Explanation-guided perturbation evaluation

This is a bounded perturbation evaluation of the frozen Phase 3A model using frozen Phase 3B explanations. It is not causal evidence about real attacks and is not an exact reproduction of the paper.

- Frozen test sample: 20 instances (5 TP, 5 FN, 5 TN, 5 FP).
- Raw representation: 30-call sequences; target features are adjacent syscall 2-grams.
- Guided intervention: process top-10 explanation-ranked bigrams, replace the second syscall of each actionable pair using a training-only normal-frequency neutral policy; each raw position is edited at most once.
- Random control: same number of edits, deterministic positions, same neutral policy.
- Prediction: frozen Phase 3A TF-IDF, chi-square selector, and MLP only.

## Limitations
The Phase 3B LIME results come from a vendored LIME-compatible implementation because the official package was unavailable in the offline runtime. The dataset is a strict reconstruction rather than the authors' exact 52,656-instance dataset. Results are descriptive only; no causal interpretation or significance claim is made.
