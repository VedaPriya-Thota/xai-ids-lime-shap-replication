# Phase 3C — Explanation-guided perturbation evaluation

This is a bounded perturbation evaluation of the frozen Phase 3A model using frozen Phase 3B explanations. It is not causal evidence about real attacks and is not an exact reproduction of the paper.

- Frozen test sample: 20 instances (5 TP, 5 FN, 5 TN, 5 FP).
- Raw representation: 30-call sequences; target features are adjacent syscall 2-grams.
- Guided intervention: process top-10 explanation-ranked bigrams, replace the second syscall of each actionable pair using a training-only normal-frequency neutral policy; each raw position is edited at most once.
- Random control: same number of edits, deterministic positions, same neutral policy.
- Prediction: frozen Phase 3A TF-IDF, chi-square selector, and MLP only.

## Limitations
The Phase 3B LIME results come from a vendored LIME-compatible implementation because the official package was unavailable in the offline runtime. The dataset is a strict reconstruction rather than the authors' exact 52,656-instance dataset. Results are descriptive only; no causal interpretation or significance claim is made.

## Aggregate method summary

| method          |   n_instances |   actionable_cases |   mean_actionable_feature_rate |   mean_actionable_edits |   mean_abs_guided_delta |   mean_abs_random_delta |   guided_label_flips |   random_label_flips |
|:----------------|--------------:|-------------------:|-------------------------------:|------------------------:|------------------------:|------------------------:|---------------------:|---------------------:|
| lime_compatible |            20 |                 15 |                          0.125 |                    4.6  |                0.234926 |                0.15669  |                    9 |                    6 |
| shap_kernel     |            20 |                 20 |                          0.32  |                   11.25 |                0.352012 |                0.300972 |                    9 |                    8 |

## Aggregate group summary

| method          | outcome_group   |   n_instances |   actionable_cases |   mean_actionable_feature_rate |   mean_actionable_edits |   mean_guided_delta |   mean_random_delta |   mean_abs_guided_delta |   mean_abs_random_delta |   guided_label_flips |   random_label_flips |
|:----------------|:----------------|--------------:|-------------------:|-------------------------------:|------------------------:|--------------------:|--------------------:|------------------------:|------------------------:|---------------------:|---------------------:|
| lime_compatible | FN              |             5 |                  2 |                           0.04 |                     1   |          0.00473978 |         0.0144056   |               0.0390167 |             0.0219629   |                    0 |                    0 |
| lime_compatible | FP              |             5 |                  5 |                           0.24 |                     7.6 |         -0.438019   |        -0.276901    |               0.438019  |             0.326189    |                    4 |                    3 |
| lime_compatible | TN              |             5 |                  3 |                           0.06 |                     3.8 |          0.108616   |         1.02688e-05 |               0.108616  |             0.000357518 |                    1 |                    0 |
| lime_compatible | TP              |             5 |                  5 |                           0.16 |                     6   |         -0.324452   |        -0.199836    |               0.354054  |             0.278249    |                    4 |                    3 |
| shap_kernel     | FN              |             5 |                  5 |                           0.3  |                     6.8 |         -0.0631389  |        -0.0611439   |               0.0830559 |             0.0644754   |                    0 |                    0 |
| shap_kernel     | FP              |             5 |                  5 |                           0.38 |                    14.4 |         -0.583723   |        -0.53449     |               0.583723  |             0.53449     |                    4 |                    4 |
| shap_kernel     | TN              |             5 |                  5 |                           0.28 |                    11.6 |          0.0829694  |         0.0744626   |               0.10042   |             0.0917119   |                    0 |                    0 |
| shap_kernel     | TP              |             5 |                  5 |                           0.32 |                    12.2 |         -0.640848   |        -0.513211    |               0.640848  |             0.513211    |                    5 |                    4 |