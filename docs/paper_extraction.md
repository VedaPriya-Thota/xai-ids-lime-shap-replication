# Paper extraction

**Paper:** Gaspar, D., Silva, P., & Silva, C. (2024). *Explainable AI for Intrusion Detection Systems:
LIME and SHAP Applicability on Multi-Layer Perceptron.* IEEE Access, 12, 30164-30175.
DOI: 10.1109/ACCESS.2024.3368377

**Reading coverage:** journal pp. 30168-30171 read directly (Sections IV-V.D, Figs 2-9, Table 1).
Pages 1-3 and 9-12 are **not yet read**; anything marked "not found" must be re-checked there.

Match status values: **Confirmed** (in paper) / **Derived** (computed from paper numbers) /
**Not found** (assumption used) / **Unverified** (only from a secondary summary) / **Deviation** (we differ on purpose).

## A. Confirmed from the paper

| Paper item | Extracted detail | Evidence/source | Replication implementation | Match status |
|---|---|---|---|---|
| Task | Binary IDS: 0 = normal, 1 = anomaly | Sec. V.C, p. 30171 | `label_map` normal 0 / attack 1 | Confirmed |
| Dataset | Generated version of ADFA-LD, IoT IDS context | Sec. IV.A p. 30168; Sec. V.A p. 30170 | Rebuild from raw ADFA-LD | Deviation (A1) |
| Instance | Sequence of 30 system calls | Sec. V.A, p. 30170 | `window_size: 30` | Confirmed |
| Vocabulary | 149 distinct system calls | Sec. V.A, p. 30170 | Report our count in the audit | Confirmed (ours may differ) |
| Dataset size | 52,656 instances | Sec. V.C text, p. 30171 | Report our window count | Confirmed (ours will differ) |
| Attack types | Password brute force, DoS, network scan | Sec. IV.A, V.A | Attack_Data_Master, mapping unverified | Confirmed, mapping unverified (A5) |
| Vectorization | TF-IDF with n=2 (bigrams) | Sec. V.A, p. 30170 | `TfidfVectorizer(ngram_range=(2,2))` | Confirmed |
| Initial features | 2,805 n-grams | Fig. 8, p. 30170 | Report our initial count | Confirmed (ours may differ) |
| Feature reduction | Chi-squared test to 150 features | Sec. V.A, Fig. 8 | `SelectKBest(chi2, k=150)`, train only | Confirmed |
| Feature identity | Features are bigrams (e.g. '91 195') although text says "system calls" | Fig. 8 vs. Sec. V.D | Features are bigram columns | Confirmed; terminology mismatch noted |
| Model | TensorFlow/Keras MLPClassifier, input + 3 dense layers | Sec. V.C, p. 30170-30171 | Keras MLP | Confirmed |
| Activations | Dense 1 and 2 ReLU, dense 3 sigmoid | Sec. V.C | Same | Confirmed |
| Input dimension | 150 | Sec. V.C | `input_dim: 150` | Confirmed |
| Confusion matrix (text) | TP 24,038; FN 2,832; TN 25,311; FP 475 (sum 52,656) | Sec. V.C, p. 30171 | Reference values | Confirmed |
| Figure 9 | Cell labels appear inconsistent with the text (25,311 sits in a Positive/Positive cell) | Fig. 9, p. 30171 | Use text values as reference | Confirmed inconsistency |
| Table 1 | Accuracy 0.9371, Precision 0.9806, Sensitivity 0.8946, Specificity 0.9816, F1 0.9356 | Table 1, p. 30171 | Comparison table (Phase 6) | Confirmed |
| Evaluation basis | Counts cover all 52,656 instances; no split described in pages read | Fig. 9, Sec. V.C | Held-out test set | Deviation (A8) |
| LIME | `LimeTabularExplainer` given data, feature names, class names; top-10 features per instance | Sec. IV.A, p. 30168 | Same explainer, top-10 | Confirmed |
| SHAP | `KernelExplainer`, top-10 features per instance | Sec. IV.A, p. 30168 | Same explainer, top-10 | Confirmed |
| Explained set | Explanations for all instances of the dataset | Sec. IV.A-C | First TP/TN/FN/FP of test set | Deviation (A17) |
| Perturbation batch 1 | Remove the 10 most important features from every instance, re-run model | Fig. 5, p. 30169 | Stage B | Confirmed |
| Perturbation batch 2 | Iteratively replace class-exclusive features (X from TP-only, Y from FN-only sets) in FN/TP instances; repeat for TN/FP | Fig. 6, p. 30169 | Stage C (stretch) | Confirmed |
| Perturbation batch 3 | Same as batch 2 but X and Y chosen randomly | Fig. 6, p. 30169 | Stage C (stretch) | Confirmed |
| Example result | `setsockopt` replaced by `accept`: 90% of affected FN became TP | Sec. V.D, p. 30171 | Optional comparison target | Confirmed |
| Class-exclusive features | Some features appear only in FN (or TP) explanations | Sec. V.D | Optional analysis | Confirmed |

## B. Derived or unverified

| Paper item | Detail | Source | Status |
|---|---|---|---|
| Class counts | Attack 26,870 (TP+FN); normal 25,786 (TN+FP) | Derived from the text counts | Derived (not stated directly) |
| Accuracy check | (24,038 + 25,311) / 52,656 = 0.9371 | Derived | Consistent with Table 1 |
| Survey | 24 participants comparing LIME vs SHAP | Secondary summary only | Unverified; survey not replicated (A22) |
| Other perturbation results | e.g. `nfsservctl` replacement moving ~100% of some FP to TN | Secondary summary only | Unverified until pp. 9-12 read |

## C. Not found in pages read (assumptions used)

| Paper item | Our replication value | Assumption ID |
|---|---|---|
| Hidden-layer widths | 128, 64 | A11 |
| Optimizer, learning rate, batch size, epochs | Adam, 1e-3, 128, max 100 | A12 |
| Early stopping | val_loss, patience 10 | A13 |
| Train/validation/test split, stratification | 70/15/15 stratified | A8 |
| Random seed | 42 | A23 |
| Window generation, stride, short-trace handling | stride 30, drop short traces | A2, A3 |
| Class balancing | none | A6 |
| LIME num_samples, kernel width, discretization | 5000, default, false | A15 |
| SHAP background size and sampling | 50 random training samples | A16 |
| How a system call is "removed"/"replaced" at TF-IDF level | Set bigram column to 0 | A19 |
| Package versions, hardware | See requirements.txt / docs/environment.md | A24 |
