# ASSUMPTIONS

This is an **independent replication**. The authors' generated ADFA-LD dataset, code, and package
versions are not available to us. Every value below that the paper does not state is **our choice**,
not the authors' setting. "Not found" means *not found in the paper pages we have read so far
(journal pp. 30168-30171)*; pages 1-3 and 9-12 must be checked before the final report says
"not reported".

| ID | Item | Paper status | Our value | Why | Potential impact |
|----|------|--------------|-----------|-----|------------------|
| A1 | Dataset | Generated version of ADFA-LD, not public | Rebuild from raw ADFA-LD | Only available source | Counts, class balance, and difficulty will differ from the paper |
| A2 | Window stride | Not found (paper only says instances are 30 calls) | 30 (non-overlapping) | Simplest; no duplicated content between windows | Fewer windows than an overlapping scheme; window count will not match 52,656 |
| A3 | Traces shorter than 30 calls | Not found | Dropped, count and report them | Padding would add an artificial "pad" token | Slight loss of very short traces |
| A4 | Window labels | Not found | Every window of an attack trace = attack; of a normal trace = normal | Only trace-level labels exist in ADFA-LD | Some attack windows may look benign (label noise); can lower recall |
| A5 | Which raw folders are normal/attack | Not found | Training_Data_Master + Validation_Data_Master = normal; Attack_Data_Master = attack | ADFA-LD's documented layout | The paper names brute force, DoS and network scan; how these map to ADFA-LD attack folders is unverified |
| A6 | Class balance / resampling | Classes ~balanced (derived from confusion matrix); no resampling described | No resampling; report actual ratio | Do not alter data without a stated reason | Raw ADFA-LD may not be balanced after windowing; affects recall/precision |
| A7 | **Split unit** | Not found | `group_by: trace` (trace-disjoint stratified split) | Keeps every trace entirely within one partition and avoids cross-partition trace leakage | Metrics may differ from an unknown author split |
| A8 | Split proportions | Not found; paper's counts cover the full dataset | 70/15/15 trace-disjoint stratified, seed 42 | Allows validation for early stopping and a clean held-out test | Metrics are on a test subset, so compare rates, not raw counts |
| A9 | TF-IDF settings | TF-IDF, n=2 | Bigrams of whitespace-separated IDs, no lowercasing, sklearn defaults otherwise | Syscall IDs are numeric tokens | Initial bigram count will differ from 2,805 |
| A10 | Chi-squared selection | k=150 | `SelectKBest(chi2, k=150)` fit on train only | Prevents leakage | None expected beyond A1 |
| A11 | Hidden-layer widths | Not found (2 ReLU dense layers) | 128, 64 | Common small baseline | Affects capacity and metrics |
| A12 | Optimizer, LR, batch size, epochs | Not found | Adam, 1e-3, 128, max 100 | Common defaults | Affects convergence and metrics |
| A13 | Early stopping | Not found | monitor val_loss, patience 10, restore best weights | Common practice | Prevents overfitting; not the authors' protocol |
| A14 | Decision threshold | Not found | 0.5 | Standard for sigmoid | Changes TP/FN balance |
| A15 | LIME settings | LimeTabularExplainer, top-10 | num_samples=5000, random_state=42, default kernel width, `discretize_continuous=false` | Configurable; discretization concern for sparse TF-IDF columns | Rankings depend on these |
| A16 | SHAP settings | KernelExplainer, top-10 | 50 random training samples as background, default nsamples, seed 42 | KernelExplainer is expensive; small background is tractable | Attribution values depend on background |
| A17 | Explained instances | Paper explains all instances | First TP, TN, FN, FP in test-set order | Deterministic, cheap | Comparable examples, not the paper's exact rows |
| A18 | Ranking rule | "10 most important features" | Rank by absolute weight | Direction is reported separately | Zeroing a feature already equal to 0 is a no-op; we will count such cases (decide in Phase 5) |
| A19 | Meaning of "remove" | Paper speaks of removing system calls | Set the TF-IDF bigram column to 0, no re-normalization | Simplest valid version in feature space | Not identical to editing the raw sequence; other bigrams and IDF are unchanged |
| A20 | Random baseline | Third batch replaces randomly | Sample k features from non-top-ranked columns, seed 42 | Needed to judge whether top features matter more than random ones | Depends on seed |
| A21 | Stage C (exclusive replacement) | Described | Stretch goal, approximate in feature space | TF-IDF does not map back to raw edits | Not an exact recreation |
| A22 | Survey | Reported | Not replicated | Needs participants and ethics approval | We claim technical experiments only |
| A23 | Seed | Not found | 42 everywhere (Python, NumPy, TensorFlow, split, LIME, SHAP) | Reproducibility | GPU/multithread nondeterminism may remain |
| A24 | Software versions | Not found | See requirements.txt (TF 2.16.2 / Keras 3.3.3 on Python 3.12) | TF 2.15.1 has no Python 3.12 build | Minor numerical differences possible |

## Frozen decisions for Phase 2.6
- **A7:** `group_by: trace` is the active baseline. No window-level random split is used as the primary protocol.
- **A18/A19:** how to treat top-ranked features that are already 0 in the instance (decide in Phase 5).
