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
| A11 | Hidden-layer widths | Not found (2 ReLU dense layers) | 64, 32 | Common small baseline | Affects capacity and metrics |
| A12 | Optimizer, LR, batch size, epochs | Not found | Adam, 1e-3, 256, max 100 | Common defaults | Affects convergence and metrics |
| A13 | Early stopping | Not found | scikit-learn `MLPClassifier` internal early stopping: `early_stopping=True`, `validation_fraction=0.1` (10% of the training split, not a separately held-out validation set), `n_iter_no_change=10` | scikit-learn's built-in mechanism, used as-is | Prevents overfitting; not the authors' protocol |
| A14 | Decision threshold | Not found | 0.5 | Standard for sigmoid | Changes TP/FN balance |
| A15 | LIME settings | LimeTabularExplainer, top-10 | Vendored LIME-compatible local-surrogate reconstruction (the official `lime==0.2.0.1` package could not be installed in the offline execution environment; see `docs/lime_official_package_verification.md`); num_samples=5000, random_state=42, default kernel width, `discretize_continuous=false` | Official package unavailable offline; vendored implementation exposes the same API/settings | Rankings depend on these settings, and the surrogate is a documented reconstruction, not the official package |
| A16 | SHAP settings | KernelExplainer, top-10 | Actual `shap.KernelExplainer`; 100 train-only background rows (50 normal + 50 attack, deterministic seeded selection), `nsamples=512` (explicit, not SHAP's default), `l1_reg="num_features(150)"`, seed 42 | KernelExplainer is expensive; a small stratified background is tractable | Attribution values depend on background |
| A17 | Explained instances | Paper explains all instances | Exactly 20 test instances: 5 TP, 5 FN, 5 TN, 5 FP, selected via seeded random sampling within each outcome group (`g.sample(n=5, random_state=42)`, then sorted by window index) | Deterministic, cheap, covers all four outcome types | Comparable examples, not the paper's exact rows |
| A18 | Ranking rule | "10 most important features" | Rank by absolute weight | Direction is reported separately | — |
| A19 | Meaning of "remove" | Paper speaks of removing system calls | Edit the raw 30-call syscall sequence: replace the second syscall of an exact adjacent occurrence of a top-10 2-gram feature with a deterministic neutral-replacement syscall (ID 3, the most frequent syscall in training-normal raw windows), then re-tokenize into 2-grams and re-run the frozen TF-IDF → chi-square → MLP pipeline | Operates on the same representation the model actually sees the sequence in; no direct feature-space edits | Not identical to editing a TF-IDF column; other bigrams adjacent to an edited position can also change |
| A20 | Random baseline | Third batch replaces randomly | Matched random baseline: the same number of raw-position edits as the guided run, applied to an independent copy of the original sequence, using the same neutral-replacement policy, with a deterministic seed derived from base seed 42 + method + instance ID | Needed to judge whether guided edits matter more than random ones | Depends on seed |
| A21 | Phase 3C scope | Paper describes class-exclusive/random replacement experiments | Additional reconstruction analysis (guided vs. matched-random raw-sequence edits, as A19/A20), not a reproduction of a specific paper experiment | The paper's own perturbation protocol intervenes in feature space with class-exclusive/random feature sets; ours intervenes on the raw sequence instead | Not a claimed reproduction of the paper's perturbation figures |
| A22 | Survey | Reported | Not replicated | Needs participants and ethics approval | We claim technical experiments only |
| A23 | Seed | Not found | 42 everywhere (Python, NumPy, scikit-learn `random_state`, split, LIME-compatible surrogate, SHAP) | Reproducibility | Multithread nondeterminism may remain |
| A24 | Software versions | Not found | See `requirements.txt`; the frozen MLP itself uses scikit-learn 1.4.2 / NumPy 1.26.4 on Python 3.12 (TensorFlow/Keras are listed in `requirements.txt` but are not used by the frozen Phase 3A reconstruction) | Documents the actual runtime, not just the file that lists it | Minor numerical differences possible across scikit-learn versions |
| A25 | Framework | Paper: TensorFlow/Keras | scikit-learn `MLPClassifier` | Single-library pipeline (the same library already used for TF-IDF/chi-square); avoids GPU/CUDA cross-version non-determinism in a CPU-only, offline environment | Framework itself differs from the paper; documented in `docs/phase1_result_comparison.md` |

## Frozen decisions for Phase 2.6
- **A7:** `group_by: trace` is the active baseline. No window-level random split is used as the primary protocol.
- **A18/A19:** a top-10 feature is "actionable" only if its 2-gram occurs literally, adjacently, in the instance's raw 30-call sequence; non-actionable features (no matching adjacent occurrence) are recorded, not silently skipped (see `docs/phase1_result_comparison.md`, Section D).
