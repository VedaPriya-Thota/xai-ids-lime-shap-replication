# XAI for IDS: LIME and SHAP on MLP

An independent, course-project reconstruction of the intrusion-detection and explainability workflow described in Gaspar, Silva, and Silva (2024). The project rebuilds the paper's syscall-window → MLP → LIME/SHAP pipeline on publicly available raw ADFA-LD data and compares the resulting experimental behavior with the paper's reported results.

## Original research paper

> Gaspar, D., Silva, P., & Silva, C. (2024). *Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron.* IEEE Access, 12, 30164–30175. DOI: [10.1109/ACCESS.2024.3368377](https://doi.org/10.1109/ACCESS.2024.3368377)

This paper was selected because it describes a self-contained, methodologically explicit pipeline — syscall windowing, bigram TF-IDF, chi-square feature selection, an MLP classifier, and LIME/SHAP explanations — that is well suited to a course project on explainable AI for intrusion detection, and because it reports concrete, checkable numbers (Table 1, a confusion matrix, and worked LIME/SHAP examples) against which a reconstruction can be compared.

## Current status

**Phase 1 complete; Phase 2 not started.**

> **Strict reconstruction disclaimer:** this project does **not** contain the exact author-generated 52,656-instance dataset. The exact dataset and complete author preprocessing/split procedure were not recovered. Results are therefore independent reconstruction results, not exact numerical replication of the paper. The reconstruction results **are** compared against the paper's reported results (see `docs/phase1_result_comparison.md`) as a required replication deliverable; because the dataset/protocol differ, this comparison assesses similarity of behavior rather than a controlled head-to-head performance comparison.

## Project objective

The course project reproduces/reconstructs the key experiment of the paper and compares the resulting experimental behavior with the paper's reported results. It covers: intrusion detection as binary classification, MLP classification, syscall 2-gram representation, TF-IDF, chi-square feature selection, LIME, SHAP, explanation comparison, and explanation-guided perturbation analysis.

Because the exact author-generated 52,656-instance dataset and complete preprocessing/split protocol were not recovered, this is a **strict independent reconstruction** using the publicly available raw ADFA-LD data, not an exact numerical replication. Every point where the reconstruction had to deviate from or assume beyond the paper is documented (see `ASSUMPTIONS.md` and `docs/phase1_result_comparison.md`).

## What we achieved

- Audited the available public ADFA-LD raw data (`docs/paper_dataset_forensics.md`, `reports/phase2_5_dataset_audit.md`/`.json`).
- Built a trace-disjoint reconstruction using 30-call non-overlapping windows.
- Converted syscall sequences into adjacent 2-gram transitions.
- Applied train-only TF-IDF.
- Applied train-only chi-square feature selection to 150 features.
- Trained and froze an MLP classifier (Phase 3A).
- Generated deterministic LIME-compatible explanations (Phase 3B).
- Generated Kernel SHAP explanations (Phase 3B).
- Compared top-10 explanation features between the two methods.
- Performed explanation-guided raw-sequence perturbation analysis (Phase 3C).
- Compared guided perturbations against matched random perturbations.
- Preserved frozen artifacts and reproducibility metadata (hashes, seeds, configs).
- Added a paper-vs-reconstruction result comparison (`docs/phase1_result_comparison.md`).
- Added dataset/provenance forensic documentation (`docs/paper_dataset_forensics.md`).
- Added a reproducibility test suite (67 tests).

## End-to-end pipeline

```
ADFA-LD raw traces
  → 30-call windows (non-overlapping, incomplete tails dropped)
  → adjacent syscall 2-grams (e.g. "221_197")
  → train-only TF-IDF
  → chi-square feature selection
  → 150 selected features
  → MLP classifier
  → LIME / Kernel SHAP explanations
  → explanation comparison (top-10 Jaccard / sign agreement)
  → explanation-guided perturbation analysis
```

- **Windows** — each raw syscall trace is cut into fixed-length sequences of 30 calls; this matches the paper's stated instance length.
- **2-grams** — each window is turned into adjacent-pair tokens (syscall *i* followed by syscall *i+1*), matching the paper's bigram representation.
- **TF-IDF** — the 2-gram tokens are vectorized with a TF-IDF fitted only on the training split, to avoid leakage from validation/test data.
- **Chi-square selection** — the TF-IDF features are reduced to the paper's reported 150 features using a chi-square test fitted on training data only.
- **MLP** — a small feed-forward classifier predicts normal vs. attack from the 150 selected features.
- **LIME / SHAP** — for a fixed, deterministic sample of test windows, both methods produce a ranked top-10 list of features driving each prediction.
- **Explanation comparison** — the two methods' top-10 lists are compared for overlap and sign agreement.
- **Perturbation analysis** — an additional analysis that edits the raw syscall sequence at explanation-guided positions and checks how the model's predicted attack probability moves, compared to matched random edits.

## Dataset and forensic findings

The exact author-generated 52,656-instance dataset was not publicly recoverable, so the reconstruction is built from a public raw ADFA-LD mirror (`https://www.kaggle.com/datasets/yianqaq/adfa-ld`; see `data/README.md` for the recorded provenance and archive checksum). Full forensic detail is in [`docs/paper_dataset_forensics.md`](docs/paper_dataset_forensics.md).

| Reconstruction | Value |
|---|---:|
| Valid raw traces | 5,951 |
| — normal | 5,205 |
| — attack | 746 |
| Total windows | 88,828 |
| — normal windows | 78,599 |
| — attack windows | 10,229 |
| Observed syscall IDs | 175 (range 1–340) |
| Window size / stride | 30 / 30 (non-overlapping, incomplete tails dropped) |
| Split | Trace-disjoint, seed 42, 15% validation, 15% test |

| Paper | Value |
|---|---:|
| Instances | 52,656 |
| Syscall vocabulary | 149 |
| Instance length | 30 system calls |
| Representation | 2-gram, TF-IDF |
| Initial bigrams | 2,805 |
| Selected features | 150 (chi-square) |

The 149-vs-175 vocabulary difference is an expected consequence of using a different source/provenance mirror rather than the authors' own generated dataset; it does not imply that the exact paper dataset was recovered. See `docs/paper_dataset_forensics.md` and `reports/phase2_5_dataset_audit.md` for the full audit.

## Methodology

### Data preparation
Raw ADFA-LD traces are labelled (Training/Validation = normal, Attack = attack), cut into non-overlapping 30-call windows, and split trace-disjointly (no trace's windows appear in more than one split) with seed 42.

### 2-gram representation
Each window becomes a sequence of adjacent syscall-pair tokens, e.g. `221_197` for the transition from syscall 221 to syscall 197.

### TF-IDF
The 2-gram tokens are vectorized with TF-IDF, fitted on the training split only and applied (transform-only) to validation/test.

### Chi-square feature selection
`SelectKBest(chi2, k=150)` is fitted on training data only and reduces the TF-IDF space to the paper's reported 150 features.

### MLP
The frozen reconstruction MLP uses:

| Setting | Value |
|---|---|
| Input features | 150 |
| Hidden layers | 64 ReLU → 32 ReLU |
| Output | logistic/sigmoid |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 256 |
| Max iterations | 100 |
| Early stopping | train-only, internal |
| Class weighting / resampling | none |
| Seed | 42 |

The paper specifies two ReLU hidden dense layers and a sigmoid output but does not report hidden-layer widths or complete training hyperparameters; the widths and training settings above are therefore independent reconstruction choices, not values taken from the paper.

### LIME
Explanations are produced with a documented **vendored LIME-compatible local-surrogate reconstruction** (deterministic perturbation, exponential locality weighting, weighted linear surrogate), because the official `lime==0.2.0.1` package could not be installed in the offline execution environment. See `docs/lime_official_package_verification.md`.

### SHAP
Explanations use the actual `shap.KernelExplainer` (SHAP 0.50.0) with a train-only background of 100 windows (50 normal + 50 attack), seed 42.

## Results

Frozen Phase 3A reconstruction test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.937391 |
| Balanced accuracy | 0.850313 |
| Normal precision | 0.964654 |
| Normal recall | 0.964336 |
| Normal F1 | 0.964495 |
| Attack precision | 0.734481 |
| Attack recall | 0.736291 |
| Attack F1 | 0.735385 |
| Macro F1 | 0.849940 |
| ROC-AUC | 0.959989 |
| PR-AUC | 0.739275 |

Confusion matrix (13,736 held-out test windows): TN = 11,681, FP = 432, FN = 428, TP = 1,195.

Paper's reported values (Table 1 and confusion-matrix discussion, over its full 52,656-instance dataset):

| Metric | Paper |
|---|---:|
| Accuracy | 0.9371 |
| Precision | 0.9806 |
| Sensitivity | 0.8946 |
| Specificity | 0.9816 |
| F1 | 0.9356 |
| TP | 24,038 |
| FN | 2,832 |
| TN | 25,311 |
| FP | 475 |

This comparison is required by the course and assesses how closely the reconstructed methodology reproduces the paper's reported behavior. It is **not a controlled head-to-head performance comparison**, because dataset support size, class distribution, split/evaluation basis, framework (scikit-learn vs. TensorFlow/Keras), and some training settings differ. Neither result is presented as "better" or "worse." Full detail: [`docs/phase1_result_comparison.md`](docs/phase1_result_comparison.md).

## XAI results

### LIME
Vendored LIME-compatible local-surrogate reconstruction, top-10 features per instance, seed 42. The official `lime==0.2.0.1` package was unavailable in the offline environment.

### Kernel SHAP
Actual `shap.KernelExplainer`, train-only background of 100 rows (50 normal + 50 attack), top-10 features per instance.

Both methods explain the same 20 deterministic frozen test windows: 5 TP, 5 FN, 5 TN, 5 FP, selected with seed 42.

### Explanation comparison

| Statistic | Value |
|---|---:|
| Mean top-10 Jaccard overlap | 0.08264 |
| Median top-10 Jaccard overlap | 0.05263 |
| Mean sign agreement (overlapping nonzero features) | 0.40476 |

These are descriptive results for the selected 20-instance sample only and should not be generalized to the full test set. The explanatory features are **syscall 2-gram transitions** (e.g. `221_197`), not isolated syscall IDs.

## Additional Phase 3C analysis

**Additional reconstruction analysis — not a direct reproduction of the paper's perturbation experiment.** The intervention operates on raw 30-call syscall sequences (replacing the second syscall in a matched top-10 bigram occurrence with a documented neutral-replacement token) rather than the paper's feature-space class-exclusive/random replacement protocol.

| Method | Guided mean Δ | Random mean Δ | Guided flips | Random flips | Direction consistency | Actionable |
|---|---:|---:|---:|---:|---:|---:|
| LIME-compatible | 0.234926 | 0.156690 | 9/20 | 6/20 | 65% | 15/20 |
| Kernel SHAP | 0.352012 | 0.300972 | 9/20 | 8/20 | 95% | 20/20 |

This is a descriptive analysis on a 20-instance sample, with no statistical significance test performed; results measure model-output sensitivity under the documented intervention policy, not causal syscall importance.

## Reproducibility

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q
```

The frozen test suite was last verified at **67 passed**. Random seed **42** is used throughout (data split, TF-IDF/chi-square fitting, MLP training, LIME/SHAP sampling, perturbation selection).

Dataset acquisition, expected raw-data layout, and archive provenance are documented in the sections below and in `data/README.md`. Rerunning Phase 2.5/3A/3B/3C against a locally supplied copy of the raw data is expected to reproduce the same reconstruction protocol, but byte-identical outputs are not guaranteed beyond what the frozen reproducibility metadata (seeds, hashes) documents.

### One documented environment, explicit artifact directories

The repository uses a single documented Python environment (`requirements.txt`) for the whole canonical workflow:

```
dataset → Phase 3A → Phase 3A artifact directory → Phase 3B → Phase 3B artifact directory → Phase 3C
```

Because a serialized model/vectorizer/selector can become unloadable if pickled under a different NumPy/scikit-learn version than the one installed later, Phase 3B and Phase 3C do not hard-require the historical frozen artifact directories. Both accept explicit artifact-directory overrides — `--phase3a-dir` (Phase 3B and 3C), `--phase3b-dir` (Phase 3C), and `--experiment-dir` for each phase's own output — so a freshly regenerated Phase 3A/3B run (produced under this same documented environment) can feed the next phase directly, without a second virtual environment, manual package-version workarounds, or overwriting the historical frozen directories. Omitting these flags preserves the original default behavior (reading/writing the historical frozen directories).

Public raw ADFA-LD mirror used for the reconstruction:

`https://www.kaggle.com/datasets/yianqaq/adfa-ld`

Place the extracted dataset at `data/raw/ADFA-LD/` (or pass `--raw-data-dir` / the `ADFA_LD_RAW_DIR` environment variable). **Do not commit raw data.** Expected raw folders: `Training_Data_Master/` and `Validation_Data_Master/` (normal), `Attack_Data_Master/` (attack). See `data/README.md` for the recorded archive SHA-256 and extracted layout.

Frozen trace-manifest SHA-256: `765289887ac57f0a7f08b0d2761c9507e65f8ff98805ab69dc08a6cc312bf4f1`.

Commands for each stage (only needed if regenerating frozen artifacts, which is not required for Phase 1 submission):

```bash
# Phase 2.5 / dataset audit
python scripts/run_reconciliation.py --config configs/baseline.yaml --raw-data-dir /path/to/ADFA-LD --no-fit

# Phase 3A — MLP reconstruction (writes a fresh artifact directory; never overwrites the frozen one)
PYTHONPATH=. python scripts/run_phase3a.py --config configs/baseline.yaml --raw-data-dir /path/to/ADFA-LD --experiment-dir results/experiments/phase3a_rerun

# Phase 3B — LIME/SHAP explanations (reads the Phase 3A rerun above; writes its own fresh directory)
PYTHONPATH=. python scripts/run_phase3b.py --raw-data-dir /path/to/ADFA-LD --nsamples 512 --lime-num-samples 5000 \
    --phase3a-dir results/experiments/phase3a_rerun --experiment-dir results/experiments/phase3b_rerun

# Phase 3C — explanation-guided perturbation evaluation (reads both rerun directories above)
PYTHONPATH=. python scripts/run_phase3c.py --raw-data-dir /path/to/ADFA-LD \
    --phase3a-dir results/experiments/phase3a_rerun --phase3b-dir results/experiments/phase3b_rerun --experiment-dir results/experiments/phase3c_rerun
```

Omit the `--phase3a-dir`/`--phase3b-dir`/`--experiment-dir` flags to use the historical frozen directories instead (the original default behavior).

## Repository structure

| Path | Contents |
|---|---|
| `src/` | Core pipeline code: data loading, preprocessing, Phase 3A/3B/3C implementations, config, seeding, utilities. |
| `scripts/` | Command-line entry points that run each phase (`run_reconciliation.py`, `run_phase3a.py`, `run_phase3b.py`, `run_phase3c.py`). |
| `lime/` | Vendored LIME-compatible local-surrogate implementation used in place of the official package. |
| `tests/` | Reproducibility test suite (67 tests). |
| `configs/` | Frozen pipeline configuration (`baseline.yaml`) and the reconciliation-audit config. |
| `data/` | `README.md` with dataset provenance; `processed/` holds the frozen trace manifest; raw data is never committed. |
| `results/experiments/` | Frozen outputs: `phase3a_strict_reconstructed_mlp/` (model, predictions, metrics, metadata), `phase3b_xai_explanations/` (LIME/SHAP explanations, agreement stats), `phase3c_perturbation_evaluation/` (perturbation tables, figures, metadata). |
| `results/` (other) | `models/`, `figures/`, `tables/`, `logs/`, `explanations/` — supporting frozen outputs and audit artifacts. |
| `reports/` | Dataset audit (`phase2_5_dataset_audit.md`/`.json`) and the one-page `phase1_replication_summary.md`. |
| `docs/` | Paper extraction, dataset forensics, replication protocol, paper-vs-reconstruction comparison, and the LIME-verification plan. |
| `notebooks/` | Reserved for exploratory notebooks (currently empty). |

## Limitations

1. The exact 52,656-instance author-generated dataset was not recovered.
2. The complete original preprocessing/split protocol was not recovered.
3. The reconstruction uses public raw ADFA-LD data, not the authors' own generated dataset.
4. Observed vocabulary is 175 syscall IDs versus the paper's reported 149.
5. MLP framework and training settings are an independent reconstruction choice where the paper did not report complete details.
6. The official LIME package was unavailable offline; a documented vendored compatibility implementation was used instead.
7. The XAI comparison uses 20 selected test instances, not the full test set.
8. Phase 3C is an additional sensitivity analysis, not a reproduction of the paper's exact perturbation protocol.
9. Perturbation results are descriptive/model-sensitivity measurements, not causal evidence.
10. No statistical significance test was performed on the XAI or perturbation comparisons.

These are documented deviations from an unrecoverable exact replication, not project failures: the pipeline, model, and both explanation methods were implemented and evaluated end-to-end, and every deviation above is traceable to a documented, specific cause.

## Course Phase 1 deliverable

This repository addresses the course's Phase 1 requirements as follows:

- **Model implemented** — frozen scikit-learn MLP classifier (`src/phase3a.py`, `results/experiments/phase3a_strict_reconstructed_mlp/`).
- **XAI methods implemented** — vendored LIME-compatible reconstruction and actual Kernel SHAP (`src/phase3b.py`, `results/experiments/phase3b_xai_explanations/`).
- **Dataset reconstruction documented** — [`docs/paper_dataset_forensics.md`](docs/paper_dataset_forensics.md), `reports/phase2_5_dataset_audit.md`/`.json`.
- **2–3 main results reconstructed/analysed** — MLP classification result, LIME/SHAP top-10 explanations, and the additional Phase 3C perturbation-sensitivity analysis (see `docs/phase1_result_comparison.md`, Section E).
- **Paper-vs-reconstruction comparison** — [`docs/phase1_result_comparison.md`](docs/phase1_result_comparison.md).
- **GitHub-ready source code** — `src/`, `scripts/`, `tests/`.
- **Requirements** — `requirements.txt`.
- **Dataset/provenance instructions** — `data/README.md`, this README's Reproducibility section.
- **Random seed** — 42, used throughout and documented in `docs/phase1_result_comparison.md`.
- **Reproducibility tests** — `tests/` (67 passed).
- **One-page replication summary** — [`reports/phase1_replication_summary.md`](reports/phase1_replication_summary.md).

Additional documentation: [`AI_USAGE.md`](AI_USAGE.md) (AI-use and human-verification record), [`ASSUMPTIONS.md`](ASSUMPTIONS.md) (documented reconstruction assumptions), [`docs/lime_official_package_verification.md`](docs/lime_official_package_verification.md) (official-LIME limitation and future verification plan).

## References

Gaspar, D., Silva, P., & Silva, C. (2024). *Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron.* IEEE Access, 12, 30164–30175. DOI: [10.1109/ACCESS.2024.3368377](https://doi.org/10.1109/ACCESS.2024.3368377)

ADFA-LD (public raw mirror used for this reconstruction): `https://www.kaggle.com/datasets/yianqaq/adfa-ld` — see `data/README.md` for recorded provenance and archive checksum.
