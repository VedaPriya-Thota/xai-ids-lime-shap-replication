# XAI for IDS: LIME and SHAP on MLP

## Project objective

This course project reconstructs and evaluates the methodology of:

> Gaspar, D., Silva, P., & Silva, C. (2024). *Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron.* IEEE Access, 12, 30164–30175. DOI: `10.1109/ACCESS.2024.3368377`.

The project focuses on a 30-system-call representation, adjacent syscall 2-grams, TF-IDF, chi-square feature selection to 150 features, an MLP classifier, LIME/SHAP explanations, and explanation-guided perturbation analysis.

## Current status

**Phase 1 complete; Phase 2 not started.**

The repository also contains the frozen reconstruction evidence generated during the completed technical baseline/XAI stages. No new Phase 2 research experiment is started by this documentation finalization.

> **Strict reconstruction disclaimer:** this project does **not** contain the exact author-generated 52,656-instance dataset. The exact dataset and complete author preprocessing/split procedure were not recovered. Results are therefore independent reconstruction results, not exact numerical replication of the paper.

## Data acquisition

Public raw ADFA-LD mirror used for the reconstruction:

`https://www.kaggle.com/datasets/yianqaq/adfa-ld`

Place the extracted dataset at:

`data/raw/ADFA-LD/`

or pass another location using `--raw-data-dir` or the `ADFA_LD_RAW_DIR` environment variable. **Do not commit raw data.** Record the archive SHA-256, download date, source, and extracted layout in `data/README.md`. The supplied archive checksum used for the frozen reconstruction is recorded there.

Expected raw folders:

- `Training_Data_Master/` — normal
- `Validation_Data_Master/` — normal
- `Attack_Data_Master/` — attack

## Reproducibility protocol

The frozen reconstruction uses:

- 30 calls per window;
- stride 30 and non-overlapping windows;
- incomplete tails dropped;
- trace-disjoint stratified splitting;
- seed **42**;
- 15% validation and 15% test fractions;
- no class weighting or resampling;
- train-only TF-IDF fitting;
- train-only chi-square selection to 150 features.

The frozen trace-manifest SHA-256 is:
`765289887ac57f0a7f08b0d2761c9507e65f8ff98805ab69dc08a6cc312bf4f1`.

## Environment and installation

Use the Python environment specified by `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

The requirements file contains the project's pinned environment. Note that the completed offline Phase 3B/3C execution could not install the official LIME package; the repository therefore uses the vendored compatibility implementation for those frozen results. See `docs/lime_official_package_verification.md`.

## Commands

Run commands from the repository root.

### 1. Tests

```bash
python -m pytest tests/ -q
```

The frozen test suite was last verified at **60 passed**.

### 2. Phase 2.5 / dataset audit

```bash
python scripts/run_reconciliation.py --config configs/baseline.yaml --raw-data-dir /path/to/ADFA-LD --no-fit
```

### 3. Phase 3A — frozen MLP reconstruction

```bash
PYTHONPATH=. python scripts/run_phase3a.py --config configs/baseline.yaml --raw-data-dir /path/to/ADFA-LD
```

### 4. Phase 3B — frozen LIME/SHAP explanations

```bash
PYTHONPATH=. python scripts/run_phase3b.py --raw-data-dir /path/to/ADFA-LD --nsamples 512 --lime-num-samples 5000
```

**Important:** the LIME output in the current offline environment is from the vendored LIME-compatible implementation, not official `lime==0.2.0.1`.

### 5. Phase 3C — explanation-guided perturbation evaluation

```bash
PYTHONPATH=. python scripts/run_phase3c.py --raw-data-dir /path/to/ADFA-LD
```

Do not rerun these commands merely to reproduce the documentation unless you intend to regenerate the frozen experimental artifacts. The Phase 1 finalization task itself does not require experiment reruns.

## Important outputs

- `results/experiments/phase3a_strict_reconstructed_mlp/` — frozen MLP model, predictions, metrics, and reproducibility metadata.
- `results/experiments/phase3b_xai_explanations/` — selected instances, LIME-compatible explanations, SHAP explanations, agreement statistics, and metadata.
- `results/experiments/phase3c_perturbation_evaluation/` — perturbation tables, paired comparisons, figures, and metadata.
- `reports/phase2_5_dataset_audit.md` / `.json` — dataset audit.
- `data/processed/trace_manifest.csv` — frozen trace manifest.

## Documentation

- [`docs/paper_dataset_forensics.md`](docs/paper_dataset_forensics.md) — dataset/provenance investigation.
- [`docs/phase1_result_comparison.md`](docs/phase1_result_comparison.md) — paper-versus-reconstruction comparison.
- [`reports/phase1_replication_summary.md`](reports/phase1_replication_summary.md) — concise replication summary.
- [`docs/lime_official_package_verification.md`](docs/lime_official_package_verification.md) — official-LIME limitation and future verification procedure.
- [`AI_USAGE.md`](AI_USAGE.md) — AI-use and human-verification record.
- [`ASSUMPTIONS.md`](ASSUMPTIONS.md) — documented reconstruction assumptions.

## Expected limitations

1. The exact 52,656-instance author-generated dataset was unavailable.
2. The exact paper preprocessing and split protocol was not fully recoverable.
3. The completed LIME explanations use a vendored compatibility implementation because the official package was unavailable offline.
4. The 20-instance XAI/perturbation sample is not the full test set.
5. Perturbation results are descriptive model-sensitivity measurements, not causal evidence.
6. Paper and reconstruction metrics are not directly numerically comparable because their datasets/protocols differ.

See `docs/phase1_result_comparison.md` for the complete comparison and explicit claims the project does not make.
