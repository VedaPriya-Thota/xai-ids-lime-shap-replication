# Phase 2.5 / Phase 2.6 execution status

The real-data reconciliation audit has now been executed against the supplied local raw ADFA-LD mirror.
Phase 2.6 freezes and documents the strict independent reconstruction. No MLP, LIME, SHAP, balancing,
TF-IDF fitting, or chi-squared feature experiment was run as part of the frozen audit.

## Frozen baseline

**trace-disjoint stratified split, seed 42, 15% validation, 15% test, 30-call non-overlapping windows,
stride 30, incomplete tails dropped.**

No class weights, undersampling, oversampling, SMOTE, or other balancing is applied.

## Real-data audit

Raw source: public raw ADFA-LD mirror supplied for this project.

Observed inventory:

- Training_Data_Master: 833 valid traces
- Validation_Data_Master: 4,372 valid traces
- Attack_Data_Master: 746 valid traces
- Total valid traces: 5,951
- Hidden ignored files: 1 (`.DS_Store`)
- Blank, no-valid-call, malformed, unreadable, and too-short files: 0
- Raw syscall vocabulary: 175 distinct IDs, range 1–340

Baseline window results:

- 88,828 total windows
- 78,599 normal windows
- 10,229 attack windows
- 11.5155% attack windows

Trace-disjoint split:

| split | normal traces | attack traces | normal windows | attack windows |
|---|---:|---:|---:|---:|
| train | 3,643 | 522 | 54,671 | 6,882 |
| validation | 781 | 112 | 11,815 | 1,724 |
| test | 781 | 112 | 12,113 | 1,623 |

Integrity checks: trace-ID overlap = 0; duplicate `(trace_id, start)` windows = 0.

## Commands executed

Final test suite:

```text
python -m pytest tests/ -q
46 passed, exit 0
```

Real-data reconciliation (executed once for Phase 2.6 validation, with feature fitting disabled):

```text
python scripts/run_reconciliation.py --config configs/baseline.yaml --raw-data-dir /tmp/adfa/ADFA-LD --no-fit
exit 0
```

The raw override points to the supplied extracted archive and is equivalent in dataset content to the
configured `data/raw/ADFA-LD` source layout. `--no-fit` is intentional: Phase 2.6 does not create
TF-IDF/chi-squared feature matrices.

The generated reconciliation tables/figures/log are the actual outputs of that run. The Phase 2.6
artifact files were then generated from those completed audit outputs and the same raw trace inventory;
this artifact-generation step does not run the reconciliation experiments again.

## Audit artifacts

- `reports/phase2_5_dataset_audit.md`
- `reports/phase2_5_dataset_audit.json`
- `data/processed/trace_manifest.csv`
- `results/tables/reconciliation_*.csv` (9 tables)
- `results/figures/reconciliation_*.png` (4 figures)
- `results/logs/reconciliation.log`

The manifest contains trace metadata and split assignment only; it contains no raw syscall sequences.
The JSON records deterministic split-assignment SHA-256 hashes and a deterministic raw-tree SHA-256.

## Paper reconciliation

The paper references remain unchanged:

- 52,656 instances (reported)
- 26,870 attack instances (derived from TP + FN)
- 25,786 normal instances (derived from TN + FP)
- 149 syscall IDs (reported)
- 2,805 initial bigrams (reported)
- 150 selected features (reported)

The reconstruction is explicitly **an independent reconstruction, not the exact paper dataset**.
The 88,828-window result is not claimed to be the authors' dataset, and no data manipulation was
performed to approach the paper's counts.

Forensic provenance remains:

**OPTION C — RELATED CODE ONLY.**

The ARCADIAN-IoT / Device Behaviour Monitoring lineage is a verified project-level lead, but an exact
52,656-instance author dataset and paper-specific preprocessing pipeline were not recovered.

## Files and documentation updated in Phase 2.6

- `configs/baseline.yaml`
- `ASSUMPTIONS.md`
- `README.md`
- `data/README.md`
- `docs/replication_protocol.md`
- `docs/paper_dataset_forensics.md` retained/referenced
- `reports/PHASE2_5_EXECUTION_STATUS.md`
- `src/audit_artifacts.py`
- `src/reconcile_dataset.py`
- `scripts/run_reconciliation.py`
- `tests/test_reconcile_dataset.py`
- generated audit artifacts listed above

## Stop boundary

Phase 2.6 ends here. Do not train the MLP, fit the Phase 3 feature pipeline, run LIME/SHAP,
or perform balancing experiments until this frozen baseline and its reconciliation are reviewed.
