# Phase 2.5 Real-Data Audit / Phase 2.6 Frozen Baseline

Generated: `2026-09-25T17:00:51.519460+00:00`  
Command: `python scripts/run_reconciliation.py --config configs/baseline.yaml --raw-data-dir /tmp/adfa/ADFA-LD --no-fit`

**This is an independent reconstruction, not the exact paper dataset.** The public raw ADFA-LD mirror is not claimed to be the authors' generated 52,656-instance dataset.

## Frozen baseline

**trace-disjoint stratified split, seed 42, 15% validation, 15% test, 30-call non-overlapping windows, stride 30, incomplete tails dropped.** No balancing, class weights, undersampling, oversampling, or SMOTE.

Configured raw path: `/tmp/adfa/ADFA-LD`  
Resolved raw root: `/tmp/adfa/ADFA-LD`  
Raw-tree SHA-256: `38454c054b5d50a082d52dca2e3b0c1bdafad8dafaf43085b42b966967383a13`

## Raw inventory

| Source | TXT files | Valid traces | Min | Q25 | Median | Q75 | Max | Hidden ignored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training_Data_Master | 833 | 833 | 79 | 131 | 201 | 410 | 2948 | 0 |
| Validation_Data_Master | 4372 | 4372 | 77 | 162.75 | 343 | 452 | 4494 | 1 |
| Attack_Data_Master (all families) | 746 | 746 | 75 | 139 | 290 | 560.75 | 2712 | 0 |
| **Total** | **5951** | **5951** | | | | | | **1** |

Blank/no-valid/malformed/unreadable/too-short files: **0/0/0/0/0**.

## Vocabulary

Observed: **175 IDs**, range **1–340**. Paper reference: **149 IDs (reported; reference only)**.

## Windows and splits

Baseline windows: **88,828 total = 78,599 normal + 10,229 attack (11.52% attack)**.

| Split | Normal traces | Attack traces | Normal windows | Attack windows |
|---|---:|---:|---:|---:|
| train | 3643 | 522 | 54671 | 6882 |
| validation | 781 | 112 | 11815 | 1724 |
| test | 781 | 112 | 12113 | 1623 |

## Integrity checks

- Trace-ID overlap across splits: **0**
- Duplicate `(trace_id, start)` windows: **0**
- Manifest rows: **5,951**
- Manifest window sum matches baseline: **True**
- Split window sum matches baseline: **True**
- Raw data read-only: **True**

## Paper-vs-reconstruction reconciliation

| Quantity | Paper | Reconstruction | Status |
|---|---:|---:|---|
| Instances | 52,656 | 88,828 | reported paper value; independent reconstruction |
| Attack instances | 26,870 | 10,229 | derived paper value (TP+FN) |
| Normal instances | 25,786 | 78,599 | derived paper value (TN+FP) |
| Syscall IDs | 149 | 171 | reported paper value |
| Initial bigrams | 2,805 | not fitted | reported paper value |
| Selected features | 150 | not fitted | reported paper value |

The reconstruction is not altered to approach paper counts.

## Unresolved limitations

1. Authors' generated dataset and exact preprocessing/split artifacts were not recovered; forensic provenance remains **OPTION C — RELATED CODE ONLY**.
2. The raw mirror does not reproduce 52,656 instances under the frozen non-overlapping policy.
3. The raw mirror has 175 observed syscall IDs versus 149 reported.
4. Paper class supports are derived from the reported confusion matrix.
5. No MLP, TF-IDF/chi-squared model experiment, LIME, or SHAP was run in this audit.

## Artifacts

- `reports/phase2_5_dataset_audit.md`
- `reports/phase2_5_dataset_audit.json`
- `data/processed/trace_manifest.csv`
