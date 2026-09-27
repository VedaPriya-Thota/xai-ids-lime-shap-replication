# Data

Raw ADFA-LD is **not included** and must never be committed. Obtain it from the official UNSW source
or an authorized mirror, and record which one you used and the download date below.

Place it under `data/raw/ADFA-LD/` (or point `data.raw_data_dir` / `--raw-data-dir` / the
`ADFA_LD_RAW_DIR` environment variable at wherever you extracted it). The loader searches this
folder (and one level of subdirectory below it) for the standard ADFA-LD directories rather than
requiring one exact layout:
- `Training_Data_Master/` and `Validation_Data_Master/` (labelled normal)
- `Attack_Data_Master/` (labelled attack)

## Public raw mirror used for Phase 2.5

Source: <https://www.kaggle.com/datasets/yianqaq/adfa-ld> (`archive.zip`).

This is a **public raw ADFA-LD mirror**, not the paper authors' own generated, windowed
52,656-instance dataset, and it is not proven to be byte-identical to the official UNSW
distribution. It is treated throughout this project as a mirror of the raw trace files only.
Third-party CSVs or notebooks bundled inside a Kaggle archive are not paper-author data and are
never read by the loader (only the three folders above are).

Before running the reconciliation audit or any later stage against a copy of this mirror, record
here (placeholders are intentionally retained when the value is not available):
- **Source used:** `https://www.kaggle.com/datasets/yianqaq/adfa-ld` (public raw ADFA-LD Kaggle mirror).
- **Download date:** not recorded in the available repository provenance.
- **Archive checksum** (supplied local `archive(1).zip`): `686825e85e2cd513e5b6462adcc55c3c6d81e8bf2bac24dd912a91fa014a6454`
- **Extracted folder structure** (output of `find data/raw/ADFA-LD -maxdepth 2 -type d`): top-level directories `Training_Data_Master/`, `Validation_Data_Master/`, `Attack_Data_Master/`.

Integrity/inventory checks (file counts, blank/malformed/too-short files, hidden files ignored) are
produced by `scripts/run_reconciliation.py` once real data is present; see
`results/tables/reconciliation_raw_inventory.csv` after running it.

Raw files must never be committed to this repository (see `.gitignore`), regardless of the
mirror's own license/size terms.
