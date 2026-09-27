# Paper Dataset Forensics

## 1. Objective

This document records a bounded forensic investigation of the public/project provenance relevant to the dataset-generation procedure used in:

Diogo Gaspar, Paulo Silva, Catarina Silva (2024), “Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron,” IEEE Access, DOI: 10.1109/ACCESS.2024.3368377.

The objective is **not** to make the local dataset reach 52,656 instances. The objective is to determine what public evidence exists for the authors' or ARCADIAN project's actual data/preprocessing procedure, whether a paper-specific 52,656-instance artifact can be recovered, and which parts can defensibly be connected to the target paper.

Scope was deliberately limited to:
- inspection of the supplied repository snapshot without changing its source/config/tests/raw data;
- inspection of the supplied Phase 2.5 documentation and tests;
- public inspection of the named GitHub/GitLab/ARCADIAN provenance leads;
- searches for the paper's numerical and implementation fingerprints;
- classification of candidate evidence as A/B/C/D.

No MLP, LIME, SHAP, balancing, class weighting, SMOTE, oversampling, undersampling, or new preprocessing experiment was executed.

## 2. Target Paper

Target paper:

- **Authors:** Diogo Gaspar, Paulo Silva, Catarina Silva
- **Year:** 2024
- **Title:** Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron
- **Venue:** IEEE Access, Vol. 12, pp. 30164–30175
- **DOI:** 10.1109/ACCESS.2024.3368377
- **Paper URL:** https://doi.org/10.1109/ACCESS.2024.3368377

The paper fingerprint supplied for this forensic pass is:

| Fingerprint | Value | Status |
|---|---:|---|
| Dataset | generated version of ADFA-LD | directly reported in the paper |
| Instances | 52,656 | directly reported in the paper |
| Calls per instance | 30 | directly reported in the paper |
| Distinct system calls | 149 | directly reported in the paper |
| Representation | 2-gram | directly reported in the paper |
| TF-IDF | yes | directly reported in the paper |
| Initial n-grams | 2,805 | directly reported in the paper |
| Chi-square feature selection | yes | directly reported in the paper |
| Final selected features | 150 | directly reported in the paper |
| Model | MLP | directly reported in the paper |
| XAI | LIME and SHAP | directly reported in the paper |
| TP | 24,038 | directly reported in the paper |
| FN | 2,832 | directly reported in the paper |
| TN | 25,311 | directly reported in the paper |
| FP | 475 | directly reported in the paper |
| Attack support | 26,870 | derived from TP + FN; not independently reported |
| Normal support | 25,786 | derived from TN + FP; not independently reported |

The inferred supports are therefore 26,870 attack and 25,786 normal, with an inferred attack share of approximately 51.03%. These support values are **derived fingerprints**, not separately reported class-support statements.

A separate public ARCADIAN-IoT project record lists the target IEEE Access paper among the project's peer-reviewed publications and identifies the same three authors and DOI. This establishes the paper's project context but does not by itself expose the paper's dataset-generation code. Source: https://cordis.europa.eu/project/id/101020259/results

## 3. Current Repository State

The inspected repository was the supplied ZIP snapshot:

`xai-ids-lime-shap-replication (1).zip`

The ZIP contains the replication scaffold and Phase 2.5 diagnostic implementation, including:

- `README.md`
- `ASSUMPTIONS.md`
- `docs/paper_extraction.md`
- `docs/replication_protocol.md`
- `data/README.md`
- `configs/baseline.yaml`
- `configs/reconciliation.yaml`
- `src/config.py`
- `src/data.py`
- `src/preprocess.py`
- `src/reconcile_dataset.py`
- `src/seed.py`
- `src/utils.py`
- `scripts/run_reconciliation.py`
- `tests/conftest.py`
- `tests/test_reconcile_dataset.py`
- `reports/PHASE2_5_EXECUTION_STATUS.md`

The ZIP does **not** contain the real Phase 2.5 reconciliation CSV/PNG/log outputs or the raw ADFA-LD files. The `results/tables`, `results/figures`, `results/logs`, `data/raw`, `data/processed`, and `data/artifacts` locations in the ZIP are represented only by `.gitkeep` files.

This is important because the current verified local Phase 2.5 results supplied for this investigation were produced outside the supplied ZIP snapshot. The ZIP's own documentation describes an earlier state in which the real-data audit had not yet been executed. Therefore, the current verified counts supplied for this forensic task are treated as **existing local audit evidence supplied by the project owner**, while the ZIP is treated as a repository snapshot that must not be mistaken for the latest audit state.

### Raw trace discovery

`src/data.py` implements:

1. `find_adfa_root()` checks the configured raw directory itself and exactly one immediate child directory.
2. The root must contain the configured normal directories and attack directory.
3. The loader recursively scans the three configured dataset groups.
4. Only non-hidden `.txt` files are parsed.
5. Hidden files are counted as ignored.
6. Non-`.txt` files are skipped by the loader.
7. Trace identifiers are relative-path based:
   `<source_group>/<relative_path_without_extension>`.
8. Normal labels are `0`; attack labels are `1`.
9. Whitespace-separated non-negative integer syscall IDs are retained.
10. malformed/negative tokens are dropped and counted.
11. unreadable files are marked unreadable.
12. no raw data are written or downloaded by the loader.

The project documentation therefore describes the raw dataset as read-only and locally supplied.

### Labels

`src/data.py` assigns:
- `Training_Data_Master` and `Validation_Data_Master` as normal through the configured normal directories;
- `Attack_Data_Master` as attack;
- label `0` for normal and `1` for attack.

The assignment is configuration-driven, not inferred from trace contents.

### Window generation

`src/data.py` creates windows using:

- `window_size` from `data.window_size`;
- `stride` from `data.stride`;
- starts `0, stride, 2*stride, ...` while a complete `window_size` segment remains.

The supplied baseline configuration contains `window_size: 30` and `stride: 30`.

The implementation therefore drops incomplete trailing calls rather than padding them.

### Split process

The supplied ZIP snapshot has an important historical/configuration discrepancy.

`src/data.py` implements only `stratified_group_by_trace`. It first stratifies **trace IDs** and then assigns all windows from each trace to the same partition.

However, the supplied `configs/baseline.yaml` still contains:

`group_by: window`

and `ASSUMPTIONS.md` describes a random window-level split as an open decision.

The Phase 2.5 test suite, meanwhile, explicitly tests the primary policy as `stratified_group_by_trace`, and the current local Phase 2.5 results supplied for this investigation report zero trace-ID overlap.

Therefore the **supplied ZIP snapshot contains stale/inconsistent documentation relative to the current verified baseline**. The current forensic investigation does not change that baseline and does not attempt to reconcile it by editing the repository.

### Outputs

`build_from_config()` constructs the trace/window tables in memory and writes nothing.

The reconciliation runner is explicitly designed to write only:

- `results/tables/reconciliation_*.csv`
- `results/figures/reconciliation_*.png`
- `results/logs/reconciliation.log`

The runner does not write to `data/processed` or `data/artifacts`, and it does not modify `configs/baseline.yaml`.

## 4. Completed Local Phase 2.5 Evidence

The current verified Phase 2.5 evidence supplied for this forensic investigation is:

### Tests

`python -m pytest tests/ -q`

Result:

- **45 passed**
- exit code `0`

The tests use synthetic ADFA-LD-shaped fixtures. They do not establish equivalence to the paper dataset.

### Real-data reconciliation

`python scripts/run_reconciliation.py --config configs/baseline.yaml`

Current verified results supplied for this investigation:

- `Training_Data_Master`: 833 valid `.txt` traces
- `Validation_Data_Master`: 4,372 valid `.txt` traces
- `Attack_Data_Master`: 746 valid `.txt` traces
- attack traces are distributed across 61 run-folders covering Adduser, Hydra_FTP, Hydra_SSH, Java_Meterpreter, Meterpreter, and Web_Shell
- total valid traces: 5,951
- normal traces: 5,205
- attack traces: 746
- no blank, malformed, unreadable, or too-short files
- one hidden `.DS_Store` was ignored
- raw syscall vocabulary: 175 distinct IDs
- ID range: 1–340

Current verified primary baseline:

- window size: 30
- stride: 30
- incomplete tails dropped
- non-overlapping windows
- trace-disjoint stratified split
- validation fraction: 0.15
- test fraction: 0.15
- seed: 42
- no class weights
- no undersampling
- no oversampling
- no SMOTE
- no balancing

Current verified windows:

- total: 88,828
- normal: 78,599
- attack: 10,229
- attack proportion: 11.52%

Split counts:

| Split | Normal | Attack | Total |
|---|---:|---:|---:|
| Train | 54,671 | 6,882 | 61,553 |
| Validation | 11,815 | 1,724 | 13,539 |
| Test | 12,113 | 1,623 | 13,736 |

Integrity:

- trace-ID overlap across splits: 0
- duplicate `(trace_id, start)` window pairs: 0

The local raw mirror therefore does not reproduce:

- the paper's 52,656 total instances;
- the paper-derived near-balanced class support;
- the paper's 149-ID vocabulary.

This discrepancy is preserved as evidence. No attempt was made to force the local data toward the paper's count.

### What the 45 tests validate

The supplied tests cover:

- window-count and start-index arithmetic for multiple strides;
- covered syscall-pair and position masks;
- non-overlap boundary handling;
- dropping windows containing excluded vocabulary IDs;
- top-k syscall ordering;
- raw inventory accounting for blank, malformed, short, hidden and non-text files;
- external reference counts being recorded rather than enforced;
- scenario counts against brute-force implementations;
- source-group scenario accounting;
- trace-length filtering without deleting data;
- dropped partial trailing windows;
- top-k/UNK vocabulary diagnostics;
- TF-IDF and chi-square fitting behavior;
- deterministic feature fitting;
- deterministic subsampling diagnostics;
- sampled real-window construction;
- use of the primary trace-disjoint split;
- skipped fitting when a scenario is too small;
- syscall frequency tables;
- balance-feasibility arithmetic;
- descriptive distance calculations;
- summary-table structure/content;
- invalid reconciliation-config rejection;
- output generation for reconciliation;
- preservation of preprocessing/artifact directories;
- missing-dataset error behavior.

These tests validate implementation logic against synthetic fixtures and brute-force calculations. They do **not** validate the authors' dataset-generation procedure.

## 5. Public Provenance Trail

The principal public trail consists of:

1. **Target IEEE Access paper**
   - DOI: https://doi.org/10.1109/ACCESS.2024.3368377
   - The target paper reports the 52,656-instance generated ADFA-LD dataset and its 30-call / 2-gram / TF-IDF / chi-square / 150-feature pipeline.

2. **ARCADIAN-IoT project results**
   - URL: https://cordis.europa.eu/project/id/101020259/results
   - The European Commission CORDIS record lists the target paper as an ARCADIAN-IoT project publication with the same three authors and DOI.
   - This establishes project context, not the exact source-code pipeline.

3. **vitalinarh / Host-Intrusion-Detection-System**
   - URL: https://github.com/vitalinarh/Host-Intrusion-Detection-System
   - Public GitHub repository.
   - Current indexed page shows `main`, 29 commits, and the top-level paths `data/`, `federated/`, `resources/`, `trace_module/`, `.gitignore`, `README.md`, `dbm.py`, and `requirements.txt`.
   - The README describes a Device Behaviour Monitoring system developed in the ARCADIAN-IoT scope and based on system-call analysis for IoT devices.
   - The README instructs users to clone `https://github.com/vitalinarh/device_behaviour_monitoring`, install Perf Linux tools and Python requirements, and run `dbm.py`.
   - The README exposes configuration concepts including `DETECTION_THRESHOLD`, `SYSCALL_LIMIT`, `PAUSE_TIME`, `SAVE_SYSCALLS`, `FILTER_OUT`, and `FILTER_IN`.
   - Public page: https://github.com/vitalinarh/Host-Intrusion-Detection-System

4. **vitalinarh / device_behaviour_monitoring**
   - URL: https://github.com/vitalinarh/device_behaviour_monitoring
   - This exact GitHub URL is referenced by the Host-Intrusion-Detection-System README.
   - The repository itself was not directly retrievable through the available public-web fetch path during this investigation; therefore its nested source files, commit hashes, branch history, tags and artifacts were not independently inspected.
   - The related 2025 paper by Diogo Gaspar and Paulo Silva explicitly cites “Device Behaviour Monitoring” at `https://gitlab.com/arcadian_iot/device_behaviour_monitoring` as reference 24, and the same paper is within the ARCADIAN-IoT context. This is strong project-level linkage, but it does not prove that the 2024 XAI paper's exact dataset pipeline is contained in the repository.

5. **ARCADIAN-IoT GitLab lead**
   - URL: https://gitlab.com/arcadian_iot/device_behaviour_monitoring
   - The repository was not directly retrievable in this web session.
   - No exact branch/tag/commit, preprocessing file contents, dataset artifact, or serialized model could therefore be verified from this source.

6. **Related 2025 system-call IDS publication**
   - URL: https://link.springer.com/article/10.1007/s10207-024-00926-9
   - Authors include Vitalina Holubenko, Diogo Gaspar, Rúben Leal and Paulo Silva.
   - The publication references the ARCADIAN-IoT Device Behaviour Monitoring project and describes an IoT HIDS based on system-call traces.
   - This provides additional project/author context but is not the target 2024 paper's implementation artifact.

The public evidence therefore establishes a coherent ARCADIAN-IoT/system-call-monitoring lineage, but not a defensible byte-level or code-level identity between the public monitoring project and the target paper's 52,656-row generated dataset.

## 6. Repositories and Versions Inspected

### 6.1 `vitalinarh/Host-Intrusion-Detection-System`

**URL:** https://github.com/vitalinarh/Host-Intrusion-Detection-System

**Visible branch:** `main`

**Visible repository history:** 29 commits

**Visible tags/releases:** the repository page exposes branch/tag navigation, but the branch/tag/release pages were not retrievable through the available web fetch path. No hash is recorded here because none was independently verified.

**Visible top-level paths:**

- `data/`
- `federated/`
- `resources/`
- `trace_module/`
- `.gitignore`
- `README.md`
- `dbm.py`
- `requirements.txt`

**Observed behavior/content:**

- ARCADIAN-IoT project context is explicitly stated.
- Device Behaviour Monitoring is described as an IDS based on system-call analysis.
- Federated Learning is part of the stated system.
- Perf Linux Tool is required for system-call log extraction.
- `dbm.py` is the stated execution entry point.
- Runtime configuration includes intrusion threshold and system-call queue/tracing controls.
- The repository page labels the project with topics including `mlp`, `perf`, `system-calls`, `ids`, and `hids`.

**Access limitations:**

- Direct clicks to `dbm.py`, `data/`, `resources/`, `trace_module/`, commit history, branches and tags returned cache-miss/fetch failures.
- Consequently, no nested file content, commit hash, commit date, branch-specific implementation, tag-specific implementation, or artifact contents are claimed.
- The `.gitignore` file is visible as a path but its contents were not retrieved.
- No dataset artifact was visible at the indexed top level.

### 6.2 `vitalinarh/device_behaviour_monitoring`

**URL:** https://github.com/vitalinarh/device_behaviour_monitoring

**Branch/tag/commit:** unresolved.

**Relevant path named by the public README:** `dbm.py` at repository root.

**Observed behavior/content:** only the clone/run reference from the Host-Intrusion-Detection-System README was independently visible.

**Access limitations:** direct repository retrieval was unavailable through the current web access path. No nested source, history, branches, tags, releases, serialized artifacts or dataset files were verified.

### 6.3 ARCADIAN-IoT GitLab `device_behaviour_monitoring`

**URL:** https://gitlab.com/arcadian_iot/device_behaviour_monitoring

**Branch/tag/commit:** unresolved.

**Relevant lead:** the project is explicitly cited as “Device Behaviour Monitoring” in the related 2025 publication by Diogo Gaspar and Paulo Silva.

**Access limitations:** repository content was not directly retrievable in this session. No source file, commit, tag, release, artifact or dataset was therefore promoted to stronger provenance than the public citation supports.

### 6.4 ARCADIAN-IoT project record

**URL:** https://cordis.europa.eu/project/id/101020259/results

**Version:** project record; no repository commit version.

**Observed behavior/content:** lists the target IEEE Access paper by Diogo Gaspar, Paulo Silva and Catarina Silva, including DOI `10.1109/ACCESS.2024.3368377`.

**Access limitations:** project-level publication metadata does not contain the target paper's source preprocessing pipeline.

### 6.5 Supplied replication repository snapshot

**Artifact:** `xai-ids-lime-shap-replication (1).zip`

**Version/commit:** not available in the ZIP; no `.git/` directory was present.

**Observed paths:** see Section 3.

**Access limitation:** this is a ZIP snapshot, not the live working tree. Its documentation is stale relative to the current verified Phase 2.5 state supplied in the task.

## 7. Search Terms and Artifacts Investigated

The public-web investigation used the requested provenance leads and searches centered on:

- `ADFA`
- `ADFA-LD`
- `52656`
- `52,656`
- `26870`
- `25786`
- `149`
- `2805`
- `150`
- `30`
- `2-gram`
- `ngram`
- `TfidfVectorizer`
- `TF-IDF`
- `chi2`
- `SelectKBest`
- `LIME`
- `SHAP`
- `KernelExplainer`
- `MLP`
- `system call`
- `trace`
- `preprocess`
- `dataset`
- `feature`
- `training`
- `validation`
- `attack`
- `normal`
- `data`
- `resources`
- `trace_module`
- `dbm.py`
- `preprocessing`
- `dataset loaders`
- `notebooks`
- `csv`
- `npy`
- `npz`
- `pkl`
- `joblib`
- `json`
- `.gitignore`
- Git LFS
- historical commits
- branches
- tags
- releases
- `Diogo Gaspar`
- `Paulo Silva`
- `Catarina Silva`
- `IEEE Access`
- `10.1109/ACCESS.2024.3368377`
- `ARCADIAN-IoT`
- `device_behaviour_monitoring`
- `Host-Intrusion-Detection-System`

The requested serialized artifact classes (`.csv`, `.npy`, `.npz`, `.pkl`, `.joblib`, `.json`) were not recovered as paper-specific dataset artifacts through the accessible public index.

## 8. Author/Project Preprocessing Found

### Found: project-level system-call acquisition/monitoring

The public Host-Intrusion-Detection-System page describes a Device Behaviour Monitoring system in the ARCADIAN-IoT project, using system-call analysis for IoT intrusion detection. It requires Perf Linux Tool for system-call log extraction and exposes configuration for syscall tracing/saving and intrusion detection.

This is genuine **related project preprocessing/collection infrastructure**, but it is not the same thing as proving the target paper's 52,656-instance feature-generation pipeline.

### Not found: verified target-paper preprocessing code

No accessible public source was found that could be defensibly quoted as implementing all of:

1. conversion of ADFA-LD traces into exactly 30-call instances;
2. the exact paper's instance-generation policy;
3. a resulting 52,656-row dataset;
4. 2-gram extraction yielding exactly 2,805 initial features;
5. TF-IDF;
6. chi-square selection to exactly 150 features;
7. the exact paper dataset used for the reported confusion matrix.

Similarity at the conceptual level is not enough to promote the evidence to Category A.

### Important provenance distinction

The ARCADIAN project relationship is real and relevant:

- the target paper is listed as an ARCADIAN-IoT project publication;
- a related 2025 Diogo Gaspar / Paulo Silva publication cites Device Behaviour Monitoring;
- the public Device Behaviour Monitoring repository describes system-call IDS infrastructure.

However, none of those facts proves that the 2024 paper's generated 52,656-instance dataset was produced by the currently public repository code.

## 9. Candidate Dataset Artifacts

| Candidate | Evidence found | Status |
|---|---|---|
| Public raw ADFA-LD mirror used locally | 5,951 valid traces; 175 observed syscall IDs; 88,828 baseline windows | related raw source, not paper artifact |
| `data/` in Host-Intrusion-Detection-System | top-level directory visible | contents inaccessible; no dataset artifact verified |
| `resources/` in Host-Intrusion-Detection-System | top-level directory visible | contents inaccessible; no paper artifact verified |
| `trace_module/` in Host-Intrusion-Detection-System | top-level directory visible | contents inaccessible; system-call monitoring is described by README |
| root `dbm.py` | visible as a source path | source contents inaccessible through current fetch path |
| `device_behaviour_monitoring/dbm.py` | referenced by public README clone/run instructions | exact source inaccessible |
| GitLab Device Behaviour Monitoring | publicly named project and cited by related paper | exact artifact inaccessible |
| 52,656-row CSV/NPY/NPZ/PKL/joblib/JSON artifact | no verified public artifact found | unresolved / not recovered |
| 2,805-feature artifact | no verified paper-specific artifact found | unresolved |
| 150-feature artifact tied to the paper | no verified artifact found | unresolved |

**No 52,656-row artifact was recovered.**

## 10. Provenance Classification

The requested strict categories are applied as follows.

### A. VERIFIED PAPER ARTIFACT

**None recovered.**

There is no defensibly proven 52,656-instance file or paper-specific source pipeline available from the accessible public evidence.

### B. VERIFIED AUTHOR/PROJECT CODE, BUT NOT PROVEN PAPER PIPELINE

**ARCADIAN-IoT / Device Behaviour Monitoring lineage — Category B.**

Evidence supporting B:

- the target paper is an ARCADIAN-IoT project publication;
- the public Host-Intrusion-Detection-System repository explicitly describes Device Behaviour Monitoring as ARCADIAN-IoT work;
- that repository is centered on system-call analysis and includes the `trace_module`, `data`, `resources`, `dbm.py`, and federated components;
- the Host-Intrusion-Detection-System README points to `vitalinarh/device_behaviour_monitoring`;
- a related 2025 publication by Diogo Gaspar and Paulo Silva cites Device Behaviour Monitoring directly and is itself in the same ARCADIAN-IoT/system-call IDS research lineage.

Why it remains B rather than A:

- the exact 2024 paper dataset is not exposed;
- the exact 52,656-row artifact is not recovered;
- the exact 30-call instance-generation procedure is not verified in accessible source;
- the exact 2,805-to-150 feature-generation pipeline is not verified in accessible source;
- no verified commit was tied to the paper's experiment date;
- the repository owner/name visible in the accessible GitHub page is `vitalinarh`, not the target paper's author list;
- the nested source and history could not be independently fetched in this session.

### C. RELATED ADFA-LD PROCESSING

The public ADFA-LD repositories/projects found through general searches are not promoted to author/paper provenance merely because they use ADFA-LD, system calls, n-grams, or intrusion detection.

In particular, generic ADFA-LD processing projects using LSTM, 3-grams, padding, or balancing are not evidence for the target paper's pipeline.

### D. UNRELATED

General XAI/IDS repositories using LIME/SHAP with CICIDS, NSL-KDD, UNSW-NB15, or unrelated network-flow features are unrelated to the target paper's ADFA-LD provenance and were not used as evidence.

## 11. Paper Numerical Fingerprints

The numerical fingerprints are intentionally separated by evidence type.

| Value | Classification | Meaning |
|---:|---|---|
| 52,656 | directly reported in the paper | total instances |
| 30 | directly reported in the paper | system calls per instance |
| 149 | directly reported in the paper | distinct system calls |
| 2,805 | directly reported in the paper | initial 2-gram feature count |
| 150 | directly reported in the paper | final selected feature count |
| 24,038 | directly reported in the paper | TP |
| 2,832 | directly reported in the paper | FN |
| 25,311 | directly reported in the paper | TN |
| 475 | directly reported in the paper | FP |
| 26,870 | derived from paper-reported values | TP + FN |
| 25,786 | derived from paper-reported values | TN + FP |
| 51.03% | derived from paper-reported values | 26,870 / 52,656 |
| 5,951 | observed in our local raw dataset | valid raw traces |
| 5,205 | observed in our local raw dataset | valid normal traces |
| 746 | observed in our local raw dataset | valid attack traces |
| 175 | observed in our local raw dataset | raw syscall vocabulary |
| 1–340 | observed in our local raw dataset | raw syscall ID range |
| 88,828 | observed in our local raw dataset | current primary baseline windows |
| 78,599 | observed in our local raw dataset | current primary normal windows |
| 10,229 | observed in our local raw dataset | current primary attack windows |
| 11.52% | observed in our local raw dataset | current primary attack proportion |
| 2,805 from public author/project code | unresolved | no accessible exact artifact/code evidence |
| 52,656-row artifact | unresolved | not recovered |
| 149-ID author-generated vocabulary | unresolved | not recovered from author code |

The public ARCADIAN code trail is therefore **qualitatively related but numerically unverified** against the paper fingerprint.

## 12. Exact 52,656-Instance Dataset Status

**Status: NOT RECOVERED.**

No defensibly proven 52,656-instance artifact was found.

The current local reconstruction produces 88,828 windows under the deliberately fixed primary policy. This is not evidence that the paper is wrong, and it is not evidence that the local reconstruction should be changed to 52,656.

The difference is diagnostically useful:

- local raw trace count is 5,951;
- local raw syscall vocabulary is 175, not 149;
- local non-overlapping 30-call window count is 88,828, not 52,656;
- local inferred attack share is 11.52%, while the paper's confusion-matrix-derived support is approximately 51.03% attack.

These observations demonstrate that the paper's **generated version** of ADFA-LD is not reproduced by the current raw-mirror/non-overlapping reconstruction.

They do not identify the missing generation operation.

Possible mechanisms such as overlap, filtering, balancing, sampling, alternative labeling, or different source data are hypotheses only and must not be presented as the authors' procedure without evidence.

## 13. What Can Be Reproduced

The following can be reproduced or verified at the level supported by current evidence:

1. The target paper's public numerical fingerprint.
2. The existence of an ARCADIAN-IoT project relationship.
3. The existence of a public Device Behaviour Monitoring/system-call IDS code lineage.
4. The public Host-Intrusion-Detection-System repository's top-level structure and README-described system behavior.
5. The local raw ADFA-LD mirror inventory.
6. The current primary trace-disjoint 30-call/stride-30 baseline.
7. The current Phase 2.5 test suite behavior and its synthetic-fixture validation.
8. The fact that the local raw mirror does not reproduce the paper's 52,656/149/near-balanced fingerprint under the current baseline.

## 14. What Cannot Yet Be Reproduced

The following remain unresolved:

1. The exact authors' 52,656-instance generated dataset.
2. The exact instance-generation rule.
3. The exact treatment of partial traces/windows.
4. The exact split construction used for the paper's reported 52,656 counts.
5. Any resampling/balancing/filtering operation, if one existed.
6. The exact route from raw ADFA-LD traces to exactly 149 observed system calls.
7. The exact 2,805 initial bigram feature vocabulary.
8. The exact train/test or full-dataset feature-fitting boundary used by the authors.
9. The exact serialized dataset or preprocessing artifacts.
10. The exact author commit/version corresponding to the 2024 experiment.
11. The exact package versions and execution environment.
12. A code-level proof that the public Device Behaviour Monitoring repository generated the target paper dataset.

No missing item has been filled by assumption.

## 15. Recommended Next Step

**Do not change the primary baseline and do not start MLP/LIME/SHAP/Phase 3 yet.**

The next bounded action should be a **source-level provenance recovery pass on an actual clone/snapshot of the two Device Behaviour Monitoring repositories**, because the public web interface currently exposes the project relationship and top-level repository structure but does not permit reliable inspection of the nested source/history required to prove or disprove paper equivalence.

Specifically, obtain an inspectable repository snapshot for:

- `https://github.com/vitalinarh/device_behaviour_monitoring`
- `https://gitlab.com/arcadian_iot/device_behaviour_monitoring`

and then inspect, without executing preprocessing:

- `dbm.py`
- `trace_module/`
- `resources/`
- `data/`
- preprocessing/dataset modules
- notebooks
- `.gitignore`
- Git LFS pointers
- serialized artifacts
- all branches/tags/releases
- commits around 2023–2024

The critical forensic search should be for an exact chain:

`raw syscall traces -> instance construction -> 30 calls -> generated dataset -> 2-grams -> 2,805 -> TF-IDF -> chi-square -> 150`

with a paper-specific commit, artifact, or result file tying that chain to the 2024 experiment.

If such evidence is found, stop before executing any new preprocessing and review it against the primary baseline.

## 16. Claims We Must Not Make

The project must **not** claim any of the following on the present evidence:

- “We reproduced the authors' 52,656-instance dataset.”
- “The authors used stride 30.”
- “The authors used stride 1/overlapping windows.”
- “The authors balanced the dataset.”
- “The authors did not balance the dataset.”
- “The authors used 70/15/15.”
- “The authors used a trace-disjoint split.”
- “The authors used a window-level split.”
- “The authors used a particular random seed.”
- “The public ARCADIAN repository is the exact code used for this paper.”
- “The public ARCADIAN repository generated the 52,656 rows.”
- “The 149-system-call vocabulary was recovered from the authors' code.”
- “The 2,805 bigrams were recovered from the authors' code.”
- “A public CSV/NPY/NPZ/PKL/joblib/JSON artifact is the paper dataset.”
- “The current 88,828-window dataset is the paper dataset.”
- “The paper's class supports were directly reported as 26,870 attack and 25,786 normal.” They are derived from the reported confusion-matrix counts.
- “Similarity between public ARCADIAN code and the paper pipeline proves exact provenance.”
- “Absence of an artifact from the currently accessible web index proves that the authors never published it.”

### Selected conclusion: OPTION C — Related code only

The investigation recovered meaningful ARCADIAN-IoT / Device Behaviour Monitoring / system-call IDS project evidence, including a public repository lineage and independent project/publication linkage to the target authors and ARCADIAN-IoT. However, the exact 52,656-instance artifact and the paper-specific preprocessing pipeline were **not** recovered or proven.

Therefore the defensible conclusion is:

**OPTION C — RELATED CODE ONLY.**

No new preprocessing, baseline modification, MLP training, LIME/SHAP execution, balancing, class weighting, or Phase 3 work should begin on the strength of this provenance pass alone.
