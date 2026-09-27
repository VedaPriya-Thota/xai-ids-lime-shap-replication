# Phase 1 Result Comparison: Paper vs. Strict Independent Reconstruction

## Scope

This document compares the published study by Gaspar, Silva, and Silva (2024) with the frozen reconstruction in this repository. The comparison is deliberately conservative: values are reported as **paper-reported**, **derived**, or **reconstruction** values, and differences are not treated as evidence of an implementation error unless the underlying protocol is known to be identical.

> **Replication status:** this is a strict independent reconstruction using public raw ADFA-LD traces. The authors' generated 52,656-instance dataset and complete preprocessing/split procedure were not recovered. Therefore, this is **not an exact numerical replication** of the paper.

## A. Dataset and protocol comparison

| Item | Published paper | Frozen reconstruction | Comparison |
|---|---|---|---|
| Dataset source | Generated version of ADFA-LD for the IoT IDS study | Public raw ADFA-LD mirror from Kaggle, rebuilt locally | Same dataset family; source artifact differs |
| Dataset size | 52,656 instances | 88,828 non-overlapping windows | Not numerically identical |
| Syscall vocabulary | 149 distinct system calls | 175 observed syscall IDs, range 1–340 | Differs |
| Instance representation | 30 system calls per instance | 30 system calls per window | Aligned |
| Window stride / overlap | Not fully reported | Stride 30; non-overlapping; incomplete tails dropped | Reconstruction assumption |
| Feature representation | 2-gram / bigram TF-IDF | Adjacent 2-gram tokens + train-only TF-IDF | Aligned at the documented representation level |
| Initial feature count | 2,805 n-grams | Determined from the reconstructed training data | Depends on reconstructed corpus |
| Feature selection | Chi-square selection to 150 features | Train-only `SelectKBest(chi2, k=150)` | Aligned |
| Split | Not fully reported in the recovered paper material | Trace-disjoint stratified split; seed 42; 15% validation; 15% test | Reconstruction choice |
| Class support | From paper confusion matrix: attack = 24,038 + 2,832 = 26,870; normal = 25,311 + 475 = 25,786; attack share ≈ 51.03% | 78,599 normal windows and 10,229 attack windows; attack share = 11.52% | Different dataset/protocol distributions |
| MLP framework | TensorFlow/Keras | scikit-learn `MLPClassifier` | Framework differs |
| MLP architecture | 150 inputs; two ReLU hidden dense layers; sigmoid output | 150 inputs; 64 ReLU → 32 ReLU → logistic output | Architecture family aligned; hidden widths are an independent reconstruction choice |
| MLP training details | Complete widths/training settings not reported in the recovered paper material | Adam; learning rate 0.001; batch size 256; max 100 iterations; train-only 10% internal early stopping; patience 10; seed 42 | Reconstruction choices |
| LIME | `LimeTabularExplainer`; top-10 features | Vendored LIME-compatible local-surrogate reconstruction; top-10 features | Method intent aligned; official package unavailable offline |
| SHAP | `KernelExplainer`; top-10 features | Actual `shap.KernelExplainer`, SHAP 0.50.0; train-only 100-row 50/50 background; top-10 | Method aligned; package/version environment differs |
| Perturbation logic | Paper describes feature-removal and class-exclusive/random replacement experiments | Raw 30-call sequence intervention with matched random edits and frozen transform/model | Related evaluation goal, but author-specific intervention implementation was unavailable |

### Important feature terminology

The explanatory features are **adjacent syscall transitions (2-grams)**, such as `221_197`, rather than isolated syscall IDs. A feature therefore represents the transition from syscall 221 to syscall 197 at an adjacent position in the 30-call sequence.

## B. MLP result comparison

### Published paper

The paper reports the following values in Table 1 and the accompanying confusion-matrix discussion:

| Metric / count | Paper |
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
| Total | 52,656 |

The paper's reported confusion-matrix counts imply attack support of 26,870 and normal support of 25,786. These support values are derived from the reported counts, not separately stated class totals.

### Frozen reconstruction

| Metric / count | Reconstruction |
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
| PR-AUC / average precision | 0.739275 |
| TN | 11,681 |
| FP | 432 |
| FN | 428 |
| TP | 1,195 |
| Test windows | 13,736 |

The reconstruction confusion matrix is therefore:

| Actual \ Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 11,681 | 432 |
| Attack | 428 | 1,195 |

The numerical values should **not** be interpreted as a direct reproduction of the paper's performance because the dataset size, class distribution, split protocol, framework, and some training details differ.

## C. XAI comparison

The paper uses LIME and Kernel SHAP to obtain top-10 explanatory features for instances. The reconstruction evaluates exactly 20 deterministic frozen test windows: 5 TP, 5 FN, 5 TN, and 5 FP, selected with seed 42.

| XAI item | Paper | Reconstruction |
|---|---|---|
| LIME | `LimeTabularExplainer` | Vendored LIME-compatible local-surrogate reconstruction |
| SHAP | `KernelExplainer` | Actual `shap.KernelExplainer` |
| Features | Top-10 features | Top-10 selected syscall 2-gram transitions |
| LIME/SHAP comparison set | Paper explains the dataset instances | 20 deterministic selected test instances |
| Mean top-10 Jaccard overlap | Not reported as a directly comparable value | 0.08264 |
| Median top-10 Jaccard overlap | Not reported as a directly comparable value | 0.05263 |
| Mean sign agreement on overlapping nonzero features | Not reported as a directly comparable value | 0.40476 |

The low mean Jaccard overlap (0.08264) is a descriptive result for the selected 20 instances only. It does not establish that one explanation method is generally better than the other.

### LIME package limitation

The official `lime==0.2.0.1` package could not be installed in the offline execution environment because network/DNS access was unavailable. Phase 3B and the downstream Phase 3C evaluation therefore use the documented **vendored LIME-compatible local-surrogate reconstruction**. It should not be described as execution of the official external LIME package.

## D. Perturbation comparison

The reconstruction operates on the original 30-call sequences rather than directly editing the transformed 150-dimensional feature vectors. For each actionable top-10 2-gram, the second syscall in an exact adjacent occurrence is replaced using a deterministic neutral replacement policy derived only from training-normal raw windows. A matched random control makes the same number of raw-position edits under the same replacement policy.

### Aggregate guided vs. matched-random results

| Method | Guided mean | Random mean | Difference | Guided flips | Random flips | Direction consistency |
|---|---:|---:|---:|---:|---:|---:|
| LIME-compatible | Mean absolute attack-probability delta = 0.234926 | 0.156690 | +0.078237 | 9/20 | 6/20 | 65% |
| Kernel SHAP | Mean absolute attack-probability delta = 0.352012 | 0.300972 | +0.051040 | 9/20 | 8/20 | 95% |

Additional actionability statistics:

| Method | Actionable cases | Mean actionable top-10 features | Mean raw edits |
|---|---:|---:|---:|
| LIME-compatible | 15/20 (75%) | 1.25 | 4.60 |
| Kernel SHAP | 20/20 (100%) | 3.20 | 11.25 |

These are **descriptive paired results on a 20-instance selected sample**. No statistical significance test was run. The intervention changes adjacent 2-grams around edited positions, so the procedure does not isolate one feature in a strictly causal sense. The results therefore measure model-output sensitivity under the documented intervention policy, not causal syscall importance.

## E. Claims We Do Not Make

This project does **not** claim:

1. that the authors' exact 52,656-instance dataset was recovered or reproduced;
2. that the paper's exact preprocessing or split procedure was recovered;
3. that the official external `lime==0.2.0.1` package was executed in the reported offline Phase 3B/3C runs;
4. that the reconstruction metrics are directly comparable numerical replicas of the paper's metrics;
5. that changing a syscall transition demonstrates a causal relationship with real attack behaviour;
6. that the 20-instance XAI results generalize to the full test set;
7. that LIME or SHAP is universally superior to the other method.

## Reproducibility anchors

- Seed: **42**.
- Window size: **30** calls.
- Stride: **30**.
- Split: **trace-disjoint stratified**, validation 15%, test 15%.
- Trace manifest SHA-256: `765289887ac57f0a7f08b0d2761c9507e65f8ff98805ab69dc08a6cc312bf4f1`.
- Split-assignment SHA-256: `5c2b0745cccde2c060ce139fe92e383510ab4237eb0d53b64b9c5f29024e04ba`.

For provenance details, see `docs/paper_dataset_forensics.md` and the Phase 2.5/2.6 audit report.
