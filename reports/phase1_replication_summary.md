# Phase 1 Replication Summary

## 1. Objective

This project reconstructs the main modeling and explainability workflow of Gaspar, Silva, and Silva (2024), *Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron*. The goal is reproducibility and interpretation of the documented method, not forced agreement with the published numbers.

## 2. Dataset and Reproducibility Challenge

The exact author-generated 52,656-instance dataset was not publicly recovered. An independent reconstruction was therefore built from a public raw ADFA-LD mirror. The reconstruction contains 5,951 valid raw traces (5,205 normal and 746 attack) and produces 88,828 non-overlapping 30-call windows. The raw syscall vocabulary contains 175 observed IDs. This differs from the paper's reported 149 IDs and 52,656 instances, so the result is not an exact numerical replication.

## 3. Independent Reconstruction Protocol

Each 30-call window is converted to adjacent syscall 2-grams, represented with train-only TF-IDF, reduced to 150 features using train-only chi-square selection, and classified with a frozen scikit-learn MLP. The split is trace-disjoint and stratified with seed 42, using 15% validation and 15% test data. No class weighting or resampling is used. The MLP uses 64 and 32 ReLU hidden units, Adam, learning rate 0.001, batch size 256, maximum 100 iterations, and train-only early stopping. These hidden widths and training settings are independent reconstruction choices because the paper does not provide the complete settings.

## 4. MLP Results

On the frozen reconstruction test partition, attack F1 is **0.735385**, ROC-AUC is **0.959989**, and PR-AUC is **0.739275**. The test confusion matrix is TN=11,681, FP=432, FN=428, and TP=1,195. The paper reports accuracy 0.9371, precision 0.9806, sensitivity 0.8946, specificity 0.9816, and F1 0.9356, with a different 52,656-instance confusion matrix. Because the datasets and evaluation protocols differ, these values are not presented as direct numerical replicas.

## 5. XAI and Perturbation Results

Twenty deterministic test windows (5 TP, 5 FN, 5 TN, 5 FP) were explained with top-10 2-gram transitions. Mean LIME-compatible/Kernel-SHAP top-10 Jaccard overlap was **0.08264**, indicating limited overlap in this selected sample; mean sign agreement on overlapping nonzero features was 0.40476. The official LIME package could not be installed in the offline environment, so the project used a vendored LIME-compatible local-surrogate reconstruction. For perturbation evaluation, explanation-guided raw-sequence edits produced larger descriptive mean absolute attack-probability changes than matched random edits for both methods: 0.234926 vs. 0.156690 for LIME-compatible explanations, and 0.352012 vs. 0.300972 for Kernel SHAP.

## 6. Limitations

The exact author dataset and complete preprocessing/split procedure were unavailable. The LIME implementation is not the official external package. Only 20 test instances were used for the XAI comparison and perturbation evaluation. Raw sequence edits can change neighboring 2-grams, and the perturbation results are model-sensitivity measurements rather than causal evidence. No statistical significance test was performed.

## 7. Conclusion

The project provides a reproducible, frozen independent reconstruction of the paper's main pipeline using public raw ADFA-LD data. It reproduces the documented 30-call, bigram TF-IDF, 150-feature, MLP, LIME/SHAP workflow at the methodological level while documenting the unavoidable dataset and environment deviations. The reported results should therefore be read as **strict reconstruction results, not exact numerical replication of the paper**.
