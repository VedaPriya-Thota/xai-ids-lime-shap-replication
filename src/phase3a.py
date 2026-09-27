"""Phase 3A: leakage-controlled 2-gram TF-IDF + chi2 + MLP baseline.

This module consumes the frozen Phase 2.6 trace manifest. It never writes to raw data or
regenerates/reassigns the manifest. All feature fitting is performed on TRAIN windows only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from .config import PROJECT_ROOT, load_config, resolve_path
from .data import LABEL_ATTACK, LABEL_NORMAL, find_adfa_root, load_traces
from .seed import set_global_seed

EXPERIMENT_NAME = "phase3a_strict_reconstructed_mlp"
PAPER_DOI = "10.1109/ACCESS.2024.3368377"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_manifest(manifest: pd.DataFrame, cfg: dict[str, Any]) -> None:
    required = {
        "trace_id", "source_relative_path", "source_group", "label", "usable",
        "raw_length", "complete_window_count", "leftover_call_count", "assigned_split",
        "window_size", "stride", "tail_policy", "seed",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Frozen manifest is missing required columns: {sorted(missing)}")
    d = cfg["data"]
    if set(manifest["assigned_split"].dropna().unique()) != {"train", "validation", "test"}:
        raise ValueError("Frozen manifest must contain train, validation, and test assignments")
    if int(d["window_size"]) != 30 or int(d["stride"]) != 30:
        raise ValueError("Phase 3A requires the frozen 30-call / stride-30 baseline")
    if str(d.get("short_trace_policy", "drop")) != "drop":
        raise ValueError("Phase 3A requires incomplete tails/traces to be dropped")
    if int(cfg["seed"]) != 42:
        raise ValueError("Phase 3A requires frozen seed 42")
    if not ((manifest["window_size"] == 30).all() and (manifest["stride"] == 30).all()):
        raise ValueError("Manifest window policy differs from the frozen 30/30 policy")
    if not (manifest["tail_policy"].astype(str).eq("drop").all()):
        raise ValueError("Manifest tail policy differs from the frozen drop policy")
    if not (manifest["seed"] == 42).all():
        raise ValueError("Manifest seed differs from frozen seed 42")


def load_frozen_manifest(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen Phase 2.6 manifest not found: {path}")
    digest = sha256_file(path)
    df = pd.read_csv(path)
    _require_manifest(df, cfg)
    if df["trace_id"].duplicated().any():
        raise ValueError("Frozen manifest contains duplicate trace_id rows")
    if df["trace_id"].isna().any() or df["assigned_split"].isna().any():
        raise ValueError("Frozen manifest contains missing trace IDs or split assignments")
    assignment_rows = [
        f"{r.trace_id}|{int(r.label)}|{r.assigned_split}"
        for r in df.sort_values("trace_id").itertuples(index=False)
    ]
    assignment_hash = sha256_lines(assignment_rows)
    return df, digest, assignment_hash


def _verify_manifest_against_raw(manifest: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], Path]:
    d = cfg["data"]
    root = find_adfa_root(resolve_path(d["raw_data_dir"]), list(d["normal_directories"]) + [d["attack_directory"]])
    meta, calls_lists, _ = load_traces(root, list(d["normal_directories"]), d["attack_directory"])
    raw = {k: np.asarray(v, dtype=np.int64) for k, v in calls_lists.items()}
    current = meta.set_index("trace_id")
    m = manifest.set_index("trace_id")
    if set(m.index) != set(current.index):
        raise ValueError("Frozen manifest trace IDs do not exactly match the current raw dataset inventory")
    for tid, row in m.iterrows():
        cur = current.loc[tid]
        if int(row["label"]) != int(cur["label"]) or int(row["raw_length"]) != int(cur["n_calls"]):
            raise ValueError(f"Raw dataset differs from frozen manifest for trace {tid}")
        expected_windows = int(row["raw_length"]) // 30
        expected_leftover = int(row["raw_length"]) % 30
        if int(row["complete_window_count"]) != expected_windows or int(row["leftover_call_count"]) != expected_leftover:
            raise ValueError(f"Frozen manifest window metadata is inconsistent for trace {tid}")
    return raw, root


def _window_texts(calls: dict[str, np.ndarray], manifest: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[int]], pd.DataFrame]:
    texts = {"train": [], "validation": [], "test": []}
    labels = {"train": [], "validation": [], "test": []}
    rows: list[dict[str, Any]] = []
    for r in manifest.sort_values("trace_id").itertuples(index=False):
        seq = calls[r.trace_id]
        for start in range(0, len(seq) - 29, 30):
            window = seq[start:start + 30]
            # Precise token representation: adjacent calls are "callA_callB".
            tokens = [f"{int(a)}_{int(b)}" for a, b in zip(window[:-1], window[1:])]
            text = " ".join(tokens)
            split = str(r.assigned_split)
            texts[split].append(text)
            labels[split].append(int(r.label))
            rows.append({"trace_id": r.trace_id, "start": start, "label": int(r.label), "split": split})
    return texts, labels, pd.DataFrame(rows)


def _fit_features(train_texts: list[str], train_y: np.ndarray, other_texts: dict[str, list[str]], cfg: dict[str, Any]):
    if not train_texts:
        raise ValueError("No training windows available")
    vec = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"[^\s]+", lowercase=False, norm="l2", use_idf=True)
    # Input documents already contain the 2-gram tokens; vectorizer therefore indexes those exact tokens.
    x_train = vec.fit_transform(train_texts)
    if x_train.shape[1] < 150:
        raise ValueError(f"Training corpus contains only {x_train.shape[1]} distinct 2-gram tokens; required k=150")
    selector = SelectKBest(chi2, k=150)
    x_train_sel = selector.fit_transform(x_train, train_y)
    matrices = {"train": x_train_sel}
    for split in ("validation", "test"):
        x = vec.transform(other_texts[split])
        matrices[split] = selector.transform(x)
    return vec, selector, matrices


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["normal", "attack"], output_dict=True, zero_division=0)
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_normal": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_normal": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_normal": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision_pr_auc": float(average_precision_score(y_true, y_prob)),
        "classification_report": report,
        "probability_summary": {
            "min": float(np.min(y_prob)), "q25": float(np.quantile(y_prob, .25)),
            "median": float(np.median(y_prob)), "q75": float(np.quantile(y_prob, .75)), "max": float(np.max(y_prob)),
            "mean": float(np.mean(y_prob)), "std": float(np.std(y_prob)),
        },
    }


def run_phase3a(config_path: str = "configs/baseline.yaml", experiment_dir: str | None = None, raw_data_dir: str | None = None) -> Path:
    cfg = load_config(config_path)
    if raw_data_dir is not None:
        cfg["data"]["raw_data_dir"] = raw_data_dir
    seed = int(cfg["seed"])
    set_global_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    manifest_path = resolve_path(cfg["data"]["processed_data_dir"]) / "trace_manifest.csv"
    manifest, manifest_hash, assignment_hash = load_frozen_manifest(manifest_path, cfg)
    calls, raw_root = _verify_manifest_against_raw(manifest, cfg)
    texts, labels, windows = _window_texts(calls, manifest)

    # Explicit partition integrity checks.
    if windows["trace_id"].groupby(windows["trace_id"]).apply(lambda s: s.nunique()).max() != 1:
        raise RuntimeError("A trace appears in multiple partitions")
    if len(windows) != 88828:
        raise RuntimeError(f"Frozen baseline window count changed: expected 88828, got {len(windows)}")
    if {s: len(texts[s]) for s in texts} != {"train": 61553, "validation": 13539, "test": 13736}:
        raise RuntimeError("Window split counts differ from frozen Phase 2.5 counts")

    out = resolve_path(experiment_dir or f"results/experiments/{EXPERIMENT_NAME}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Phase 3A experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "preprocessing").mkdir()

    vec, selector, matrices = _fit_features(texts["train"], np.asarray(labels["train"], dtype=int), {"validation": texts["validation"], "test": texts["test"]}, cfg["features"])

    # Save preprocessing artifacts before model training.
    joblib.dump(vec, out / "preprocessing" / "tfidf_vectorizer.joblib")
    joblib.dump(selector, out / "preprocessing" / "chi2_selector.joblib")
    names = np.asarray(vec.get_feature_names_out())[selector.get_support()]
    scores = np.asarray(selector.scores_)[selector.get_support()]
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    feature_df = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "model_column_index": np.arange(len(order))[np.argsort(np.argsort(np.arange(len(order))))],
        "feature_name": names[order],
        "chi2_score": scores[order],
    })
    # Model columns remain selector output order; record that explicitly.
    model_names = pd.DataFrame({"model_column_index": np.arange(len(names)), "feature_name": names})
    model_names.to_csv(out / "preprocessing" / "selected_feature_names.csv", index=False)
    feature_df.to_csv(out / "preprocessing" / "selected_feature_ranking.csv", index=False)

    model_cfg = {
        "framework": "scikit-learn MLPClassifier",
        "architecture": "150 input -> 128 ReLU -> 64 ReLU -> 1 logistic output",
        "hidden_layer_sizes": [64, 32],
        "hidden_activation": "relu",
        "output_activation": "logistic (binary MLPClassifier output)",
        "solver": "adam", "learning_rate_init": 0.001, "batch_size": 128,
        "max_iter": 100, "random_state": seed, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10,
        "class_weight": None, "sample_weight": None, "resampling": None,
        "decision_threshold": 0.5,
        "classification_loss": "log_loss",
        "parameter_status": "independent reconstruction default for unspecified architecture/training hyperparameters",
    }
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=0.001,
                          batch_size=256, max_iter=100, random_state=seed, early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=10, shuffle=True, verbose=False)
    model.fit(matrices["train"], np.asarray(labels["train"], dtype=int))

    predictions = {}
    metrics = {}
    for split in ("validation", "test"):
        y = np.asarray(labels[split], dtype=int)
        prob = model.predict_proba(matrices[split])[:, 1]
        pred = (prob >= 0.5).astype(int)
        predictions[split] = (y, pred, prob)
        metrics[split] = _metrics(y, prob, 0.5)
        pd.DataFrame({"window_index": np.arange(len(y)), "label": y, "predicted_label": pred, "predicted_attack_probability": prob}).to_csv(out / f"{split}_predictions.csv", index=False)
        pd.DataFrame(metrics[split]["confusion_matrix"], index=["actual_normal", "actual_attack"], columns=["pred_normal", "pred_attack"]).to_csv(out / f"{split}_confusion_matrix.csv")

    joblib.dump(model, out / "mlp_model.joblib")
    pd.DataFrame({"iteration": np.arange(1, len(model.loss_curve_) + 1), "loss": model.loss_curve_}).to_csv(out / "training_history.csv", index=False)

    counts = {}
    for split in ("train", "validation", "test"):
        y = np.asarray(labels[split], dtype=int)
        counts[split] = {"total": int(len(y)), "normal": int((y == 0).sum()), "attack": int((y == 1).sum())}

    meta = {
        "experiment": EXPERIMENT_NAME, "paper_doi": PAPER_DOI,
        "status": "strict reconstructed ADFA-LD baseline, not the authors' exact 52,656-instance dataset",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": project_relative(resolve_path(config_path)),
        "manifest_path": project_relative(manifest_path), "manifest_sha256": manifest_hash,
        "split_assignment_sha256": assignment_hash,
        "raw_source_dir": project_relative(raw_root),
        "raw_data_modified": False,
        "window_policy": {"window_size": 30, "stride": 30, "tail_policy": "drop"},
        "split_policy": {"method": "trace-disjoint stratified split", "seed": seed, "validation_fraction": .15, "test_fraction": .15},
        "feature_pipeline": {
            "token_form": "callA_callB", "ngram_order": 2, "tfidf_fit_split": "train_only",
            "chi2_fit_split": "train_only", "requested_k": 150, "selected_k": int(matrices["train"].shape[1]),
            "vectorizer": {"lowercase": False, "norm": "l2", "use_idf": True},
        },
        "model": model_cfg,
        "class_distribution_by_split": counts,
        "leakage_checks": {
            "trace_disjoint": True, "tfidf_fit_on_validation": False, "tfidf_fit_on_test": False,
            "chi2_fit_on_validation": False, "chi2_fit_on_test": False, "validation_labels_used_for_fit": False,
            "test_labels_used_for_fit": False, "test_metrics_used_for_selection": False,
            "class_weight_used": False, "resampling_used": False,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": __import__("sklearn").__version__, "joblib": joblib.__version__},
        "random_seeds": {"global": seed, "numpy": seed, "python": seed, "model_random_state": seed},
    }
    (out / "reproducibility_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out / "config_snapshot.json").write_text(json.dumps({"model": model_cfg, "features": cfg["features"], "split": cfg["split"], "data": {"window_size": 30, "stride": 30, "tail_policy": "drop"}}, indent=2, sort_keys=True), encoding="utf-8")

    report = [
        "# Phase 3A — Strict Reconstructed ADFA-LD MLP Baseline", "",
        "**This is a strict reconstructed ADFA-LD baseline, not the authors' exact 52,656-instance dataset.**", "",
        f"Paper: Gaspar, Silva, and Silva (2024), DOI {PAPER_DOI}.", "",
        "## Pipeline", "- Frozen 30-call, stride-30, non-overlapping windows; incomplete tails dropped.",
        "- 2-gram tokens use the exact form `callA_callB` (e.g. `1_2`).", "- TF-IDF fitted on TRAIN only.",
        "- chi-square SelectKBest fitted on TRAIN only, k=150.", "- Validation/test are transformed only after fitting.",
        "", "## MLP", "- Paper-reported: Keras/TensorFlow model; two hidden dense layers with ReLU; sigmoid output; binary prediction.",
        "- Independent defaults: 64 and 32 hidden units, Adam, learning rate 0.001, batch size 256, max 100 iterations with early stopping on an internal 10% subset of TRAIN only, seed 42.",
        "- No class weights or resampling.", "", "## Results", "",
    ]
    for split in ("validation", "test"):
        m = metrics[split]
        report += [f"### {split.title()}", f"- Accuracy: {m['accuracy']:.6f}", f"- Balanced accuracy: {m['balanced_accuracy']:.6f}",
                   f"- Precision (normal/attack): {m['precision_normal']:.6f} / {m['precision_attack']:.6f}",
                   f"- Recall (normal/attack): {m['recall_normal']:.6f} / {m['recall_attack']:.6f}",
                   f"- F1 (normal/attack): {m['f1_normal']:.6f} / {m['f1_attack']:.6f}",
                   f"- Macro F1: {m['macro_f1']:.6f}", f"- ROC-AUC: {m['roc_auc']:.6f}",
                   f"- PR-AUC / average precision: {m['average_precision_pr_auc']:.6f}", f"- Confusion matrix [normal, attack]: {m['confusion_matrix']}", ""]
    report += ["## Leakage and reproducibility", "- Trace-disjoint manifest assignments were verified.",
               "- TF-IDF and chi-square were fit on training windows only.", "- Validation/test labels were not used for fitting or model selection.",
               "- Test metrics were computed after the model configuration was finalized.",
               "- Raw data and Phase 2.5/2.6 artifacts were not modified.", "", "## Scope boundary",
               "LIME and SHAP have **not** been run in Phase 3A."]
    (out / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out
