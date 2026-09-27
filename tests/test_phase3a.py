from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase3a import _fit_features, _require_manifest
from src.config import load_config, PROJECT_ROOT


def _docs(n=160):
    return [" ".join(f"{i}_{i+1}" for i in range(2, 2 + n))]


def test_tfidf_and_chi2_are_train_only_and_select_exactly_150():
    train = [" ".join(f"{i}_{i+1}" for i in range(1, 180)) for _ in range(6)]
    # This token occurs only in validation/test and therefore must never enter the fitted vocabulary.
    other = {"validation": ["999_1000 1_2"], "test": ["999_1000 3_4"]}
    y = np.array([0, 1, 0, 1, 0, 1])
    vec, selector, matrices = _fit_features(train, y, other, {"k_best": 150})
    assert matrices["train"].shape[1] == 150
    assert selector.k == 150
    assert "999_1000" not in vec.vocabulary_


def test_manifest_rejects_changed_window_policy():
    cfg = load_config()
    good = pd.DataFrame({
        "trace_id": ["a", "b", "c"], "source_relative_path": ["a.txt", "b.txt", "c.txt"],
        "source_group": ["Training_Data_Master"] * 3, "label": [0, 0, 1], "usable": [True] * 3,
        "raw_length": [60, 60, 60], "complete_window_count": [2, 2, 2], "leftover_call_count": [0, 0, 0],
        "assigned_split": ["train", "validation", "test"], "window_size": [30, 30, 30],
        "stride": [30, 30, 30], "tail_policy": ["drop"] * 3, "seed": [42, 42, 42],
    })
    _require_manifest(good, cfg)
    bad = good.copy(); bad.loc[0, "stride"] = 1
    with pytest.raises(ValueError, match="window policy"):
        _require_manifest(bad, cfg)


def test_frozen_manifest_is_trace_disjoint_and_has_expected_counts():
    p = PROJECT_ROOT / "data/processed/trace_manifest.csv"
    df = pd.read_csv(p)
    assert df.groupby("trace_id")["assigned_split"].nunique().max() == 1
    assert df.shape[0] == 5951
    assert df["assigned_split"].value_counts().to_dict() == {"train": 4165, "validation": 893, "test": 893}


def test_baseline_has_no_balancing_or_class_weight():
    cfg = load_config()
    model = cfg["model"]
    training = cfg["training"]
    assert model.get("class_weight") in (None, "none")
    assert training.get("class_weight") in (None, "none")
    assert training.get("resampling") in (None, "none")
