"""Raw ADFA-LD discovery and loading, plus the primary trace-disjoint stratified split.

This module never downloads anything. It only reads files that are already on disk under the
configured `data.raw_data_dir`, and it never writes to that directory.

Trace identifiers are relative-path based (`<source_dir>/<...>/<filename-without-suffix>`), never
absolute machine paths, so they are stable across machines and reruns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABEL_NORMAL = 0
LABEL_ATTACK = 1

META_COLUMNS = [
    "trace_id", "relative_path", "source_group", "attack_family", "label",
    "n_calls", "read_status", "malformed_token_count", "usable", "exclusion_reason",
]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw ADFA-LD folder cannot be found under the configured path.

    This tool never downloads the dataset for you.
    """


class SplitError(ValueError):
    """A trace-disjoint stratified split could not be produced for the given data/parameters."""


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------
def find_adfa_root(raw_dir: str | Path, required_subdirs: list[str]) -> Path:
    """Locate the ADFA-LD root: either `raw_dir` itself, or exactly one immediate subdirectory of
    it, whichever directly contains every name in `required_subdirs`.

    Raises DatasetNotFoundError with an actionable message (and explicitly states that this tool
    never downloads the dataset) if no such directory is found.
    """
    raw_dir = Path(raw_dir)
    candidates: list[Path] = [raw_dir]
    if raw_dir.is_dir():
        candidates += sorted(c for c in raw_dir.iterdir() if c.is_dir())
    for c in candidates:
        if all((c / s).is_dir() for s in required_subdirs):
            return c
    raise DatasetNotFoundError(
        f"ADFA-LD folder not found under '{raw_dir}'. Expected the subdirectories {required_subdirs} "
        f"either directly inside that path, or inside one immediate subdirectory of it (e.g. "
        f"'{raw_dir}/ADFA-LD'). This tool never downloads the dataset for you: download the raw "
        "ADFA-LD archive yourself (see data/README.md for the source and required layout), extract "
        "it locally, and point data.raw_data_dir at the extracted folder (or set the "
        "ADFA_LD_RAW_DIR environment variable / pass --raw-data-dir)."
    )


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
def _classify_and_parse(text: str) -> tuple[list[int], str, int]:
    """Parse whitespace-separated syscall IDs from one trace file's text.

    Returns (valid_calls, read_status, malformed_token_count).
    read_status is one of: "empty" (no tokens at all), "no_valid_calls" (tokens present but none
    parse as a non-negative integer), "ok" (at least one valid call).
    A token that is not a non-negative integer (non-numeric, or negative) counts as malformed and
    is dropped from the sequence; it never silently becomes a valid call.
    """
    tokens = text.split()
    if not tokens:
        return [], "empty", 0
    calls: list[int] = []
    malformed = 0
    for tok in tokens:
        try:
            value = int(tok)
        except ValueError:
            malformed += 1
            continue
        if value < 0:
            malformed += 1
            continue
        calls.append(value)
    if not calls:
        return [], "no_valid_calls", malformed
    return calls, "ok", malformed


def _family_of(group_dir: str, attack_dir: str, rel_parts: tuple[str, ...]) -> str:
    """Attack family name from the first path component under Attack_Data_Master (e.g. 'Adduser_1'
    -> 'Adduser'). Empty string for normal traces; 'unknown' for an attack file with no run folder.
    """
    if group_dir != attack_dir:
        return ""
    return re.sub(r"_\d+$", "", rel_parts[0]) if len(rel_parts) > 1 else "unknown"


def load_traces(root: Path, normal_dirs: list[str], attack_dir: str) -> tuple[pd.DataFrame, dict[str, list[int]], dict[str, Any]]:
    """Read every non-hidden `.txt` file under `root/<normal_dirs>` and `root/<attack_dir>` (read-only).

    Returns (meta, calls_lists, info):
      meta: one row per considered .txt file (trace_id, relative_path, source_group, attack_family,
        label, n_calls, read_status, malformed_token_count, usable, exclusion_reason).
      calls_lists: trace_id -> list[int] of valid syscall IDs, for traces with n_calls > 0 only.
      info: {"n_hidden_files_ignored": int} - hidden files (dotfiles) are counted but never parsed.
        Non-.txt files are silently skipped here; callers that need to count them (e.g. the
        reconciliation raw-inventory report) rescan the directories themselves.
    """
    root = Path(root)
    rows: list[dict[str, Any]] = []
    calls_lists: dict[str, list[int]] = {}
    n_hidden = 0

    for group in list(normal_dirs) + [attack_dir]:
        base = root / group
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith("."):
                n_hidden += 1
                continue
            if p.suffix.lower() != ".txt":
                continue
            rel = p.relative_to(base)
            family = _family_of(group, attack_dir, rel.parts)
            label = LABEL_ATTACK if group == attack_dir else LABEL_NORMAL
            trace_id = (Path(group) / rel).with_suffix("").as_posix()
            try:
                text = p.read_text(encoding="utf-8")
                calls, status, malformed = _classify_and_parse(text)
            except (UnicodeDecodeError, OSError):
                calls, status, malformed = [], "unreadable", 0
            usable = len(calls) > 0
            reason = "" if usable else status
            rows.append({
                "trace_id": trace_id, "relative_path": (Path(group) / rel).as_posix(),
                "source_group": group, "attack_family": family, "label": label,
                "n_calls": len(calls), "read_status": status, "malformed_token_count": malformed,
                "usable": usable, "exclusion_reason": reason,
            })
            if calls:
                calls_lists[trace_id] = calls

    meta = pd.DataFrame(rows, columns=META_COLUMNS)
    if len(meta):
        meta["label"] = meta["label"].astype(int)
        meta["n_calls"] = meta["n_calls"].astype(int)
    return meta, calls_lists, {"n_hidden_files_ignored": n_hidden}


# --------------------------------------------------------------------------------------
# Trace-disjoint stratified split
# --------------------------------------------------------------------------------------
def _two_stage_stratified(
    ids: np.ndarray, labels: np.ndarray, test_size: float, validation_size: float, seed: int, group_label: str = "items",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of `ids` (grouped by their own label) into (train, validation, test).

    Splitting happens on `ids` directly (e.g. trace IDs), so whatever those IDs identify stays
    entirely within one partition - this is what makes the primary split trace-disjoint when `ids`
    are trace IDs. Deterministic for a given seed. Raises SplitError if any class has too few items
    to produce three non-empty partitions at the requested proportions.
    """
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    if len(ids) == 0:
        raise SplitError(f"no {group_label} to split")
    if not (0 < test_size < 1) or not (0 < validation_size < 1) or test_size + validation_size >= 1:
        raise SplitError(f"invalid split proportions: test_size={test_size}, validation_size={validation_size}")

    classes = np.unique(labels)
    if len(classes) < 2:
        raise SplitError(f"only one class present among {group_label}; cannot stratify")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_test = int(round(n * test_size))
        n_val = int(round(n * validation_size))
        n_train = n - n_test - n_val
        if n_train < 1 or n_test < 1 or n_val < 1:
            raise SplitError(
                f"class {c!r} has only {n} {group_label} (test_size={test_size}, "
                f"validation_size={validation_size}); too few to give every partition at least one"
            )
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test:n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return ids[train_idx], ids[val_idx], ids[test_idx]


# --------------------------------------------------------------------------------------
# Primary (non-reconciliation) window/split pipeline
# --------------------------------------------------------------------------------------
def _window_starts(n_calls: int, window_size: int, stride: int) -> np.ndarray:
    if n_calls < window_size:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_calls - window_size + 1, stride, dtype=np.int64)


@dataclass
class LoadedDataset:
    meta: pd.DataFrame
    calls: dict[str, np.ndarray]
    root: Path


def build_from_config(cfg: dict[str, Any]) -> tuple[LoadedDataset, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Build the primary trace and window tables for `cfg` (window_size/stride from cfg["data"]).

    Returns (loaded_dataset, traces_df, windows_df, calls). `traces_df` holds one row per usable
    trace (n_calls >= window_size), sorted by trace_id. `windows_df` holds one row per generated
    window (trace_id, start, label, source_group, attack_family). Read-only; writes nothing.
    """
    d = cfg["data"]
    raw_dir = resolve_path_for_data(d["raw_data_dir"])
    normal_dirs = list(d.get("normal_directories") or d.get("normal_dirs") or [])
    attack_dir = d.get("attack_directory") or next(iter(d.get("attack_dirs") or []), None)
    if not normal_dirs or not attack_dir:
        raise SplitError("config.data must define normal_directories/normal_dirs and attack_directory/attack_dirs")

    root = find_adfa_root(raw_dir, normal_dirs + [attack_dir])
    meta, calls_lists, _info = load_traces(root, normal_dirs, attack_dir)
    calls = {k: np.asarray(v, dtype=np.int64) for k, v in calls_lists.items()}

    window_size = int(d.get("window_size", 30))
    stride = int(d.get("stride", window_size))
    traces = meta[meta["n_calls"] >= window_size].sort_values("trace_id").reset_index(drop=True)

    rows = []
    for r in traces.itertuples(index=False):
        n = len(calls.get(r.trace_id, ()))
        for start in _window_starts(n, window_size, stride).tolist():
            rows.append({"trace_id": r.trace_id, "start": start, "label": r.label,
                        "source_group": r.source_group, "attack_family": r.attack_family})
    windows = pd.DataFrame(rows, columns=["trace_id", "start", "label", "source_group", "attack_family"])

    return LoadedDataset(meta=meta, calls=calls, root=root), traces, windows, calls


def make_split(
    traces: pd.DataFrame, windows: pd.DataFrame, policy: str, validation_size: float, test_size: float, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each trace (and every window generated from it) to train/validation/test.

    Only "stratified_group_by_trace" is implemented: traces are split (stratified by label) via
    `_two_stage_stratified`, and every window inherits its trace's split - so no trace's windows
    can appear in more than one partition.
    """
    if policy != "stratified_group_by_trace":
        raise ValueError(f"unsupported split policy: {policy!r}")
    ids = traces["trace_id"].to_numpy()
    labels = traces["label"].to_numpy()
    train_ids, val_ids, test_ids = _two_stage_stratified(ids, labels, test_size, validation_size, seed, "traces")
    assign = {}
    for name, arr in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
        for t in arr.tolist():
            assign[t] = name

    traces = traces.copy()
    traces["split"] = traces["trace_id"].map(assign)
    windows = windows.copy()
    windows["split"] = windows["trace_id"].map(assign)
    return traces, windows


def resolve_path_for_data(path: str) -> Path:
    """Thin indirection to src.config.resolve_path, kept local to avoid a hard import cycle."""
    from .config import resolve_path
    return resolve_path(path)
