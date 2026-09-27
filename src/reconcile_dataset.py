"""Phase 2.5: dataset reconciliation between our raw-ADFA-LD reconstruction and the paper's generated dataset.

!!! EXPLORATORY DIAGNOSTICS ONLY !!!
Nothing here recovers, approximates or claims the authors' generated dataset. Every scenario is a labelled
exploratory experiment. Distances to the paper's values are descriptive columns, never optimisation targets.
Raw data are only read. Nothing is written to data/processed or data/artifacts.

Experiments
  A raw inventory            B window-generation grid       C source-group inclusion
  D trace-length filters     E syscall-vocabulary handling  F bigram vocabulary (every scenario)
  G balance feasibility (no balanced dataset is generated)

Counting is vectorised (see window_starts / covered_pair_mask) and unit-tested against brute force.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    LABEL_ATTACK, LABEL_NORMAL, SplitError, _two_stage_stratified, find_adfa_root, load_traces,
)
from .preprocess import fit_features
from .audit_artifacts import write_audit_artifacts

log = logging.getLogger(__name__)

EXPLORATORY_LABEL = "EXPLORATORY reconciliation experiment - not a paper-replication result"
SUMMARY_COLUMNS = [
    "scenario_id", "normal_trace_sources", "attack_sources", "window_size", "stride", "length_filter",
    "vocabulary_handling", "normal_windows", "attack_windows", "total_windows", "attack_percentage",
    "unique_syscalls", "unique_bigrams", "distance_to_paper_total", "distance_to_paper_attack_ratio", "notes",
]


class ReconciliationConfigError(ValueError):
    """The reconciliation config has an invalid value."""


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class TraceSet:
    """All raw traces as numpy arrays (read-only view of the dataset)."""
    meta: pd.DataFrame                 # trace_id, raw_path, label, source_group, attack_family, n_calls, read_status, ...
    calls: dict[str, np.ndarray]       # trace_id -> int64 array of syscall IDs
    root: Path
    outside_root_entries: list[Path]   # third-party entries next to the dataset root (ignored)
    hidden_ignored: int


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    group: str                          # reference | grid | source | length | vocab
    normal_sources: tuple[str, ...]
    attack_families: tuple[str, ...] | None   # None = all attack families
    window_size: int
    stride: int
    min_trace_len: int = 0
    vocab_handling: str = "none"        # none | top_unk | top_drop
    notes: str = ""

    def key(self) -> tuple:
        return (self.normal_sources, self.attack_families, self.window_size, self.stride, self.min_trace_len, self.vocab_handling)


# --------------------------------------------------------------------------------------
# Pure counting helpers (unit-tested against brute force)
# --------------------------------------------------------------------------------------
def count_windows(n_calls: int, window_size: int, stride: int) -> int:
    """Number of complete windows of `window_size` taken every `stride` calls (incomplete final windows dropped)."""
    return 0 if n_calls < window_size else (n_calls - window_size) // stride + 1


def window_starts(n_calls: int, window_size: int, stride: int) -> np.ndarray:
    return np.arange(0, n_calls - window_size + 1, stride, dtype=np.int64) if n_calls >= window_size else np.empty(0, np.int64)


def covered_position_mask(n_calls: int, starts: np.ndarray, window_size: int) -> np.ndarray:
    """True for every call position that lies inside at least one window."""
    if len(starts) == 0:
        return np.zeros(n_calls, dtype=bool)
    d = np.bincount(starts, minlength=n_calls + 1) - np.bincount(starts + window_size, minlength=n_calls + 1)
    return np.cumsum(d)[:n_calls] > 0


def covered_pair_mask(n_calls: int, starts: np.ndarray, window_size: int) -> np.ndarray:
    """True for pair index i when calls i and i+1 lie in the SAME window (bigrams never cross a window boundary)."""
    if n_calls < 2 or len(starts) == 0:
        return np.zeros(max(n_calls - 1, 0), dtype=bool)
    d = np.bincount(starts, minlength=n_calls + 1) - np.bincount(starts + window_size - 1, minlength=n_calls + 1)
    return np.cumsum(d)[: n_calls - 1] > 0


def drop_windows_with_excluded(calls: np.ndarray, starts: np.ndarray, window_size: int, excluded_lut: np.ndarray) -> np.ndarray:
    """Remove window starts whose window contains any excluded syscall ID."""
    if len(starts) == 0:
        return starts
    ex = excluded_lut[calls]
    cs = np.concatenate([[0], np.cumsum(ex)])
    return starts[(cs[starts + window_size] - cs[starts]) == 0]


def top_k_ids(counts: np.ndarray, k: int) -> np.ndarray:
    """IDs of the k most frequent syscalls (ties broken by smaller ID); only IDs with count > 0 are eligible."""
    ids = np.flatnonzero(counts > 0)
    order = np.lexsort((ids, -counts[ids]))
    return ids[order][:k]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_traceset(cfg: dict[str, Any]) -> TraceSet:
    """Read all raw traces (read-only). Raises DatasetNotFoundError with an actionable message if absent."""
    from .config import resolve_path

    d = cfg["data"]
    normal_dirs, attack_dir = list(d["normal_directories"]), d["attack_directory"]
    raw_dir = resolve_path(d["raw_data_dir"])
    root = find_adfa_root(raw_dir, normal_dirs + [attack_dir])
    meta, calls_lists, info = load_traces(root, normal_dirs, attack_dir)
    calls = {k: np.asarray(v, dtype=np.int64) for k, v in calls_lists.items()}
    outside = sorted(c for c in root.parent.iterdir() if c != root and not c.name.startswith(".")) if root != raw_dir else []
    return TraceSet(meta=meta, calls=calls, root=root, outside_root_entries=outside, hidden_ignored=info["n_hidden_files_ignored"])


# --------------------------------------------------------------------------------------
# A. Raw inventory
# --------------------------------------------------------------------------------------
def _family_of(group_dir: str, attack_dir: str, rel_parts: tuple[str, ...]) -> str:
    import re
    if group_dir != attack_dir:
        return ""
    return re.sub(r"_\d+$", "", rel_parts[0]) if len(rel_parts) > 1 else "unknown"


def count_ignored_files(root: Path, group_dirs: list[str], attack_dir: str) -> dict[tuple[str, str], dict[str, int]]:
    """Files inside the dataset folders that are not traces: hidden files and non-.txt files, per (source_group, family)."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for g in group_dirs:
        base = root / g
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            key = (g, _family_of(g, attack_dir, p.relative_to(base).parts))
            rec = out.setdefault(key, {"hidden": 0, "non_txt": 0})
            if p.name.startswith("."):
                rec["hidden"] += 1
            elif p.suffix.lower() != ".txt":
                rec["non_txt"] += 1
    return out


def _count_files_recursive(paths: list[Path]) -> int:
    n = 0
    for p in paths:
        n += 1 if p.is_file() else sum(1 for f in p.rglob("*") if f.is_file())
    return n


def raw_inventory(ts: TraceSet, cfg: dict[str, Any], rec: dict[str, Any]) -> pd.DataFrame:
    """Experiment A: per-source counts of valid/blank/malformed/too-short/ignored files plus length quantiles."""
    d = cfg["data"]
    attack_dir, normal_dirs = d["attack_directory"], list(d["normal_directories"])
    w = int(rec["window_size"])
    ref = rec.get("external_reference_counts", {})
    ignored = count_ignored_files(ts.root, normal_dirs + [attack_dir], attack_dir)
    m = ts.meta

    def row(label_src: str, family: str, g: pd.DataFrame, ref_count, hidden: int, non_txt: int, note: str = "") -> dict[str, Any]:
        valid = g["n_calls"] > 0
        n = g["n_calls"][valid]
        q = n.quantile([0.25, 0.5, 0.75]) if len(n) else pd.Series([np.nan] * 3, index=[0.25, 0.5, 0.75])
        observed = int(valid.sum())
        return {
            "source_group": label_src, "attack_family": family, "n_txt_files": len(g), "n_valid_traces": observed,
            "n_blank_files": int((g["read_status"] == "empty").sum()),
            "n_no_valid_calls": int((g["read_status"] == "no_valid_calls").sum()),
            "n_files_with_malformed_tokens": int((g["malformed_token_count"] > 0).sum()),
            "n_unreadable": int((g["read_status"] == "unreadable").sum()),
            "n_too_short_for_window": int((valid & (g["n_calls"] < w)).sum()),
            "n_hidden_files_ignored": hidden, "n_non_txt_files_ignored": non_txt,
            "total_calls": int(g["n_calls"].sum()),
            "min_len": int(n.min()) if len(n) else 0, "q25_len": q.iloc[0], "median_len": q.iloc[1], "q75_len": q.iloc[2],
            "max_len": int(n.max()) if len(n) else 0,
            "external_reference_count": ref_count,
            "observed_minus_reference": (observed - ref_count) if ref_count is not None else np.nan,
            "note": note,
        }

    rows = []
    for g in normal_dirs:
        sub = m[m["source_group"] == g]
        h = sum(v["hidden"] for (sg, _), v in ignored.items() if sg == g)
        nt = sum(v["non_txt"] for (sg, _), v in ignored.items() if sg == g)
        rows.append(row(g, "", sub, ref.get(g), h, nt))
    for fam in sorted(m.loc[m["label"] == LABEL_ATTACK, "attack_family"].unique()):
        sub = m[(m["source_group"] == attack_dir) & (m["attack_family"] == fam)]
        v = ignored.get((attack_dir, fam), {"hidden": 0, "non_txt": 0})
        rows.append(row(attack_dir, fam, sub, None, v["hidden"], v["non_txt"]))
    normal_sub, attack_sub = m[m["label"] == LABEL_NORMAL], m[m["label"] == LABEL_ATTACK]

    def ign(pred) -> tuple[int, int]:
        sel = [v for (sg, _), v in ignored.items() if pred(sg)]
        return sum(v["hidden"] for v in sel), sum(v["non_txt"] for v in sel)

    normal_ref = sum(ref[g] for g in normal_dirs) if all(g in ref for g in normal_dirs) else None
    rows.append(row("SUBTOTAL_NORMAL", "", normal_sub, normal_ref, *ign(lambda g: g in normal_dirs)))
    rows.append(row("SUBTOTAL_ATTACK", "", attack_sub, ref.get(attack_dir), *ign(lambda g: g == attack_dir)))
    rows.append(row("TOTAL", "", m, ref.get("total"), *ign(lambda g: True),
                    "External reference counts are quoted from published ADFA-LD documentation and are NOT treated as ground truth."))
    if ts.outside_root_entries:
        empty = m.iloc[0:0]
        rows.append({**row("OUTSIDE_DATASET_ROOT_IGNORED", "", empty, None, 0, 0,
                           "Third-party entries next to the dataset root (" + ", ".join(p.name for p in ts.outside_root_entries) + "); never read."),
                     "n_txt_files": _count_files_recursive(ts.outside_root_entries)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------------------
def validate_rec_config(rec: dict[str, Any]) -> None:
    if int(rec["window_size"]) < 2:
        raise ReconciliationConfigError("window_size must be >= 2")
    if not rec["strides"] or any(int(s) < 1 for s in rec["strides"]):
        raise ReconciliationConfigError("strides must be a non-empty list of positive integers")
    if list(rec["incomplete_policies"]) != ["drop"]:
        raise ReconciliationConfigError("incomplete_policies: only ['drop'] is implemented")
    if any(int(t) < 0 for t in rec["min_trace_length_thresholds"]):
        raise ReconciliationConfigError("min_trace_length_thresholds must be >= 0")
    if int(rec["top_k_syscalls"]) < 1:
        raise ReconciliationConfigError("top_k_syscalls must be >= 1")


def build_scenarios(rec: dict[str, Any], normal_dirs: list[str], families: list[str]) -> list[Scenario]:
    validate_rec_config(rec)
    w = int(rec["window_size"])
    all_normal = tuple(normal_dirs)
    tr, va = rec["source_groups"]["training"], rec["source_groups"]["validation"]
    sc = [Scenario("baseline_all_sources_s30", "reference", all_normal, None, w, 30,
                   notes="Current primary data reconstruction (non-overlapping windows); reference row.")]
    for s in rec["strides"]:
        sc.append(Scenario(f"grid_w{w}_s{int(s)}_drop", "grid", all_normal, None, w, int(s),
                           notes="Overlapping windows are a diagnostic only, never the final baseline." if int(s) < w else "Non-overlapping."))
    sc += [
        Scenario("src_training_normal_plus_all_attack", "source", (tr,), None, w, 30),
        Scenario("src_validation_normal_plus_all_attack", "source", (va,), None, w, 30),
        Scenario("src_training_validation_normal_plus_all_attack", "source", (tr, va), None, w, 30),
    ]
    sc += [Scenario(f"src_family_{f}_vs_normal", "source", all_normal, (f,), w, 30, notes="Single attack family versus all normal traces.")
           for f in families]
    sc += [Scenario(f"len_min{int(t)}", "length", all_normal, None, w, 30, min_trace_len=int(t),
                    notes="Traces shorter than the threshold are excluded in memory only.") for t in rec["min_trace_length_thresholds"]]
    k = int(rec["top_k_syscalls"])
    sc += [
        Scenario(f"vocab_top{k}_unk", "vocab", all_normal, None, w, 30, vocab_handling="top_unk",
                 notes=f"Diagnostic: {k} most frequent training IDs kept, others mapped to one UNK token. Frequency is an assumption."),
        Scenario(f"vocab_top{k}_drop_windows", "vocab", all_normal, None, w, 30, vocab_handling="top_drop",
                 notes=f"Diagnostic: {k} most frequent training IDs kept, windows containing other IDs removed. Frequency is an assumption."),
    ]
    return sc


# --------------------------------------------------------------------------------------
# Scenario analysis
# --------------------------------------------------------------------------------------
def _in_sources(m: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    """Traces that belong to the scenario's normal sources / attack families (before any length filter)."""
    normal = (m["label"] == LABEL_NORMAL) & m["source_group"].isin(sc.normal_sources)
    attack_ok = pd.Series(True, index=m.index) if sc.attack_families is None else m["attack_family"].isin(sc.attack_families)
    attack = (m["label"] == LABEL_ATTACK) & attack_ok
    return m[normal | attack]


def _select_traces(ts: TraceSet, sc: Scenario) -> tuple[pd.DataFrame, int]:
    in_sources = _in_sources(ts.meta, sc)
    keep = in_sources["n_calls"] >= sc.min_trace_len
    return in_sources[keep].sort_values("trace_id"), int((~keep).sum())


def _source_label(row) -> str:
    return f"{row.source_group}::{row.attack_family}" if row.label == LABEL_ATTACK else row.source_group


def _sample_windows(calls_list, starts_list, labels, window_size, cap, seed):
    """Deterministically pick up to `cap` windows (uniformly at random over all windows) -> (texts, labels, total_windows)."""
    counts = np.array([len(s) for s in starts_list], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return [], np.empty(0, dtype=int), 0
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(total, size=cap, replace=False)) if total > cap else np.arange(total)
    cum = np.cumsum(counts)
    ti = np.searchsorted(cum, pick, side="right")
    off = pick - (cum[ti] - counts[ti])
    texts, ys = [], []
    for t, o in zip(ti.tolist(), off.tolist()):
        st = int(starts_list[t][o])
        texts.append(" ".join(map(str, calls_list[t][st:st + window_size])))
        ys.append(labels[t])
    return texts, np.asarray(ys), total


def analyze_scenario(ts: TraceSet, sc: Scenario, feat_cfg: dict[str, Any], split_cfg: dict[str, Any], rec: dict[str, Any],
                     seed: int, do_fit: bool = True) -> dict[str, Any]:
    """Compute every reconciliation metric for one scenario. Read-only; trains no model."""
    w, s = sc.window_size, sc.stride
    sel, n_excluded_len = _select_traces(ts, sc)
    ids = sel["trace_id"].tolist()
    labels = sel["label"].to_numpy()
    src_labels = [_source_label(r) for r in sel.itertuples(index=False)]
    calls = {t: ts.calls[t] for t in ids if t in ts.calls}
    base_starts = {t: window_starts(len(calls.get(t, [])), w, s) for t in ids}
    usable = [i for i, t in enumerate(ids) if len(base_starts[t]) > 0]

    res: dict[str, Any] = {"scenario": sc, "n_traces_selected": len(ids), "n_traces_excluded_by_min_len": n_excluded_len,
                           "n_traces_excluded_by_min_len_normal": 0, "n_traces_excluded_by_min_len_attack": 0,
                           "fit_status": "not_run", "split_status": "ok"}
    in_src = _in_sources(ts.meta, sc)
    short = in_src[in_src["n_calls"] < sc.min_trace_len]
    res["n_traces_excluded_by_min_len_normal"] = int((short["label"] == LABEL_NORMAL).sum())
    res["n_traces_excluded_by_min_len_attack"] = int((short["label"] == LABEL_ATTACK).sum())

    # ---- trace-disjoint stratified split (same policy/seed as the primary baseline), needed for vocab + fit ----
    split: dict[str, list[int]] | None = None
    try:
        u_ids = np.array([ids[i] for i in usable])
        u_lab = np.array([labels[i] for i in usable])
        tr, va, te = _two_stage_stratified(u_ids, u_lab, float(split_cfg["test_size"]), float(split_cfg["validation_size"]),
                                            int(split_cfg["seed"]), "traces")
        pos = {t: i for i, t in enumerate(ids)}
        split = {"train": [pos[t] for t in tr], "validation": [pos[t] for t in va], "test": [pos[t] for t in te]}
    except SplitError as exc:
        res["split_status"] = f"skipped: {str(exc).splitlines()[0]}"
    except ValueError as exc:  # e.g. no traces at all
        res["split_status"] = f"skipped: {exc}"

    # ---- vocabulary handling ----
    calls_used, excluded_lut, unk = dict(calls), None, int(rec["unk_token_id"])
    res.update({"vocab_ids_excluded": 0, "vocab_excluded_id_list": "", "vocab_share_calls_excluded": 0.0})
    if sc.vocab_handling != "none":
        if split is None:
            raise ValueError(f"Scenario {sc.scenario_id} needs a train partition but the split failed: {res['split_status']}")
        max_id = max(int(c.max()) for c in calls.values())
        counts = np.zeros(max_id + 1, dtype=np.int64)
        for i in split["train"]:
            counts += np.bincount(calls[ids[i]], minlength=max_id + 1)
        kept = top_k_ids(counts, int(rec["top_k_syscalls"]))
        all_seen = np.zeros(max_id + 1, dtype=np.int64)
        for c in calls.values():
            all_seen += np.bincount(c, minlength=max_id + 1)
        excluded_ids = np.setdiff1d(np.flatnonzero(all_seen > 0), kept)
        excluded_lut = np.zeros(max_id + 1, dtype=bool)
        excluded_lut[excluded_ids] = True
        res["vocab_ids_excluded"] = int(len(excluded_ids))
        res["vocab_excluded_id_list"] = " ".join(map(str, excluded_ids.tolist()))
        res["vocab_share_calls_excluded"] = float(all_seen[excluded_ids].sum() / max(all_seen.sum(), 1))
        if sc.vocab_handling == "top_unk":
            mapper = np.arange(max_id + 1, dtype=np.int64)
            mapper[excluded_ids] = unk
            calls_used = {t: mapper[c] for t, c in calls.items()}
    starts_used: dict[str, np.ndarray] = {}
    for t in ids:
        st = base_starts[t]
        if sc.vocab_handling == "top_drop":
            st = drop_windows_with_excluded(calls[t], st, w, excluded_lut)
        starts_used[t] = st

    # ---- counts, coverage, vocab sizes ----
    n_win = {LABEL_NORMAL: 0, LABEL_ATTACK: 0}
    traces_contrib = {LABEL_NORMAL: 0, LABEL_ATTACK: 0}
    by_source: dict[str, int] = {}
    partial_windows = dropped_calls = 0
    used_ids: list[np.ndarray] = []
    all_ids_included: list[np.ndarray] = []
    pair_codes: list[np.ndarray] = []
    base = max(int(max((c.max() for c in calls_used.values()), default=0)), unk) + 2
    for t, y, sl in zip(ids, labels, src_labels):
        c, st = calls_used.get(t, np.empty(0, np.int64)), starts_used[t]
        n = len(c)
        n_win[int(y)] += len(st)
        by_source[sl] = by_source.get(sl, 0) + len(st)
        if len(st):
            traces_contrib[int(y)] += 1
        if n:
            bs = base_starts[t]                       # windows before any vocabulary handling
            partial_windows += len(range(0, n, s)) - len(bs)
            dropped_calls += n - (int(bs[-1]) + w if len(bs) else 0)
            all_ids_included.append(c)
        if len(st):
            used_ids.append(c[covered_position_mask(n, st, w)])
            pm = covered_pair_mask(n, st, w)
            pair_codes.append((c[:-1][pm] * base + c[1:][pm]))
    normal_w, attack_w = n_win[LABEL_NORMAL], n_win[LABEL_ATTACK]
    total_w = normal_w + attack_w
    res.update({
        "normal_windows": normal_w, "attack_windows": attack_w, "total_windows": total_w,
        "attack_percentage": round(100.0 * attack_w / total_w, 4) if total_w else float("nan"),
        "normal_traces_contributing": traces_contrib[LABEL_NORMAL], "attack_traces_contributing": traces_contrib[LABEL_ATTACK],
        "unique_syscalls": int(len(np.unique(np.concatenate(used_ids)))) if used_ids else 0,
        "unique_syscalls_all_calls_of_included_traces": int(len(np.unique(np.concatenate(all_ids_included)))) if all_ids_included else 0,
        "unique_bigrams": int(len(np.unique(np.concatenate(pair_codes)))) if pair_codes else 0,
        "windows_by_source": by_source, "partial_trailing_windows_dropped": partial_windows, "dropped_trailing_calls": dropped_calls,
    })
    if split is not None:
        for name in ("train", "validation", "test"):
            res[f"windows_{name}"] = int(sum(len(starts_used[ids[i]]) for i in split[name]))

    # ---- train-only TF-IDF + chi2 fit (diagnostic): vocabulary sizes, selected bigrams, all-zero windows ----
    fit_cols = {"tfidf_train_vocab_size": np.nan, "chi2_selected_features": np.nan, "selected_bigram_names": "",
                "pct_windows_all_zero_train": np.nan, "pct_windows_all_zero_test": np.nan,
                "fit_train_windows_used": np.nan, "fit_test_windows_used": np.nan, "fit_subsampled": False}
    res.update(fit_cols)
    if not do_fit:
        res["fit_status"] = "skipped (fit_features=false)"
    elif split is None:
        res["fit_status"] = res["split_status"]
    else:
        cap = int(rec["zero_check"]["max_windows_per_partition"])
        parts = {}
        for j, name in enumerate(("train", "test")):
            idx = split[name]
            parts[name] = _sample_windows([calls_used[ids[i]] for i in idx], [starts_used[ids[i]] for i in idx],
                                          [int(labels[i]) for i in idx], w, cap, seed + j)
        (tr_txt, tr_y, tr_total), (te_txt, te_y, te_total) = parts["train"], parts["test"]
        if len(set(tr_y.tolist())) < 2 or not te_txt:
            res["fit_status"] = "skipped: train partition lacks a class or test partition is empty"
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = fit_features(tr_txt, tr_y, {"test": te_txt}, feat_cfg)
            names = fitted["vectorizer"].get_feature_names_out()[fitted["selector"].get_support()]
            xtr, xte = fitted["matrices"]["train"], fitted["matrices"]["test"]
            res.update({
                "tfidf_train_vocab_size": int(fitted["vocab_size"]), "chi2_selected_features": int(fitted["k_actual"]),
                "selected_bigram_names": ";".join(names),
                "pct_windows_all_zero_train": round(100.0 * float((xtr.getnnz(axis=1) == 0).mean()), 4),
                "pct_windows_all_zero_test": round(100.0 * float((xte.getnnz(axis=1) == 0).mean()), 4),
                "fit_train_windows_used": len(tr_txt), "fit_test_windows_used": len(te_txt),
                "fit_subsampled": bool(tr_total > cap or te_total > cap),
                "fit_status": "ok" + (" (fewer features than requested)" if any("fewer than the requested" in str(x.message) for x in caught) else ""),
            })
    return res


def run_scenarios(ts: TraceSet, scenarios: list[Scenario], cfg: dict[str, Any], rec: dict[str, Any], seed: int,
                  do_fit: bool = True) -> list[dict[str, Any]]:
    """Analyse every scenario; identical parameter sets are computed once and reused (labelled by their own ID)."""
    memo: dict[tuple, dict[str, Any]] = {}
    out = []
    for sc in scenarios:
        if sc.key() not in memo:
            log.info("Analysing %s", sc.scenario_id)
            memo[sc.key()] = analyze_scenario(ts, sc, cfg["features"], cfg["split"], rec, seed, do_fit)
        out.append({**memo[sc.key()], "scenario": sc})
    return out


# --------------------------------------------------------------------------------------
# Result tables
# --------------------------------------------------------------------------------------
def _fmt_sources(sc: Scenario) -> tuple[str, str]:
    return ";".join(sc.normal_sources), ("Attack_Data_Master (all families)" if sc.attack_families is None
                                         else "Attack_Data_Master:" + ";".join(sc.attack_families))


def _fmt_length(sc: Scenario) -> str:
    return "none" if sc.min_trace_len == 0 else f"min_trace_len>={sc.min_trace_len}"


def _fmt_vocab(sc: Scenario, k: int) -> str:
    return {"none": "none", "top_unk": f"top{k}_train_ids_others_to_UNK", "top_drop": f"top{k}_train_ids_drop_windows_with_others"}[sc.vocab_handling]


def descriptive_distances(total: int, attack_pct: float, paper: dict[str, Any]) -> tuple[float, float]:
    """Absolute differences to the paper's total and attack ratio. Descriptive only, never an objective."""
    paper_ratio = paper["attack_instances"] / paper["total_instances"]
    d_total = abs(int(total) - int(paper["total_instances"]))
    d_ratio = abs(attack_pct / 100.0 - paper_ratio) if attack_pct == attack_pct else float("nan")
    return d_total, round(d_ratio, 4) if d_ratio == d_ratio else d_ratio


def summary_table(results: list[dict[str, Any]], rec: dict[str, Any]) -> pd.DataFrame:
    paper, k = rec["paper_reference"], int(rec["top_k_syscalls"])
    rows = []
    for r in results:
        sc: Scenario = r["scenario"]
        ns, at = _fmt_sources(sc)
        d_total, d_ratio = descriptive_distances(r["total_windows"], r["attack_percentage"], paper)
        rows.append({
            "scenario_id": sc.scenario_id, "normal_trace_sources": ns, "attack_sources": at, "window_size": sc.window_size,
            "stride": sc.stride, "length_filter": _fmt_length(sc), "vocabulary_handling": _fmt_vocab(sc, k),
            "normal_windows": r["normal_windows"], "attack_windows": r["attack_windows"], "total_windows": r["total_windows"],
            "attack_percentage": r["attack_percentage"], "unique_syscalls": r["unique_syscalls"], "unique_bigrams": r["unique_bigrams"],
            "distance_to_paper_total": d_total, "distance_to_paper_attack_ratio": d_ratio,
            "notes": f"EXPLORATORY: {sc.notes}".strip() + " Distances are descriptive, not objectives.",
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _common(r: dict[str, Any], k: int) -> dict[str, Any]:
    sc: Scenario = r["scenario"]
    ns, at = _fmt_sources(sc)
    return {"scenario_id": sc.scenario_id, "normal_trace_sources": ns, "attack_sources": at, "window_size": sc.window_size,
            "stride": sc.stride, "length_filter": _fmt_length(sc), "vocabulary_handling": _fmt_vocab(sc, k)}


def window_grid_table(results, rec) -> pd.DataFrame:
    """Experiment B."""
    k = int(rec["top_k_syscalls"])
    rows = []
    for r in results:
        sc = r["scenario"]
        if sc.group not in ("grid",):
            continue
        rows.append({
            "scenario_id": sc.scenario_id, "window_size": sc.window_size, "stride": sc.stride, "incomplete_policy": "drop",
            "total_windows": r["total_windows"], "normal_windows": r["normal_windows"], "attack_windows": r["attack_windows"],
            "attack_percentage": r["attack_percentage"],
            "contributing_traces_normal": r["normal_traces_contributing"], "contributing_traces_attack": r["attack_traces_contributing"],
            "contributing_traces_total": r["normal_traces_contributing"] + r["attack_traces_contributing"],
            "unique_syscall_ids": r["unique_syscalls"], "unique_bigrams": r["unique_bigrams"],
            "pct_windows_all_zero_train": r["pct_windows_all_zero_train"], "pct_windows_all_zero_test": r["pct_windows_all_zero_test"],
            "fit_train_windows_used": r["fit_train_windows_used"], "fit_test_windows_used": r["fit_test_windows_used"],
            "fit_subsampled": r["fit_subsampled"], "fit_status": r["fit_status"], "experiment_label": EXPLORATORY_LABEL,
        })
    return pd.DataFrame(rows)


def source_scenario_table(results, rec) -> pd.DataFrame:
    """Experiment C (baseline row included for reference); keeps per-source window counts."""
    k = int(rec["top_k_syscalls"])
    paper = rec["paper_reference"]
    sel = [r for r in results if r["scenario"].group in ("source", "reference")]
    labels = sorted({s for r in sel for s in r["windows_by_source"]})
    rows = []
    for r in sel:
        d_total, d_ratio = descriptive_distances(r["total_windows"], r["attack_percentage"], paper)
        row = {**_common(r, k), "normal_traces_contributing": r["normal_traces_contributing"],
               "attack_traces_contributing": r["attack_traces_contributing"], "normal_windows": r["normal_windows"],
               "attack_windows": r["attack_windows"], "total_windows": r["total_windows"], "attack_percentage": r["attack_percentage"],
               "normal_to_attack_window_ratio": round(r["normal_windows"] / r["attack_windows"], 3) if r["attack_windows"] else float("nan"),
               "distance_to_paper_total": d_total, "distance_to_paper_attack_ratio": d_ratio}
        for lab in labels:
            row[f"windows__{lab}"] = r["windows_by_source"].get(lab, 0)
        row["experiment_label"] = EXPLORATORY_LABEL
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["closest_attack_ratio_to_paper_in_table"] = df["distance_to_paper_attack_ratio"] == df["distance_to_paper_attack_ratio"].min()
        df["closest_total_to_paper_in_table"] = df["distance_to_paper_total"] == df["distance_to_paper_total"].min()
    return df


def length_filter_table(results, rec) -> pd.DataFrame:
    """Experiment D (baseline row = no filter)."""
    k = int(rec["top_k_syscalls"])
    rows = []
    for r in results:
        sc = r["scenario"]
        if sc.group not in ("length", "reference"):
            continue
        rows.append({
            "scenario_id": sc.scenario_id, "min_trace_len": sc.min_trace_len, "window_size": sc.window_size, "stride": sc.stride,
            "traces_excluded_normal": r["n_traces_excluded_by_min_len_normal"], "traces_excluded_attack": r["n_traces_excluded_by_min_len_attack"],
            "normal_windows": r["normal_windows"], "attack_windows": r["attack_windows"], "total_windows": r["total_windows"],
            "attack_percentage": r["attack_percentage"], "unique_syscalls": r["unique_syscalls"], "unique_bigrams": r["unique_bigrams"],
            "partial_trailing_windows_dropped": r["partial_trailing_windows_dropped"], "dropped_trailing_calls": r["dropped_trailing_calls"],
            "experiment_label": EXPLORATORY_LABEL,
        })
    return pd.DataFrame(rows)


def vocabulary_scenario_table(results, rec) -> pd.DataFrame:
    """Experiment E scenarios (baseline row = all observed IDs)."""
    k = int(rec["top_k_syscalls"])
    rows = []
    for r in results:
        sc = r["scenario"]
        if sc.group not in ("vocab", "reference"):
            continue
        rows.append({
            "scenario_id": sc.scenario_id, "vocabulary_handling": _fmt_vocab(sc, k), "requested_vocabulary_size": k if sc.vocab_handling != "none" else "all observed",
            "ids_excluded": r["vocab_ids_excluded"], "share_of_all_calls_excluded": round(r["vocab_share_calls_excluded"], 6),
            "excluded_id_list": r["vocab_excluded_id_list"], "normal_windows": r["normal_windows"], "attack_windows": r["attack_windows"],
            "total_windows": r["total_windows"], "attack_percentage": r["attack_percentage"],
            "unique_syscalls_in_windows": r["unique_syscalls"], "unique_bigrams": r["unique_bigrams"],
            "paper_distinct_syscalls": rec["paper_reference"]["distinct_syscalls"],
            "note": "UNK counts as one extra token" if sc.vocab_handling == "top_unk" else "", "experiment_label": EXPLORATORY_LABEL,
        })
    return pd.DataFrame(rows)


def bigram_scenario_table(results, rec) -> pd.DataFrame:
    """Experiment F: bigram vocabulary for every scenario, compared with the paper's 2,805 and 150."""
    paper = rec["paper_reference"]
    rows = []
    for r in results:
        sc = r["scenario"]
        v = r["tfidf_train_vocab_size"]
        rows.append({
            "scenario_id": sc.scenario_id, "raw_observed_bigrams": r["unique_bigrams"], "tfidf_train_vocab_size": v,
            "chi2_requested": int(paper["selected_features"]), "chi2_selected_features": r["chi2_selected_features"],
            "paper_initial_bigrams": paper["initial_bigrams"],
            "tfidf_vocab_minus_paper_initial": (v - paper["initial_bigrams"]) if v == v else np.nan,
            "paper_selected_features": paper["selected_features"],
            "pct_windows_all_zero_train": r["pct_windows_all_zero_train"], "pct_windows_all_zero_test": r["pct_windows_all_zero_test"],
            "fit_train_windows_used": r["fit_train_windows_used"], "fit_subsampled": r["fit_subsampled"], "fit_status": r["fit_status"],
            "selected_bigram_names": r["selected_bigram_names"], "experiment_label": EXPLORATORY_LABEL,
        })
    return pd.DataFrame(rows)


def syscall_frequency_table(ts: TraceSet, rec: dict[str, Any], split_cfg: dict[str, Any], all_normal: tuple[str, ...]) -> pd.DataFrame:
    """Experiment E: every observed syscall ID with its frequency by class and source group; low-frequency and top-k flags.

    Ranks/top-k flags use the calls of the TRAIN partition of the trace-disjoint split (never the test partition).
    """
    m = ts.meta
    max_id = max(int(c.max()) for c in ts.calls.values())
    src_of = {r.trace_id: _source_label(r) for r in m.itertuples(index=False)}
    lab_of = dict(zip(m["trace_id"], m["label"]))
    per_source: dict[str, np.ndarray] = {}
    for t, c in ts.calls.items():
        per_source.setdefault(src_of[t], np.zeros(max_id + 1, np.int64))
        per_source[src_of[t]] += np.bincount(c, minlength=max_id + 1)
    total = sum(per_source.values())
    normal = sum(v for s, v in per_source.items() if "::" not in s)
    attack = total - normal
    ntraces = np.zeros(max_id + 1, np.int64)
    for c in ts.calls.values():
        ntraces[np.unique(c)] += 1

    usable = m[m["n_calls"] >= int(rec["window_size"])].sort_values("trace_id")
    ids, lab = usable["trace_id"].to_numpy(), usable["label"].to_numpy()
    tr, _, _ = _two_stage_stratified(ids, lab, float(split_cfg["test_size"]), float(split_cfg["validation_size"]), int(split_cfg["seed"]), "traces")
    train_counts = np.zeros(max_id + 1, np.int64)
    for t in tr:
        train_counts += np.bincount(ts.calls[t], minlength=max_id + 1)
    top = set(top_k_ids(train_counts, int(rec["top_k_syscalls"])).tolist())
    observed = np.flatnonzero(total > 0)
    order = np.lexsort((observed, -total[observed]))
    rank_all = {int(i): r + 1 for r, i in enumerate(observed[order])}
    obs_train = np.flatnonzero(train_counts > 0)
    rank_train = {int(i): r + 1 for r, i in enumerate(obs_train[np.lexsort((obs_train, -train_counts[obs_train]))])}
    thr = int(rec["low_frequency_threshold"])
    rows = []
    for i in observed.tolist():
        row = {"syscall_id": i, "total_count": int(total[i]), "share_of_all_calls": round(float(total[i] / total.sum()), 8),
               "normal_count": int(normal[i]), "attack_count": int(attack[i])}
        for s in sorted(per_source):
            row[f"count__{s}"] = int(per_source[s][i])
        row.update({"n_traces_containing": int(ntraces[i]), "rank_overall": rank_all[i], "train_partition_count": int(train_counts[i]),
                    "rank_train_partition": rank_train.get(i, np.nan), f"in_top{int(rec['top_k_syscalls'])}_train": i in top,
                    "low_frequency": bool(total[i] < thr), "only_in_one_class": bool((normal[i] == 0) != (attack[i] == 0)),
                    "experiment_label": EXPLORATORY_LABEL})
        rows.append(row)
    return pd.DataFrame(rows)


def balance_feasibility_table(ts: TraceSet, results, rec) -> pd.DataFrame:
    """Experiment G: what a balanced 52,656-window dataset would require. Nothing is generated or resampled."""
    paper, w = rec["paper_reference"], int(rec["window_size"])
    target = paper["total_instances"] // 2
    m = ts.meta
    k = int(rec["top_k_syscalls"])
    rows = []
    for r in results:
        sc: Scenario = r["scenario"]
        if sc.group not in ("source", "reference") or sc.stride != w:
            continue
        sel, _ = _select_traces(ts, sc)
        a_len = sel.loc[sel["label"] == LABEL_ATTACK, "n_calls"].to_numpy()
        per_stride = {s: int(sum(count_windows(int(n), w, s) for n in a_len)) for s in range(1, w + 1)}
        reaching = [s for s, v in per_stride.items() if v >= target]
        a, n = r["attack_windows"], r["normal_windows"]
        rows.append({
            **{k_: v for k_, v in _common(r, k).items() if k_ in ("scenario_id", "normal_trace_sources", "attack_sources")},
            "window_size": w, "stride": sc.stride, "normal_windows": n, "attack_windows": a,
            "balanced_target_per_class": target, "attack_windows_vs_target_deficit": target - a,
            "attack_windows_support_target": bool(a >= target), "attack_share_of_target": round(a / target, 4),
            "normal_windows_surplus_over_target": n - target, "largest_balanced_nonoverlap_total": 2 * min(a, n),
            "paper_attack_instances": paper["attack_instances"], "attack_deficit_vs_paper_attack": paper["attack_instances"] - a,
            "attack_windows_at_stride_1": per_stride[1],
            "largest_stride_reaching_target_attack": max(reaching) if reaching else np.nan,
            "interpretation": ("non-overlapping attack windows are insufficient; reaching the target would need overlap, a different "
                               "segmentation, resampling, or another undisclosed step (hypotheses, not findings)") if a < target
            else "non-overlapping attack windows suffice for the target", "experiment_label": EXPLORATORY_LABEL,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures (matplotlib imported lazily so the module imports without it)
# --------------------------------------------------------------------------------------
def make_figures(ts: TraceSet, results, rec, freq: pd.DataFrame, fig_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paper, w = rec["paper_reference"], int(rec["window_size"])
    paths = []

    grid = sorted([r for r in results if r["scenario"].group == "grid"], key=lambda r: r["scenario"].stride)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = [r["scenario"].stride for r in grid]
    for key, lab, col in (("total_windows", "total", "k"), ("normal_windows", "normal", "tab:blue"), ("attack_windows", "attack", "tab:red")):
        ax.plot(x, [r[key] for r in grid], "o-", label=lab, color=col)
    ax.axhline(paper["total_instances"], ls="--", color="gray", lw=1, label=f"paper total {paper['total_instances']:,} (reference only)")
    ax.axhline(paper["attack_instances"], ls=":", color="tab:red", lw=1, label=f"paper attack {paper['attack_instances']:,} (reference only)")
    ax.set_yscale("log"); ax.set_xlabel(f"stride (window size {w}; stride {w} = non-overlapping)"); ax.set_ylabel("number of windows (log)")
    ax.set_title("EXPLORATORY: window count vs stride"); ax.legend(fontsize=8); fig.tight_layout()
    p = fig_dir / "reconciliation_window_count_vs_stride.png"; fig.savefig(p, dpi=200); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(results) + 1.5)))
    seen, labs, vals = set(), [], []
    for r in results:
        if r["scenario"].scenario_id in seen:
            continue
        seen.add(r["scenario"].scenario_id); labs.append(r["scenario"].scenario_id); vals.append(r["attack_percentage"])
    ax.barh(range(len(labs)), vals, color="tab:red", alpha=0.7)
    ax.axvline(100 * paper["attack_instances"] / paper["total_instances"], ls="--", color="k", label="paper attack share (reference only)")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("attack windows (%)"); ax.set_title("EXPLORATORY: attack share by scenario"); ax.legend(fontsize=8); fig.tight_layout()
    p = fig_dir / "reconciliation_attack_ratio_by_scenario.png"; fig.savefig(p, dpi=200); plt.close(fig); paths.append(p)

    m = ts.meta[ts.meta["n_calls"] > 0]
    groups = [(f"{r.source_group}\n{r.attack_family}" if r.attack_family else r.source_group) for r in m.itertuples(index=False)]
    order = sorted(set(groups))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(order)), 4.8))
    ax.boxplot([m["n_calls"].to_numpy()[np.array(groups) == g] for g in order], showfliers=True)
    ax.set_xticklabels(order)
    ax.axhline(w, ls="--", color="k", lw=1, label=f"window size {w}")
    ax.set_yscale("log"); ax.set_ylabel("trace length (calls, log)"); ax.set_title("EXPLORATORY: trace length by source group")
    ax.tick_params(axis="x", labelsize=7); ax.legend(fontsize=8); fig.tight_layout()
    p = fig_dir / "reconciliation_trace_length_by_class.png"; fig.savefig(p, dpi=200); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for col, lab, colr in (("total_count", "all", "k"), ("normal_count", "normal", "tab:blue"), ("attack_count", "attack", "tab:red")):
        v = np.sort(freq[col].to_numpy())[::-1]
        v = v[v > 0]
        ax.loglog(np.arange(1, len(v) + 1), v, "-", label=lab, color=colr)
    ax.axvline(paper["distinct_syscalls"], ls="--", color="gray", label=f"rank {paper['distinct_syscalls']} (paper vocabulary, reference only)")
    ax.axhline(int(rec["low_frequency_threshold"]), ls=":", color="orange", label="low-frequency threshold")
    ax.set_xlabel("syscall rank by frequency"); ax.set_ylabel("call count"); ax.set_title(f"EXPLORATORY: syscall frequency ({len(freq)} observed IDs)")
    ax.legend(fontsize=8); fig.tight_layout()
    p = fig_dir / "reconciliation_syscall_frequency.png"; fig.savefig(p, dpi=200); plt.close(fig); paths.append(p)
    return paths


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def run_reconciliation(cfg: dict[str, Any], rec: dict[str, Any], results_dir: Path, do_fit: bool | None = None, write_audit: bool = False) -> dict[str, Any]:
    """Run experiments A-G and write tables/figures under results_dir. Never touches data/processed or data/artifacts."""
    validate_rec_config(rec)
    fit = bool(rec["zero_check"]["fit_features"]) if do_fit is None else do_fit
    ts = load_traceset(cfg)
    tab_dir, fig_dir = results_dir / "tables", results_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    families = sorted(ts.meta.loc[ts.meta["label"] == LABEL_ATTACK, "attack_family"].unique())
    scenarios = build_scenarios(rec, list(cfg["data"]["normal_directories"]), families)
    results = run_scenarios(ts, scenarios, cfg, rec, int(cfg["seed"]), fit)

    tables = {
        "reconciliation_raw_inventory": raw_inventory(ts, cfg, rec),
        "reconciliation_window_grid": window_grid_table(results, rec),
        "reconciliation_source_group_scenarios": source_scenario_table(results, rec),
        "reconciliation_length_filter_scenarios": length_filter_table(results, rec),
        "reconciliation_syscall_frequency": syscall_frequency_table(ts, rec, cfg["split"], tuple(cfg["data"]["normal_directories"])),
        "reconciliation_vocabulary_scenarios": vocabulary_scenario_table(results, rec),
        "reconciliation_bigram_scenarios": bigram_scenario_table(results, rec),
        "reconciliation_balance_feasibility": balance_feasibility_table(ts, results, rec),
        "reconciliation_summary": summary_table(results, rec),
    }
    for name, df in tables.items():
        df.to_csv(tab_dir / f"{name}.csv", index=False)
    figs = make_figures(ts, results, rec, tables["reconciliation_syscall_frequency"], fig_dir)
    out = {"tables": tables, "figures": figs, "results": results, "traceset": ts}
    if write_audit:
        out["audit_artifacts"] = write_audit_artifacts(
            out, cfg, rec, results_dir.parent / "reports",
            Path(cfg["data"]["processed_data_dir"]),
            str(cfg.get("_audit_command", "python scripts/run_reconciliation.py --config configs/baseline.yaml")),
        )
    return out
