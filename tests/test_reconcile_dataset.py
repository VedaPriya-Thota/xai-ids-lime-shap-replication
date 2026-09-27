"""Tests for src/reconcile_dataset.py using SYNTHETIC data only (never real ADFA-LD).

Every reconciliation calculation is checked against a plain brute-force implementation.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import ATTACK_DIR, NORMAL_DIRS, make_synthetic_adfa
from src import reconcile_dataset as rd
from src.data import DatasetNotFoundError, SplitError, _two_stage_stratified
from src.config import PROJECT_ROOT


@pytest.fixture
def rec():
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "reconciliation.yaml"))
    cfg["strides"] = [1, 5, 10, 30]
    cfg["min_trace_length_thresholds"] = [30, 100, 150]
    cfg["zero_check"] = {"max_windows_per_partition": 400, "fit_features": True}
    cfg["top_k_syscalls"] = 30
    cfg["low_frequency_threshold"] = 200   # synthetic IDs 26-40 (attack only) have ~126 calls each
    return cfg


@pytest.fixture
def ts(synthetic_cfg):
    return rd.load_traceset(synthetic_cfg)


def brute_windows(calls, w, s):
    return [tuple(calls[i:i + w]) for i in range(0, len(calls) - w + 1, s)]


# ---------------- pure counting helpers vs brute force ----------------
@pytest.mark.parametrize("n,w,s", [(0, 30, 30), (10, 30, 30), (30, 30, 30), (31, 30, 1), (100, 30, 30), (100, 30, 7), (65, 30, 10), (200, 30, 1)])
def test_count_windows_and_starts_match_brute_force(n, w, s):
    calls = list(range(n))
    expected = brute_windows(calls, w, s)
    assert rd.count_windows(n, w, s) == len(expected)
    assert list(rd.window_starts(n, w, s)) == [c[0] for c in expected]


@pytest.mark.parametrize("w,s", [(30, 1), (30, 5), (30, 29), (30, 30), (10, 3), (10, 15), (30, 40)])
def test_covered_pairs_and_positions_match_brute_force(w, s):
    rng = np.random.default_rng(0)
    calls = rng.integers(0, 6, size=97)
    starts = rd.window_starts(len(calls), w, s)
    exp_pairs = set()
    covered = np.zeros(len(calls), bool)
    for st in starts:
        win = calls[st:st + w]
        exp_pairs.update(zip(win[:-1].tolist(), win[1:].tolist()))
        covered[st:st + w] = True
    pm = rd.covered_pair_mask(len(calls), starts, w)
    got_pairs = set(zip(calls[:-1][pm].tolist(), calls[1:][pm].tolist()))
    assert got_pairs == exp_pairs
    np.testing.assert_array_equal(rd.covered_position_mask(len(calls), starts, w), covered)


def test_non_overlapping_windows_do_not_count_boundary_bigrams():
    calls = np.arange(60)
    starts = rd.window_starts(60, 30, 30)
    pm = rd.covered_pair_mask(60, starts, 30)
    assert not pm[29]              # pair (29, 30) crosses the window boundary
    assert pm.sum() == 2 * 29


def test_drop_windows_with_excluded_and_top_k_ids():
    calls = np.array([1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1])
    starts = rd.window_starts(len(calls), 4, 4)                  # [0, 4, 8]
    lut = np.zeros(10, bool)
    lut[9] = True
    assert list(rd.drop_windows_with_excluded(calls, starts, 4, lut)) == [0, 8]
    counts = np.array([0, 5, 5, 7, 0, 2])
    assert list(rd.top_k_ids(counts, 3)) == [3, 1, 2]           # frequency, then smaller id first; zeros are ineligible
    assert list(rd.top_k_ids(counts, 10)) == [3, 1, 2, 5]


# ---------------- experiment A: inventory ----------------
def test_raw_inventory_counts_all_file_kinds(tmp_path, rec):
    from src.config import load_config
    root = make_synthetic_adfa(tmp_path / "raw" / "ADFA-LD")
    (root / NORMAL_DIRS[0] / "blank.txt").write_text("   \n")
    (root / NORMAL_DIRS[0] / "malformed.txt").write_text(" ".join(["3"] * 40 + ["x", "-1"]))
    (root / NORMAL_DIRS[1] / "short.txt").write_text(" ".join(["4"] * 12))
    (root / NORMAL_DIRS[1] / ".DS_Store").write_text("junk")
    (root / NORMAL_DIRS[1] / "notes.md").write_text("not a trace")
    (root / ATTACK_DIR / "Adduser_1" / "allbad.txt").write_text("a b c")
    (tmp_path / "raw" / "TRAINING").mkdir()
    (tmp_path / "raw" / "TRAINING" / "third_party.txt").write_text("1 2 3")
    cfg = load_config()
    cfg["data"].update(raw_data_dir=str(tmp_path / "raw"))
    ts = rd.load_traceset(cfg)
    inv = rd.raw_inventory(ts, cfg, rec).set_index(["source_group", "attack_family"])
    t0, v0 = inv.loc[(NORMAL_DIRS[0], "")], inv.loc[(NORMAL_DIRS[1], "")]
    assert t0.n_txt_files == 32 and t0.n_valid_traces == 31 and t0.n_blank_files == 1
    assert t0.n_files_with_malformed_tokens == 1 and t0.n_too_short_for_window == 0
    assert v0.n_txt_files == 31 and v0.n_too_short_for_window == 1 and v0.n_hidden_files_ignored == 1 and v0.n_non_txt_files_ignored == 1
    adduser = inv.loc[(ATTACK_DIR, "Adduser")]
    assert adduser.n_no_valid_calls == 1 and adduser.n_valid_traces == 10
    tot = inv.loc[("TOTAL", "")]
    assert tot.n_hidden_files_ignored == 1 and tot.n_non_txt_files_ignored == 1
    assert tot.n_txt_files == len(ts.meta)
    out = inv.loc[("OUTSIDE_DATASET_ROOT_IGNORED", "")]
    assert out.n_txt_files == 1 and out.n_valid_traces == 0
    assert inv.loc[("SUBTOTAL_NORMAL", ""), "n_valid_traces"] + inv.loc[("SUBTOTAL_ATTACK", ""), "n_valid_traces"] == tot.n_valid_traces


def test_inventory_reference_counts_are_recorded_not_enforced(ts, synthetic_cfg, rec):
    inv = rd.raw_inventory(ts, synthetic_cfg, rec)
    tot = inv[inv.source_group == "TOTAL"].iloc[0]
    assert tot.external_reference_count == rec["external_reference_counts"]["total"]
    assert tot.observed_minus_reference == tot.n_valid_traces - tot.external_reference_count   # no assertion that they match


# ---------------- scenarios and per-scenario counts vs brute force ----------------
def brute_scenario(ts, sc):
    n_win = {0: 0, 1: 0}
    pairs, ids = set(), set()
    for r in ts.meta.itertuples(index=False):
        if r.label == 0 and r.source_group not in sc.normal_sources:
            continue
        if r.label == 1 and sc.attack_families is not None and r.attack_family not in sc.attack_families:
            continue
        if r.n_calls < sc.min_trace_len or r.trace_id not in ts.calls:
            continue
        for win in brute_windows(ts.calls[r.trace_id].tolist(), sc.window_size, sc.stride):
            n_win[r.label] += 1
            ids.update(win)
            pairs.update(zip(win[:-1], win[1:]))
    return n_win, len(ids), len(pairs)


def _run(ts, sc, cfg, rec, fit=False):
    return rd.analyze_scenario(ts, sc, cfg["features"], cfg["split"], rec, cfg["seed"], do_fit=fit)


@pytest.mark.parametrize("stride", [1, 5, 10, 30])
def test_window_grid_counts_match_brute_force(ts, synthetic_cfg, rec, stride):
    sc = rd.Scenario(f"g{stride}", "grid", tuple(NORMAL_DIRS), None, 30, stride)
    r = _run(ts, sc, synthetic_cfg, rec)
    n_win, n_ids, n_pairs = brute_scenario(ts, sc)
    assert (r["normal_windows"], r["attack_windows"]) == (n_win[0], n_win[1])
    assert r["total_windows"] == n_win[0] + n_win[1]
    assert r["unique_syscalls"] == n_ids and r["unique_bigrams"] == n_pairs
    assert r["attack_percentage"] == round(100 * n_win[1] / (n_win[0] + n_win[1]), 4)


def test_more_overlap_never_decreases_windows_or_bigrams(ts, synthetic_cfg, rec):
    res = [_run(ts, rd.Scenario(f"g{s}", "grid", tuple(NORMAL_DIRS), None, 30, s), synthetic_cfg, rec) for s in (30, 10, 5, 1)]
    assert [r["total_windows"] for r in res] == sorted(r["total_windows"] for r in res)
    assert [r["unique_bigrams"] for r in res] == sorted(r["unique_bigrams"] for r in res)


def test_source_group_scenarios(ts, synthetic_cfg, rec):
    only_train = rd.Scenario("t", "source", (NORMAL_DIRS[0],), None, 30, 30)
    only_val = rd.Scenario("v", "source", (NORMAL_DIRS[1],), None, 30, 30)
    both = rd.Scenario("b", "source", tuple(NORMAL_DIRS), None, 30, 30)
    fam = rd.Scenario("f", "source", tuple(NORMAL_DIRS), ("Adduser",), 30, 30)
    rt, rv, rb, rf = (_run(ts, s, synthetic_cfg, rec) for s in (only_train, only_val, both, fam))
    assert rt["normal_windows"] + rv["normal_windows"] == rb["normal_windows"]
    assert rt["attack_windows"] == rv["attack_windows"] == rb["attack_windows"]
    assert 0 < rf["attack_windows"] < rb["attack_windows"]
    for r, sc in ((rt, only_train), (rv, only_val), (rb, both), (rf, fam)):
        n_win, _, _ = brute_scenario(ts, sc)
        assert (r["normal_windows"], r["attack_windows"]) == (n_win[0], n_win[1])
    assert set(rt["windows_by_source"]) - {"Attack_Data_Master::Adduser", "Attack_Data_Master::Hydra_FTP",
                                            "Attack_Data_Master::Web_Shell"} == {NORMAL_DIRS[0]}     # source group preserved
    assert set(rf["windows_by_source"]) & {"Attack_Data_Master::Hydra_FTP"} == set()
    assert sum(rb["windows_by_source"].values()) == rb["total_windows"]


@pytest.mark.parametrize("min_len", [0, 100, 150, 10_000])
def test_length_filters_exclude_traces_in_memory_only(ts, synthetic_cfg, rec, min_len):
    n_before = len(ts.meta)
    sc = rd.Scenario("l", "length", tuple(NORMAL_DIRS), None, 30, 30, min_trace_len=min_len)
    r = _run(ts, sc, synthetic_cfg, rec)
    n_win, n_ids, n_pairs = brute_scenario(ts, sc)
    assert (r["normal_windows"], r["attack_windows"]) == (n_win[0], n_win[1])
    assert r["unique_syscalls"] == n_ids and r["unique_bigrams"] == n_pairs
    assert r["n_traces_excluded_by_min_len"] == int((ts.meta["n_calls"] < min_len).sum())
    assert len(ts.meta) == n_before                                 # nothing deleted


def test_partial_trailing_windows_and_dropped_calls(ts, synthetic_cfg, rec):
    r = _run(ts, rd.Scenario("b", "reference", tuple(NORMAL_DIRS), None, 30, 30), synthetic_cfg, rec)
    exp_calls = sum(n - 30 * (n // 30) for n in ts.meta["n_calls"])
    exp_windows = sum(1 for n in ts.meta["n_calls"] if n % 30)
    assert r["dropped_trailing_calls"] == exp_calls and r["partial_trailing_windows_dropped"] == exp_windows


# ---------------- vocabulary handling ----------------
def _vocab_scenario(mode):
    return rd.Scenario(f"v_{mode}", "vocab", tuple(NORMAL_DIRS), None, 30, 30, vocab_handling=mode)


def test_vocab_top_k_unk_and_drop_match_brute_force(ts, synthetic_cfg, rec):
    base = _run(ts, rd.Scenario("base", "reference", tuple(NORMAL_DIRS), None, 30, 30), synthetic_cfg, rec)
    r_unk = _run(ts, _vocab_scenario("top_unk"), synthetic_cfg, rec)
    r_drop = _run(ts, _vocab_scenario("top_drop"), synthetic_cfg, rec)
    # brute-force: rebuild the train partition and the top-k set
    meta = ts.meta[ts.meta["n_calls"] >= 30].sort_values("trace_id")
    tr, _, _ = _two_stage_stratified(meta["trace_id"].to_numpy(), meta["label"].to_numpy(), 0.15, 0.15, 42, "traces")
    counts = np.zeros(200, int)
    for t in tr:
        counts += np.bincount(ts.calls[t], minlength=200)
    ids = np.flatnonzero(counts > 0)
    kept = set(ids[np.lexsort((ids, -counts[ids]))][:30].tolist())
    assert r_unk["vocab_ids_excluded"] == len({i for c in ts.calls.values() for i in c.tolist()} - kept) > 0
    unk = 100000
    win_unk, win_drop, ids_unk, pairs_unk, pairs_drop = 0, 0, set(), set(), set()
    for r in ts.meta.itertuples(index=False):
        for win in brute_windows(ts.calls[r.trace_id].tolist(), 30, 30):
            mapped = tuple(i if i in kept else unk for i in win)
            win_unk += 1
            ids_unk.update(mapped)
            pairs_unk.update(zip(mapped[:-1], mapped[1:]))
            if all(i in kept for i in win):
                win_drop += 1
                pairs_drop.update(zip(win[:-1], win[1:]))
    assert r_unk["total_windows"] == base["total_windows"] == win_unk          # UNK keeps every window
    assert r_unk["unique_syscalls"] == len(ids_unk) <= 31 and r_unk["unique_bigrams"] == len(pairs_unk)
    assert r_drop["total_windows"] == win_drop < base["total_windows"]
    assert r_drop["unique_bigrams"] == len(pairs_drop) <= base["unique_bigrams"]
    assert r_drop["unique_syscalls"] <= 30
    assert 0 < r_unk["vocab_share_calls_excluded"] < 1


# ---------------- experiment F (fit) and zero-window check ----------------
def test_fit_columns_and_all_zero_percentage(ts, synthetic_cfg, rec):
    sc = rd.Scenario("b", "reference", tuple(NORMAL_DIRS), None, 30, 30)
    r = _run(ts, sc, synthetic_cfg, rec, fit=True)
    assert r["fit_status"].startswith("ok")
    assert r["tfidf_train_vocab_size"] <= r["unique_bigrams"]
    assert r["chi2_selected_features"] == min(150, r["tfidf_train_vocab_size"])
    assert len(r["selected_bigram_names"].split(";")) == r["chi2_selected_features"]
    assert 0 <= r["pct_windows_all_zero_test"] <= 100 and 0 <= r["pct_windows_all_zero_train"] <= 100
    assert r["fit_subsampled"] is False
    r2 = _run(ts, sc, synthetic_cfg, rec, fit=True)                   # deterministic
    assert r["selected_bigram_names"] == r2["selected_bigram_names"] and r["pct_windows_all_zero_test"] == r2["pct_windows_all_zero_test"]


def test_fit_subsampling_is_deterministic_and_flagged(ts, synthetic_cfg, rec):
    rec2 = {**rec, "zero_check": {"max_windows_per_partition": 60, "fit_features": True}}
    sc = rd.Scenario("g1", "grid", tuple(NORMAL_DIRS), None, 30, 1)
    a, b = _run(ts, sc, synthetic_cfg, rec2, fit=True), _run(ts, sc, synthetic_cfg, rec2, fit=True)
    assert a["fit_subsampled"] is True and a["fit_train_windows_used"] == 60 and a["fit_test_windows_used"] == 60
    assert a["selected_bigram_names"] == b["selected_bigram_names"]


def test_sample_windows_returns_real_windows():
    calls = [np.arange(100), np.arange(100, 170)]
    starts = [rd.window_starts(100, 30, 30), rd.window_starts(70, 30, 30)]
    texts, ys, total = rd._sample_windows(calls, starts, [0, 1], 30, cap=1000, seed=1)
    assert total == 5 and len(texts) == 5 and list(ys) == [0, 0, 0, 1, 1]
    assert texts[0] == " ".join(map(str, range(30))) and texts[3] == " ".join(map(str, range(100, 130)))
    t2, _, _ = rd._sample_windows(calls, starts, [0, 1], 30, cap=2, seed=1)
    assert len(t2) == 2 and set(t2) <= set(texts)


def test_split_uses_primary_policy(ts, synthetic_cfg, rec):
    """The scenario split equals the primary trace-disjoint split (same function, seed and trace ordering)."""
    sc = rd.Scenario("b", "reference", tuple(NORMAL_DIRS), None, 30, 30)
    r = _run(ts, sc, synthetic_cfg, rec)
    from src.data import build_from_config, make_split
    _, traces, windows, _ = build_from_config(synthetic_cfg)
    _, wa = make_split(traces, windows, "stratified_group_by_trace", 0.15, 0.15, 42)
    for name in ("train", "validation", "test"):
        assert r[f"windows_{name}"] == int((wa["split"] == name).sum())


def test_family_scenario_with_too_few_traces_reports_skipped_fit(synthetic_cfg, rec, tmp_path):
    root = make_synthetic_adfa(tmp_path / "tiny" / "ADFA-LD", families=("Adduser",), runs_per_family=1, files_per_run=2)
    from src.config import load_config
    cfg = load_config()
    cfg["data"].update(raw_data_dir=str(root))
    ts2 = rd.load_traceset(cfg)
    r = _run(ts2, rd.Scenario("f", "source", tuple(NORMAL_DIRS), ("Adduser",), 30, 30), cfg, rec, fit=True)
    assert r["fit_status"].startswith("skipped") and r["split_status"].startswith("skipped")
    assert r["attack_windows"] > 0                                   # counts still reported
    with pytest.raises(ValueError):                                   # vocabulary handling needs a split
        _run(ts2, rd.Scenario("v", "vocab", tuple(NORMAL_DIRS), ("Adduser",), 30, 30, vocab_handling="top_unk"), cfg, rec)


# ---------------- experiment E: frequency table ----------------
def test_syscall_frequency_table(ts, synthetic_cfg, rec):
    df = rd.syscall_frequency_table(ts, rec, synthetic_cfg["split"], tuple(NORMAL_DIRS))
    all_calls = np.concatenate(list(ts.calls.values()))
    assert set(df["syscall_id"]) == set(np.unique(all_calls).tolist())
    row = df.set_index("syscall_id")
    for i in (1, 12, 30, 40):
        assert row.loc[i, "total_count"] == int((all_calls == i).sum())
    assert (row["normal_count"] + row["attack_count"] == row["total_count"]).all()
    src_cols = [c for c in df.columns if c.startswith("count__")]
    assert set(src_cols) == {f"count__{NORMAL_DIRS[0]}", f"count__{NORMAL_DIRS[1]}", "count__Attack_Data_Master::Adduser",
                             "count__Attack_Data_Master::Hydra_FTP", "count__Attack_Data_Master::Web_Shell"}
    assert (row[src_cols].sum(axis=1) == row["total_count"]).all()
    assert row["rank_overall"].min() == 1 and row["low_frequency"].dtype == bool
    assert row["in_top30_train"].sum() == 30
    assert row["low_frequency"].any()                                   # guard: an empty selection would make max() NaN
    assert row.loc[row["low_frequency"], "total_count"].max() < rec["low_frequency_threshold"]
    assert (row.loc[~row["low_frequency"], "total_count"] >= rec["low_frequency_threshold"]).all()
    assert row.loc[40, "only_in_one_class"] and not row.loc[12, "only_in_one_class"]      # 26-40 attack only; 10-25 shared


# ---------------- experiment G: balance feasibility ----------------
def test_balance_feasibility_arithmetic(ts, synthetic_cfg, rec):
    scs = [rd.Scenario("baseline", "reference", tuple(NORMAL_DIRS), None, 30, 30)]
    results = rd.run_scenarios(ts, scs, synthetic_cfg, rec, 42, do_fit=False)
    df = rd.balance_feasibility_table(ts, results, rec)
    row = df.iloc[0]
    target = rec["paper_reference"]["total_instances"] // 2
    assert target == 26328 and row.balanced_target_per_class == 26328
    assert row.attack_windows_vs_target_deficit == 26328 - row.attack_windows > 0
    assert row.attack_windows_support_target is np.False_ or row.attack_windows_support_target is False
    assert row.normal_windows_surplus_over_target == row.normal_windows - 26328
    assert row.largest_balanced_nonoverlap_total == 2 * min(row.normal_windows, row.attack_windows)
    assert row.attack_deficit_vs_paper_attack == 26870 - row.attack_windows
    attack_lens = ts.meta.loc[ts.meta.label == 1, "n_calls"]
    assert row.attack_windows_at_stride_1 == int(sum(max(n - 29, 0) for n in attack_lens))
    assert pd.isna(row.largest_stride_reaching_target_attack) and "insufficient" in row.interpretation


def test_balance_feasibility_detects_sufficient_attack_windows(tmp_path, rec):
    """A synthetic attack class with very long traces: the target is reachable and a max stride is reported."""
    from src.config import load_config
    root = make_synthetic_adfa(tmp_path / "big" / "ADFA-LD", n_train=5, n_val=5, families=("Adduser",), runs_per_family=1, files_per_run=2)
    for f in (root / ATTACK_DIR / "Adduser_1").glob("*.txt"):
        f.write_text(" ".join(["5"] * 60_000))                    # 2 traces x 60,000 calls
    cfg = load_config()
    cfg["data"].update(raw_data_dir=str(root))
    ts2 = rd.load_traceset(cfg)
    res = rd.run_scenarios(ts2, [rd.Scenario("baseline", "reference", tuple(NORMAL_DIRS), None, 30, 30)], cfg, rec, 42, do_fit=False)
    row = rd.balance_feasibility_table(ts2, res, rec).iloc[0]
    assert row.attack_windows == 4000 and not row.attack_windows_support_target
    # stride 1 gives 2 * (60000 - 29) = 119,942 windows; the largest stride with >= 26,328 is 4 (2 * floor(59970/4 + 1) = 29,992)
    assert row.attack_windows_at_stride_1 == 119_942 and row.largest_stride_reaching_target_attack == 4


# ---------------- summary + distances ----------------
def test_descriptive_distances(rec):
    paper = rec["paper_reference"]
    assert rd.descriptive_distances(52656, 100 * 26870 / 52656, paper) == (0, 0.0)
    d_total, d_ratio = rd.descriptive_distances(88828, 11.5153, paper)
    assert d_total == 88828 - 52656
    assert d_ratio == round(abs(0.115153 - 26870 / 52656), 4)
    assert np.isnan(rd.descriptive_distances(0, float("nan"), paper)[1])


def test_summary_table_columns_and_content(ts, synthetic_cfg, rec):
    scs = rd.build_scenarios(rec, NORMAL_DIRS, ["Adduser", "Hydra_FTP", "Web_Shell"])
    results = rd.run_scenarios(ts, scs, synthetic_cfg, rec, 42, do_fit=False)
    summ = rd.summary_table(results, rec)
    assert list(summ.columns) == rd.SUMMARY_COLUMNS == [
        "scenario_id", "normal_trace_sources", "attack_sources", "window_size", "stride", "length_filter", "vocabulary_handling",
        "normal_windows", "attack_windows", "total_windows", "attack_percentage", "unique_syscalls", "unique_bigrams",
        "distance_to_paper_total", "distance_to_paper_attack_ratio", "notes"]
    assert len(summ) == len(scs) and summ["scenario_id"].is_unique
    assert (summ["normal_windows"] + summ["attack_windows"] == summ["total_windows"]).all()
    assert summ["notes"].str.startswith("EXPLORATORY").all()
    assert (summ["distance_to_paper_total"] == (summ["total_windows"] - 52656).abs()).all()
    ids = set(summ["scenario_id"])
    assert {"baseline_all_sources_s30", "grid_w30_s1_drop", "src_training_normal_plus_all_attack", "src_family_Adduser_vs_normal",
            "len_min100", "vocab_top30_unk", "vocab_top30_drop_windows"} <= ids
    base = summ.set_index("scenario_id")
    assert base.loc["baseline_all_sources_s30", "total_windows"] == base.loc["grid_w30_s30_drop", "total_windows"]
    assert base.loc["baseline_all_sources_s30", "total_windows"] == base.loc["src_training_validation_normal_plus_all_attack", "total_windows"]


def test_invalid_reconciliation_config_is_rejected(rec):
    for bad in ({"strides": []}, {"strides": [0, 5]}, {"incomplete_policies": ["pad"]}, {"window_size": 1}, {"top_k_syscalls": 0}):
        with pytest.raises(rd.ReconciliationConfigError):
            rd.validate_rec_config({**rec, **bad})


# ---------------- full run, missing dataset, no side effects ----------------
def _tree_hash(path):
    h = hashlib.sha1()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(path).as_posix().encode() + p.read_bytes())
    return h.hexdigest()


def test_full_run_writes_all_outputs_and_leaves_preprocessing_artifacts_untouched(synthetic_cfg, rec, tmp_path):
    from src.config import resolve_path
    proc, art = resolve_path(synthetic_cfg["data"]["processed_data_dir"]), resolve_path(synthetic_cfg["data"]["artifacts_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    sentinel_proc = proc / "train_features.npz"
    sentinel_art = art / "preprocessing_metadata.json"
    sentinel_proc.write_bytes(b"sentinel-processed")
    sentinel_art.write_text('{"sentinel": true}')
    try:
        before = (_tree_hash(proc), _tree_hash(art))
        raw_before = _tree_hash(resolve_path(synthetic_cfg["data"]["raw_data_dir"]))
        out = rd.run_reconciliation(synthetic_cfg, rec, tmp_path / "results")
        tabs = sorted(p.name for p in (tmp_path / "results" / "tables").glob("*.csv"))
        assert tabs == sorted(f"{n}.csv" for n in [
            "reconciliation_raw_inventory", "reconciliation_window_grid", "reconciliation_source_group_scenarios",
            "reconciliation_length_filter_scenarios", "reconciliation_syscall_frequency", "reconciliation_vocabulary_scenarios",
            "reconciliation_bigram_scenarios", "reconciliation_balance_feasibility", "reconciliation_summary"])
        figs = sorted(p.name for p in (tmp_path / "results" / "figures").glob("*.png"))
        assert figs == ["reconciliation_attack_ratio_by_scenario.png", "reconciliation_syscall_frequency.png",
                        "reconciliation_trace_length_by_class.png", "reconciliation_window_count_vs_stride.png"]
        assert (_tree_hash(proc), _tree_hash(art)) == before
        assert _tree_hash(resolve_path(synthetic_cfg["data"]["raw_data_dir"])) == raw_before
        for name in ("reconciliation_window_grid", "reconciliation_source_group_scenarios", "reconciliation_bigram_scenarios"):
            df = pd.read_csv(tmp_path / "results" / "tables" / f"{name}.csv")
            assert df["experiment_label"].str.startswith("EXPLORATORY").all()
        grid = pd.read_csv(tmp_path / "results" / "tables" / "reconciliation_window_grid.csv")
        assert list(grid["stride"]) == [1, 5, 10, 30] and grid["pct_windows_all_zero_test"].notna().all()
        assert "windows__Attack_Data_Master::Adduser" in pd.read_csv(tmp_path / "results" / "tables" / "reconciliation_source_group_scenarios.csv").columns
    finally:
        sentinel_proc.unlink(missing_ok=True)
        sentinel_art.unlink(missing_ok=True)


def test_missing_dataset_raises_clear_error(synthetic_cfg, rec, tmp_path):
    synthetic_cfg["data"]["raw_data_dir"] = str(tmp_path / "does_not_exist")
    with pytest.raises(DatasetNotFoundError, match="never downloads"):
        rd.run_reconciliation(synthetic_cfg, rec, tmp_path / "results")


def test_script_returns_2_without_dataset(monkeypatch, tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_reconciliation", PROJECT_ROOT / "scripts" / "run_reconciliation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "baseline.yaml"))
    cfg["data"]["raw_data_dir"] = str(tmp_path / "missing")
    cfg["results_dir"] = str(tmp_path / "results")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert mod.main(["--config", str(p)]) == 2
    err = capsys.readouterr().err
    assert "ADFA-LD folder not found" in err and "never downloads" in err


def test_phase26_audit_artifacts_schema_and_counts(tmp_path, synthetic_cfg, rec):
    from src.audit_artifacts import write_audit_artifacts
    synthetic_cfg = dict(synthetic_cfg)
    synthetic_cfg["data"] = dict(synthetic_cfg["data"])
    synthetic_cfg["data"]["processed_data_dir"] = str(tmp_path / "processed")
    synthetic_cfg["results_dir"] = str(tmp_path / "results")
    out = rd.run_reconciliation(synthetic_cfg, rec, tmp_path / "results", do_fit=False, write_audit=True)
    paths = out["audit_artifacts"]
    assert paths["markdown"].exists()
    assert paths["json"].exists()
    assert paths["manifest"].exists()
    manifest = pd.read_csv(paths["manifest"])
    assert set(["trace_id", "source_relative_path", "source_group", "label", "usable", "exclusion_reason",
                "raw_length", "complete_window_count", "leftover_call_count", "assigned_split",
                "window_size", "stride", "tail_policy", "seed"]).issubset(manifest.columns)
    assert not any("calls" in c.lower() or "sequence" in c.lower() for c in manifest.columns)
    payload = __import__("json").loads(paths["json"].read_text())
    assert payload["result_type"] == "independent reconstruction, not the exact paper dataset"
    assert payload["configuration"]["split_policy"] == "trace-disjoint stratified split"
    assert payload["windowing"]["stride"] == 30
    assert payload["integrity_checks"]["manifest_matches_baseline_window_count"] is True
