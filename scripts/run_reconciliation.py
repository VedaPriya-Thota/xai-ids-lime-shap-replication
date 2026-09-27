"""Phase 2.5: EXPLORATORY dataset reconciliation (paper's generated dataset vs our raw-ADFA-LD reconstruction).

Run from the project root:
    python scripts/run_reconciliation.py [--config configs/baseline.yaml] [--reconciliation-config configs/reconciliation.yaml] [--no-fit]

Reads the raw traces (read-only) and writes ONLY to results/tables, results/figures and results/logs.
The audit writes only its declared audit manifest under data/processed; it never writes to data/artifacts or changes configs/baseline.yaml.
No model is trained; no balancing, resampling or class weighting is applied.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.config import ConfigError, load_config, resolve_path
from src.data import DatasetNotFoundError
from src.reconcile_dataset import ReconciliationConfigError, run_reconciliation
from src.seed import set_global_seed
from src.utils import setup_logging

log = logging.getLogger("reconciliation")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--reconciliation-config", default="configs/reconciliation.yaml")
    ap.add_argument("--no-fit", action="store_true", help="skip the TF-IDF/chi-squared fits (fast; those columns stay empty)")
    ap.add_argument(
        "--raw-data-dir", default=None,
        help="override data.raw_data_dir (path to the extracted ADFA-LD folder or its parent); "
             "the ADFA_LD_RAW_DIR environment variable is used if this is not given",
    )
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        rec_path = resolve_path(args.reconciliation_config)
        rec = yaml.safe_load(open(rec_path, encoding="utf-8"))
    except (ConfigError, OSError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    raw_override = args.raw_data_dir or os.environ.get("ADFA_LD_RAW_DIR")
    if raw_override:
        cfg["data"]["raw_data_dir"] = raw_override
    cfg["_audit_command"] = "python scripts/run_reconciliation.py --config configs/baseline.yaml --raw-data-dir " + str(cfg["data"]["raw_data_dir"]) + " --no-fit"
    res_dir = resolve_path(cfg["results_dir"])
    setup_logging(res_dir / "logs" / "reconciliation.log")
    set_global_seed(cfg["seed"])
    print("EXPLORATORY dataset reconciliation: diagnostics only; nothing here recovers or approximates the authors' dataset.")
    try:
        out = run_reconciliation(cfg, rec, res_dir, do_fit=False if args.no_fit else None, write_audit=True)
    except (DatasetNotFoundError, ReconciliationConfigError, ValueError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        log.error("%s", exc, extra={"file_only": True})
        return 2

    paper = rec["paper_reference"]
    t = out["tables"]
    inv, summ = t["reconciliation_raw_inventory"], t["reconciliation_summary"]
    tot = inv[inv.source_group == "TOTAL"].iloc[0]
    lines = [
        "",
        "=== A. Raw inventory (observed; external counts are references, not ground truth) ===",
        f"Valid traces: {int(tot.n_valid_traces):,} (normal {int(inv[inv.source_group == 'SUBTOTAL_NORMAL'].n_valid_traces.iloc[0]):,}, "
        f"attack {int(inv[inv.source_group == 'SUBTOTAL_ATTACK'].n_valid_traces.iloc[0]):,}); blank {int(tot.n_blank_files)}, "
        f"no valid calls {int(tot.n_no_valid_calls)}, malformed-token files {int(tot.n_files_with_malformed_tokens)}, "
        f"unreadable {int(tot.n_unreadable)}, too short {int(tot.n_too_short_for_window)}, hidden ignored {int(tot.n_hidden_files_ignored)}",
        f"Paper reference (descriptive only): {paper['total_instances']:,} instances, attack share "
        f"{100 * paper['attack_instances'] / paper['total_instances']:.1f}%, {paper['distinct_syscalls']} syscalls, "
        f"{paper['initial_bigrams']:,} initial bigrams, {paper['selected_features']} selected",
        "",
        "=== Scenario summary (EXPLORATORY; distances are descriptive, not objectives) ===",
        summ[["scenario_id", "stride", "length_filter", "vocabulary_handling", "normal_windows", "attack_windows", "attack_percentage",
              "unique_syscalls", "unique_bigrams", "distance_to_paper_total", "distance_to_paper_attack_ratio"]].to_string(index=False),
        "",
        "=== G. Balance feasibility (no balanced dataset is generated) ===",
        t["reconciliation_balance_feasibility"][["scenario_id", "attack_windows", "balanced_target_per_class",
                                                 "attack_windows_support_target", "attack_windows_at_stride_1",
                                                 "largest_stride_reaching_target_attack"]].to_string(index=False),
        "",
        f"Tables : {res_dir / 'tables'}/reconciliation_*.csv (9 files)",
        f"Figures: {res_dir / 'figures'}/reconciliation_*.png (4 files)",
        f"Log    : {res_dir / 'logs' / 'reconciliation.log'}",
        "Audit artifact: reports/phase2_5_dataset_audit.md, reports/phase2_5_dataset_audit.json, data/processed/trace_manifest.csv.",
        "Primary preprocessing artifacts under data/processed/data/artifacts are otherwise untouched.",
    ]
    for ln in lines:
        print(ln)
        log.info(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
