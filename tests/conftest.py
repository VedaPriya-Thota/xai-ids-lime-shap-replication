"""Shared test fixtures. Every fixture here builds a SYNTHETIC ADFA-LD-shaped directory tree with
NumPy-generated integer sequences; nothing here reads or requires the real dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NORMAL_DIRS = ["Training_Data_Master", "Validation_Data_Master"]
ATTACK_DIR = "Attack_Data_Master"
DEFAULT_FAMILIES = ("Adduser", "Hydra_FTP", "Web_Shell")

# Syscall ID pools: normal traces draw from [0, 25]; attack traces draw from [10, 40], weighted
# toward the shared [10, 25] range so that IDs 26-40 are individually rare (exercises the
# low-frequency / vocabulary-truncation diagnostics) while every ID in both ranges still appears
# at least once across the default-sized synthetic dataset.
_NORMAL_POOL_HI = 25          # normal IDs: 0..25 inclusive
_ATTACK_POOL_LO = 10          # attack IDs: 10..40 inclusive
_ATTACK_POOL_HI = 40
_ATTACK_ONLY_LO = 26          # IDs >= this are never used in normal traces


def _write_trace(path: Path, ids) -> None:
    path.write_text(" ".join(str(int(i)) for i in ids), encoding="utf-8")


def make_synthetic_adfa(
    root: Path,
    n_train: int = 30,
    n_val: int = 30,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    runs_per_family: int = 5,
    files_per_run: int = 2,
    seed: int = 0,
) -> Path:
    """Build a synthetic ADFA-LD-shaped tree under `root` and return `root`.

    Trace lengths are randint(40, 150) (comfortably above the window size of 30). Normal traces
    draw syscall IDs uniformly from [0, 25]; attack traces draw from [10, 40], weighted so the
    attack-only IDs (26-40) are rare relative to the shared IDs (10-25).
    """
    root = Path(root)
    rng = np.random.default_rng(seed)

    for dir_name, n_files, prefix in ((NORMAL_DIRS[0], n_train, "UTD"), (NORMAL_DIRS[1], n_val, "UVD")):
        d = root / dir_name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n_files + 1):
            length = int(rng.integers(40, 150))
            ids = rng.integers(0, _NORMAL_POOL_HI + 1, size=length)
            _write_trace(d / f"{prefix}-{i:04d}.txt", ids)

    pool = np.arange(_ATTACK_POOL_LO, _ATTACK_POOL_HI + 1)
    weights = np.where(pool < _ATTACK_ONLY_LO, 3.0, 0.2)
    weights = weights / weights.sum()
    for fam in families:
        for r in range(1, runs_per_family + 1):
            run_dir = root / ATTACK_DIR / f"{fam}_{r}"
            run_dir.mkdir(parents=True, exist_ok=True)
            for j in range(1, files_per_run + 1):
                length = int(rng.integers(40, 150))
                ids = rng.choice(pool, size=length, p=weights)
                _write_trace(run_dir / f"UAD-{j:04d}.txt", ids)

    return root


@pytest.fixture(scope="session")
def _synthetic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_adfa") / "ADFA-LD"
    make_synthetic_adfa(root)
    return root


@pytest.fixture
def synthetic_cfg(_synthetic_root):
    """A full, normalized config pointing at the shared synthetic dataset.

    Only data.raw_data_dir is overridden; data.processed_data_dir and data.artifacts_dir are left
    at their real (repo-relative) defaults deliberately, so tests can assert that a reconciliation
    run never touches the primary pipeline's real output directories even while reading synthetic
    raw data.
    """
    from src.config import load_config

    cfg = load_config()
    cfg["data"]["raw_data_dir"] = str(_synthetic_root)
    return cfg
