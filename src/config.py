"""Configuration loading and path resolution.

baseline.yaml (Phase 1) nests dataset/output paths under `paths:` and uses `normal_dirs` /
`attack_dirs` / `*_frac` names. The Phase 2.5 reconciliation modules (written against a planned
but not-yet-implemented Phase 2 contract) expect a flatter shape: `cfg["results_dir"]`,
`cfg["data"]["raw_data_dir" | "processed_data_dir" | "artifacts_dir" | "normal_directories" |
"attack_directory"]`, and `cfg["split"]["test_size" | "validation_size" | "seed"]`.

`load_config` reads the YAML file as-is and then *additively* derives the missing flat keys from
the nested ones already present, so configs/baseline.yaml itself is never rewritten and every
original key is preserved. This normalization is the "smallest justified correction" needed to
make reconcile_dataset.py / run_reconciliation.py importable and runnable against the existing
Phase 1 config file.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """The configuration file is missing a required key or has an invalid value."""


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root. Absolute paths are returned unchanged."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _normalize(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    paths = cfg.get("paths", {})

    cfg.setdefault("results_dir", paths.get("results_dir", "results"))

    d = cfg.setdefault("data", {})
    d.setdefault("raw_data_dir", paths.get("raw_data_dir", "data/raw/ADFA-LD"))
    d.setdefault("processed_data_dir", paths.get("processed_dir", "data/processed"))
    d.setdefault("artifacts_dir", paths.get("artifacts_dir", "data/artifacts"))

    if "normal_directories" not in d:
        normal_dirs = d.get("normal_dirs")
        if not normal_dirs:
            raise ConfigError("config.data must define normal_directories or normal_dirs")
        d["normal_directories"] = list(normal_dirs)

    if "attack_directory" not in d:
        attack_dirs = d.get("attack_dirs")
        if not attack_dirs:
            raise ConfigError("config.data must define attack_directory or attack_dirs")
        if len(attack_dirs) != 1:
            raise ConfigError(
                "attack_directory normalization requires exactly one entry in data.attack_dirs "
                f"(got {attack_dirs!r}); set data.attack_directory explicitly instead"
            )
        d["attack_directory"] = attack_dirs[0]

    sp = cfg.setdefault("split", {})
    if "test_size" not in sp:
        if "test_frac" not in sp:
            raise ConfigError("config.split must define test_size or test_frac")
        sp["test_size"] = sp["test_frac"]
    if "validation_size" not in sp:
        if "val_frac" not in sp:
            raise ConfigError("config.split must define validation_size or val_frac")
        sp["validation_size"] = sp["val_frac"]
    sp.setdefault("seed", cfg.get("seed"))

    return cfg


def load_config(path: str | Path = "configs/baseline.yaml") -> dict[str, Any]:
    """Load a YAML config and normalize it (see module docstring). Raises ConfigError on problems."""
    p = resolve_path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    try:
        with open(p, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {p} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {p} did not parse to a mapping")
    if "seed" not in raw:
        raise ConfigError("config is missing required key: seed")
    return _normalize(raw)
