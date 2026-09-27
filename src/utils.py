"""Small, dependency-light helpers: logging setup, directory creation, safe JSON writes."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_logging(log_path: str | Path, level: int = logging.INFO) -> None:
    """Configure the root logger to write to both `log_path` and stderr. Creates parent dirs."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if setup_logging is called more than once in the same process.
    root.handlers = [h for h in root.handlers if not isinstance(h, (logging.FileHandler, logging.StreamHandler))]

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(sh)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str, sort_keys=True), encoding="utf-8")
    return p
