from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3a import run_phase3a

p = argparse.ArgumentParser(description="Run Phase 3A strict reconstructed ADFA-LD MLP baseline once.")
p.add_argument("--config", default="configs/baseline.yaml")
p.add_argument("--raw-data-dir", default=None, help="Override raw data path in memory only; baseline config is not modified.")
p.add_argument("--experiment-dir", default=None)
args = p.parse_args()
print(run_phase3a(args.config, args.experiment_dir, raw_data_dir=args.raw_data_dir))
