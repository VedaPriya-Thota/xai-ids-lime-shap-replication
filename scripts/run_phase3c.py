from __future__ import annotations
import argparse
from src.phase3c import run_phase3c

p=argparse.ArgumentParser()
p.add_argument('--raw-data-dir', required=True)
a=p.parse_args()
run_phase3c(a.raw_data_dir)
print('Phase 3C completed')
