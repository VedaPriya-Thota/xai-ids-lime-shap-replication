from __future__ import annotations
import argparse
from src.phase3c import run_phase3c

p=argparse.ArgumentParser()
p.add_argument('--raw-data-dir', required=True)
p.add_argument('--phase3a-dir',default=None,
               help="Phase 3A artifact directory to consume. Defaults to the historical frozen "
                    "directory; pass a freshly regenerated Phase 3A output directory to avoid "
                    "loading an artifact serialized under an incompatible NumPy/scikit-learn version.")
p.add_argument('--phase3b-dir',default=None,
               help="Phase 3B artifact directory to consume. Defaults to the historical frozen "
                    "directory; pass a freshly regenerated Phase 3B output directory to match a "
                    "non-default --phase3a-dir.")
p.add_argument('--experiment-dir',default=None,
               help="Output directory for this Phase 3C run. Defaults to the historical Phase 3C "
                    "directory; pass an explicit directory to avoid overwriting it.")
a=p.parse_args()
run_phase3c(a.raw_data_dir, phase3a_dir=a.phase3a_dir, phase3b_dir=a.phase3b_dir, experiment_dir=a.experiment_dir)
print('Phase 3C completed')
