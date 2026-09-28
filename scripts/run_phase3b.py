from __future__ import annotations
import argparse
from src.phase3b import run_phase3b

p=argparse.ArgumentParser()
p.add_argument('--raw-data-dir',required=True)
p.add_argument('--nsamples',type=int,default=512)
p.add_argument('--lime-num-samples',type=int,default=5000)
p.add_argument('--phase3a-dir',default=None,
               help="Phase 3A artifact directory to consume (model/vectorizer/selector/predictions). "
                    "Defaults to the historical frozen directory; pass a freshly regenerated Phase 3A "
                    "output directory to avoid loading an artifact serialized under an incompatible "
                    "NumPy/scikit-learn version.")
p.add_argument('--experiment-dir',default=None,
               help="Output directory for this Phase 3B run. Defaults to the historical Phase 3B "
                    "directory; pass an explicit directory to avoid overwriting it.")
a=p.parse_args()
run_phase3b(a.raw_data_dir, nsamples=a.nsamples, lime_num_samples=a.lime_num_samples,
            phase3a_dir=a.phase3a_dir, experiment_dir=a.experiment_dir)
print('Phase 3B completed')
