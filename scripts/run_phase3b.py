from __future__ import annotations
import argparse
from src.phase3b import run_phase3b

p=argparse.ArgumentParser()
p.add_argument('--raw-data-dir',required=True)
p.add_argument('--nsamples',type=int,default=512)
p.add_argument('--lime-num-samples',type=int,default=5000)
a=p.parse_args()
run_phase3b(a.raw_data_dir, nsamples=a.nsamples, lime_num_samples=a.lime_num_samples)
print('Phase 3B completed')
