from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
from src.phase3b import EXPECTED_MANIFEST_SHA, EXPECTED_ASSIGNMENT_SHA, select_instances, sha256_file

ROOT=Path(__file__).resolve().parents[1]
P3A=ROOT/'results/experiments/phase3a_strict_reconstructed_mlp'

def test_frozen_manifest_and_split_hashes():
    assert sha256_file(ROOT/'data/processed/trace_manifest.csv') == EXPECTED_MANIFEST_SHA
    m=pd.read_csv(ROOT/'data/processed/trace_manifest.csv')
    h=hashlib.sha256()
    for r in m.sort_values('trace_id').itertuples(index=False):
        h.update(f'{r.trace_id}|{int(r.label)}|{r.assigned_split}\n'.encode())
    assert h.hexdigest() == EXPECTED_ASSIGNMENT_SHA

def test_exactly_20_deterministic_outcome_selection():
    pred=pd.read_csv(P3A/'test_predictions.csv')
    windows=pd.DataFrame({'trace_id':[f't{i}' for i in range(len(pred))], 'start':np.arange(len(pred))})
    a=select_instances(pred,windows); b=select_instances(pred,windows)
    assert len(a)==20 and a.outcome_group.value_counts().to_dict()=={'TP':5,'FN':5,'TN':5,'FP':5}
    assert a[['selection_id','window_index']].equals(b[['selection_id','window_index']])

def test_phase3a_artifacts_exist_and_selected_features_are_150():
    names=pd.read_csv(P3A/'preprocessing/selected_feature_names.csv')
    assert len(names)==150
    for p in [P3A/'mlp_model.joblib',P3A/'preprocessing/tfidf_vectorizer.joblib',P3A/'preprocessing/chi2_selector.joblib',P3A/'test_predictions.csv']:
        assert p.exists()

def test_no_raw_syscall_columns_in_phase3b_tables_if_present():
    out=ROOT/'results/experiments/phase3b_xai_explanations'
    if not out.exists(): return
    forbidden={'syscall_sequence','raw_syscalls','calls','raw_sequence','window_calls'}
    for p in out.glob('*.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        assert not forbidden & cols
