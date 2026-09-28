from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
from src.phase3c import (EXPECTED_MANIFEST_SHA, EXPECTED_ASSIGNMENT_SHA, sha256_file,
                         assignment_hash, choose_neutral, guided_edits, random_edits,
                         protected_paths, P3A_REL, P3B_REL)

ROOT=Path(__file__).resolve().parents[1]

def test_all_protected_artifacts_exist_and_manifest_hashes():
    p=protected_paths()
    assert all(x.exists() for x in p.values())
    assert sha256_file(p['manifest']) == EXPECTED_MANIFEST_SHA
    assert assignment_hash(pd.read_csv(p['manifest'])) == EXPECTED_ASSIGNMENT_SHA


def test_protected_paths_defaults_to_historical_directories():
    p = protected_paths()
    assert p["model"] == ROOT / P3A_REL / "mlp_model.joblib"
    assert p["selected_instances"] == ROOT / P3B_REL / "selected_test_instances.csv"


def test_protected_paths_honors_explicit_phase3a_and_phase3b_dirs(tmp_path):
    custom_p3a = tmp_path / "custom_phase3a_rerun"
    custom_p3b = tmp_path / "custom_phase3b_rerun"
    p = protected_paths(custom_p3a, custom_p3b)
    assert p["model"] == custom_p3a / "mlp_model.joblib"
    assert p["vectorizer"] == custom_p3a / "preprocessing/tfidf_vectorizer.joblib"
    assert p["selected_instances"] == custom_p3b / "selected_test_instances.csv"
    assert p["lime"] == custom_p3b / "lime_local_explanations.csv"
    # Files/manifest paths that don't come from a Phase 3A/3B artifact directory are unaffected.
    assert p["manifest"] == ROOT / "data/processed/trace_manifest.csv"

def test_raw_window_perturbation_preserves_length_and_positions_unique():
    seq=np.array([1,2,3,4,5,6,7,8,9,10]*3,dtype=np.int64)
    freq={99:100, 98:90, 97:80, 1:70}
    feats=pd.DataFrame([
        {'rank':1,'feature_name':'1_2','signed_attribution':1.0},
        {'rank':2,'feature_name':'3_4','signed_attribution':.5},
        {'rank':3,'feature_name':'50_60','signed_attribution':.2},
    ])
    out, actions, _, _=guided_edits(seq,feats,freq)
    assert len(out)==30
    pos=[a['position'] for a in actions]
    assert len(pos)==len(set(pos))
    assert all(0<=x<30 for x in pos)

def test_exact_target_bigram_detection_and_neutral_policy():
    seq=np.array([1,2,3,4]+[5]*26,dtype=np.int64)
    freq={5:100,6:90,7:80}
    feats=pd.DataFrame([{'rank':1,'feature_name':'1_2','signed_attribution':1.0}])
    out, actions, _, _=guided_edits(seq,feats,freq)
    assert len(actions)==1
    assert actions[0]['position']==1
    assert out[1] != 2
    assert out[1] != 2

def test_random_edit_count_and_determinism():
    seq=np.arange(30,dtype=np.int64)
    freq={100:100,101:90,102:80}
    a,la=random_edits(seq,4,123,freq,[(1,2),(3,4)])
    b,lb=random_edits(seq,4,123,freq,[(1,2),(3,4)])
    assert np.array_equal(a,b)
    assert la==lb
    assert len(la)==4
    assert len({x['position'] for x in la})==4
    assert len(a)==30

def test_no_fit_or_raw_sequence_columns_in_phase3c_if_present():
    out=ROOT/'results/experiments/phase3c_perturbation_evaluation'
    if not out.exists(): return
    forbidden={'syscall_sequence','raw_syscalls','calls','raw_sequence','window_calls','full_sequence'}
    for p in out.glob('*.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        assert not forbidden & cols

def test_actionable_feature_rate_is_bounded_by_top10():
    from src.phase3c import guided_edits
    seq=np.array([1,2,1,2,1,2]+[5]*24,dtype=np.int64)
    freq={5:100,6:90}
    feats=pd.DataFrame([{'rank':1,'feature_name':'1_2','signed_attribution':1.0},{'rank':2,'feature_name':'5_5','signed_attribution':.5}])
    _,actions,_,actionable_features=guided_edits(seq,feats,freq)
    assert len(actionable_features) <= 10
    assert len(actions) >= len(actionable_features)
