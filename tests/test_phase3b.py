from pathlib import Path
import hashlib
import shutil
import pandas as pd
import numpy as np
from src.phase3b import EXPECTED_MANIFEST_SHA, EXPECTED_ASSIGNMENT_SHA, select_instances, sha256_file, verify_frozen_inputs
from src.config import load_config

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


def test_verify_frozen_inputs_accepts_explicit_non_default_phase3a_dir(tmp_path):
    """Phase 3B must be able to load a Phase 3A artifact set from an arbitrary directory, not only
    the historical frozen path. This is the behavior that lets a freshly regenerated Phase 3A run
    (produced under the repository's documented environment, and therefore guaranteed to unpickle
    correctly in that same environment) feed Phase 3B directly, instead of Phase 3B being hard-wired
    to a single historical directory whose serialized artifacts may predate the documented
    environment's package versions.

    The artifacts here are freshly fit under the current environment (not copies of the historical
    frozen pickle) so this test exercises the actual reproducibility fix rather than depending on
    whether the historical pickle happens to be loadable in whatever environment runs the suite.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.neural_network import MLPClassifier

    custom_p3a = tmp_path / "phase3a_rerun"
    (custom_p3a / "preprocessing").mkdir(parents=True)

    # Reuse the real frozen selection/prediction *data* (not code-version-dependent pickles): the
    # feature-name table and the test-set predictions must match EXPECTED_CM for verification to
    # pass, exactly as a genuine same-methodology Phase 3A rerun would reproduce.
    shutil.copy(P3A / "preprocessing" / "selected_feature_names.csv", custom_p3a / "preprocessing" / "selected_feature_names.csv")
    shutil.copy(P3A / "test_predictions.csv", custom_p3a / "test_predictions.csv")

    # Freshly fit (under the current environment) placeholder vectorizer/selector/model with the
    # required shape (150 selected features), standing in for a real Phase 3A rerun's artifacts.
    docs = [" ".join(f"{i}_{i+1}" for i in range(1, 200)) for _ in range(8)]
    y = np.array([0, 1] * 4)
    vec = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"[^\s]+", lowercase=False, norm="l2", use_idf=True)
    x = vec.fit_transform(docs)
    selector = SelectKBest(chi2, k=150)
    x_sel = selector.fit_transform(x, y)
    model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5, random_state=42)
    model.fit(x_sel, y)
    joblib.dump(vec, custom_p3a / "preprocessing" / "tfidf_vectorizer.joblib")
    joblib.dump(selector, custom_p3a / "preprocessing" / "chi2_selector.joblib")
    joblib.dump(model, custom_p3a / "mlp_model.joblib")

    cfg = load_config()
    verified_custom = verify_frozen_inputs(cfg, custom_p3a)
    # Loaded successfully from a directory other than the historical default, with a model/
    # vectorizer/selector that were fit (and pickled) entirely under the current environment.
    assert verified_custom["paths"]["model"] == custom_p3a / "mlp_model.joblib"
    assert hasattr(verified_custom["model"], "predict_proba")
    assert verified_custom["selector"].get_support().sum() == 150


