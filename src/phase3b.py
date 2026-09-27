"""Phase 3B: frozen-model LIME and SHAP explanations.

No Phase 3A artifact is written. The model and preprocessing are loaded read-only.
The raw ADFA-LD mirror is read only to reconstruct train/test feature rows from the
frozen manifest; no preprocessing transformer is fitted in this phase.
"""
from __future__ import annotations

import hashlib, json, os, platform, random, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .config import PROJECT_ROOT, load_config, resolve_path
from .data import find_adfa_root, load_traces

try:
    from lime.lime_tabular import LimeTabularExplainer
    import lime
    LIME_IMPORT_VERSION = getattr(lime, "__version__", "unknown")
except ImportError as exc:
    raise RuntimeError("LIME is required for Phase 3B") from exc

SEED = 42
EXP_DIR_NAME = "results/experiments/phase3b_xai_explanations"
PHASE3A_DIR = "results/experiments/phase3a_strict_reconstructed_mlp"
EXPECTED_MANIFEST_SHA = "765289887ac57f0a7f08b0d2761c9507e65f8ff98805ab69dc08a6cc312bf4f1"
EXPECTED_ASSIGNMENT_SHA = "5c2b0745cccde2c060ce139fe92e383510ab4237eb0d53b64b9c5f29024e04ba"
EXPECTED_CM = [[11681, 432], [428, 1195]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(values: list[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(v.encode())
        h.update(b"\n")
    return h.hexdigest()


def rel(path: Path) -> str:
    try: return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError: return str(path.resolve())


def verify_frozen_inputs(cfg: dict[str, Any], p3a: Path) -> dict[str, Any]:
    paths = {
        "model": p3a / "mlp_model.joblib",
        "vectorizer": p3a / "preprocessing/tfidf_vectorizer.joblib",
        "selector": p3a / "preprocessing/chi2_selector.joblib",
        "feature_names": p3a / "preprocessing/selected_feature_names.csv",
        "test_predictions": p3a / "test_predictions.csv",
        "manifest": PROJECT_ROOT / "data/processed/trace_manifest.csv",
        "audit": PROJECT_ROOT / "reports/phase2_5_dataset_audit.json",
        "config": PROJECT_ROOT / "configs/baseline.yaml",
    }
    missing = [rel(p) for p in paths.values() if not p.exists()]
    if missing: raise FileNotFoundError(f"Protected Phase 3B inputs missing: {missing}")
    manifest_sha = sha256_file(paths["manifest"])
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(f"Manifest SHA mismatch: {manifest_sha} != {EXPECTED_MANIFEST_SHA}")
    manifest = pd.read_csv(paths["manifest"])
    assignment = [f"{r.trace_id}|{int(r.label)}|{r.assigned_split}" for r in manifest.sort_values("trace_id").itertuples(index=False)]
    assignment_sha = sha256_lines(assignment)
    if assignment_sha != EXPECTED_ASSIGNMENT_SHA:
        raise RuntimeError(f"Split-assignment SHA mismatch: {assignment_sha} != {EXPECTED_ASSIGNMENT_SHA}")
    names = pd.read_csv(paths["feature_names"])
    if len(names) != 150 or names["feature_name"].isna().any():
        raise RuntimeError(f"Expected exactly 150 selected feature names, got {len(names)}")
    pred = pd.read_csv(paths["test_predictions"])
    cm = pd.crosstab(pred["label"], pred["predicted_label"]).reindex(index=[0,1], columns=[0,1], fill_value=0).values.tolist()
    if cm != EXPECTED_CM:
        raise RuntimeError(f"Frozen test confusion matrix mismatch: {cm} != {EXPECTED_CM}")
    model = joblib.load(paths["model"])
    vec = joblib.load(paths["vectorizer"])
    selector = joblib.load(paths["selector"])
    if getattr(selector, "k", None) != 150 or selector.get_support().sum() != 150:
        raise RuntimeError("Frozen chi-square selector does not select exactly 150 features")
    if not hasattr(model, "predict_proba") or not hasattr(vec, "transform") or not hasattr(selector, "transform"):
        raise RuntimeError("Frozen Phase 3A artifacts are incompatible with Phase 3B")
    protected_hashes = {k: sha256_file(v) for k,v in paths.items()}
    return {"paths": paths, "manifest": manifest, "predictions": pred, "feature_names": names,
            "model": model, "vectorizer": vec, "selector": selector, "hashes": protected_hashes,
            "manifest_sha256": manifest_sha, "assignment_sha256": assignment_sha}


def reconstruct_windows(manifest: pd.DataFrame, cfg: dict[str, Any], raw_dir: str | Path):
    d = cfg["data"]
    root = find_adfa_root(resolve_path(raw_dir), list(d["normal_directories"]) + [d["attack_directory"]])
    meta, calls_lists, _ = load_traces(root, list(d["normal_directories"]), d["attack_directory"])
    current = meta.set_index("trace_id")
    m = manifest.set_index("trace_id")
    if set(current.index) != set(m.index): raise RuntimeError("Current raw trace inventory differs from frozen manifest")
    for tid, r in m.iterrows():
        cur = current.loc[tid]
        if int(cur.label) != int(r.label) or int(cur.n_calls) != int(r.raw_length):
            raise RuntimeError(f"Raw trace differs from frozen manifest: {tid}")
    texts, rows = [], []
    for r in manifest.sort_values("trace_id").itertuples(index=False):
        seq = np.asarray(calls_lists[r.trace_id], dtype=np.int64)
        for start in range(0, len(seq)-29, 30):
            w = seq[start:start+30]
            text = " ".join(f"{int(a)}_{int(b)}" for a,b in zip(w[:-1], w[1:]))
            texts.append(text)
            rows.append({"trace_id": r.trace_id, "start": int(start), "label": int(r.label), "split": r.assigned_split})
    windows = pd.DataFrame(rows)
    return root, texts, windows


def select_instances(test_pred: pd.DataFrame, test_windows: pd.DataFrame) -> pd.DataFrame:
    p = test_pred.copy()
    if len(p) != len(test_windows): raise RuntimeError("Frozen test prediction count does not match reconstructed test windows")
    p["trace_id"] = test_windows["trace_id"].to_numpy()
    p["window_start"] = test_windows["start"].to_numpy()
    p["outcome_group"] = np.select([
        (p.label==1)&(p.predicted_label==1), (p.label==1)&(p.predicted_label==0),
        (p.label==0)&(p.predicted_label==0), (p.label==0)&(p.predicted_label==1)], ["TP","FN","TN","FP"], default="UNKNOWN")
    if set(p.outcome_group) != {"TP","FN","TN","FP"}: raise RuntimeError("Missing outcome group")
    parts=[]
    for group in ["TP","FN","TN","FP"]:
        g=p[p.outcome_group==group]
        if len(g)<5: raise RuntimeError(f"Outcome group {group} has only {len(g)} rows")
        parts.append(g.sample(n=5, random_state=SEED).sort_values("window_index"))
    out=pd.concat(parts, ignore_index=True)
    out.insert(0,"selection_id",[f"phase3b_{i:02d}" for i in range(1,21)])
    return out[["selection_id","window_index","trace_id","window_start","label","predicted_label","predicted_attack_probability","outcome_group"]]


def build_matrices(texts, windows, vec, selector):
    split_idx = {s: np.flatnonzero(windows.split.to_numpy()==s) for s in ["train","validation","test"]}
    x = {}
    for s in split_idx:
        raw = vec.transform([texts[i] for i in split_idx[s]])
        x[s] = selector.transform(raw).toarray().astype(np.float64)
    return x, split_idx


def top_rows(method, selected, values, X, names):
    rows=[]
    for i in range(len(selected)):
        vals=np.asarray(values[i], dtype=float)
        order=np.argsort(-np.abs(vals), kind="stable")[:10]
        for rank,j in enumerate(order,1):
            rows.append({"selection_id":selected.iloc[i].selection_id,"window_index":int(selected.iloc[i].window_index),
                         "trace_id":selected.iloc[i].trace_id,"window_start":int(selected.iloc[i].window_start),
                         "true_label":int(selected.iloc[i].label),"predicted_label":int(selected.iloc[i].predicted_label),
                         "predicted_attack_probability":float(selected.iloc[i].predicted_attack_probability),
                         "outcome_group":selected.iloc[i].outcome_group,"rank":rank,"feature_name":names[j],
                         "feature_value":float(X[i,j]),"signed_attribution":float(vals[j]),"absolute_attribution":float(abs(vals[j]))})
    return pd.DataFrame(rows)


def run_phase3b(raw_data_dir: str, nsamples: int=512, lime_num_samples: int=5000):
    cfg=load_config("configs/baseline.yaml")
    random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"]=str(SEED)
    p3a=PROJECT_ROOT/PHASE3A_DIR
    verified=verify_frozen_inputs(cfg,p3a)
    manifest=verified["manifest"]
    root,texts,windows=reconstruct_windows(manifest,cfg,raw_data_dir)
    if len(windows)!=88828: raise RuntimeError(f"Frozen window count changed: {len(windows)}")
    x, split_idx=build_matrices(texts,windows,verified["vectorizer"],verified["selector"])
    if x["train"].shape[1]!=150: raise RuntimeError("Final feature matrix is not 150 columns")
    test_pred=verified["predictions"]
    selected=select_instances(test_pred,windows[windows.split=="test"].reset_index(drop=True))
    out=PROJECT_ROOT/EXP_DIR_NAME
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite existing Phase 3B directory: {out}")
    (out/"figures").mkdir(parents=True)
    selected.to_csv(out/"selected_test_instances.csv",index=False)

    # Deterministic 50/50 training-only SHAP background.
    train_pred=pd.DataFrame({"window_index":np.arange(len(split_idx["train"])),"label":windows.iloc[split_idx["train"]].label.to_numpy()})
    bg_parts=[]
    for label in [0,1]:
        idx=np.flatnonzero(train_pred.label.to_numpy()==label)
        rng=np.random.default_rng(SEED+label)
        chosen=rng.choice(idx,size=min(50,len(idx)),replace=False)
        bg_parts.append(chosen)
    bg_local=np.concatenate(bg_parts)
    background=x["train"][bg_local]
    bg_rows=windows.iloc[split_idx["train"]].iloc[bg_local].copy()
    bg_rows.insert(0,"background_id",[f"shap_bg_{i:03d}" for i in range(len(bg_rows))])
    bg_rows["class_name"]=bg_rows.label.map({0:"normal",1:"attack"})
    bg_rows[["background_id","trace_id","start","label","class_name"]].rename(columns={"start":"window_start"}).to_csv(out/"shap_background_instances.csv",index=False)

    names=verified["feature_names"].sort_values("model_column_index")["feature_name"].tolist()
    model=verified["model"]
    predict_fn=lambda a: model.predict_proba(np.asarray(a,dtype=float))[:,1]
    selected_local_idx=selected.window_index.to_numpy()
    # window_index in frozen test prediction order corresponds to reconstructed test order.
    test_windows=windows[windows.split=="test"].reset_index(drop=True)
    test_x=x["test"]
    Xsel=test_x[selected_local_idx]

    # LIME: full train-only transformed matrix; no validation/test rows in explainer data.
    lime_explainer=LimeTabularExplainer(x["train"],mode="classification",feature_names=names,class_names=["normal","attack"],
                                        discretize_continuous=False,random_state=SEED)
    lime_vals=[]; lime_meta=[]; lime_failures=[]
    for i in range(len(selected)):
        try:
            exp=lime_explainer.explain_instance(Xsel[i],predict_fn,labels=(1,),num_features=10,num_samples=lime_num_samples)
            vals=np.zeros(150,dtype=float)
            for j,w in exp.local_exp[1]: vals[j]=w
            lime_vals.append(vals)
            lime_meta.append({"selection_id":selected.iloc[i].selection_id,"intercept":float(exp.intercept[1]),"local_surrogate_score":float(exp.score[1]),"local_pred":float(exp.local_pred[1])})
        except Exception as e:
            lime_failures.append({"selection_id":selected.iloc[i].selection_id,"error":repr(e)})
            lime_vals.append(np.zeros(150,dtype=float)); lime_meta.append({"selection_id":selected.iloc[i].selection_id,"intercept":None,"local_surrogate_score":None,"local_pred":None})
    if lime_failures: raise RuntimeError(f"LIME explanation failures: {lime_failures}")
    lime_arr=np.vstack(lime_vals)
    lime_df=top_rows("lime",selected,lime_arr,Xsel,names)
    lime_df.to_csv(out/"lime_local_explanations.csv",index=False)
    pd.DataFrame(lime_meta).to_csv(out/"lime_surrogate_metadata.csv",index=False)

    # SHAP KernelExplainer, attack-class probability only, training-only 100-row background.
    np.random.seed(SEED)
    explainer=shap.KernelExplainer(predict_fn,background)
    shap_result=explainer.shap_values(Xsel,nsamples=nsamples,l1_reg="num_features(150)",silent=True)
    if isinstance(shap_result,list): shap_arr=np.asarray(shap_result[-1])
    else: shap_arr=np.asarray(shap_result)
    if shap_arr.ndim==3: shap_arr=shap_arr[:,:,0]
    if shap_arr.shape != Xsel.shape: raise RuntimeError(f"Unexpected SHAP shape {shap_arr.shape}, expected {Xsel.shape}")
    shap_df=top_rows("shap",selected,shap_arr,Xsel,names)
    shap_df.to_csv(out/"shap_local_explanations.csv",index=False)
    base=np.asarray(explainer.expected_value).reshape(-1)
    attack_base=float(base[-1]) if len(base)>1 else float(base[0])

    def global_summary(df,col):
        return df.groupby("feature_name",as_index=False)["absolute_attribution"].mean().rename(columns={"absolute_attribution":"mean_absolute_attribution"}).sort_values("mean_absolute_attribution",ascending=False,kind="stable").reset_index(drop=True)
    lime_global=global_summary(lime_df,"signed_attribution"); shap_global=global_summary(shap_df,"signed_attribution")
    lime_global.insert(0,"rank",np.arange(1,len(lime_global)+1)); shap_global.insert(0,"rank",np.arange(1,len(shap_global)+1))
    lime_global.to_csv(out/"lime_global_summary.csv",index=False); shap_global.to_csv(out/"shap_global_summary.csv",index=False)

    agreements=[]
    for sid in selected.selection_id:
        l=lime_df[lime_df.selection_id==sid].sort_values("rank"); s=shap_df[shap_df.selection_id==sid].sort_values("rank")
        lm=set(l.feature_name); sm=set(s.feature_name); inter=lm&sm; union=lm|sm
        sign=[]
        for f in inter:
            a=float(l.loc[l.feature_name==f,"signed_attribution"].iloc[0]); b=float(s.loc[s.feature_name==f,"signed_attribution"].iloc[0])
            if a!=0 and b!=0: sign.append(np.sign(a)==np.sign(b))
        agreements.append({"selection_id":sid,"outcome_group":selected.loc[selected.selection_id==sid,"outcome_group"].iloc[0],
                           "lime_top10_set":"|".join(sorted(lm)),"shap_top10_set":"|".join(sorted(sm)),"intersection_size":len(inter),"union_size":len(union),
                           "jaccard_overlap":len(inter)/len(union) if union else 1.0,"sign_agreement_overlap":float(np.mean(sign)) if sign else None})
    agree=pd.DataFrame(agreements); agree.to_csv(out/"lime_shap_agreement.csv",index=False)

    # Aggregate figures.
    for title,df,path in [("LIME mean absolute local weight",lime_global,out/"figures/lime_global_top_features.png"),("SHAP mean absolute local value",shap_global,out/"figures/shap_global_top_features.png")]:
        top=df.head(10).iloc[::-1]; plt.figure(figsize=(9,6)); plt.barh(top.feature_name,top.mean_absolute_attribution); plt.xlabel("Mean absolute attribution"); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()
    plt.figure(figsize=(7,5)); plt.hist(agree.jaccard_overlap,bins=np.linspace(0,1,11),edgecolor="black"); plt.xlabel("LIME–SHAP top-10 Jaccard overlap"); plt.ylabel("Number of instances"); plt.title("Per-instance explanation agreement"); plt.tight_layout(); plt.savefig(out/"figures/jaccard_overlap_distribution.png",dpi=160); plt.close()

    # Representative per outcome: use first selected row of each group, both methods.
    for group in ["TP","FN","TN","FP"]:
        row=selected[selected.outcome_group==group].iloc[0]; sid=row.selection_id
        for method,df,path in [("LIME",lime_df,out/f"figures/{group.lower()}_lime_top10.png"),("SHAP",shap_df,out/f"figures/{group.lower()}_shap_top10.png")]:
            d=df[df.selection_id==sid].sort_values("rank").iloc[::-1]; plt.figure(figsize=(9,6)); plt.barh(d.feature_name,d.signed_attribution); plt.xlabel("Signed attribution"); plt.title(f"{method} representative {group} ({sid})"); plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()

    versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":__import__("sklearn").__version__,"joblib":joblib.__version__,"shap":shap.__version__,"lime":LIME_IMPORT_VERSION,"matplotlib":__import__("matplotlib").__version__}
    metadata={"phase":"3B","status":"sampled XAI reconstruction; not all test instances explained","paper_doi":"10.1109/ACCESS.2024.3368377",
              "generated_at_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"manifest_sha256":verified["manifest_sha256"],"split_assignment_sha256":verified["assignment_sha256"],
              "protected_input_hashes":verified["hashes"],"raw_source_dir":str(root),"raw_data_modified":False,
              "selection":{"n_per_group":5,"total":20,"groups":["TP","FN","TN","FP"],"random_seed":42,"rule":"random sample from each frozen outcome group"},
              "shap":{"method":"KernelExplainer","background_total":len(background),"background_normal":int((bg_rows.label==0).sum()),"background_attack":int((bg_rows.label==1).sum()),"background_seed":42,"nsamples":nsamples,"l1_reg":"num_features(150)","expected_value_attack":attack_base},
              "lime":{"method":"lime.lime_tabular.LimeTabularExplainer","training_data":"full transformed TRAIN matrix (61,553 x 150)","num_features":10,"num_samples":lime_num_samples,"random_state":42,"discretize_continuous":False,"class_names":["normal","attack"],"explained_class":"attack_probability","kernel_width":"library/default compatibility value"},
              "leakage_checks":{"background_train_only":True,"lime_training_data_train_only":True,"transformer_fit_performed":False,"model_fit_performed":False,"test_labels_used_for_fitting":False,"validation_used_for_fitting":False,"balancing_used":False,"raw_sequences_in_outputs":False},
              "software":versions}
    (out/"reproducibility_metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True),encoding="utf-8")
    report=f'''# Phase 3B — LIME and SHAP Explanations\n\nThis is a sampled XAI reconstruction, not an all-instance explanation.\n\n**This is a strict reconstructed ADFA-LD baseline, not the authors' exact 52,656-instance dataset.**\n\nPhase 3A model and preprocessing artifacts were loaded frozen and were not retrained or refit.\n\n## Selection\n20 frozen test windows: 5 TP, 5 FN, 5 TN, 5 FP, selected with seed 42.\n\n## LIME\n- LimeTabularExplainer-compatible implementation: `{LIME_IMPORT_VERSION}`\n- Training data: full TRAIN transformed matrix (61,553 × 150)\n- class_names: `['normal', 'attack']`\n- num_features: 10\n- num_samples: {lime_num_samples}\n- random_state: 42\n- discretize_continuous: false\n\n## SHAP\n- KernelExplainer only\n- Background: 100 TRAIN rows, 50 normal + 50 attack\n- seed: 42\n- nsamples: {nsamples}\n- l1_reg: `num_features(150)`\n- explained output: attack probability\n\n## Limitations\nThe exact author-generated dataset was not recovered. LIME/SHAP are computed for only 20 sampled test instances. The external `lime` package was unavailable in the offline runtime, so the repository uses a vendored `LimeTabularExplainer` compatibility implementation exposing the requested API; this is a documented implementation deviation and should not be described as the exact external package runtime.\n\nNo perturbation analysis, retraining, balancing, preprocessing refit, or raw-data modification was performed.\n'''
    (out/"README.md").write_text(report,encoding="utf-8")
    return out
