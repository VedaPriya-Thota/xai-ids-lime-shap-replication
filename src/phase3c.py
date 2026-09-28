"""Phase 3C: raw-sequence perturbation evaluation for Phase 3A/3B artifacts.

This module performs no fitting. It reconstructs raw 30-call windows, applies explanation-guided
and matched random raw-position edits, then reuses only Phase 3A/3B TF-IDF/chi2/model/explanation
artifacts loaded read-only from an artifact directory (the historical frozen directories by
default, or explicit `phase3a_dir`/`phase3b_dir` pointing at runs produced under the repository's
documented environment).
"""
from __future__ import annotations

import hashlib, json, os, platform, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config, resolve_dir_or_default, resolve_path
from .data import find_adfa_root, load_traces

SEED = 42
OUT_REL = Path("results/experiments/phase3c_perturbation_evaluation")
P3A_REL = Path("results/experiments/phase3a_strict_reconstructed_mlp")
P3B_REL = Path("results/experiments/phase3b_xai_explanations")
EXPECTED_MANIFEST_SHA = "765289887ac57f0a7f08b0d2761c9507e65f8ff98805ab69dc08a6cc312bf4f1"
EXPECTED_ASSIGNMENT_SHA = "5c2b0745cccde2c060ce139fe92e383510ab4237eb0d53b64b9c5f29024e04ba"
EXPECTED_CM = [[11681, 432], [428, 1195]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assignment_hash(manifest: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for r in manifest.sort_values("trace_id").itertuples(index=False):
        h.update(f"{r.trace_id}|{int(r.label)}|{r.assigned_split}\n".encode())
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def protected_paths(p3a_dir: Path | None = None, p3b_dir: Path | None = None) -> dict[str, Path]:
    p3a = resolve_dir_or_default(p3a_dir, P3A_REL)
    p3b = resolve_dir_or_default(p3b_dir, P3B_REL)
    return {
        "baseline_config": PROJECT_ROOT / "configs/baseline.yaml",
        "manifest": PROJECT_ROOT / "data/processed/trace_manifest.csv",
        "audit": PROJECT_ROOT / "reports/phase2_5_dataset_audit.json",
        "audit_md": PROJECT_ROOT / "reports/phase2_5_dataset_audit.md",
        "model": p3a / "mlp_model.joblib",
        "vectorizer": p3a / "preprocessing/tfidf_vectorizer.joblib",
        "selector": p3a / "preprocessing/chi2_selector.joblib",
        "feature_names": p3a / "preprocessing/selected_feature_names.csv",
        "test_predictions": p3a / "test_predictions.csv",
        "selected_instances": p3b / "selected_test_instances.csv",
        "lime": p3b / "lime_local_explanations.csv",
        "shap": p3b / "shap_local_explanations.csv",
        "p3b_metadata": p3b / "reproducibility_metadata.json",
    }


def verify_inputs(p3a_dir: Path | None = None, p3b_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    paths = protected_paths(p3a_dir, p3b_dir)
    missing = [rel(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Protected Phase 3C inputs missing: {missing}")
    hashes = {k: sha256_file(v) for k, v in paths.items()}
    if hashes["manifest"] != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Manifest SHA-256 mismatch")
    manifest = pd.read_csv(paths["manifest"])
    if assignment_hash(manifest) != EXPECTED_ASSIGNMENT_SHA:
        raise RuntimeError("Split-assignment SHA-256 mismatch")
    names = pd.read_csv(paths["feature_names"])
    if len(names) != 150:
        raise RuntimeError(f"Expected 150 selected features, got {len(names)}")
    pred = pd.read_csv(paths["test_predictions"])
    cm = pd.crosstab(pred.label, pred.predicted_label).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values.tolist()
    if cm != EXPECTED_CM:
        raise RuntimeError(f"Frozen confusion matrix mismatch: {cm}")
    sel = pd.read_csv(paths["selected_instances"])
    counts = sel.outcome_group.value_counts().to_dict()
    if len(sel) != 20 or counts != {"TP": 5, "FN": 5, "TN": 5, "FP": 5}:
        raise RuntimeError(f"Frozen selection mismatch: {len(sel)} rows, {counts}")
    for f in ["lime", "shap"]:
        df = pd.read_csv(paths[f])
        if len(df) != 200 or df.groupby("selection_id").size().min() != 10 or df.groupby("selection_id").size().max() != 10:
            raise RuntimeError(f"Phase 3B {f} output is not exactly 10 features per selected instance")
    model = joblib.load(paths["model"])
    vec = joblib.load(paths["vectorizer"])
    selector = joblib.load(paths["selector"])
    if selector.get_support().sum() != 150 or getattr(selector, "k", None) != 150:
        raise RuntimeError("Frozen selector does not select exactly 150 features")
    if not all(hasattr(x, "transform") for x in [vec, selector]) or not hasattr(model, "predict_proba"):
        raise RuntimeError("Frozen Phase 3A artifacts incompatible")
    return {"paths": paths, "hashes": hashes, "manifest": manifest, "predictions": pred, "selected": sel,
            "lime": pd.read_csv(paths["lime"]), "shap": pd.read_csv(paths["shap"]),
            "feature_names": names, "model": model, "vectorizer": vec, "selector": selector}, hashes


def reconstruct_raw_windows(manifest: pd.DataFrame, cfg: dict[str, Any], raw_data_dir: str | Path) -> dict[tuple[str, int], np.ndarray]:
    d = cfg["data"]
    root = find_adfa_root(resolve_path(raw_data_dir), list(d["normal_dirs"]) + list(d["attack_dirs"]))
    meta, calls, _ = load_traces(root, list(d["normal_dirs"]), d["attack_dirs"][0])
    current = meta.set_index("trace_id")
    m = manifest.set_index("trace_id")
    if set(current.index) != set(m.index):
        raise RuntimeError("Current raw trace inventory differs from frozen manifest")
    for tid, r in m.iterrows():
        cur = current.loc[tid]
        if int(cur.label) != int(r.label) or int(cur.n_calls) != int(r.raw_length):
            raise RuntimeError(f"Raw trace differs from frozen manifest: {tid}")
    result: dict[tuple[str, int], np.ndarray] = {}
    for r in manifest.itertuples(index=False):
        seq = np.asarray(calls[r.trace_id], dtype=np.int64)
        for start in range(0, len(seq) - 29, 30):
            result[(r.trace_id, int(start))] = seq[start:start + 30].copy()
    return result


def tokens(seq: np.ndarray) -> str:
    return " ".join(f"{int(a)}_{int(b)}" for a, b in zip(seq[:-1], seq[1:]))


def transform_predict(seq: np.ndarray, vec, selector, model) -> tuple[float, int]:
    X = selector.transform(vec.transform([tokens(seq)]))
    p = float(model.predict_proba(X)[:, 1][0])
    return p, int(p >= 0.5)


def normal_training_frequency(manifest: pd.DataFrame, raw_windows: dict[tuple[str, int], np.ndarray]) -> dict[int, int]:
    freq: dict[int, int] = {}
    train_normal = manifest[(manifest.assigned_split == "train") & (manifest.label == 0)]
    for r in train_normal.itertuples(index=False):
        for start in range(0, int(r.raw_length) - 29, 30):
            for c in raw_windows[(r.trace_id, start)]:
                c = int(c); freq[c] = freq.get(c, 0) + 1
    return freq


def choose_neutral(freq: dict[int, int], original: int, preceding: int | None, target_pair: tuple[int, int]) -> int:
    for cand, _count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if cand == original:
            continue
        if preceding is not None and (preceding, cand) == target_pair:
            continue
        return cand
    raise RuntimeError("No valid neutral replacement candidate")


def guided_edits(original: np.ndarray, features: pd.DataFrame, freq: dict[int, int]) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    work = original.copy()
    edited: set[int] = set()
    edit_log: list[dict[str, Any]] = []
    actionable: list[dict[str, Any]] = []
    actionable_features: set[str] = set()
    for row in features.sort_values("rank").itertuples(index=False):
        a, b = map(int, str(row.feature_name).split("_", 1))
        found = False
        for i in range(29):
            if int(work[i]) == a and int(work[i + 1]) == b and (i + 1) not in edited:
                found = True
                pos = i + 1
                old = int(work[pos])
                new = choose_neutral(freq, old, int(work[i]), (a, b))
                if new == old:
                    continue
                before_pair_changes = [(j, f"{int(work[j])}_{int(work[j+1])}") for j in range(29) if j == i or j == pos-1]
                work[pos] = new
                edited.add(pos)
                after_pair_changes = [(j, f"{int(work[j])}_{int(work[j+1])}") for j in range(29) if j == i or j == pos-1]
                cascaded = ";".join(f"{oldp}->{newp}" for (j,oldp),(j2,newp) in zip(before_pair_changes,after_pair_changes) if oldp != newp)
                actionable_features.add(str(row.feature_name))
                actionable.append({"rank": int(row.rank), "feature_name": row.feature_name, "target_a": a, "target_b": b, "position": pos,
                                   "before_syscall": old, "after_syscall": new})
                edit_log.append({"rank": int(row.rank), "feature_name": row.feature_name, "position": pos,
                                 "before_syscall": old, "after_syscall": new, "neighbor_before": a,
                                 "cascaded_neighbor_changes": cascaded, "original_target_pair": f"{a}_{b}", "status":"edited"})
        if not found:
            edit_log.append({"rank": int(row.rank), "feature_name": row.feature_name, "position": None,
                             "before_syscall": None, "after_syscall": None, "neighbor_before": None,
                             "cascaded_neighbor_changes": "", "original_target_pair": f"{a}_{b}", "status": "not_actionable"})
    return work, actionable, edit_log, actionable_features

def random_edits(original: np.ndarray, n_edits: int, seed: int, freq: dict[int, int], target_pairs: list[tuple[int, int]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    work = original.copy(); edited: set[int] = set(); rng = np.random.default_rng(seed); log=[]
    eligible = list(range(30)); rng.shuffle(eligible)
    for j in range(n_edits):
        candidates = [p for p in eligible if p not in edited]
        if not candidates: raise RuntimeError("Insufficient random edit positions")
        pos = candidates[0]; edited.add(pos)
        old=int(work[pos]); pair=target_pairs[j % len(target_pairs)] if target_pairs else (int(work[pos-1]) if pos else -1, old)
        preceding=int(work[pos-1]) if pos else None
        new=choose_neutral(freq, old, preceding, pair)
        work[pos]=new
        log.append({"position":pos,"before_syscall":old,"after_syscall":new,"target_constraint":f"{pair[0]}_{pair[1]}"})
    return work,log


def cascaded_changes(before: np.ndarray, after: np.ndarray, edited_positions: list[int]) -> list[str]:
    changed=[]
    for i,(a,b) in enumerate(zip(before[:-1],before[1:])):
        c,d=after[i],after[i+1]
        if int(a)!=int(c) or int(b)!=int(d):
            changed.append(f"{int(a)}_{int(b)}->{int(c)}_{int(d)}")
    return changed


def run_phase3c(raw_data_dir: str | Path, phase3a_dir: str | Path | None = None,
                 phase3b_dir: str | Path | None = None, experiment_dir: str | Path | None = None) -> Path:
    """Run Phase 3C against Phase 3A/3B artifact directories.

    `phase3a_dir`/`phase3b_dir` default to the historical frozen directories so the existing
    default workflow is unchanged. Pass explicit directories (e.g. freshly regenerated Phase 3A/3B
    outputs produced under the repository's documented environment) to consume those instead,
    without touching the frozen historical directories. `experiment_dir` likewise defaults to the
    historical Phase 3C output location and can be overridden the same way.
    """
    cfg=load_config("configs/baseline.yaml")
    random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"]=str(SEED)
    p3a_resolved = resolve_dir_or_default(phase3a_dir, P3A_REL)
    verified, before_hashes = verify_inputs(phase3a_dir, phase3b_dir)
    out=resolve_dir_or_default(experiment_dir, OUT_REL)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite existing Phase 3C directory: {out}")
    out.mkdir(parents=True); (out/"figures").mkdir()
    raw_windows=reconstruct_raw_windows(verified["manifest"],cfg,raw_data_dir)
    selected=verified["selected"]
    for r in selected.itertuples(index=False):
        key=(r.trace_id,int(r.window_start))
        if key not in raw_windows or len(raw_windows[key])!=30:
            raise RuntimeError(f"Selected window cannot be reconstructed at length 30: {key}")
    freq=normal_training_frequency(verified["manifest"],raw_windows)
    neutral_default=sorted(freq.items(),key=lambda kv:(-kv[1],kv[0]))[0]
    vec,selector,model=verified["vectorizer"],verified["selector"],verified["model"]
    names=verified["feature_names"].sort_values("model_column_index")["feature_name"].tolist()
    feature_rank={str(r.feature_name):int(r.rank) for r in pd.concat([verified["lime"],verified["shap"]])[['feature_name','rank']].drop_duplicates().itertuples(index=False)}
    all_rows=[]; edit_rows=[]; pair_rows=[]
    for method,edf in [("lime_compatible",verified["lime"]),("shap_kernel",verified["shap"])]:
        for r in selected.itertuples(index=False):
            sid=r.selection_id; original=raw_windows[(r.trace_id,int(r.window_start))]
            feats=edf[edf.selection_id==sid].sort_values("rank")
            guided,guided_actions,glog,actionable_features=guided_edits(original,feats,freq)
            p0,y0=transform_predict(original,vec,selector,model)
            frozen_row=verified["predictions"][verified["predictions"].window_index==r.window_index].iloc[0]
            if not np.isclose(p0,float(frozen_row.predicted_attack_probability),rtol=1e-10,atol=1e-12):
                raise RuntimeError(f"Reconstructed original probability mismatch for {sid}")
            if int(frozen_row.label)!=int(r.label) or int(frozen_row.predicted_label)!=int(r.predicted_label):
                raise RuntimeError(f"Frozen prediction label mismatch for {sid}")
            pg,yg=transform_predict(guided,vec,selector,model)
            if guided_actions:
                seed=int(hashlib.sha256(f"{SEED}|{method}|{sid}".encode()).hexdigest()[:8],16)
                random_seq,rlog=random_edits(original,len(guided_actions),seed,freq,[tuple(map(int,str(x).split("_"))) for x in feats.feature_name])
                pr,yr=transform_predict(random_seq,vec,selector,model)
            else:
                seed=int(hashlib.sha256(f"{SEED}|{method}|{sid}".encode()).hexdigest()[:8],16); random_seq=original.copy(); rlog=[]; pr,yr=p0,y0
            signed=float(feats[feats.feature_name.isin([a['feature_name'] for a in guided_actions])].signed_attribution.sum()) if guided_actions else 0.0
            direction_consistent = bool((pg-p0)*(-signed) > 0) if signed != 0 and pg != p0 else (bool(pg==p0) if signed==0 else False)
            row={"selection_id":sid,"method":method,"outcome_group":r.outcome_group,"trace_id":r.trace_id,"window_start":int(r.window_start),
                 "true_label":int(r.label),"original_predicted_label":int(r.predicted_label),"original_attack_probability":p0,
                 "guided_attack_probability":pg,"random_attack_probability":pr,"guided_probability_delta":pg-p0,"random_probability_delta":pr-p0,
                 "absolute_guided_delta":abs(pg-p0),"absolute_random_delta":abs(pr-p0),"guided_label_flip":int(yg!=y0),"random_label_flip":int(yr!=y0),
                 "guided_edit_count":len(guided_actions),"random_edit_count":len(rlog),"actionable_top10_count":len(actionable_features),
                 "actionable_edit_count":len(guided_actions),
                 "actionable_feature_rate":len(actionable_features)/10.0,"explanation_signed_attribution_sum":signed,
                 "direction_consistent_with_explanation":direction_consistent,"random_seed":seed}
            all_rows.append(row)
            for x in glog: edit_rows.append({"selection_id":sid,"method":method,"intervention":"guided",**x})
            for x in rlog: edit_rows.append({"selection_id":sid,"method":method,"intervention":"random",**x})
    per=pd.DataFrame(all_rows); edits=pd.DataFrame(edit_rows)
    per.to_csv(out/"per_instance_perturbations.csv",index=False); edits.to_csv(out/"per_edit_log.csv",index=False)
    method_summary=per.groupby("method").agg(n_instances=("selection_id","size"),actionable_cases=("actionable_top10_count",lambda s:int((s>0).sum())),mean_actionable_feature_rate=("actionable_feature_rate","mean"),mean_actionable_edits=("actionable_edit_count","mean"),mean_abs_guided_delta=("absolute_guided_delta","mean"),mean_abs_random_delta=("absolute_random_delta","mean"),guided_label_flips=("guided_label_flip","sum"),random_label_flips=("random_label_flip","sum")).reset_index()
    method_summary.to_csv(out/"aggregate_method_summary.csv",index=False)
    group_summary=per.groupby(["method","outcome_group"]).agg(n_instances=("selection_id","size"),actionable_cases=("actionable_top10_count",lambda s:int((s>0).sum())),mean_actionable_feature_rate=("actionable_feature_rate","mean"),mean_actionable_edits=("actionable_edit_count","mean"),mean_guided_delta=("guided_probability_delta","mean"),mean_random_delta=("random_probability_delta","mean"),mean_abs_guided_delta=("absolute_guided_delta","mean"),mean_abs_random_delta=("absolute_random_delta","mean"),guided_label_flips=("guided_label_flip","sum"),random_label_flips=("random_label_flip","sum")).reset_index()
    group_summary.to_csv(out/"aggregate_group_summary.csv",index=False)
    paired=per[["selection_id","method","outcome_group","absolute_guided_delta","absolute_random_delta","guided_probability_delta","random_probability_delta","guided_label_flip","random_label_flip"]].copy()
    paired["abs_delta_difference"]=paired.absolute_guided_delta-paired.absolute_random_delta
    paired.to_csv(out/"paired_guided_vs_random.csv",index=False)
    neutral_meta={"derivation":"highest-frequency syscall ID across TRAIN-only NORMAL 30-call windows; candidates exclude the original syscall and any candidate that recreates the target pair at the edited position","default_most_frequent_id":int(neutral_default[0]),"default_training_frequency":int(neutral_default[1]),"training_normal_call_count":int(sum(freq.values())),"unique_ids":len(freq)}
    (out/"neutral_replacement_metadata.json").write_text(json.dumps(neutral_meta,indent=2,sort_keys=True),encoding="utf-8")
    (out/"config_snapshot.json").write_text(json.dumps({"seed":SEED,"policy":"top-10 bigram raw-position intervention; second syscall replaced; one edit per raw position; matched random count","window_size":30,"vectorization":"frozen TF-IDF + frozen chi2 selector","threshold":0.5},indent=2),encoding="utf-8")
    (out/"input_hashes.json").write_text(json.dumps(before_hashes,indent=2,sort_keys=True),encoding="utf-8")
    # figures
    labels=method_summary.method.tolist(); x=np.arange(len(labels)); width=.35
    plt.figure(); plt.bar(x-width/2,method_summary.mean_abs_guided_delta,width,label="guided"); plt.bar(x+width/2,method_summary.mean_abs_random_delta,width,label="random"); plt.xticks(x,labels,rotation=15); plt.ylabel("Mean absolute probability delta"); plt.tight_layout(); plt.legend(); plt.savefig(out/"figures/guided_vs_random_absolute_delta.png",dpi=160); plt.close()
    plt.figure();
    for i,m in enumerate(labels):
        g=group_summary[group_summary.method==m]; plt.bar(np.arange(4)+i*.35,g.guided_label_flips/g.n_instances,width=.35,label=m)
    plt.xticks(np.arange(4)+.175,["FN","FP","TN","TP"]); plt.ylabel("Guided label-flip rate"); plt.legend(); plt.tight_layout(); plt.savefig(out/"figures/label_flip_rates.png",dpi=160); plt.close()
    plt.figure();
    for m in labels:
        g=group_summary[group_summary.method==m].set_index("outcome_group"); vals=[g.loc[k,"mean_guided_delta"] for k in ["TP","FN","TN","FP"]]; plt.plot(["TP","FN","TN","FP"],vals,marker="o",label=m)
    plt.ylabel("Mean guided probability delta"); plt.tight_layout(); plt.legend(); plt.savefig(out/"figures/probability_delta_by_outcome_group.png",dpi=160); plt.close()
    plt.figure();
    for m in labels:
        g=group_summary[group_summary.method==m].set_index("outcome_group"); plt.bar([k+"/"+m for k in g.index],g.actionable_cases/g.n_instances)
    plt.ylabel("Actionable-case rate"); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(out/"figures/actionable_feature_rate.png",dpi=160); plt.close()
    after={k:sha256_file(v) for k,v in protected_paths(phase3a_dir, phase3b_dir).items()}
    if after != before_hashes: raise RuntimeError("Protected artifact hash changed during Phase 3C")
    p3b_resolved = resolve_dir_or_default(phase3b_dir, P3B_REL)
    metadata={"phase":"3C","generated_at_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"protected_hashes_before":before_hashes,"protected_hashes_after":after,"raw_data_modified":False,"model_fit":False,"preprocessing_fit":False,"explainer_fit":False,"balancing":False,"methods":["lime_compatible","shap_kernel"],"limitations":["sampled 20 test instances only","vendored LIME-compatible implementation from Phase 3B","perturbations are model-input interventions, not causal claims"],"phase3a_source_dir":rel(p3a_resolved),"phase3b_source_dir":rel(p3b_resolved),"software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"joblib":joblib.__version__,"matplotlib":__import__('matplotlib').__version__}}
    (out/"reproducibility_metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True),encoding="utf-8")
    report=f'''# Phase 3C — Explanation-guided perturbation evaluation\n\nThis is a bounded perturbation evaluation of the frozen Phase 3A model using frozen Phase 3B explanations. It is not causal evidence about real attacks and is not an exact reproduction of the paper.\n\n- Frozen test sample: 20 instances (5 TP, 5 FN, 5 TN, 5 FP).\n- Raw representation: 30-call sequences; target features are adjacent syscall 2-grams.\n- Guided intervention: process top-10 explanation-ranked bigrams, replace the second syscall of each actionable pair using a training-only normal-frequency neutral policy; each raw position is edited at most once.\n- Random control: same number of edits, deterministic positions, same neutral policy.\n- Prediction: frozen Phase 3A TF-IDF, chi-square selector, and MLP only.\n\n## Limitations\nThe Phase 3B LIME results come from a vendored LIME-compatible implementation because the official package was unavailable in the offline runtime. The dataset is a strict reconstruction rather than the authors' exact 52,656-instance dataset. Results are descriptive only; no causal interpretation or significance claim is made.\n'''
    (out/"README.md").write_text(report,encoding="utf-8")
    (out/"RESULTS.md").write_text(report+"\n## Aggregate method summary\n\n"+method_summary.to_markdown(index=False)+"\n\n## Aggregate group summary\n\n"+group_summary.to_markdown(index=False),encoding="utf-8")
    return out
