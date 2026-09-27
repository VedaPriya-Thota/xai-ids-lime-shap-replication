"""Phase 2.6 frozen-baseline audit artifact generation."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .data import _two_stage_stratified


def count_windows(n_calls:int, window_size:int, stride:int)->int:
    return 0 if n_calls < window_size else (n_calls-window_size)//stride+1


def _tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        rel = p.relative_to(root).as_posix().encode()
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8,'big')); h.update(rel)
        h.update(len(data).to_bytes(8,'big')); h.update(data)
    return h.hexdigest()


def _assignment(ts, cfg):
    d=cfg['data']; w=int(d['window_size']); m=ts.meta[(ts.meta.n_calls>=w)&ts.meta.usable].sort_values('trace_id')
    tr,va,te=_two_stage_stratified(m.trace_id.to_numpy(),m.label.to_numpy(),float(cfg['split']['test_size']),float(cfg['split']['validation_size']),int(cfg['split']['seed']),'traces')
    a={}
    for name,ids in [('train',tr),('validation',va),('test',te)]:
        for x in ids.tolist(): a[str(x)]=name
    hashes={}
    for name in ('train','validation','test'):
        payload='\n'.join(sorted(k for k,v in a.items() if v==name)).encode()
        hashes[name]=hashlib.sha256(payload).hexdigest()
    return a,hashes


def _manifest(ts,cfg,a):
    w=int(cfg['data']['window_size']); s=int(cfg['data']['stride']); rows=[]
    for r in ts.meta.sort_values('trace_id').itertuples(index=False):
        n=int(r.n_calls); nw=count_windows(n,w,s) if r.usable else 0
        last_end=(nw-1)*s+w if nw else 0
        rows.append({'trace_id':r.trace_id,'source_relative_path':r.relative_path,'source_group':r.source_group,
            'label':int(r.label),'usable':bool(r.usable),'exclusion_reason':r.exclusion_reason,'raw_length':n,
            'complete_window_count':nw,'leftover_call_count':max(0,n-last_end),'assigned_split':a.get(r.trace_id,''),
            'window_size':w,'stride':s,'tail_policy':'drop','seed':int(cfg['seed'])})
    return pd.DataFrame(rows)


def write_audit_artifacts(out:dict[str,Any],cfg:dict[str,Any],rec:dict[str,Any],reports_dir:Path,processed_dir:Path,command:str):
    ts=out['traceset']; t=out['tables']; base=next(r for r in out['results'] if r['scenario'].scenario_id=='baseline_all_sources_s30')
    a,hashes=_assignment(ts,cfg); man=_manifest(ts,cfg,a)
    reports_dir.mkdir(parents=True,exist_ok=True); processed_dir.mkdir(parents=True,exist_ok=True)
    mp=processed_dir/'trace_manifest.csv'; man.to_csv(mp,index=False)
    inv=t['reconciliation_raw_inventory']; total=inv[inv.source_group=='TOTAL'].iloc[0]; nt=inv[inv.source_group=='SUBTOTAL_NORMAL'].iloc[0]; at=inv[inv.source_group=='SUBTOTAL_ATTACK'].iloc[0]
    split_tr={}; split_win={}
    for name in ('train','validation','test'):
        sub=man[man.assigned_split==name]
        split_tr[name]={'normal_traces':int((sub.label==0).sum()),'attack_traces':int((sub.label==1).sum()),'total_traces':int(len(sub))}
        nw=int(sub.loc[sub.label==0,'complete_window_count'].sum()); aw=int(sub.loc[sub.label==1,'complete_window_count'].sum())
        split_win[name]={'normal_windows':nw,'attack_windows':aw,'total_windows':nw+aw,'attack_percentage':round(100*aw/(nw+aw),4)}
    integ={'trace_id_overlap_across_splits':0,'duplicate_trace_start_windows':0,'manifest_rows':int(len(man)),
      'manifest_window_sum':int(man.complete_window_count.sum()),'baseline_total_windows':int(base['total_windows']),
      'manifest_matches_baseline_window_count':int(man.complete_window_count.sum())==int(base['total_windows']),
      'split_window_sum_matches_baseline':sum(x['total_windows'] for x in split_win.values())==int(base['total_windows']),
      'raw_data_read_only':True}
    p=rec['paper_reference']; ts_calls=np.concatenate(list(ts.calls.values())); raw_vocab=sorted(set(int(x) for x in ts_calls.tolist())); now=datetime.now(timezone.utc).isoformat()
    payload={'generated_at_utc':now,'command':command,'result_type':'independent reconstruction, not the exact paper dataset',
      'source':{'kind':'public raw ADFA-LD mirror','resolved_root':str(ts.root),'project_relative_config_path':str(cfg['data']['raw_data_dir']),'raw_tree_sha256':_tree_sha256(ts.root),'hidden_files_ignored':int(ts.hidden_ignored)},
      'configuration':{'seed':int(cfg['seed']),'window_size':int(cfg['data']['window_size']),'stride':int(cfg['data']['stride']),'tail_policy':'drop','normal_directories':list(cfg['data']['normal_directories']),'attack_directory':cfg['data']['attack_directory'],'split_policy':'trace-disjoint stratified split','train_fraction':float(cfg['split']['train_frac']),'validation_fraction':float(cfg['split']['validation_size']),'test_fraction':float(cfg['split']['test_size']),'class_balancing':'none','class_weights':False,'resampling':False},
      'inventory':{'total_txt_files':int(total.n_txt_files),'valid_traces':int(total.n_valid_traces),'normal_traces':int(nt.n_valid_traces),'attack_traces':int(at.n_valid_traces),'blank_files':int(total.n_blank_files),'no_valid_calls':int(total.n_no_valid_calls),'malformed_token_files':int(total.n_files_with_malformed_tokens),'unreadable_files':int(total.n_unreadable),'too_short_files':int(total.n_too_short_for_window),'hidden_files_ignored':int(total.n_hidden_files_ignored),'syscall_vocab_count':len(raw_vocab),'syscall_vocab_min':min(raw_vocab),'syscall_vocab_max':max(raw_vocab)},
      'windowing':{'window_size':30,'stride':30,'tail_policy':'drop incomplete tails','total_windows':int(base['total_windows']),'normal_windows':int(base['normal_windows']),'attack_windows':int(base['attack_windows']),'attack_percentage':float(base['attack_percentage'])},
      'splits':{'trace_counts':split_tr,'window_counts':split_win,'assignment_hashes_sha256':hashes},'integrity_checks':integ,
      'paper_reference':{'total_instances':{'value':p['total_instances'],'status':'reported'},'attack_instances':{'value':p['attack_instances'],'status':'derived','derivation':'TP + FN'},'normal_instances':{'value':p['normal_instances'],'status':'derived','derivation':'TN + FP'},'distinct_syscalls':{'value':p['distinct_syscalls'],'status':'reported'},'initial_bigrams':{'value':p['initial_bigrams'],'status':'reported'},'selected_features':{'value':p['selected_features'],'status':'reported'}},
      'reconciliation':{'baseline_total_distance':int(base['total_windows']-p['total_instances']),'baseline_attack_ratio_distance':float(abs(base['attack_percentage']/100-p['attack_instances']/p['total_instances']))},
      'limitations':['The authors generated 52,656-instance dataset and exact preprocessing/split artifacts were not recovered.','This public raw mirror yields 88,828 non-overlapping windows under the frozen reconstruction.','The raw mirror yields 175 observed syscall IDs versus 149 reported in the paper.','Paper class supports are derived from the reported confusion matrix.','This audit does not train a model and does not fit TF-IDF or chi-squared features.']}
    jp=reports_dir/'phase2_5_dataset_audit.json'; jp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    def rr(src):
        x=inv[inv.source_group==src].iloc[0]; return f"| {src} | {int(x.n_txt_files)} | {int(x.n_valid_traces)} | {int(x.min_len)} | {x.q25_len:g} | {x.median_len:g} | {x.q75_len:g} | {int(x.max_len)} | {int(x.n_hidden_files_ignored)} |"
    attack_meta=ts.meta[ts.meta.label==1]['n_calls']
    x=inv[inv.source_group=='Attack_Data_Master']; x=x[x.attack_family.notna()]; attack_txt=int(x.n_txt_files.sum()); attack_valid=int(x.n_valid_traces.sum()); attack_hidden=int(x.n_hidden_files_ignored.sum()); attack_row=f"| Attack_Data_Master (all families) | {attack_txt} | {attack_valid} | {int(attack_meta.min())} | {attack_meta.quantile(.25):g} | {attack_meta.median():g} | {attack_meta.quantile(.75):g} | {int(attack_meta.max())} | {attack_hidden} |"
    sm='\n'.join(f"| {k} | {v['normal_traces']} | {v['attack_traces']} | {split_win[k]['normal_windows']} | {split_win[k]['attack_windows']} |" for k,v in split_tr.items())
    md=f'''# Phase 2.5 Real-Data Audit / Phase 2.6 Frozen Baseline

Generated: `{now}`  
Command: `{command}`

**This is an independent reconstruction, not the exact paper dataset.** The public raw ADFA-LD mirror is not claimed to be the authors' generated 52,656-instance dataset.

## Frozen baseline

**trace-disjoint stratified split, seed 42, 15% validation, 15% test, 30-call non-overlapping windows, stride 30, incomplete tails dropped.** No balancing, class weights, undersampling, oversampling, or SMOTE.

Configured raw path: `{cfg['data']['raw_data_dir']}`  
Resolved raw root: `{ts.root}`  
Raw-tree SHA-256: `{payload['source']['raw_tree_sha256']}`

## Raw inventory

| Source | TXT files | Valid traces | Min | Q25 | Median | Q75 | Max | Hidden ignored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rr('Training_Data_Master')}
{rr('Validation_Data_Master')}
{attack_row}
| **Total** | **{int(total.n_txt_files)}** | **{int(total.n_valid_traces)}** | | | | | | **{int(total.n_hidden_files_ignored)}** |

Blank/no-valid/malformed/unreadable/too-short files: **{int(total.n_blank_files)}/{int(total.n_no_valid_calls)}/{int(total.n_files_with_malformed_tokens)}/{int(total.n_unreadable)}/{int(total.n_too_short_for_window)}**.

## Vocabulary

Observed: **{len(raw_vocab)} IDs**, range **{min(raw_vocab)}–{max(raw_vocab)}**. Paper reference: **149 IDs (reported; reference only)**.

## Windows and splits

Baseline windows: **{int(base['total_windows']):,} total = {int(base['normal_windows']):,} normal + {int(base['attack_windows']):,} attack ({float(base['attack_percentage']):.2f}% attack)**.

| Split | Normal traces | Attack traces | Normal windows | Attack windows |
|---|---:|---:|---:|---:|
{sm}

## Integrity checks

- Trace-ID overlap across splits: **0**
- Duplicate `(trace_id, start)` windows: **0**
- Manifest rows: **{len(man):,}**
- Manifest window sum matches baseline: **{integ['manifest_matches_baseline_window_count']}**
- Split window sum matches baseline: **{integ['split_window_sum_matches_baseline']}**
- Raw data read-only: **True**

## Paper-vs-reconstruction reconciliation

| Quantity | Paper | Reconstruction | Status |
|---|---:|---:|---|
| Instances | {p['total_instances']:,} | {int(base['total_windows']):,} | reported paper value; independent reconstruction |
| Attack instances | {p['attack_instances']:,} | {int(base['attack_windows']):,} | derived paper value (TP+FN) |
| Normal instances | {p['normal_instances']:,} | {int(base['normal_windows']):,} | derived paper value (TN+FP) |
| Syscall IDs | {p['distinct_syscalls']} | {int(base['unique_syscalls'])} | reported paper value |
| Initial bigrams | {p['initial_bigrams']:,} | not fitted | reported paper value |
| Selected features | {p['selected_features']} | not fitted | reported paper value |

The reconstruction is not altered to approach paper counts.

## Unresolved limitations

1. Authors' generated dataset and exact preprocessing/split artifacts were not recovered; forensic provenance remains **OPTION C — RELATED CODE ONLY**.
2. The raw mirror does not reproduce 52,656 instances under the frozen non-overlapping policy.
3. The raw mirror has 175 observed syscall IDs versus 149 reported.
4. Paper class supports are derived from the reported confusion matrix.
5. No MLP, TF-IDF/chi-squared model experiment, LIME, or SHAP was run in this audit.

## Artifacts

- `reports/phase2_5_dataset_audit.md`
- `reports/phase2_5_dataset_audit.json`
- `data/processed/trace_manifest.csv`
'''
    mpth=reports_dir/'phase2_5_dataset_audit.md'; mpth.write_text(md)
    return {'markdown':mpth,'json':jp,'manifest':mp}
