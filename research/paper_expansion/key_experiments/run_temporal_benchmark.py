from __future__ import annotations
"""Fit-free preflight for the frozen T temporal-context experiment.

This version intentionally has no model-fitting code."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PILOT = EXP.parent
OUT = HERE / "T_temporal_v1"
PREFLIGHT = OUT / "preflight_v1"
CACHE = PILOT / "session_feature_cache_v1.joblib"
RUNNER = EXP / "runner.py"
PROTOCOL = HERE / "protocol_key_v1.md"
ADDENDUM = HERE / "T_temporal_protocol_v1.md"
IDENTITY = EXP / "shared_identity_audit_v1.json"
AXIS_RUN = PILOT / "runs" / "axis_window_v1"
POLICIES = ("primary", "blank_normal")
SPLITS = ("record42", "timeblock42")
FEATURES = ("imu_only", "axis_window")
MODELS = ("RF100", "HistGB")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def hash_obj(value) -> str:
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def load_axis_builder():
    src=PILOT / "run_axis_window_control.py"
    name="roadsens_axis_builder_frozen"
    spec=importlib.util.spec_from_file_location(name,src)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load existing axis-window builder")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def runtime():
    import sklearn, numpy, pandas
    expected=Path(r"C:\Users\Administrator\miniconda3\python.exe").resolve()
    if Path(sys.executable).resolve()!=expected: raise RuntimeError(f"wrong interpreter {sys.executable}")
    if sklearn.__version__!="1.9.0": raise RuntimeError(f"wrong sklearn {sklearn.__version__}")
    return {"executable":sys.executable,"python":sys.version,"sklearn":sklearn.__version__,"numpy":numpy.__version__,"pandas":pandas.__version__}

def window_members_by_anchor(session: dict) -> dict[int, frozenset[int]]:
    """Return raw full-cadence row IDs contributing to each sampled anchor.

    Uses finite timestamps, stable (time, source-row) ordering and the exact
    closed-left interval semantics of run_pilot.causal_window_features.
    """
    times=np.asarray(session["times_full"],dtype=np.float64)
    anchors=np.asarray(session["source_row"],dtype=np.int64)
    order=np.flatnonzero(np.isfinite(times))
    order=order[np.argsort(times[order],kind="stable")]
    sorted_t=times[order]
    position=np.full(len(times),-1,dtype=np.int64)
    position[order]=np.arange(len(order),dtype=np.int64)
    out={}
    for sr in anchors:
        sr=int(sr)
        if sr<0 or sr>=len(times) or position[sr]<0: continue
        now=times[sr]
        left=int(np.searchsorted(sorted_t,now-1.0,side="left"))
        right=int(position[sr])+1
        out[sr]=frozenset(int(x) for x in order[left:right])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preflight",action="store_true",required=True,help="fit-free only")
    args=ap.parse_args()
    import joblib
    sys.path.insert(0,str(EXP))
    import runner as r
    rt=runtime()
    cache,frozen,splitqa=r.validate_immutable_inputs()
    specs,rows=r.build_all_splits(cache["sessions"],frozen,splitqa)
    audit=json.loads(IDENTITY.read_text(encoding="utf-8"))
    if audit.get("status")!="shared_identity_audit_verified": raise ValueError("shared identity certificate not verified")
    binding=audit["binding"]
    for key,expected in (("cache_sha256",r.sha256(r.CACHE)),("run_pilot_sha256",r.sha256(r.PILOT/"run_pilot.py")),("shared_runner_sha256",r.sha256(EXP/"runner.py")),("protocol_sha256",r.sha256(EXP/"protocol_v1.md")),("frozen_split_authority_sha256",r.sha256(EXP/"frozen_split_authority_v1.md"))):
        if binding.get(key)!=expected: raise ValueError(f"shared identity certificate stale: {key}")
    # Verify current folds/fit IDs against the prior exact identity certificate.
    canonical_folds={}
    for policy in POLICIES:
        table=rows[policy]
        for sid in SPLITS:
            item=specs[(policy,sid)]
            identity_key=f"{policy}/{sid}"
            old=audit["derived_identity"]["split_fit_test_sha256"].get(identity_key)
            if old is None: raise ValueError(f"identity certificate lacks {identity_key}")
            base_folds=[]
            for fi,(tr0,te) in enumerate(item["folds"]):
                fit0=r.cap_train_indices(tr0,table.y,fi,42,20000)
                expected=old[fi]
                got={"train":hash_obj(tr0.tolist()),"fit":hash_obj(fit0.tolist()),"test":hash_obj(te.tolist())}
                if got!=expected: raise ValueError(f"frozen source split identity differs: {identity_key}, fold{fi+1}")
                base_folds.append((tr0.copy(),te.copy()))
            canonical_folds[(policy,sid)]=base_folds
    # Feature construction reuses the existing, immutable source-cadence builder.
    axis=load_axis_builder()
    pilot_parent=r._load_modules()[0]
    axis_sessions,axis_names=axis.build_axis_features(cache["sessions"],pilot_parent)
    if len(axis_names)!=60 or len(set(axis_names))!=60: raise ValueError("axis feature schema is not exactly 60 unique features")
    anchor_windows={s["session_key"]:window_members_by_anchor(s) for s in cache["sessions"]}
    fold_meta=[]; arrays={}
    class_support_ok=True; raw_overlap_ok=True; paired_fit_ok=True
    for policy in POLICIES:
        table=rows[policy]
        if len(table.y)!=(32885 if policy=="primary" else 61730):
            raise ValueError(f"unexpected frozen population size for {policy}: {len(table.y)}")
        # Assert the two feature matrices share the same ordered policy rows.
        x_imu,imu_cols=r.feature_matrix(cache["sessions"],table,"imu_only")
        x_axis=np.concatenate([axis_sessions[int(owner)]["axis_window"][int(local)][None,:]
                               for owner,local in zip(table.owners,table.local_rows)],axis=0)
        # Exact row-order dimensions; both matrices use one frozen PolicyRows table.
        if x_imu.shape!=(len(table.y),12) or x_axis.shape!=(len(table.y),60): raise ValueError(f"bad feature dimensions: {policy}")
        for sid in SPLITS:
            for fi,(train_base,test) in enumerate(canonical_folds[(policy,sid)]):
                train=train_base.copy()
                removed=[]
                if sid=="timeblock42":
                    test_raw={}
                    for idx in test:
                        g=str(table.groups[idx]); sr=int(table.source_rows[idx])
                        mem=anchor_windows[g].get(sr)
                        if mem is None: raise ValueError(f"missing raw-window membership {g}:{sr}")
                        test_raw.setdefault(g,set()).update(mem)
                    keep=[]
                    for idx in train:
                        g=str(table.groups[idx]); sr=int(table.source_rows[idx])
                        mem=anchor_windows[g].get(sr)
                        if mem is None: raise ValueError(f"missing candidate raw-window membership {g}:{sr}")
                        (removed if mem & test_raw.get(g,set()) else keep).append(int(idx))
                    train=np.asarray(keep,dtype=np.int64)
                    removed=np.asarray(removed,dtype=np.int64)
                    # Authoritative separation check across each session.
                    train_raw={}
                    for idx in train:
                        g=str(table.groups[idx]); train_raw.setdefault(g,set()).update(anchor_windows[g][int(table.source_rows[idx])])
                    if any(train_raw.get(g,set()) & raw for g,raw in test_raw.items()):
                        raw_overlap_ok=False; raise AssertionError(f"raw source-row windows overlap after purge: {policy}/fold{fi+1}/{g}")
                else:
                    removed=np.asarray([],dtype=np.int64)
                fit=r.cap_train_indices(train,table.y,fi,42,20000)
                if not np.array_equal(fit,r.cap_train_indices(train,table.y,fi,42,20000)): raise AssertionError("non-deterministic cap")
                def supports(ids): return {c:int(np.sum(table.y[ids]==c)) for c in r.CLASSES}
                counts={"prepurge_train":supports(train_base),"postpurge_train":supports(train),"fit":supports(fit),"test":supports(test)}
                if any(v==0 for side in counts.values() for v in side.values()): class_support_ok=False
                if len(fit)>20000: class_support_ok=False
                # Identical capped IDs are used by both representations and both estimators.
                if len(np.unique(fit))!=len(fit): raise AssertionError("duplicate fit rows")
                k=f"{policy}__{sid}__fold{fi+1}"
                arrays[k+"__train_original"]=train_base
                arrays[k+"__train_after_window_purge"]=train
                arrays[k+"__extra_removed"]=removed
                arrays[k+"__fit"]=fit
                arrays[k+"__test"]=test
                fold_meta.append({"policy":policy,"split":sid,"fold":fi+1,"fit_n":len(fit),"test_n":len(test),"extra_window_overlap_removed":len(removed),"class_support":counts,"train_original_sha256":hash_obj(train_base.tolist()),"train_postpurge_sha256":hash_obj(train.tolist()),"extra_removed_sha256":hash_obj(removed.tolist()),"fit_sha256":hash_obj(fit.tolist()),"test_sha256":hash_obj(test.tolist()),"feature_sets_share_fit_rows":True,"estimators_share_fit_rows":True})
    if not class_support_ok: raise ValueError("one or more cells lack class support after cap/purge")
    if not raw_overlap_ok: raise ValueError("raw input row overlap remains")
    # Freeze exact builder and model configurations without fitting.
    rf=r._make_estimator("RF100").get_params(deep=False)
    hgb=r._make_estimator("HistGB").get_params(deep=False)
    for model,params in (("RF100",rf),("HistGB",hgb)):
        if model=="RF100" and not (params["n_estimators"]==100 and params["max_depth"]==20 and params["min_samples_leaf"]==5 and params["n_jobs"]==4): raise ValueError("RF100 params drift")
        if model=="HistGB" and not (params["learning_rate"]==0.1 and params["max_iter"]==100 and params["max_leaf_nodes"]==31 and params["min_samples_leaf"]==20 and params["l2_regularization"]==1.0 and params["early_stopping"] is False): raise ValueError("HistGB params drift")
    axis_manifest=AXIS_RUN/"manifest.json"
    if not axis_manifest.is_file(): raise FileNotFoundError(axis_manifest)
    axis_files=[axis_manifest,AXIS_RUN/"results.json",AXIS_RUN/"primary_group.axis_window.json",AXIS_RUN/"blank_normal_group.axis_window.json"]
    for policy in POLICIES:
        for fold in range(1,6): axis_files += [AXIS_RUN/f"{policy}_group.axis_window.fold{fold}.checkpoint.json",AXIS_RUN/f"{policy}_group.axis_window.fold{fold}.joblib"]
    if any(not p.is_file() for p in axis_files): raise FileNotFoundError("historical axis-window gate artifacts incomplete")
    payload={"status":"preflight_complete_fit_free","model_fit_or_scores":False,"planned_fit_folds":80,"policies":list(POLICIES),"splits":list(SPLITS),"features":{"imu_only":12,"axis_window":len(axis_names)},"models":{"RF100":rf,"HistGB":hgb},"runtime":rt,"cache_sha256":sha(CACHE),"parent_runner_sha256":sha(PILOT/"run_pilot.py"),"shared_runner_sha256":sha(RUNNER),"key_protocol_sha256":sha(PROTOCOL),"T_protocol_sha256":sha(ADDENDUM),"driver_sha256":sha(Path(__file__).resolve()),"cache_source_count":len(cache["source_sha256"]),"source_sha256":cache["source_sha256"],"frozen_map_sha256":sha(PILOT/"session_fold_map_v1.json"),"timeblock_support_sha256":sha(PILOT/"timeblock_split_support_v3.json"),"identity_certificate_sha256":sha(IDENTITY),"axis_window_reference_sha256":{str(p.relative_to(AXIS_RUN)):sha(p) for p in axis_files},"feature_schema":axis_names,"feature_schema_sha256":hash_obj(axis_names),"window":{"seconds":1.0,"interval":"[t-1,t]","boundary":"inclusive left, stable source-row order at timestamp ties","raw_window_overlap_purged":True},"cap":{"rows":20000,"sampling_seed":"42000 + zero-based fold","same_fit_rows_across_features_models":True},"class_order_for_metrics":list(r.CLASSES),"fit_free_checks":{"frozen_source_and_cache_integrity":True,"historical_row_identity_certificate":True,"existing_split_fit_test_identities_match":True,"all_test_raw_window_overlaps_removed":raw_overlap_ok,"all_fold_class_support":class_support_ok,"estimators_match_locked_parameters":True},"planned_cells":[{"policy":p,"split":s,"feature":f,"model":m,"seed":42,"folds":5} for p in POLICIES for s in SPLITS for f in FEATURES for m in MODELS],"folds":fold_meta}
    payload["planned_cell_count"]=len(payload["planned_cells"])
    if payload["planned_cell_count"]!=16: raise AssertionError("T cell matrix does not have 16 config-fold groups")
    # Create only after all validation succeeds. Re-running is allowed only for byte-equivalent content.
    PREFLIGHT.mkdir(parents=True,exist_ok=True)
    npz=PREFLIGHT/"split_membership_v1.npz"
    tmp=npz.with_suffix(".tmp.npz"); np.savez_compressed(tmp,**arrays)
    npz_hash=sha(tmp)
    payload["split_membership_sha256"]=npz_hash
    manifest=PREFLIGHT/"preflight_manifest.json"
    encoded=(json.dumps(payload,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    if manifest.exists() or npz.exists():
        if not (manifest.exists() and npz.exists() and json.loads(manifest.read_text(encoding="utf-8"))==payload and sha(npz)==npz_hash):
            tmp.unlink(missing_ok=True); raise FileExistsError("preflight output exists with different contents; preserve and create a new version")
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(npz); manifest.write_bytes(encoded)
    print(json.dumps({"status":payload["status"],"model_fit_or_scores":False,"planned_fit_folds":payload["planned_fit_folds"],"population":{p:len(rows[p].y) for p in POLICIES},"fold_count":len(fold_meta),"extra_window_overlap_removed":sum(x["extra_window_overlap_removed"] for x in fold_meta),"class_support_ok":class_support_ok,"preflight_dir":str(PREFLIGHT),"membership_sha256":npz_hash},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
