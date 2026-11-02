"""Frozen auxiliary RoadSens phases B-E. Import-only consumer of runner.py helpers.

No fitting occurs without --run, a phase-specific root-reviewed matrix, and a
matching fit-free preflight directory. This module never edits runner.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
sys.path.insert(0, str(HERE))
import runner as r

PHASE_METHODS = {
    "B": [(m, f, s, p, seed) for m in ("ExtraTrees100", "HistGB", "LogisticRegression")
          for f in ("full_numeric", "context_only") for s in ("random", "record")
          for p in r.POLICIES for seed in (43, 44)],
    "R": [("RF100", f, s, p, seed) for f in ("full_numeric", "missing_only", "context_only", "imu_only")
          for s in ("random", "record") for p in r.POLICIES for seed in (43, 44)],
    "C": [("RF100", f, s, p, 42) for f in ("gps_values", "weather_only", "elevation_only",
          "sensor_only", "imu_plus_gps", "imu_plus_weather", "gps_mask", "full_with_mask")
          for s in ("random42", "record42", "timeblock42") for p in r.POLICIES],
    "D": [(m, "full_numeric", s, p, 42) for m in ("RF100_uncapped", "RF300_capped", "RF300_uncapped")
          for s in ("random42", "record42", "timeblock42") for p in r.POLICIES],
    "E": [("RF100", f, s, "primary_binary", 42) for f in ("full_numeric", "missing_only", "context_only", "imu_only")
          for s in ("random42", "record42", "timeblock42")],
}
EXPECTED_SKLEARN = "1.9.0"
IDENTITY_AUDIT_PATH = HERE / "shared_identity_audit_v1.json"
IDENTITY_AUDIT_SHA = None


def _phase_protocol(phase):
    return HERE / ("rf_runtime_recheck_v1.md" if phase == "R" else "protocol_v1.md")


def _runtime():
    import sklearn
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(f"auxiliary execution requires scikit-learn {EXPECTED_SKLEARN}; got {sklearn.__version__} from {sys.executable}")
    return {"executable":sys.executable,"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"sklearn":sklearn.__version__}


def _load():
    global IDENTITY_AUDIT_SHA
    cache, frozen, qa = r.validate_immutable_inputs()
    specs, rows = r.build_all_splits(cache["sessions"], frozen, qa)
    runtime=_runtime()
    map_paths={"seed42":r.PILOT/"session_fold_map_v1.json",
        "seed43":r.PILOT/"runs"/"confirmation_seed43_v1"/"session_fold_map.json",
        "seed44":r.PILOT/"runs"/"confirmation_seed44_v1"/"session_fold_map.json"}
    checkpoint_paths=[]
    for policy in r.POLICIES:
      for feature in ("full_numeric","missing_only","context_only","imu_only"):
        for split,folder,name in (("random42","pilot_v2","random"),("record42","pilot_v2","group"),
                                  ("timeblock42","timeblock_v1","timeblock")):
          checkpoint_paths.append(r.PILOT/"runs"/folder/f"{policy}_{name}.{feature}.checkpoint.json")
        for seed in (43,44):
          folder=f"confirmation_seed{seed}_v1"
          checkpoint_paths.extend([r.PILOT/"runs"/folder/f"{policy}_{name}.{feature}.checkpoint.json" for name in ("random","group")])
    binding={"runtime":runtime,"cache_sha256":r.sha256(r.CACHE),
        "run_pilot_sha256":r.sha256(r.PILOT/"run_pilot.py"),
        "shared_runner_sha256":r.sha256(HERE/"runner.py"),
        "protocol_sha256":r.sha256(HERE/"protocol_v1.md"),
        "frozen_split_authority_sha256":r.sha256(HERE/"frozen_split_authority_v1.md"),
        "source_sha256":cache["source_sha256"],
        "frozen_maps_sha256":{k:r.sha256(v) for k,v in map_paths.items()},
        "historical_checkpoint_sha256":{str(p.relative_to(r.PILOT).as_posix()):r.sha256(p) for p in sorted(set(checkpoint_paths))}}
    derived={"ordered_row_identity_sha256":{p:r.object_hash([[str(g),int(sr),str(y),int(o),int(l)] for g,sr,y,o,l in zip(t.groups,t.source_rows,t.y,t.owners,t.local_rows)]) for p,t in rows.items()},
        "split_fit_test_sha256":{f"{p}/{sid}":[{"train":r.object_hash(tr.tolist()),"fit":r.object_hash(r.cap_train_indices(tr,rows[p].y,fi,val["spec"]["seed"],20000).tolist()),"test":r.object_hash(te.tolist())} for fi,(tr,te) in enumerate(val["folds"])] for (p,sid),val in sorted(specs.items())}}
    if IDENTITY_AUDIT_PATH.exists():
        audit=json.loads(IDENTITY_AUDIT_PATH.read_text(encoding="utf-8"))
        prior_binding={k:v for k,v in audit.get("binding",{}).items() if k!="auxiliary_driver_sha256"}
        if audit.get("status")!="shared_identity_audit_verified" or prior_binding!=binding or audit.get("derived_identity")!=derived:
            raise ValueError("shared identity audit is stale; refusing to reuse or silently replace it")
    else:
        # One full parse of the historical predictions binds exact row identities.
        exact={"seed42":r._compare_seed42_original(specs,rows),
               "existing_seed43_44":r._compare_existing_confirmation_rows(specs,rows)}
        audit={"status":"shared_identity_audit_verified","binding":binding,"derived_identity":derived,"exact_comparison_report":exact,
               "note":"Historical checkpoint fold lists were parsed once and matched exactly. Reuse requires unchanged locked runtime, shared runner, source/cache, frozen maps, every checkpoint hash, and current derived rows/splits/fit IDs."}
        IDENTITY_AUDIT_PATH.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    IDENTITY_AUDIT_SHA=r.sha256(IDENTITY_AUDIT_PATH)
    return cache, frozen, qa, specs, rows


def _cols(sessions, field):
    keys = {"numeric": "numeric_columns", "context": "context_columns", "imu": "imu_columns"}
    return list(sessions[0][keys[field]])


def _mat(sessions, rows, name):
    if name in ("full_numeric", "context_only", "imu_only", "missing_only"):
        return r.feature_matrix(sessions, rows, name)
    numeric, ncols = r.feature_matrix(sessions, rows, "full_numeric")
    missing, _ = r.feature_matrix(sessions, rows, "missing_only")
    context_cols = _cols(sessions, "context")
    imu_cols = _cols(sessions, "imu")
    if name == "gps_values": names = ["location_latitude", "location_longitude"]
    elif name == "weather_only": names = _feature_names(sessions, "weather_only")
    elif name == "elevation_only": names = [c for c in ncols if c == "GIS_Elevation" or "elevation" in c.lower()]
    elif name == "sensor_only": names = [c for c in ncols if c not in context_cols]
    elif name == "imu_plus_gps": names = imu_cols + ["location_latitude", "location_longitude"]
    elif name == "imu_plus_weather":
        weather = _feature_names(sessions, "weather_only")
        names = imu_cols + weather
    elif name == "gps_mask":
        idx = [ncols.index(c) for c in ("location_latitude", "location_longitude")]
        return missing[:, idx], [f"isfinite({ncols[i]})" for i in idx]
    elif name == "full_with_mask":
        return np.concatenate((numeric, missing), axis=1), ncols + [f"isfinite({c})" for c in ncols]
    else: raise ValueError(name)
    if not names or any(c not in ncols for c in names):
        raise ValueError(f"feature {name}: expected columns missing/empty: {names}")
    ix = [ncols.index(c) for c in names]
    return numeric[:, ix], names


def _feature_names(sessions, name):
    ncols = _cols(sessions, "numeric")
    if name == "weather_only":
        # Prefer exact schema names identified from source; fail if expected four are absent.
        candidates = [c for c in ncols if any(k in c.lower() for k in ("temperature", "humidity", "wind", "cloud"))]
        if len(candidates) != 4:
            raise ValueError(f"expected exact four weather columns, found {candidates}")
        return candidates
    return [c for c in ncols if c == "GIS_Elevation" or "elevation" in c.lower()]


def _split_specs(phase, specs):
    if phase in ("B","R"): return {k: v for k, v in specs.items() if k[1] in ("random43", "random44", "record43", "record44")}
    if phase in ("C", "D"): return {k: v for k, v in specs.items() if k[1] in ("random42", "record42", "timeblock42")}
    return {k: v for k, v in specs.items() if k[1] in ("random42", "record42", "timeblock42")}


def _binary_rows(cache, frozen, qa, specs, rows):
    base = rows["primary"]
    finite_gps = np.zeros(len(base.y), dtype=bool)
    lat, lon = _cols(cache["sessions"], "numeric").index("location_latitude"), _cols(cache["sessions"], "numeric").index("location_longitude")
    for owner, sess in enumerate(cache["sessions"]):
        ids = np.flatnonzero(base.owners == owner)
        num = np.asarray(sess["numeric"], dtype=float)
        finite_gps[ids] = np.isfinite(num[base.local_rows[ids], lat]) & np.isfinite(num[base.local_rows[ids], lon])
    keep = np.isin(base.y, ["Pothole", "Bump"]) & finite_gps
    from runner import PolicyRows, _random_splits, _record_splits
    outrows = PolicyRows(base.y[keep], base.groups[keep], base.source_rows[keep], base.times[keep], base.owners[keep], base.local_rows[keep])
    old_to_new = np.full(len(base.y), -1, dtype=np.int64); old_to_new[np.flatnonzero(keep)] = np.arange(np.sum(keep))
    result = {}
    # Rebuild random folds for the binary population. Recording/timeblock assignments
    # come from the frozen original seed-42 membership, then restrict through old->new.
    result[("primary_binary", "random42")] = {"spec": {"id":"random42","kind":"random","seed":42}, "folds": _random_splits(outrows, 42), "metadata": {"population":"explicit finite-GPS Bump/Pothole"}}
    for sid in ("record42", "timeblock42"):
        old = specs[("primary", sid)]["folds"]
        folds = []
        for tr, te in old:
            tr2 = old_to_new[tr]; te2 = old_to_new[te]
            folds.append((np.sort(tr2[tr2 >= 0]), np.sort(te2[te2 >= 0])))
        result[("primary_binary", sid)] = {"spec": specs[("primary", sid)]["spec"], "folds": folds,
            "metadata": {"population":"explicit finite-GPS Bump/Pothole; original assignment restricted"}}
    for sid, spec in result.items():
        for fold, (tr, te) in enumerate(spec["folds"], 1):
            if len(np.unique(outrows.y[tr])) != 2 or len(np.unique(outrows.y[te])) != 2:
                raise ValueError(f"E infeasible class support; stop without reseed: {sid[1]} fold {fold}")
            if np.intersect1d(tr, te).size: raise AssertionError("E train/test overlap")
    return outrows, result, {"retained": int(keep.sum()), "removed": int((~keep).sum()),
        "retained_by_class": {c: int(np.sum(outrows.y == c)) for c in ("Pothole", "Bump")},
        "rows_original_primary": int(len(base.y)), "finite_gps_explicit_anomaly_rows": int(keep.sum())}


def _save_preflight(out, phase, cache, specs, rows, binary_audit=None):
    runtime=_runtime()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out / "row_indices").mkdir(); (out / "split_indices").mkdir()
    row_tables = {"primary_binary": rows} if phase == "E" else rows
    catalog=[]
    for policy, t in row_tables.items():
        ids = np.asarray([r.CLASS_TO_ID[str(y)] for y in t.y], dtype=np.int8) if phase != "E" else np.asarray([int(y == "Pothole") for y in t.y], dtype=np.int8)
        np.savez_compressed(out / "row_indices" / f"{policy}.npz", y=ids, y_text=t.y.astype(str), group=t.groups.astype(str), source_row=t.source_rows, time=t.times, owner=t.owners, local_row=t.local_rows)
    for (policy, sid), val in sorted(specs.items()):
        arrays={}
        for i,(tr,te) in enumerate(val["folds"]):
            arrays[f"train_{i}"]=tr; arrays[f"test_{i}"]=te
            seeds = (43, 44) if phase in ("B","R") else (42,)
            for seed in seeds:
                arrays[f"fit_seed{seed}_{i}"]=r.cap_train_indices(tr, row_tables[policy].y, i, seed, 20000)
            if phase == "D": arrays[f"fit_uncapped_{i}"]=tr.copy()
        path=out / "split_indices" / f"{policy}__{sid}.npz"; np.savez_compressed(path, **arrays)
        catalog.append({"policy":policy,"split_id":sid,**val["spec"],"index_file":path.relative_to(out).as_posix(),"index_file_sha256":r.sha256(path),"metadata":val["metadata"]})
    manifest={"status":"preflight_complete_fit_free","phase":phase,"model_fit_or_scores":False,"cache_sha256":r.sha256(r.CACHE),"parent_runner_sha256":r.sha256(r.PILOT/"run_pilot.py"),"shared_runner_sha256":r.sha256(HERE/"runner.py"),"auxiliary_driver_sha256":r.sha256(Path(__file__).resolve()),"shared_identity_audit_path":str(IDENTITY_AUDIT_PATH),"shared_identity_audit_sha256":IDENTITY_AUDIT_SHA,"protocol_sha256":r.sha256(HERE/"protocol_v1.md"),"phase_protocol_path":_phase_protocol(phase).name,"phase_protocol_sha256":r.sha256(_phase_protocol(phase)),"frozen_split_authority_sha256":r.sha256(HERE/"frozen_split_authority_v1.md"),"runtime_lock_sha256":r.sha256(HERE/"runtime_lock_v1.md"),"content_amendment_sha256":r.sha256(HERE/"content_overlap_amendment_v1.md"),"runtime":runtime,"policy_row_files":{p:r.sha256(out/"row_indices"/f"{p}.npz") for p in row_tables},"split_catalog":catalog,"binary_population_audit":binary_audit}
    (out/"preflight_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest


def qa(phase):
    runtime=_runtime()
    cache,frozen,q,specs,rows=_load(); selected=_split_specs(phase,specs)
    audit=None
    if phase=="E":
        binary,selected,audit=_binary_rows(cache,frozen,q,specs,rows); tables={"primary_binary":binary}
    else: tables=rows
    for (p,s),v in selected.items():
        table=tables[p]
        for fold,(tr,te) in enumerate(v["folds"],1):
            if np.intersect1d(tr,te).size: raise AssertionError(f"overlap {p}/{s}/{fold}")
            if phase=="E":
                if len(np.unique(table.y[tr]))!=2 or len(np.unique(table.y[te]))!=2: raise ValueError("E binary class support failure")
            elif np.any([not np.all(np.isin(table.y[ix],r.CLASSES)) for ix in (tr,te)]): raise ValueError("invalid labels")
    # Validate feature definitions and widths without fitting any estimator.
    features={"B":["full_numeric","context_only"],"R":["full_numeric","missing_only","context_only","imu_only"],"C":[x[1] for x in PHASE_METHODS["C"]],"D":["full_numeric"],"E":["full_numeric","missing_only","context_only","imu_only"]}[phase]
    feature_report={}
    for f in sorted(set(features)):
        x,names=_mat(cache["sessions"],tables.get("primary", tables.get("primary_binary")),f)
        feature_report[f]={"shape":list(x.shape),"columns":names}
    return {"phase":phase,"passed":True,"splits":len(selected),"methods":len(PHASE_METHODS[phase]),"folds":len(PHASE_METHODS[phase])*5,"model_fit_or_scores":False,"runtime":runtime,"shared_identity_audit_sha256":IDENTITY_AUDIT_SHA,"feature_sets":feature_report,"binary_population_audit":audit}


def _estimator(name,seed):
    from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier,RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    if name in ("RF100","RF100_uncapped"): return RandomForestClassifier(n_estimators=100,max_depth=20,min_samples_leaf=5,max_features="sqrt",class_weight="balanced",random_state=seed,n_jobs=4)
    if name=="RF300_capped": return RandomForestClassifier(n_estimators=300,max_depth=25,min_samples_leaf=1,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=4)
    if name=="RF300_uncapped": return RandomForestClassifier(n_estimators=300,max_depth=25,min_samples_leaf=1,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=4)
    if name=="ExtraTrees100": return ExtraTreesClassifier(n_estimators=100,max_depth=20,min_samples_leaf=5,max_features="sqrt",class_weight="balanced",bootstrap=False,random_state=seed,n_jobs=4)
    if name=="HistGB":
        import inspect
        from sklearn.ensemble import HistGradientBoostingClassifier
        kw=dict(learning_rate=.1,max_iter=100,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=seed)
        if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters: kw["class_weight"]="balanced"
        return HistGradientBoostingClassifier(**kw)
    if name=="LogisticRegression": return LogisticRegression(C=1.,solver="lbfgs",max_iter=2000,class_weight="balanced",tol=1e-4,random_state=seed)
    raise ValueError(name)


def _preprocess_feature(feature, sessions, table, fit, test, x):
    """Train-only preprocessing, including the underlying-value rule for masks."""
    if feature in ("missing_only", "gps_mask"):
        numeric, ncols = r.feature_matrix(sessions, table, "full_numeric")
        if feature == "missing_only": underlying = numeric
        else:
            ix = [ncols.index("location_latitude"), ncols.index("location_longitude")]
            underlying = numeric[:, ix]
        keep = np.any(np.isfinite(underlying[fit]), axis=0)
        kept = np.flatnonzero(keep).tolist()
        xtr, xte = x[fit][:, keep], x[test][:, keep]
        if xtr.shape[1] == 0:
            xtr, xte, kept = np.zeros((len(fit), 1)), np.zeros((len(test), 1)), [0]
        return xtr, xte, kept, [], {"kind":"pilot_missing_mask_underlying_train_availability"}
    return r.preprocess_train_test(x, fit, test, feature)


def _metrics(ytrue, ypred, classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    labels=np.arange(len(classes))
    return {"n":int(len(ytrue)),"accuracy":float(accuracy_score(ytrue,ypred)),
        "balanced_accuracy":float(balanced_accuracy_score(ytrue,ypred)),
        "macro_f1":float(f1_score(ytrue,ypred,labels=labels,average="macro",zero_division=0)),
        "class_order":list(classes),
        "class_support":{c:int(np.sum(ytrue==i)) for i,c in enumerate(classes)},
        "class_recall":{c:float(v) for c,v in zip(classes,recall_score(ytrue,ypred,labels=labels,average=None,zero_division=0))},
        "confusion":confusion_matrix(ytrue,ypred,labels=labels).tolist()}


def _plan(phase,cache,specs,rows,binary_rows=None,binary_specs=None):
    methods=PHASE_METHODS[phase]; result=[]
    for model,feature,split,policy,seed in methods:
        sid=split if split.endswith(("42","43","44")) else f"{split}{seed}"
        if phase in ("B","R"): sid=f"{split}{seed}"
        if phase=="E": sid=split
        key=(policy,sid)
        source_specs=binary_specs if phase=="E" else specs
        if key not in source_specs: raise KeyError(f"missing frozen split {key}")
        table=binary_rows if phase=="E" else rows[policy]
        x,columns=_mat(cache["sessions"],table,feature)
        cap=20000
        if phase=="D" and model.endswith("uncapped"): cap=None
        result.append((model,feature,sid,policy,seed,table,x,columns,source_specs[key],cap))
    return result


def run(phase,run_id,preflight_dir,matrix_path):
    runtime=_runtime()  # Fail closed before any estimator construction or fit.
    matrix=json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("root_reviewed") is not True or matrix.get("frozen") is not True or matrix.get("phase")!=phase:
        raise ValueError("matrix must be root-reviewed, frozen, and phase-matched")
    if matrix.get("expected_model_folds") != len(PHASE_METHODS[phase])*5:
        raise ValueError("matrix planned fold count does not match the frozen driver job list")
    cache,frozen,q,specs,rows=_load(); selected=_split_specs(phase,specs)
    audit=None; binary_rows=binary_specs=None
    tables=rows
    if phase=="E": binary_rows,binary_specs,audit=_binary_rows(cache,frozen,q,specs,rows); tables={"primary_binary":binary_rows}; selected=binary_specs
    pre=json.loads((preflight_dir/"preflight_manifest.json").read_text(encoding="utf-8"))
    if pre.get("status")!="preflight_complete_fit_free" or pre.get("phase")!=phase or pre.get("cache_sha256")!=r.sha256(r.CACHE) or pre.get("auxiliary_driver_sha256")!=r.sha256(Path(__file__).resolve()) or pre.get("shared_runner_sha256")!=r.sha256(HERE/"runner.py") or pre.get("phase_protocol_sha256")!=r.sha256(_phase_protocol(phase)) or pre.get("shared_identity_audit_sha256")!=r.sha256(IDENTITY_AUDIT_PATH) or pre.get("runtime",{}).get("sklearn")!=runtime["sklearn"] or pre.get("runtime",{}).get("executable")!=runtime["executable"]: raise ValueError("preflight hash/runtime/identity status mismatch")
    catalog={(x["policy"],x["split_id"]):x for x in pre.get("split_catalog",[])}
    row_tables={"primary_binary":binary_rows} if phase=="E" else rows
    for policy,table in row_tables.items():
        rowpath=preflight_dir/"row_indices"/f"{policy}.npz"
        expected_hash=pre.get("policy_row_files",{}).get(policy)
        if not expected_hash or r.sha256(rowpath)!=expected_hash: raise ValueError(f"preflight row identity hash mismatch: {policy}")
        with np.load(rowpath) as z:
            if not (np.array_equal(z["y_text"].astype(str),table.y.astype(str)) and np.array_equal(z["group"].astype(str),table.groups.astype(str)) and np.array_equal(z["source_row"],table.source_rows) and np.array_equal(z["time"],table.times) and np.array_equal(z["owner"],table.owners) and np.array_equal(z["local_row"],table.local_rows)):
                raise ValueError(f"preflight row identity differs from live rows: {policy}")
    seeds=(43,44) if phase in ("B","R") else (42,)
    for (policy,sid),sv in selected.items():
        item=catalog.get((policy,sid))
        if item is None: raise ValueError(f"preflight misses split {policy}/{sid}")
        ixpath=preflight_dir/item["index_file"]
        if r.sha256(ixpath)!=item["index_file_sha256"]: raise ValueError(f"preflight split index hash mismatch: {policy}/{sid}")
        with np.load(ixpath) as z:
            for fold,(tr,te) in enumerate(sv["folds"]):
                if not np.array_equal(z[f"train_{fold}"],tr) or not np.array_equal(z[f"test_{fold}"],te): raise ValueError(f"preflight split membership changed: {policy}/{sid}/fold{fold+1}")
                for seed in seeds:
                    expected=r.cap_train_indices(tr,row_tables[policy].y,fold,seed,20000)
                    if not np.array_equal(z[f"fit_seed{seed}_{fold}"],expected): raise ValueError(f"preflight fit membership changed: {policy}/{sid}/seed{seed}/fold{fold+1}")
                if phase=="D" and not np.array_equal(z[f"fit_uncapped_{fold}"],tr): raise ValueError(f"uncapped membership changed: {sid}/fold{fold+1}")
    out=RUNS/run_id
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out/"models").mkdir(); (out/"predictions").mkdir(); (out/"summaries").mkdir()
    (out/"preflight_manifest.json").write_bytes((preflight_dir/"preflight_manifest.json").read_bytes())
    cfg={"phase":phase,"status":"running","fit_started":False,"matrix":matrix,"matrix_sha256":r.sha256(matrix_path),"protocol_sha256":r.sha256(HERE/"protocol_v1.md"),"phase_protocol_path":_phase_protocol(phase).name,"phase_protocol_sha256":r.sha256(_phase_protocol(phase)),"driver_sha256":r.sha256(Path(__file__).resolve()),"parent_runner_sha256":r.sha256(r.PILOT/"run_pilot.py"),"shared_runner_sha256":r.sha256(HERE/"runner.py"),"shared_identity_audit_sha256":IDENTITY_AUDIT_SHA,"frozen_split_authority_sha256":r.sha256(HERE/"frozen_split_authority_v1.md"),"runtime_lock_sha256":r.sha256(HERE/"runtime_lock_v1.md"),"content_amendment_sha256":r.sha256(HERE/"content_overlap_amendment_v1.md"),"cache_sha256":r.sha256(r.CACHE),"preflight_sha256":r.sha256(preflight_dir/"preflight_manifest.json"),"binary_population_audit":audit,"runtime":runtime,"planned_methods":[{"model":m,"feature":f,"split":s,"policy":p,"seed":seed} for m,f,s,p,seed in PHASE_METHODS[phase]],"planned_folds":len(PHASE_METHODS[phase])*5,"completed_methods":0,"failure_count":0}
    (out/"run_manifest.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    failures=[]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
      for ji,(model,feature,sid,policy,seed,table,x,columns,split,cap) in enumerate(_plan(phase,cache,specs,rows,binary_rows,binary_specs),1):
        preds=[]; truths=[]; testids=[]; fold_results=[]
        for fi,(tr0,te) in enumerate(split["folds"]):
          fit=tr0.copy() if cap is None else r.cap_train_indices(tr0,table.y,fi,seed,cap)
          xtr,xte,kept,medians,prep=_preprocess_feature(feature,cache["sessions"],table,fit,te,x)
          if model=="LogisticRegression":
            from sklearn.preprocessing import StandardScaler
            scaler=StandardScaler(); xtr=scaler.fit_transform(xtr); xte=scaler.transform(xte)
            prep={**prep,"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()}
          started=time.perf_counter(); fit_seconds=None
          try:
            est=_estimator(model,seed)
            fit_params={}
            if model=="HistGB" and "class_weight" not in est.get_params():
              from sklearn.utils.class_weight import compute_sample_weight
              fit_params["sample_weight"]=compute_sample_weight("balanced",table.y[fit])
            fit_seconds=time.perf_counter()-started; fit_started=time.perf_counter()
            cfg["fit_started"]=True
            (out/"run_manifest.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
              warnings.simplefilter("always"); est.fit(xtr,table.y[fit],**fit_params)
              fit_seconds=time.perf_counter()-fit_started
              predict_started=time.perf_counter(); pred=est.predict(xte); predict_seconds=time.perf_counter()-predict_started
            classes=[str(c) for c in est.classes_]; class_id={c:i for i,c in enumerate(classes)}
            ytrue=np.asarray([class_id[str(v)] for v in table.y[te]],dtype=np.int8); ypred=np.asarray([class_id[str(v)] for v in pred],dtype=np.int8)
            metric_classes=tuple(classes)
            old_classes=r.CLASSES
            # Phase E is binary: score only the two observed classes.
            met=_metrics(ytrue,ypred,classes)
            artifact=out/"models"/policy/sid/feature/f"{model}.fold{fi+1}.joblib"; artifact.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump({"estimator":est,"feature_columns":columns,"kept_columns":kept,"medians":medians,"preprocessing":prep,"classes":classes},artifact,compress=3)
            n_iter=getattr(est,"n_iter_",None)
            if hasattr(n_iter,"tolist"): n_iter=n_iter.tolist()
            elif n_iter is not None: n_iter=int(n_iter)
            fold_results.append({"fold":fi+1,**met,"fit_n":int(len(fit)),"train_before_cap_n":int(len(tr0)),"fit_rows_sha256":r.object_hash(fit.tolist()),"test_rows_sha256":r.object_hash(te.tolist()),"model_artifact":artifact.relative_to(out).as_posix(),"model_sha256":r.sha256(artifact),"estimator_params":est.get_params(deep=False),"preprocess_sha256":r.object_hash({"kept":kept,"medians":medians,"preprocess":prep}),"warnings":[{"category":w.category.__name__,"message":str(w.message)} for w in caught],"fit_wall_seconds":fit_seconds,"predict_wall_seconds":predict_seconds,"n_iter":n_iter})
            preds.append(ypred); truths.append(ytrue); testids.append(te.copy())
          except Exception as e:
            fail={"phase":phase,"policy":policy,"split":sid,"feature":feature,"model":model,"fold":fi+1,"type":type(e).__name__,"error":str(e)}; failures.append(fail); fold_results.append({"fold":fi+1,"status":"failed",**fail})
        mdir=out/"summaries"/policy/sid; mdir.mkdir(parents=True,exist_ok=True)
        if preds:
          rid=np.concatenate(testids); yt=np.concatenate(truths); yp=np.concatenate(preds)
          order=np.argsort(rid,kind="stable"); rid,yt,yp=rid[order],yt[order],yp[order]
          np.savez_compressed(out/"predictions"/f"{policy}__{sid}__{feature}__{model}.npz",test_row=rid,y_true=yt,y_pred=yp,class_order=np.asarray(classes,dtype=str))
          pooled=_metrics(yt,yp,classes)
        else: pooled=None
        summary={"phase":phase,"policy":policy,"split":sid,"feature":feature,"model":model,"status":"complete" if len(preds)==5 else "partial_failure","folds":fold_results,"class_order":classes if preds else None,"pooled_oof":pooled}
        (mdir/f"{feature}__{model}.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        cfg["fit_started"]=True; cfg["completed_methods"]=ji; cfg["failure_count"]=len(failures)
        (out/"run_manifest.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if failures:(out/"failures.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    cfg["status"]="phase_complete_with_failures" if failures else "phase_complete"
    (out/"run_manifest.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__); mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--qa",action="store_true"); mode.add_argument("--preflight",action="store_true"); mode.add_argument("--run",action="store_true")
    ap.add_argument("--phase",required=True,choices=list(PHASE_METHODS)); ap.add_argument("--run-id"); ap.add_argument("--preflight-dir",type=Path); ap.add_argument("--matrix",type=Path)
    a=ap.parse_args()
    if a.qa: print(json.dumps(qa(a.phase),ensure_ascii=False,indent=2)); return
    if not a.run_id: ap.error("--preflight and --run require --run-id")
    if a.preflight:
      _runtime(); cache,frozen,q,specs,rows=_load(); selected=_split_specs(a.phase,specs); audit=None
      if a.phase=="E": binary,selected,audit=_binary_rows(cache,frozen,q,specs,rows); rows=binary
      out=RUNS/a.run_id; print(_save_preflight(out,a.phase,cache,selected,rows,audit)["status"]); print(out); return
    if not a.matrix or not a.preflight_dir: ap.error("--run requires --matrix and --preflight-dir")
    print(run(a.phase,a.run_id,a.preflight_dir,a.matrix))

if __name__=="__main__": main()
