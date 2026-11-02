from __future__ import annotations
"""Root-reviewed fitting driver for T. Requires hash-bound review and gate authorization."""
import argparse, hashlib, importlib.util, json, sys, time, warnings
from pathlib import Path
import joblib, numpy as np
HERE=Path(__file__).resolve().parent; EXP=HERE.parent; PILOT=EXP.parent
OUT=HERE/"T_temporal_v3"; PRE=HERE/"T_temporal_v1"/"preflight_v1"; AXIS_RUN=HERE/"T_axis_reference_runtime19_v1"; HIST_AXIS_RUN=PILOT/"runs"/"axis_window_v1"; RUNTIME_AMENDMENT=HERE/"T_runtime_amendment_v1.md"
REVIEW=OUT/"review_T_v3.json"; CONTINUE=OUT/"continue_T_v3.json"
POLICIES=("primary","blank_normal"); SPLITS=("record42","timeblock42"); FEATURES=("imu_only","axis_window"); MODELS=("RF100","HistGB")

def sha(p):
 h=hashlib.sha256();
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def reference_artifacts():
 files=[AXIS_RUN/"manifest.json",AXIS_RUN/"results.json",AXIS_RUN/"primary_group.axis_window.json",AXIS_RUN/"blank_normal_group.axis_window.json"]
 for policy in POLICIES:
  for fold in range(1,6):
   stem=f"{policy}_group.axis_window.fold{fold}"
   files.extend([AXIS_RUN/f"{stem}.checkpoint.json",AXIS_RUN/f"{stem}.joblib"])
 if any(not p.is_file() for p in files): raise FileNotFoundError("same-runtime original axis reference is incomplete")
 result=json.loads((AXIS_RUN/"results.json").read_text(encoding="utf-8"))
 if result.get("method")!="axis_window" or result.get("formal_fit_started") is not True or any(len(result.get("policy_results",{}).get(p,[]))!=5 for p in POLICIES):
  raise ValueError("same-runtime original axis results are not complete")
 return {p.relative_to(AXIS_RUN).as_posix():sha(p) for p in files}
def load_axis_builder():
 src=PILOT/"run_axis_window_control.py"; spec=importlib.util.spec_from_file_location("roadsens_axis_builder_frozen_fit",src)
 if spec is None or spec.loader is None: raise RuntimeError("cannot load frozen axis-window builder")
 mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod
def get_review(pre_sha,driver_sha,axis_builder_sha,reference_sha,runtime_amendment_sha):
 if not REVIEW.is_file(): raise FileNotFoundError(f"root review file required: {REVIEW}")
 x=json.loads(REVIEW.read_text(encoding="utf-8"))
 if x.get("status")!="approved" or x.get("preflight_manifest_sha256")!=pre_sha or x.get("preflight_driver_sha256")!=driver_sha or x.get("fit_driver_sha256")!=sha(Path(__file__).resolve()) or x.get("T_protocol_sha256")!=sha(HERE/"T_temporal_protocol_v1.md") or x.get("axis_builder_sha256")!=axis_builder_sha or x.get("planned_fit_folds")!=80 or x.get("reference_artifact_sha256")!=reference_sha or x.get("runtime_amendment_sha256")!=runtime_amendment_sha:
  raise ValueError("review approval is absent or does not bind this exact preflight/code/matrix")
 return x
def load_data():
 import sklearn
 from threadpoolctl import threadpool_limits
 if Path(sys.executable).resolve()!=Path(r"C:\Users\Administrator\miniconda3\python.exe").resolve() or sklearn.__version__!="1.9.0": raise RuntimeError("locked Miniconda/sklearn runtime required")
 sys.path.insert(0,str(EXP)); import runner as r
 cache,frozen,qa=r.validate_immutable_inputs(); specs,rows=r.build_all_splits(cache["sessions"],frozen,qa)
 manifest=json.loads((PRE/"preflight_manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="preflight_complete_fit_free" or manifest.get("model_fit_or_scores") is not False: raise ValueError("fit-free preflight is not valid")
 live_hashes={
  "parent_runner_sha256":sha(r.PILOT/"run_pilot.py"),
  "frozen_map_sha256":sha(r.PILOT/"session_fold_map_v1.json"),
  "timeblock_support_sha256":sha(r.PILOT/"timeblock_split_support_v3.json"),
  "identity_certificate_sha256":sha(EXP/"shared_identity_audit_v1.json"),
  "key_protocol_sha256":sha(HERE/"protocol_key_v1.md"),
  "T_protocol_sha256":sha(HERE/"T_temporal_protocol_v1.md"),
 }
 for key,value in live_hashes.items():
  if manifest.get(key)!=value: raise ValueError(f"preflight binding changed: {key}")
 if manifest.get("cache_sha256")!=sha(r.CACHE) or manifest.get("shared_runner_sha256")!=sha(EXP/"runner.py") or manifest.get("driver_sha256")!=sha(HERE/"run_temporal_benchmark.py"): raise ValueError("preflight source/code hashes changed")
 axis_hashes=manifest.get("axis_window_reference_sha256",{})
 for relative,expected in axis_hashes.items():
  if sha(HIST_AXIS_RUN/relative)!=expected: raise ValueError(f"historical axis-window reference changed: {relative}")
 runtime_amendment_sha=sha(RUNTIME_AMENDMENT)
 reference_sha=reference_artifacts()
 reference_results=json.loads((AXIS_RUN/"results.json").read_text(encoding="utf-8"))
 if reference_results.get("session_cache_sha256")!=manifest.get("cache_sha256") or reference_results.get("run_pilot_sha256")!=manifest.get("parent_runner_sha256") or reference_results.get("session_fold_map_sha256")!=manifest.get("frozen_map_sha256"): raise ValueError("same-runtime reference source/split provenance mismatch")
 axis_builder_sha=sha(r.PILOT/"run_axis_window_control.py")
 membership=PRE/"split_membership_v1.npz"
 if sha(membership)!=manifest.get("split_membership_sha256"): raise ValueError("preflight split membership checksum mismatch")
 pre_sha=sha(PRE/"preflight_manifest.json"); get_review(pre_sha,manifest["driver_sha256"],axis_builder_sha,reference_sha,runtime_amendment_sha)
 maps=np.load(membership)
 axis=load_axis_builder(); parent=r._load_modules()[0]; built,names=axis.build_axis_features(cache["sessions"],parent)
 if names!=manifest.get("feature_schema"): raise ValueError("axis feature schema differs from reviewed preflight")
 matrices={}
 for policy in POLICIES:
  table=rows[policy]; xi,ci=r.feature_matrix(cache["sessions"],table,"imu_only")
  xa=np.concatenate([built[int(o)]["axis_window"][int(l)][None,:] for o,l in zip(table.owners,table.local_rows)],axis=0)
  if xi.shape!=(len(table.y),12) or xa.shape!=(len(table.y),60): raise ValueError("feature shape changed after review")
  matrices[(policy,"imu_only")]=(table,xi); matrices[(policy,"axis_window")]=(table,xa)
 return r,cache,rows,matrices,maps,manifest,pre_sha,reference_sha,runtime_amendment_sha
def metrics(y,pred,classes):
 from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,recall_score
 ids={c:i for i,c in enumerate(classes)}; yt=np.asarray([ids[str(v)] for v in y],dtype=np.int8); yp=np.asarray([ids[str(v)] for v in pred],dtype=np.int8); labels=np.arange(len(classes))
 return {"n":int(len(y)),"accuracy":float(accuracy_score(yt,yp)),"balanced_accuracy":float(balanced_accuracy_score(yt,yp)),"macro_f1":float(f1_score(yt,yp,labels=labels,average="macro",zero_division=0)),"class_order":list(classes),"class_support":{c:int(np.sum(yt==i)) for i,c in enumerate(classes)},"class_recall":{c:float(v) for c,v in zip(classes,recall_score(yt,yp,labels=labels,average=None,zero_division=0))},"confusion":confusion_matrix(yt,yp,labels=labels).tolist()}
def row_identity(table,ids): return [{"row":int(i),"session":str(table.groups[i]),"source_row":int(table.source_rows[i])} for i in ids]
def fit_one(r,table,x,fit,test,policy,sid,feature,model,fold):
 from threadpoolctl import threadpool_limits
 xxtr,xxte,kept,medians,prep=r.preprocess_train_test(x,fit,test,feature)
 est=r._make_estimator(model); fit_kw={}
 if model=="HistGB" and "class_weight" not in est.get_params():
  from sklearn.utils.class_weight import compute_sample_weight
  fit_kw["sample_weight"]=compute_sample_weight("balanced",table.y[fit])
 with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter("always"); t0=time.perf_counter(); est.fit(xxtr,table.y[fit],**fit_kw); fit_s=time.perf_counter()-t0; pred=est.predict(xxte); pred_s=time.perf_counter()-t0-fit_s
  warns=[{"category":w.category.__name__,"message":str(w.message)} for w in caught]
 pred=np.asarray([str(v) for v in pred],dtype=object); truth=np.asarray([str(v) for v in table.y[test]],dtype=object); met=metrics(truth,pred,r.CLASSES)
 out=OUT/"models"/policy/sid/feature; out.mkdir(parents=True,exist_ok=True); model_path=out/f"{model}.fold{fold}.joblib"; joblib.dump(est,model_path,compress=3)
 pred_dir=OUT/"predictions"/policy/sid; pred_dir.mkdir(parents=True,exist_ok=True); pred_path=pred_dir/f"{feature}__{model}.fold{fold}.npz"
 np.savez_compressed(pred_path,row=test.astype(np.int64),source_row=table.source_rows[test].astype(np.int64),session=table.groups[test].astype(str),y_true=truth.astype(str),y_pred=pred.astype(str),y_true_id=np.asarray([r.CLASS_TO_ID[y] for y in truth],dtype=np.int8),y_pred_id=np.asarray([r.CLASS_TO_ID[y] for y in pred],dtype=np.int8),class_order=np.asarray(r.CLASSES))
 return {"policy":policy,"split":sid,"feature":feature,"model":model,"fold":fold,"fit_n":len(fit),"test_n":len(test),"metrics":met,"fit_seconds":fit_s,"predict_seconds":pred_s,"warnings":warns,"kept_feature_indices":kept,"training_imputation_medians":medians,"preprocessing":prep,"fit_rows_sha256":r.object_hash(row_identity(table,fit)),"test_rows_sha256":r.object_hash(row_identity(table,test)),"model_path":model_path.relative_to(OUT).as_posix(),"model_sha256":sha(model_path),"prediction_path":pred_path.relative_to(OUT).as_posix(),"prediction_sha256":sha(pred_path)}
def compare_gate(r,table,record,feature,policy,fold):
 cp=json.loads((AXIS_RUN/f"{policy}_group.axis_window.fold{fold}.checkpoint.json").read_text(encoding="utf-8"))
 if row_identity(table,np.asarray([x["row"] for x in cp["fit_train_rows"]],dtype=np.int64))!=cp["fit_train_rows"]: raise ValueError("bad historical fit row identity")
 # Compare the stable identity fields; the historical test rows also store outcomes.
 hist_test=[{k:x[k] for k in ("row","session","source_row")} for x in cp["test_rows"]]
 if cp["fit_train_rows"]!=record["fit_rows"] or hist_test!=record["test_rows"]: raise ValueError(f"axis reproduction fold identity mismatch {policy}/{fold}")
 new=np.load(OUT/record["prediction_path"]); got=new["y_pred"].astype(str).tolist(); want=[str(x["pred"]) for x in cp["test_rows"]]
 if got!=want: raise ValueError(f"axis reproduction prediction mismatch {policy}/fold{fold}")
 return {"identity_match":True,"prediction_match":True,"same_runtime_reference_model_sha256":sha(AXIS_RUN/cp["model_artifact"]),"new_model_sha256":record["model_sha256"]}
def run(gate_only=False,remaining=False):
 r,cache,rows,matrices,maps,pre,pre_sha,reference_sha,runtime_amendment_sha=load_data(); mode="gate" if gate_only else "remaining"
 run_dir=OUT/"fit_gate_v1" if gate_only else OUT/"fit_remaining_v1"
 if gate_only and (OUT/"fit_gate_v1").exists(): raise FileExistsError(OUT/"fit_gate_v1")
 if remaining:
  gate_path=OUT/"fit_gate_v1"/"gate_manifest.json"
  if not gate_path.is_file(): raise FileNotFoundError("gate must be completed before remaining cells")
  gate=json.loads(gate_path.read_text(encoding="utf-8"))
  if gate.get("status")!="passed": raise ValueError("RF recording reproduction gate did not pass")
  if not CONTINUE.is_file(): raise FileNotFoundError("root continuation review required after inspected RF reproduction gate")
  cont=json.loads(CONTINUE.read_text(encoding="utf-8"))
  if cont.get("status")!="approved" or cont.get("gate_manifest_sha256")!=sha(gate_path) or cont.get("preflight_manifest_sha256")!=pre_sha: raise ValueError("continuation approval does not bind passing gate/preflight")
  if run_dir.exists(): raise FileExistsError(run_dir)
 run_dir.mkdir(parents=True,exist_ok=True); (run_dir/"models").mkdir(); (run_dir/"predictions").mkdir(); (run_dir/"folds").mkdir()
 planned=[]; failures=[]; gate_results=[]
 for policy in POLICIES:
  for sid in SPLITS:
   for feature in FEATURES:
    for model in MODELS:
     if gate_only and not (sid=="record42" and feature=="axis_window" and model=="RF100"): continue
     if remaining and (sid=="record42" and feature=="axis_window" and model=="RF100"): continue
     table,x=matrices[(policy,feature)]
     for fold in range(1,6):
      key=f"{policy}__{sid}__fold{fold}"; z=maps.files
      fit=maps[key+"__fit"].astype(np.int64); test=maps[key+"__test"].astype(np.int64)
      try:
       rec=fit_one(r,table,x,fit,test,policy,sid,feature,model,fold)
       rec["fit_rows"]=row_identity(table,fit); rec["test_rows"]=row_identity(table,test)
       if gate_only: rec["gate_comparison"]=compare_gate(r,table,rec,feature,policy,fold)
       (run_dir/"folds"/f"{policy}__{sid}__{feature}__{model}__fold{fold}.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
       planned.append({k:v for k,v in rec.items() if k not in ("fit_rows","test_rows")})
      except Exception as e:
       failures.append({"policy":policy,"split":sid,"feature":feature,"model":model,"fold":fold,"error":repr(e)})
       (run_dir/"failures.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
       if gate_only: break
 if gate_only:
  status="passed" if len(planned)==10 and not failures and all(x["gate_comparison"]["prediction_match"] for x in planned) else "failed"
 else: status="phase_complete" if len(planned)==70 and not failures else "partial_with_failures"
 manifest={"status":status,"phase":"T_temporal_v3","mode":mode,"fit_started":True,"completed_folds":len(planned),"planned_folds":10 if gate_only else 70,"failure_count":len(failures),"failures":failures,"preflight_manifest_sha256":pre_sha,"preflight_driver_sha256":pre["driver_sha256"],"fit_driver_sha256":sha(Path(__file__).resolve()),"review_sha256":sha(REVIEW),"runtime_amendment_sha256":runtime_amendment_sha,"reference_artifact_sha256":reference_sha,"historical_axis_reference_runtime_audit_sha256":sha(HERE/"T_temporal_v1"/"runtime_audit_axis_window_v1.json"),"runtime":pre["runtime"],"results":planned}
 path=run_dir/("gate_manifest.json" if gate_only else "run_manifest.json"); path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"status":status,"completed_folds":len(planned),"failures":len(failures),"manifest":str(path)},ensure_ascii=False,indent=2))

if __name__=="__main__":
 ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True); g.add_argument("--gate-only",action="store_true"); g.add_argument("--remaining",action="store_true"); a=ap.parse_args(); run(gate_only=a.gate_only,remaining=a.remaining)
