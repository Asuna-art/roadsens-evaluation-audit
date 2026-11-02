from __future__ import annotations
"""Report-only aggregation for independently validated T v3 folds; performs no fits."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/"T_temporal_v3"
CLASSES=("Normal","Pothole","Bump")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def metrics_from_cm(cm):
 cm=np.asarray(cm,dtype=np.int64)
 support=cm.sum(axis=1)
 pred=cm.sum(axis=0)
 tp=np.diag(cm)
 rec=np.divide(tp,support,out=np.zeros(3,dtype=float),where=support>0)
 f1=np.divide(2*tp, support+pred, out=np.zeros(3,dtype=float),where=(support+pred)>0)
 return {"n":int(cm.sum()),"class_support":{c:int(v) for c,v in zip(CLASSES,support)},"class_recall":{c:float(v) for c,v in zip(CLASSES,rec)},"balanced_accuracy":float(rec.mean()),"macro_f1":float(f1.mean()),"confusion":cm.tolist()}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--validation",type=Path,default=OUT/"independent_validation_v1.json")
 ap.add_argument("--out-dir",type=Path,default=OUT)
 args=ap.parse_args()
 val=json.loads(args.validation.read_text(encoding="utf-8"))
 if val.get("status")!="passed" or val.get("expected_folds")!=80 or val.get("validated_folds")!=80 or val.get("error_count")!=0:
  raise ValueError("refusing to aggregate: independent validation is not passed for all 80 folds")
 records=val.get("records",[])
 keys=[(r["policy"],r["split"],r["feature"],r["model"],int(r["fold"])) for r in records]
 if len(keys)!=80 or len(set(keys))!=80: raise ValueError("expected exactly 80 unique fold records")
 grouped={}
 expected_tasks={(p,s,f,m) for p in ("primary","blank_normal") for s in ("record42","timeblock42") for f in ("imu_only","axis_window") for m in ("RF100","HistGB")}
 for rec in records: grouped.setdefault((rec["policy"],rec["split"],rec["feature"],rec["model"]),[]).append(rec)
 if set(grouped)!=expected_tasks or any(len(v)!=5 for v in grouped.values()): raise ValueError("each of 16 T configurations must have five validated folds")
 # Confirm the per-fold prediction files behind each analyzer record form an exactly-once OOF partition.
 oof_checks=[]
 for task, folds in grouped.items():
  p,sid,feature,model=task; test_keys=[]; total=0
  for rec in folds:
   run_dir=OUT/rec["fit_dir"]
   stem=f"{p}__{sid}__{feature}__{model}__fold{rec['fold']}.json"
   detail=json.loads((run_dir/"folds"/stem).read_text(encoding="utf-8"))
   npz=OUT/detail["prediction_path"]
   with np.load(npz,allow_pickle=False) as z:
    if len(z["row"])!=rec["test_n"] or tuple(z["class_order"].astype(str).tolist())!=CLASSES: raise ValueError(f"prediction artifact support/order mismatch: {stem}")
    if not np.array_equal(z["row"],np.asarray([x["row"] for x in detail["test_rows"]],dtype=np.int64)): raise ValueError(f"prediction rows mismatch: {stem}")
   test_keys.extend((x["session"],int(x["source_row"])) for x in detail["test_rows"])
   cm=np.asarray(rec["metrics"]["confusion"],dtype=np.int64)
   if cm.shape!=(3,3) or int(cm.sum())!=int(rec["test_n"]): raise ValueError(f"confusion support mismatch: {stem}")
   total+=int(rec["test_n"])
  if len(test_keys)!=len(set(test_keys)): raise ValueError(f"test recordings/rows repeated across folds for {task}")
  expected_n=32885 if p=="primary" else 61730
  if total!=expected_n: raise ValueError(f"unexpected OOF row total for {task}: {total} != {expected_n}")
  oof_checks.append({"policy":p,"split":sid,"feature":feature,"model":model,"folds":5,"oof_rows":total,"unique_recording_source_rows":len(set(test_keys))})
 # Build configuration summaries from independently recomputed fold metrics; pooled values are summed OOF confusion.
 summaries=[]
 for task in sorted(grouped):
  folds=sorted(grouped[task],key=lambda z:int(z["fold"]))
  cms=[np.asarray(x["metrics"]["confusion"],dtype=np.int64) for x in folds]
  pooled=metrics_from_cm(np.sum(cms,axis=0))
  mean_f1=float(np.mean([x["metrics"]["macro_f1"] for x in folds]))
  mean_ba=float(np.mean([x["metrics"]["balanced_accuracy"] for x in folds]))
  mean_recall={c:float(np.mean([x["metrics"]["class_recall"][c] for x in folds])) for c in CLASSES}
  summaries.append({"policy":task[0],"split":task[1],"feature":task[2],"model":task[3],"folds":5,"n_oof":pooled["n"],"fold_mean_macro_f1":mean_f1,"pooled_oof_macro_f1":pooled["macro_f1"],"fold_mean_balanced_accuracy":mean_ba,"pooled_oof_balanced_accuracy":pooled["balanced_accuracy"],"fold_mean_class_recall":mean_recall,"pooled_oof_class_recall":pooled["class_recall"],"pooled_oof_support":pooled["class_support"],"pooled_oof_confusion":pooled["confusion"]})
 output={"status":"complete_report_only_aggregate","fit_or_scores_computed":False,"validation_path":str(args.validation.resolve()),"validation_sha256":sha(args.validation),"validation_status":val["status"],"independently_validated_fold_count":len(records),"configuration_count":len(summaries),"aggregate_basis":"fold means from independent-validation metrics; pooled OOF metrics recomputed from summed 3-class confusion matrices","oof_partition_checks":oof_checks,"summaries":summaries}
 args.out_dir.mkdir(parents=True,exist_ok=True)
 (args.out_dir/"independent_aggregate_v1.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 def f(x):return f"{x:.4f}"
 lines=["# T v3 independent fold and pooled-OOF summary","","Status: **complete**. This is report-only aggregation of the 80-fold independent validation. It fits no models and computes no new predictions.","","Fold-mean and pooled OOF metrics are reported separately. The pooled values are calculated by summing each five-fold 3-class confusion matrix; they are not averages of fold scores.","","| Policy | Split | Feature | Model | OOF n | Mean fold macro-F1 | Pooled OOF macro-F1 | Mean fold BA | Pooled OOF BA | Pooled recall N / P / B |","|---|---|---|---|---:|---:|---:|---:|---:|---|"]
 for x in summaries:
  rc=x['pooled_oof_class_recall']; lines.append(f"| {x['policy']} | {x['split']} | {x['feature']} | {x['model']} | {x['n_oof']} | {f(x['fold_mean_macro_f1'])} | {f(x['pooled_oof_macro_f1'])} | {f(x['fold_mean_balanced_accuracy'])} | {f(x['pooled_oof_balanced_accuracy'])} | {f(rc['Normal'])} / {f(rc['Pothole'])} / {f(rc['Bump'])} |")
 lines += ["","All 16 configurations have five folds and an exactly-once OOF partition over the expected policy population. Rows are source recording plus source-row identity. Interpret these as within-dataset recording/time-block results, not independent-road estimates.","",f"Validation SHA-256: `{output['validation_sha256']}`. Detailed fold summaries and confusion matrices are in `independent_aggregate_v1.json`."]
 (args.out_dir/"independent_aggregate_v1.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print(json.dumps({"status":output["status"],"configurations":len(summaries),"folds":len(records),"out":str(args.out_dir)},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
