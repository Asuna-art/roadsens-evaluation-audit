"""Independent validation/report for the compact STRIDE 17-record supplement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib

import numpy as np

CLASSES = ("Bump", "Pothole")
FEATURES = ("sensor_norm_statistics10", "duration_only1")
MODELS = ("RF100_depth5_leaf2", "LogisticRegression_scaled")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    return {"n":int(len(yt)),"macro_f1":float(f1_score(yt,yp,labels=[0,1],average="macro",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(yt,yp)),
        "class_support":{c:int(np.sum(yt==i)) for i,c in enumerate(CLASSES)},
        "class_recall":{c:float(v) for i,(c,v) in enumerate(zip(CLASSES,recall_score(yt,yp,labels=[0,1],average=None,zero_division=0)))},
        "confusion":confusion_matrix(yt,yp,labels=[0,1]).tolist()}


def analyze(run_dir: Path) -> dict:
    run_dir=run_dir.resolve(); manifest=json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    errors=[]; job_results=[]; counted_folds=0
    if manifest.get("status")!="complete":errors.append("manifest is not complete")
    if manifest.get("expected_model_folds")!=60:errors.append("expected model-fold count is not 60")
    if manifest.get("failures"):errors.append("manifest contains fit failures")
    source=json.loads((run_dir/"source_rows.json").read_text(encoding="utf-8"))
    if len(source)!=17:errors.append("source row table does not contain 17 records")
    labels=np.asarray([0 if r["label"]=="Bump" else 1 for r in source],dtype=np.int8)
    record=np.asarray([r["record"] for r in source],dtype=str)
    if {c:int(np.sum(labels==i)) for i,c in enumerate(CLASSES)}!={"Bump":9,"Pothole":8}:
        errors.append("source class counts differ from frozen 9/8 composition")
    with np.load(run_dir/"features.npz",allow_pickle=False) as features:
        if features["sensor_norm_statistics10"].shape!=(17,10):errors.append("sensor feature table shape mismatch")
        if features["duration_only1"].shape!=(17,1):errors.append("duration feature table shape mismatch")
        if not np.array_equal(features["y"],labels):errors.append("feature y rows do not match source labels")
        if not np.array_equal(features["record"],record):errors.append("feature record order differs")
        if not np.isfinite(features["sensor_norm_statistics10"]).all() or not np.isfinite(features["duration_only1"]).all():errors.append("feature table has non-finite values")
    maps={}
    for seed in SEEDS:
        with np.load(run_dir/"splits"/f"seed{seed}.npz",allow_pickle=False) as d:
            folds=[(d[f"train_{i}"].copy(),d[f"test_{i}"].copy()) for i in range(5)]
        seen=[]
        for f,(tr,te) in enumerate(folds):
            if set(tr)&set(te):errors.append(f"train/test overlap seed{seed}/fold{f+1}")
            if any(not np.any(labels[tr]==i) or not np.any(labels[te]==i) for i in (0,1)):errors.append(f"missing class seed{seed}/fold{f+1}")
            seen.extend(te.tolist())
        if sorted(seen)!=list(range(17)):errors.append(f"seed{seed} test fold sets do not partition records")
        maps[seed]=folds
    for feature in FEATURES:
      for model in MODELS:
       for seed in SEEDS:
        summary_path=run_dir/f"seed{seed}_{feature}_{model}.json"
        pred_path=run_dir/"predictions"/f"seed{seed}_{feature}_{model}.npz"
        if not summary_path.is_file() or not pred_path.is_file():errors.append(f"missing job output seed{seed}/{feature}/{model}");continue
        s=json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(pred_path,allow_pickle=False) as p:
            yt=p["y_true"].copy();yp=p["y_pred"].copy();pr=p["record"].copy()
        if s.get("status")!="complete" or len(s.get("folds",[]))!=5:errors.append(f"incomplete job seed{seed}/{feature}/{model}")
        if not np.array_equal(pr,record):errors.append(f"record OOF identity differs seed{seed}/{feature}/{model}")
        if not np.array_equal(yt,labels) or np.any((yp<0)|(yp>1)):errors.append(f"OOF labels invalid seed{seed}/{feature}/{model}")
        pool=metrics(yt,yp);stored=s.get("pooled_oof") or {}
        for key in ("n","macro_f1","balanced_accuracy","class_support","class_recall","confusion"):
            if isinstance(pool.get(key),float):
                if abs(pool[key]-float(stored.get(key,-99)))>1e-12:errors.append(f"pooled metric mismatch {seed}/{feature}/{model}/{key}")
            elif pool.get(key)!=stored.get(key):errors.append(f"pooled metric mismatch {seed}/{feature}/{model}/{key}")
        fold_f1=[]
        for f,(tr,te) in enumerate(maps[seed]):
            got=metrics(yt[te],yp[te]);fr=s["folds"][f];fold_f1.append(got["macro_f1"])
            if abs(float(fr.get("macro_f1",-99))-got["macro_f1"])>1e-12:errors.append(f"fold F1 mismatch {seed}/{feature}/{model}/fold{f+1}")
            if fr.get("class_support")!=got["class_support"]:errors.append(f"fold support mismatch {seed}/{feature}/{model}/fold{f+1}")
            art=run_dir/"models"/fr.get("artifact","")
            if not art.is_file() or sha256(art)!=fr.get("artifact_sha256"):errors.append(f"model artifact missing/hash mismatch {seed}/{feature}/{model}/fold{f+1}")
            if fr.get("test_record_ids")!=record[te].tolist():errors.append(f"fold record IDs mismatch {seed}/{feature}/{model}/fold{f+1}")
            if set(fr.get("train_record_ids",[]))&set(fr.get("test_record_ids",[])):errors.append(f"record leakage seed{seed}/{feature}/{model}/fold{f+1}")
            counted_folds+=1
        if abs(float(s.get("fold_mean_macro_f1",-99))-float(np.mean(fold_f1)))>1e-12:errors.append(f"fold mean mismatch {seed}/{feature}/{model}")
        job_results.append({"seed":seed,"feature":feature,"model":model,"fold_mean_macro_f1":float(np.mean(fold_f1)),"pooled_oof":pool,"warnings":sum(len(f.get("warnings",[])) for f in s["folds"])})
    status="complete_validated" if not errors and len(job_results)==12 and counted_folds==60 else "partial_or_invalid"
    result={"status":status,"interpretation":"small 17-record Bump-vs-Pothole recording-type feasibility only; no Normal labels, no window detection, no deployment claim",
        "expected_jobs":12,"present_jobs":len(job_results),"expected_model_folds":60,"validated_model_folds":counted_folds,
        "errors":errors,"archive_sha256":manifest.get("archive_sha256"),"protocol_sha256":manifest.get("protocol_sha256"),"jobs":job_results}
    (run_dir/"validation_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[f"# STRIDE supplement validation: {status}","", "17-record Bump-vs-Pothole recording-type task only. This is not RoadSens three-class validation, window detection, or deployment evidence.","",f"Validated {len(job_results)}/12 jobs and {counted_folds}/60 model-folds; errors: {len(errors)}.","",
        "| Seed | Feature set | Model | Fold-mean macro-F1 | Pooled OOF macro-F1 | Pooled BA |","|---:|---|---|---:|---:|---:|"]
    for j in job_results:lines.append(f"| {j['seed']} | {j['feature']} | {j['model']} | {j['fold_mean_macro_f1']:.4f} | {j['pooled_oof']['macro_f1']:.4f} | {j['pooled_oof']['balanced_accuracy']:.4f} |")
    if errors:lines.extend(["","## Validation errors",""]+[f"- {e}" for e in errors[:30]])
    (run_dir/"validation_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--run-dir",type=Path,required=True);args=ap.parse_args()
    r=analyze(args.run_dir)
    print(json.dumps({k:r[k] for k in ("status","present_jobs","expected_jobs","validated_model_folds","expected_model_folds","errors")},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
