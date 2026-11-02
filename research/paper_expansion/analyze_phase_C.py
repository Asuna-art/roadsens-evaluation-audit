"""Independent fit-free audit, paired component interval, and plot for phase C."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE/"runs"/"phase_C_v2"
PF=HERE/"runs"/"phase_C_preflight_v1"
MATRIX=HERE/"phase_C_matrix_v1.json"
CLASSES=("Normal","Pothole","Bump")
POLICIES=("primary","blank_normal")
SPLITS=("random42","record42","timeblock42")
FEATURES=("gps_values","weather_only","elevation_only","sensor_only","imu_plus_gps","imu_plus_weather","gps_mask","full_with_mask")
MODEL="RF100"
MAIN_A_BOOT=HERE/"cluster_uncertainty"/"main_A_cluster_uncertainty_v1"/"component_confusions_and_draws.npz"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def objhash(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def cm_from(y,pred,nc=3):
    cm=np.zeros((nc,nc),dtype=np.int64); np.add.at(cm,(np.asarray(y,dtype=np.int64),np.asarray(pred,dtype=np.int64)),1); return cm

def scores(cm):
    sup=cm.sum(axis=1); rec=np.divide(np.diag(cm),sup,out=np.zeros(nc:=len(sup),dtype=float),where=sup>0)
    den=2*np.diag(cm)+cm.sum(axis=0)-np.diag(cm)+sup-np.diag(cm)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(nc,dtype=float),where=den>0)
    return {"macro_f1":float(np.mean(f)),"balanced_accuracy":float(np.mean(rec)),
        "class_support":{CLASSES[i]:int(sup[i]) for i in range(3)},
        "class_recall":{CLASSES[i]:float(rec[i]) for i in range(3)},"confusion":cm.tolist()}

def macro_f1(cm):
    return scores(cm)["macro_f1"]

def main():
    import sklearn
    import sys
    expected=Path(r"C:\Users\Administrator\miniconda3\python.exe")
    if Path(sys.executable).resolve()!=expected.resolve() or sklearn.__version__!="1.9.0":
        raise RuntimeError("Use locked Miniconda Python / sklearn 1.9.0 for analysis")
    OUT=RUN
    manifest=json.loads((RUN/"run_manifest.json").read_text(encoding="utf-8"))
    pre=json.loads((RUN/"preflight_manifest.json").read_text(encoding="utf-8"))
    pfp=json.loads((PF/"preflight_manifest.json").read_text(encoding="utf-8"))
    matrix=json.loads(MATRIX.read_text(encoding="utf-8"))
    if manifest.get("status")!="phase_complete" or manifest.get("failure_count")!=0 or manifest.get("completed_methods")!=48:
        raise RuntimeError("phase_C_v2 is not complete: expected 48 jobs / 240 folds / zero failures")
    if manifest.get("planned_folds")!=240 or manifest.get("driver_sha256")!="27390ec1349faa5bd5432a11b1fcc987210ca33732d89e18865b51f0b2a9132a":
        raise RuntimeError("phase C manifest differs from approved driver/matrix plan")
    if matrix.get("expected_model_folds")!=240 or matrix.get("features")!=list(FEATURES):
        raise RuntimeError("phase C matrix feature/config mismatch")
    if pfp.get("status")!="preflight_complete_fit_free" or pfp.get("model_fit_or_scores") is not False:
        raise RuntimeError("C fit-free preflight is not valid")
    if manifest.get("preflight_sha256")!=sha(PF/"preflight_manifest.json"):
        raise RuntimeError("run was not launched from the frozen preflight")
    for k in ("cache_sha256","parent_runner_sha256","shared_runner_sha256","protocol_sha256","phase_protocol_sha256"):
        if manifest.get(k)!=pfp.get({"parent_runner_sha256":"parent_runner_sha256","shared_runner_sha256":"shared_runner_sha256"}.get(k,k)):
            raise RuntimeError(f"run/preflight provenance mismatch: {k}")
    if manifest.get("matrix_sha256")!=sha(MATRIX) or manifest.get("phase_protocol_sha256")!=sha(HERE/"protocol_v1.md"):
        raise RuntimeError("live matrix/protocol hash mismatch")
    if manifest.get("runtime",{}).get("sklearn")!="1.9.0" or manifest.get("runtime",{}).get("executable")!=str(expected):
        raise RuntimeError("phase C model runtime mismatch")
    row_data={}; label_to_id={c:i for i,c in enumerate(CLASSES)}
    for pol in POLICIES:
        rp=np.load(PF/"row_indices"/f"{pol}.npz",allow_pickle=False)
        row_data[pol]={"y":rp["y"].astype(np.int8),"group":rp["group"].astype(str),"source_row":rp["source_row"].astype(np.int64)}
        if len(set(zip(row_data[pol]["group"],row_data[pol]["source_row"])))!=len(rp["y"]):
            raise RuntimeError(f"{pol}: duplicate source row identity")
    methods=[]; checked_folds=0; warning_count=0
    pooled_conf={}
    for pol in POLICIES:
        n=len(row_data[pol]["y"])
        for split in SPLITS:
            split_path=PF/"split_indices"/f"{pol}__{split}.npz"
            if pre.get("split_catalog") is None:
                raise RuntimeError("preflight has no split catalog")
            cat=next((x for x in pre["split_catalog"] if x["policy"]==pol and x["split_id"]==split),None)
            if not cat or cat["index_file_sha256"]!=sha(PF/cat["index_file"]):
                raise RuntimeError(f"{pol}/{split}: split index SHA mismatch")
            idx=np.load(split_path,allow_pickle=False)
            for feat in FEATURES:
                sp=RUN/"summaries"/pol/split/f"{feat}__{MODEL}.json"
                pp=RUN/"predictions"/f"{pol}__{split}__{feat}__{MODEL}.npz"
                if not sp.is_file() or not pp.is_file(): raise FileNotFoundError(sp if not sp.is_file() else pp)
                sm=json.loads(sp.read_text(encoding="utf-8"))
                if (sm.get("status"),sm.get("phase"),sm.get("policy"),sm.get("split"),sm.get("feature"),sm.get("model"))!=("complete","C",pol,split,feat,MODEL):
                    raise RuntimeError(f"summary identity/status mismatch: {sp}")
                order=sm.get("class_order")
                if not order or set(order)!=set(CLASSES) or len(order)!=3: raise RuntimeError(f"bad class_order: {sp}")
                with np.load(pp,allow_pickle=False) as d:
                    test_rows=d["test_row"].astype(np.int64); yt=d["y_true"].astype(np.int8); yp=d["y_pred"].astype(np.int8); np_order=d["class_order"].astype(str).tolist()
                if np_order!=order: raise RuntimeError(f"npz/summary class order differs: {pp}")
                if not np.array_equal(np.sort(test_rows),np.arange(n)) or len(np.unique(test_rows))!=n:
                    raise RuntimeError(f"OOF rows differ from frozen complete test population: {pp}")
                id_to_label={i:c for i,c in enumerate(order)}
                summary_order_ix=[CLASSES.index(c) for c in order]
                yt_names=np.asarray([id_to_label[int(i)] for i in yt],dtype=object)
                yp_names=np.asarray([id_to_label[int(i)] for i in yp],dtype=object)
                expected_names=np.asarray([CLASSES[int(i)] for i in row_data[pol]["y"][test_rows]],dtype=object)
                if not np.array_equal(yt_names,expected_names): raise RuntimeError(f"OOF y_true mismatch against fixed row policy: {pp}")
                yti=np.asarray([label_to_id[str(s)] for s in yt_names],dtype=np.int8); ypi=np.asarray([label_to_id[str(s)] for s in yp_names],dtype=np.int8)
                fold_cms=[]
                if len(sm.get("folds",[]))!=5: raise RuntimeError(f"incomplete fold summary: {sp}")
                for f,fr in enumerate(sm["folds"]):
                    te=idx[f"test_{f}"].astype(np.int64); fit=idx[f"fit_seed42_{f}"].astype(np.int64); tr=idx[f"train_{f}"].astype(np.int64)
                    fold_mask=np.isin(test_rows,te)
                    if int(fold_mask.sum())!=len(te) or not np.array_equal(np.sort(test_rows[fold_mask]),np.sort(te)):
                        raise RuntimeError(f"fold {f+1} test membership differs: {pp}")
                    k=len(te); fold_y=yti[fold_mask]; fold_pred=ypi[fold_mask]
                    if objhash(fit.tolist())!=fr["fit_rows_sha256"] or objhash(te.tolist())!=fr["test_rows_sha256"]:
                        raise RuntimeError(f"frozen fit/test row IDs do not match summary: {sp}, fold {f+1}")
                    if not set(fit).issubset(set(tr)) or len(set(fit)&set(te)) or set(fit)-set(tr):
                        raise RuntimeError(f"fit/test disjointness failure: {sp}, fold {f+1}")
                    if split=="record42" and set(row_data[pol]["group"][fit]) & set(row_data[pol]["group"][te]):
                        raise RuntimeError(f"recording split leaks a recording: {sp}, fold {f+1}")
                    cm=cm_from(fold_y,fold_pred); met=scores(cm); fold_cms.append(cm)
                    cm_ordered=cm[np.ix_(summary_order_ix,summary_order_ix)]
                    if int(fr["n"])!=k or fr["class_order"]!=order or fr["confusion"]!=cm_ordered.tolist():
                        raise RuntimeError(f"fold support/confusion/class-order mismatch: {sp}, fold {f+1}")
                    for metric in ("macro_f1","balanced_accuracy"):
                        if abs(float(fr[metric])-met[metric])>1e-12: raise RuntimeError(f"fold {metric} mismatch: {sp}, fold {f+1}")
                    support={c: int(sum(yt_names[fold_mask]==c)) for c in CLASSES}
                    if {c:int(fr["class_support"][c]) for c in CLASSES}!=support: raise RuntimeError(f"fold class support mismatch: {sp}")
                    artifact=RUN/fr["model_artifact"]
                    if sha(artifact)!=fr["model_sha256"]: raise RuntimeError(f"model artifact SHA mismatch: {artifact}")
                    warning_count+=len(fr.get("warnings",[])); checked_folds+=1
                pooled=cm_from(yti,ypi); pooled_met=scores(pooled)
                pm=sm["pooled_oof"]
                pooled_ordered=pooled[np.ix_(summary_order_ix,summary_order_ix)]
                if pm["class_order"]!=order or pm["confusion"]!=pooled_ordered.tolist(): raise RuntimeError(f"pooled confusion/class order mismatch: {sp}")
                for metric in ("macro_f1","balanced_accuracy"):
                    if abs(float(pm[metric])-pooled_met[metric])>1e-12: raise RuntimeError(f"pooled {metric} mismatch: {sp}")
                for c in CLASSES:
                    if abs(float(pm["class_recall"][c])-pooled_met["class_recall"][c])>1e-12: raise RuntimeError(f"pooled recall mismatch: {sp}/{c}")
                methods.append({"policy":pol,"split":split,"feature":feat,"model":MODEL,"class_order":order,
                    "fold_mean_macro_f1":float(np.mean([scores(c)["macro_f1"] for c in fold_cms])),
                    "fold_mean_balanced_accuracy":float(np.mean([scores(c)["balanced_accuracy"] for c in fold_cms])),
                    "pooled_oof":pooled_met,"fit_count":5,
                    "estimator_warning_count":sum(len(f.get("warnings",[])) for f in sm["folds"])})
                pooled_conf[(pol,split,feat)]=pooled
    if checked_folds!=240 or len(methods)!=48: raise RuntimeError(f"incomplete independent check: {checked_folds} folds / {len(methods)} jobs")

    # Paired fixed-component bootstrap: phase-C sensor subset minus A full numeric reference.
    boot=np.load(MAIN_A_BOOT,allow_pickle=False)
    weights=boot["bootstrap_component_weights"].astype(np.int64)
    c_ids=boot["component_ids"].astype(str)
    c_cm=boot["component_confusions"]
    if weights.shape[0]!=2000 or c_cm.shape[:4]!=(2,3,2,4): raise RuntimeError("main-A shared bootstrap evidence has an unexpected layout")
    ci_sensor={}
    for pi,pol in enumerate(POLICIES):
        # Main-A tensor order: policy, split(record=1), feature(full=0), RF100=0, component, true, pred.
        base=c_cm[pi,1,0,0]
        if base.shape[0]!=len(c_ids): raise RuntimeError("main-A component tensor and ID count differ")
        # Rebuild component-aligned confusion matrices from C record-heldout sensor predictions.
        mem=np.load(HERE/"data_audit"/"content_component_membership_v4.npz",allow_pickle=False)
        comp_rows=mem["component_id"].astype(str); score=mem[f"scoreable_global_row_{pol}"].astype(np.int64)
        comp_order={cid:i for i,cid in enumerate(c_ids)}
        if set(np.unique(comp_rows))!=set(c_ids): raise RuntimeError("component IDs differ between A bootstrap and membership freeze")
        rowcomp=np.empty(len(score[score>=0]),dtype=np.int16)
        for cr in np.flatnonzero(score>=0): rowcomp[int(score[cr])]=comp_order[comp_rows[cr]]
        sensor=np.zeros_like(base)
        d=np.load(RUN/"predictions"/f"{pol}__record42__sensor_only__RF100.npz",allow_pickle=False)
        order=d["class_order"].astype(str).tolist(); decode={i:CLASSES.index(order[i]) for i in range(3)}
        rr=d["test_row"].astype(np.int64); yt=np.asarray([decode[int(i)] for i in d["y_true"]],dtype=np.int8); yp=np.asarray([decode[int(i)] for i in d["y_pred"]],dtype=np.int8)
        for row,t,p in zip(rr,yt,yp): sensor[rowcomp[row],t,p]+=1
        if not np.array_equal(sensor.sum(axis=0),pooled_conf[(pol,"record42","sensor_only")]):
            raise RuntimeError(f"{pol}: sensor component confusions do not reconstruct pooled OOF counts")
        diffs=[]; invalid=0
        for w in weights:
            s_cm=np.einsum("c,cij->ij",w,sensor); b_cm=np.einsum("c,cij->ij",w,base)
            s_f,b_f=macro_f1(s_cm),macro_f1(b_cm)
            if np.isfinite(s_f) and np.isfinite(b_f): diffs.append(s_f-b_f)
            else: invalid+=1
        point=scores(sensor.sum(axis=0))["macro_f1"]-scores(base.sum(axis=0))["macro_f1"]
        ci_sensor[pol]={"contrast":"phase_C sensor_only minus main_A full_numeric; recording-heldout pooled OOF",
            "point_pooled_oof_macro_f1_difference":float(point),
            "percentile_ci95":[float(x) for x in np.percentile(diffs,[2.5,97.5])],
            "valid_draws":len(diffs),"degenerate_draws_not_redrawn":invalid,
            "bootstrap_weights_reused_from_main_A":True}

    input_hashes={"run_manifest":sha(RUN/"run_manifest.json"),"preflight_manifest":sha(RUN/"preflight_manifest.json"),
        "matrix":sha(MATRIX),"protocol":sha(HERE/"protocol_v1.md"),"auxiliary_driver":sha(HERE/"run_auxiliary.py"),
        "shared_runner":sha(HERE/"runner.py"),"main_A_bootstrap_npz":sha(MAIN_A_BOOT),
        "component_freeze":sha(HERE/"data_audit"/"content_component_freeze_v4.json"),
        "component_membership":sha(HERE/"data_audit"/"content_component_membership_v4.npz"),
        "A_row_primary":sha(HERE/"runs"/"main_A_preflight_v2"/"row_indices"/"primary.npz"),
        "A_row_blank_normal":sha(HERE/"runs"/"main_A_preflight_v2"/"row_indices"/"blank_normal.npz")}
    report={"status":"complete_validated","phase":"C","run_id":"phase_C_v2","jobs_validated":48,"folds_validated":240,
        "failure_count":0,"warning_count":warning_count,"runtime":manifest["runtime"],
        "class_order_note":"Metrics were decoded using each prediction file's explicit class_order; comparisons and output use Normal, Pothole, Bump.",
        "methods":methods,"sensor_only_vs_full_recording_component_bootstrap":ci_sensor,
        "bootstrap_note":"Two-policy paired pooled-OOF conditional intervals use the same 2,000 frozen main_A component weights; no retraining, fold-level CI, or p-values.",
        "input_sha256":input_hashes}
    (RUN/"independent_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (RUN/"independent_validation.md").write_text(render_md(report),encoding="utf-8")
    make_plot(report)
    report["analysis_script_sha256"]=sha(Path(__file__))
    report["plot_sha256"]=sha(RUN/"phase_C_feature_ablation.png")
    report["validation_markdown_sha256"]=sha(RUN/"independent_validation.md")
    (RUN/"independent_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"jobs":len(methods),"folds":checked_folds,"warnings":warning_count,
        "sensor_minus_full":ci_sensor,"report":str(RUN/"independent_validation.md")},ensure_ascii=False,indent=2))

def render_md(r):
    lines=["# Phase C feature-ablation validation","",
        "Phase C is an exploratory feature ablation using the fixed seed-42 random-row, recording-heldout, and time-block splits. Fold means and pooled OOF metrics are listed separately.","",
        "| Policy | Split | Feature | Fold-mean macro-F1 | Pooled OOF macro-F1 | Fold-mean balanced accuracy | Pooled OOF balanced accuracy | Class order |","|---|---|---|---:|---:|---:|---:|---|"]
    for x in r["methods"]:
        p=x["pooled_oof"]; lines.append(f"| {x['policy']} | {x['split']} | {x['feature']} | {x['fold_mean_macro_f1']:.4f} | {p['macro_f1']:.4f} | {x['fold_mean_balanced_accuracy']:.4f} | {p['balanced_accuracy']:.4f} | {', '.join(x['class_order'])} |")
    lines += ["","## Paired component interval","","| Policy | Contrast | Pooled macro-F1 difference | 95% percentile interval | Valid draws |","|---|---|---:|---:|---:|"]
    for pol,v in r["sensor_only_vs_full_recording_component_bootstrap"].items():
        lines.append(f"| {pol} | sensor-only − main_A full_numeric | {v['point_pooled_oof_macro_f1_difference']:+.4f} | [{v['percentile_ci95'][0]:+.4f}, {v['percentile_ci95'][1]:+.4f}] | {v['valid_draws']}/2000 |")
    lines += ["","This paired interval compares the phase-C sensor-only model with main_A's full_numeric reference under the same recording-heldout seed-42 test population. It conditions on both fitted models and reuses the same 2,000 frozen content-component bootstrap draws. It is not model-training uncertainty, a causal feature ablation, or evidence of independent roads/routes. The feature sets are predictive associations under this dataset and preprocessing.","",
        "All 48 feature/policy/split jobs and 240 fold outputs were checked against frozen row identities, fit/test IDs, explicit class order, recomputed fold and pooled confusion/metrics, and serialized model hashes. No estimator warnings were observed.","",
        "Feature-ablation plot: `phase_C_feature_ablation.png`. Machine-readable results and hashes: `independent_validation.json`."]
    return "\n".join(lines)+"\n"

def make_plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats=list(FEATURES); fig,axes=plt.subplots(2,3,figsize=(17,9),sharey=True)
    xx=np.arange(len(feats)); width=.34
    for pi,pol in enumerate(POLICIES):
        for si,split in enumerate(SPLITS):
            ax=axes[pi,si]
            vals=[next(x for x in r["methods"] if x["policy"]==pol and x["split"]==split and x["feature"]==feat) for feat in feats]
            fold=[x["fold_mean_macro_f1"] for x in vals]; pooled=[x["pooled_oof"]["macro_f1"] for x in vals]
            ax.bar(xx-width/2,fold,width,label="Five-fold mean",color="#3977B8",alpha=.9)
            ax.bar(xx+width/2,pooled,width,label="Pooled OOF",color="#E28E2C",alpha=.9)
            ax.set_xticks(xx,feats,rotation=42,ha="right"); ax.set_ylim(0,1.04); ax.grid(axis="y",alpha=.22)
            ax.set_title(("Primary" if pi==0 else "Blank=Normal")+" · "+split.replace("42",""))
            if si==0: ax.set_ylabel("Macro-F1")
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle("Phase C: all eight frozen feature sets\nFold-mean and pooled out-of-fold macro-F1 are distinct estimands",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(RUN/"phase_C_feature_ablation.png",dpi=220,bbox_inches="tight"); plt.close(fig)

if __name__=="__main__": main()
