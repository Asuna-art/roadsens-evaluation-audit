"""Independent aggregation of auxiliary B/C/D/E/R outputs.

This reads frozen run summaries and writes one compact JSON/Markdown report;
it never fits, changes splits, or re-encodes labels without consulting class_order.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RUNS=HERE/"runs"
PHASES={
    "R":("consistent_runtime_R_v1",32),
    "B":("phase_B_v1",48),
    "C":("phase_C_v2",48),
    "E":("phase_E_v1",12),
    "D":("phase_D_v1",18),
}
CLASSES=("Normal","Pothole","Bump")

def _read_aux(phase,run_id):
    root=RUNS/run_id
    manifest=json.loads((root/"run_manifest.json").read_text(encoding="utf-8"))
    methods=[]
    for p in sorted((root/"summaries").rglob("*.json")):
        x=json.loads(p.read_text(encoding="utf-8")); folds=[f for f in x.get("folds",[]) if "macro_f1" in f]
        if x.get("status")!="complete" or len(folds)!=5: status="partial_or_failed"
        else: status="complete"
        mean=lambda k: float(np.mean([f[k] for f in folds])) if folds and all(k in f for f in folds) else None
        pooled=x.get("pooled_oof")
        methods.append({"phase":phase,"policy":x.get("policy"),"split":x.get("split"),"feature":x.get("feature"),"model":x.get("model"),"status":status,
            "class_order":x.get("class_order"),"fold_mean_macro_f1":mean("macro_f1"),"fold_mean_balanced_accuracy":mean("balanced_accuracy"),
            "fold_mean_class_recall":{c:mean_recall(folds,c) for c in (x.get("class_order") or [])},
            "fold_macro_f1":[float(f["macro_f1"]) for f in folds],"pooled_oof":pooled,"source_summary":str(p.relative_to(root).as_posix())})
    failpath=root/"failures.json"
    failures=json.loads(failpath.read_text(encoding="utf-8")) if failpath.exists() else []
    return {"phase":phase,"run_id":run_id,"manifest_status":manifest.get("status"),"completed_methods":manifest.get("completed_methods"),"expected_methods":PHASES[phase][1],"failure_count":manifest.get("failure_count"),"failures":failures,"methods":methods}

def mean_recall(folds,cls):
    vals=[f.get("class_recall",{}).get(cls) for f in folds]
    vals=[float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

def _a_metric(policy,split,feature):
    p=RUNS/"main_A_v2"/"predictions"/policy/split/f"{feature}__RF100.json"
    d=json.loads(p.read_text(encoding="utf-8")); return d

def _gate_summary():
    """Recompute the original Track-A confirmation rule from same-runtime predictions."""
    out={}
    for policy in ("primary","blank_normal"):
        seeds={}
        for seed in (42,43,44):
            if seed==42:
                rr=_a_metric(policy,"random42","full_numeric"); gg=_a_metric(policy,"record42","full_numeric")
                cr=_a_metric(policy,"random42","context_only"); mr=_a_metric(policy,"random42","missing_only")
            else:
                run=RUNS/"consistent_runtime_R_v1"/"summaries"/policy
                rr=json.loads((run/f"random{seed}"/f"full_numeric__RF100.json").read_text(encoding="utf-8"))
                gg=json.loads((run/f"record{seed}"/f"full_numeric__RF100.json").read_text(encoding="utf-8"))
                cr=json.loads((run/f"random{seed}"/f"context_only__RF100.json").read_text(encoding="utf-8"))
                mr=json.loads((run/f"random{seed}"/f"missing_only__RF100.json").read_text(encoding="utf-8"))
            rvals=[float(x["macro_f1"]) for x in rr["folds"]]; gvals=[float(x["macro_f1"]) for x in gg["folds"]]
            delta=float(np.mean(rvals)-np.mean(gvals)); positive=int(sum(a>b for a,b in zip(rvals,gvals)))
            context=float(np.mean([x["macro_f1"] for x in cr["folds"]])); missing=float(np.mean([x["macro_f1"] for x in mr["folds"]]))
            gate=delta>=.10 and positive>=4 and max(context,missing)>=.65
            seeds[str(seed)]={"mean_random_minus_record_full_macro_f1":delta,"positive_fold_differences":positive,"context_random_fold_mean_macro_f1":context,"missing_random_fold_mean_macro_f1":missing,"seed_gate":bool(gate)}
        pass_count=sum(x["seed_gate"] for x in seeds.values())
        positive_all=all(x["mean_random_minus_record_full_macro_f1"]>0 for x in seeds.values())
        out[policy]={"seeds":seeds,"seeds_passing":pass_count,"at_least_two_of_three_pass":pass_count>=2,"all_three_mean_differences_positive":positive_all,"confirmation_rule_met":pass_count>=2 and positive_all}
    return {"rule_source":"candidate_roadsens/confirmation_plan.md; original Track A thresholds: >=0.10 full random-record mean difference, >=4/5 positive fold differences, and random context or missing macro-F1 >=0.65; at least 2/3 seeds pass in each policy and all three mean differences positive.","runtime_note":"Seed42 metrics from same-runtime main_A_v2; seeds43/44 from consistent_runtime_R_v1. Historic sklearn1.5.2 model predictions are excluded from same-runtime comparison.","policies":out}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",action="append",choices=list(PHASES)); ap.add_argument("--partial",action="store_true"); args=ap.parse_args()
    selected=args.phase or list(PHASES)
    phases={k:_read_aux(k,PHASES[k][0]) for k in selected}
    incomplete=[]
    for name,val in phases.items():
        if val["manifest_status"] not in ("phase_complete","phase_complete_with_failures") or val["completed_methods"]!=val["expected_methods"]:
            incomplete.append(f"{name}: {val['manifest_status']} {val['completed_methods']}/{val['expected_methods']}")
    if incomplete and not args.partial: raise RuntimeError("incomplete phases: "+"; ".join(incomplete))
    report={"status":"partial" if args.partial or incomplete else "complete","phases":phases,"same_runtime_confirmation":_gate_summary() if "R" in phases else None,"aggregation_note":"Fold means summarize five test folds; pooled OOF metrics are retained separately from the run JSON. Class labels are interpreted only via each method's explicit class_order. These folds are not independent datasets.","incomplete":incomplete}
    suffix="_".join(selected)
    out=RUNS/(f"auxiliary_{suffix}_validation_v1.json" if report["status"]=="partial" else "auxiliary_summary_v1.json"); out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Auxiliary run summary","", "All numbers below come from completed locked-runtime runs. Fold means and pooled OOF scores are separate; class recall names follow each estimator’s stored class order.",""]
    lines.append("| Phase | Methods | Failure count | Status |")
    lines.append("|---|---:|---:|---|")
    for ph,x in phases.items(): lines.append(f"| {ph} | {x['completed_methods']}/{x['expected_methods']} | {x['failure_count']} | {x['manifest_status']} |")
    if report["same_runtime_confirmation"]:
        lines += ["","## Track-A same-runtime confirmation","",f"Primary: confirmation rule **{'met' if report['same_runtime_confirmation']['policies']['primary']['confirmation_rule_met'] else 'not met'}**.",f"Blank-as-Normal sensitivity: confirmation rule **{'met' if report['same_runtime_confirmation']['policies']['blank_normal']['confirmation_rule_met'] else 'not met'}**.","", "Seed42 uses main_A_v2; seeds43/44 use the locked-runtime RF recheck. Older 1.5.2 confirmation predictions remain historical and are excluded.",""]
    lines.append("## Method-level results")
    for phase,x in phases.items():
        lines += ["",f"### Phase {phase}","","| Policy | Split | Model | Feature | Fold-mean macro-F1 | Fold-mean balanced acc. | Pooled OOF macro-F1 | Class order |","|---|---|---|---|---:|---:|---:|---|"]
        for m in x["methods"]:
            pooled=m.get("pooled_oof") or {}; order=", ".join(m.get("class_order") or [])
            fmt=lambda v: f"{v:.4f}" if isinstance(v,(int,float)) else "NA"
            lines.append(f"| {m['policy']} | {m['split']} | {m['model']} | {m['feature']} | {fmt(m['fold_mean_macro_f1'])} | {fmt(m['fold_mean_balanced_accuracy'])} | {fmt(pooled.get('macro_f1'))} | {order} |")
    md=RUNS/(f"auxiliary_{suffix}_validation_v1.md" if report["status"]=="partial" else "auxiliary_summary_v1.md")
    md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out)
if __name__=="__main__": main()
