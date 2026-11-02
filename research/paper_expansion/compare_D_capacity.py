"""Independent, fit-free comparison of Phase D against canonical Phase A RF100."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
POLICIES = ("primary", "blank_normal")
SPLITS = ("random42", "record42", "timeblock42")
MODELS = ("RF100_uncapped", "RF300_capped", "RF300_uncapped")

def load(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("status") != "complete" or len(d.get("folds", [])) != 5:
        raise RuntimeError(f"Incomplete summary: {path}")
    return d

rows = []
for policy in POLICIES:
    for split in SPLITS:
        a = load(RUNS / "main_A_v2" / "predictions" / policy / split / "full_numeric__RF100.json")
        a_mean = float(np.mean([x["macro_f1"] for x in a["folds"]]))
        a_pooled = float(a["pooled_oof"]["macro_f1"])
        for model in MODELS:
            d = load(RUNS / "phase_D_v1" / "summaries" / policy / split / f"full_numeric__{model}.json")
            mean = float(np.mean([x["macro_f1"] for x in d["folds"]]))
            pooled = float(d["pooled_oof"]["macro_f1"])
            rows.append({"policy": policy, "split": split, "model": model,
                         "D_fold_mean_macro_f1": mean, "D_pooled_oof_macro_f1": pooled,
                         "A_RF100_capped_fold_mean_macro_f1": a_mean,
                         "A_RF100_capped_pooled_oof_macro_f1": a_pooled,
                         "fold_mean_delta_vs_A": mean-a_mean, "pooled_delta_vs_A": pooled-a_pooled,
                         "class_order_D": d["class_order"], "class_order_A": ["Normal", "Pothole", "Bump"]})

manifest = json.loads((RUNS / "phase_D_v1" / "run_manifest.json").read_text(encoding="utf-8"))
if manifest.get("status") != "phase_complete" or manifest.get("completed_methods") != 18 or manifest.get("failure_count") != 0 or len(rows) != 18:
    raise RuntimeError(f"Phase D validation failed: {manifest.get('status')}, {manifest.get('completed_methods')}, {manifest.get('failure_count')}")
report = {"status": "validated", "phase_D_methods": len(rows), "phase_D_folds": 90,
          "phase_D_failures": 0, "baseline": "canonical main_A_v2 full_numeric RF100 (20,000-row cap), matching policy and split",
          "comparison_note": "D compares RF100 uncapped, RF300 capped at 20,000, and RF300 uncapped against A's RF100 capped baseline. Tree count and training cap vary together across some contrasts; these comparisons are descriptive and do not isolate a causal hyperparameter effect.",
          "rows": rows}
out_json = RUNS / "phase_D_capacity_vs_A_v1.json"
out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
lines = ["# Phase D capacity comparison against canonical Phase A", "",
         "All 90 Phase D folds completed with zero failures. Scores are macro-F1. Fold-mean and pooled OOF metrics are reported separately.",
         "The Phase A reference is the matching policy/split full-numeric RF100 model with its 20,000-row training cap.",
         "D varies tree count and, for uncapped configurations, training cap; these contrasts are descriptive and do not isolate causal parameter effects.", "",
         "| Policy | Split | Phase D model | D fold mean | D pooled OOF | A RF100 capped fold mean | A RF100 capped pooled OOF | Δ fold mean | Δ pooled |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['policy']} | {r['split']} | {r['model']} | {r['D_fold_mean_macro_f1']:.4f} | {r['D_pooled_oof_macro_f1']:.4f} | {r['A_RF100_capped_fold_mean_macro_f1']:.4f} | {r['A_RF100_capped_pooled_oof_macro_f1']:.4f} | {r['fold_mean_delta_vs_A']:+.4f} | {r['pooled_delta_vs_A']:+.4f} |")
out_md = RUNS / "phase_D_capacity_vs_A_v1.md"
out_md.write_text("\n".join(lines)+"\n", encoding="utf-8")
print(out_json)
print(out_md)
