"""Descriptive summary only for the completed frozen time-block diagnostic."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "timeblock_v1"


def main():
    if json.loads((RUN / "complete.json").read_text(encoding="utf-8")).get("fit_count") != 40:
        raise RuntimeError("The diagnostic is incomplete")
    rows = []
    for policy, filename in (("primary", "primary_timeblock.json"),
                             ("blank_normal", "blank_normal_timeblock.json")):
        result = json.loads((RUN / filename).read_text(encoding="utf-8"))
        for method, folds in result["methods"].items():
            if len(folds) != 5:
                raise RuntimeError(f"Expected 5 folds for {policy}/{method}")
            rows.append({
                "policy": policy, "method": method, "fold_count": 5,
                "macro_f1_mean": statistics.mean(f["macro_f1"] for f in folds),
                "balanced_accuracy_mean": statistics.mean(f["balanced_accuracy"] for f in folds),
                "pothole_recall_mean": statistics.mean(f["recall_by_class"]["Pothole"] for f in folds),
                "macro_f1_by_fold": [f["macro_f1"] for f in folds],
                "balanced_accuracy_by_fold": [f["balanced_accuracy"] for f in folds],
                "pothole_recall_by_fold": [f["recall_by_class"]["Pothole"] for f in folds],
                "fit_seconds_mean": statistics.mean(f["fit_seconds"] for f in folds),
                "predict_seconds_mean": statistics.mean(f["predict_seconds"] for f in folds),
                "train_n_before_cap_by_fold": [f["train_n_before_cap"] for f in folds],
                "test_n_by_fold": [f["test_n"] for f in folds],
            })
    out_json = RUN / "descriptive_summary.json"
    if out_json.exists():
        raise FileExistsError(out_json)
    out_json.write_text(json.dumps({"interpretation": "descriptive only; no new pass/fail gate",
                                   "fit_count": 40, "results": rows}, indent=2), encoding="utf-8")
    lines = ["# RoadSens within-record time-block diagnostic", "",
             "This frozen 5-fold diagnostic places a time block from every CSV recording in the test fold, keeps other blocks from those same recordings for training, and removes candidate train rows within an inclusive 1-second timestamp embargo of that recording's test block. It is a within-record temporal interpolation diagnostic, not independent-record, trip, route, or external validation.", "",
             "All 40 fits completed (4 methods × 5 folds × 2 label policies). Every fold had all three classes in train and test under both policies. Model config, row-level predictions, model artifacts, split supports, code/data/protocol hashes, and per-fold timings are in `runs/timeblock_v1`; exact source-row preflight is in `timeblock_runner_validation_v1.json`. This is descriptive follow-up only; it introduces no new pass/fail threshold and does not change the original pilot gates.", "",
             "Macro-F1, balanced accuracy and Pothole recall are arithmetic means over the five fixed folds. Fold points are not independent recordings or independent experiments.", "",
             "| Label policy | Method | Macro-F1 mean (folds) | Balanced accuracy mean | Pothole recall mean | Mean fit / predict seconds |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        f1 = ", ".join(f"{x:.3f}" for x in r["macro_f1_by_fold"])
        lines.append(f"| {r['policy']} | {r['method']} | {r['macro_f1_mean']:.3f} ({f1}) | {r['balanced_accuracy_mean']:.3f} | {r['pothole_recall_mean']:.3f} | {r['fit_seconds_mean']:.2f} / {r['predict_seconds_mean']:.2f} |")
    lines += ["", "This split deliberately leaves other portions of the same source CSV in training. It cannot establish performance on unseen recordings, trips, routes, or locations. Primary labels exclude unlabeled anomaly rows; `blank_normal` treats them as Normal and remains a sensitivity interpretation.", ""]
    out_md = ROOT / "timeblock_diagnostic_results.md"
    if out_md.exists():
        raise FileExistsError(out_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md)
    for r in rows:
        print(r["policy"], r["method"], round(r["macro_f1_mean"], 4), round(r["balanced_accuracy_mean"], 4),
              [round(v, 4) for v in r["macro_f1_by_fold"]])


if __name__ == "__main__":
    main()
