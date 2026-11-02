"""Independent fit-free validation and compact summary for sampling-stride S."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PILOT = EXP.parent
CACHE = PILOT / "session_feature_cache_v1.joblib"
CLASSES = ("Normal", "Pothole", "Bump")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_runtime():
    if Path(sys.executable).resolve() != Path(r"C:\Users\Administrator\miniconda3\python.exe").resolve():
        raise RuntimeError("Use locked Miniconda Python for validation")
    sys.path.insert(0, str(EXP))
    sys.path.insert(0, str(PILOT))
    import runner
    import run_sampling_sensitivity as sampling
    return runner, sampling


def validate(run_dir: Path):
    runner, sampling = load_runtime()
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    pre_dir = Path(run["preflight_dir"])
    pre_path = pre_dir / "preflight_manifest.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if run.get("status") != "phase_complete" or run.get("completed_fits") != 40 or run.get("failures"):
        raise ValueError("run is not a complete clean 40-fit run")
    if run.get("preflight_manifest_sha256") != sha(pre_path) or pre.get("driver_sha256") != run.get("driver_sha256"):
        raise ValueError("run/preflight/driver provenance mismatch")
    cache, frozen = sampling.read_locked_inputs()
    if run.get("cache_sha256") != sha(CACHE) or run.get("seed42_map_sha256") != sha(PILOT / "session_fold_map_v1.json"):
        raise ValueError("cache or split map hash changed")
    catalog = {(j["stride"], j["policy"]): j for j in pre["jobs"]}
    results = []
    for stride in (5, 20):
        for policy in sampling.POLICIES:
            table, folds = sampling.reconstruct(cache, frozen, stride, policy)
            jobmeta = catalog[(stride, policy)]
            row_path, split_path = pre_dir / jobmeta["row_file"], pre_dir / jobmeta["split_file"]
            if sha(row_path) != jobmeta["row_file_sha256"] or sha(split_path) != jobmeta["split_file_sha256"]:
                raise ValueError(f"preflight split/row hash mismatch: {stride}/{policy}")
            with np.load(row_path, allow_pickle=False) as z:
                if not np.array_equal(z["group"], table["group"]) or not np.array_equal(z["source_row"], table["source_row"]):
                    raise ValueError(f"source row identity mismatch: {stride}/{policy}")
                y0 = np.asarray([CLASSES[int(i)] for i in z["y"]], dtype=object)
                if not np.array_equal(y0, table["y"]):
                    raise ValueError(f"raw source labels mismatch: {stride}/{policy}")
            with np.load(split_path, allow_pickle=False) as z:
                for f, (tr, fit, te) in enumerate(folds):
                    if not np.array_equal(z[f"train_{f}"], tr) or not np.array_equal(z[f"fit_{f}"], fit) or not np.array_equal(z[f"test_{f}"], te):
                        raise ValueError(f"fold row arrays mismatch: {stride}/{policy}/fold{f+1}")
            colnames = cache["sessions"][0]["numeric_columns"]
            imu_indices = [colnames.index(c) for c in cache["sessions"][0]["imu_columns"]]
            for feature in sampling.FEATURES:
                job_id = f"stride{stride}__{policy}__{feature}__RF100"
                job_path = run_dir / "jobs" / f"{job_id}.json"
                payload = json.loads(job_path.read_text(encoding="utf-8"))
                pred_path = run_dir / "predictions" / f"{job_id}.npz"
                if payload["prediction_sha256"] != sha(pred_path):
                    raise ValueError(f"prediction hash mismatch: {job_id}")
                with np.load(pred_path, allow_pickle=False) as z:
                    p_rows, p_fold, y_true, y_pred = z["row"], z["fold"], z["y_true"], z["y_pred"]
                if len(np.unique(p_rows)) != len(p_rows) or len(p_rows) != len(table["y"]):
                    raise ValueError(f"OOF rows are not a unique full test partition: {job_id}")
                if not np.array_equal(y_true, np.asarray([CLASSES.index(str(table["y"][i])) for i in p_rows], dtype=np.int8)):
                    raise ValueError(f"OOF truth labels disagree with source rows: {job_id}")
                x = table["x_full"] if feature == "full_numeric" else table["x_full"][:, imu_indices]
                fold_summaries = []
                for f, (train, fit, test) in enumerate(folds):
                    mask = p_fold == f+1
                    if not np.array_equal(p_rows[mask], test):
                        raise ValueError(f"saved OOF fold membership/order mismatch: {job_id}/fold{f+1}")
                    model_rel = payload["folds"][f]["model_path"]
                    model_path = run_dir / model_rel
                    if sha(model_path) != payload["folds"][f]["model_sha256"]:
                        raise ValueError(f"model artifact hash mismatch: {job_id}/fold{f+1}")
                    model = joblib.load(model_path)
                    xtr, xte, kept, medians, prep = runner.preprocess_train_test(x, fit, test, feature)
                    decoded = model.predict(xte)
                    stored = np.asarray([CLASSES[int(i)] for i in y_pred[mask]], dtype=object)
                    if not np.array_equal(decoded, stored):
                        raise ValueError(f"reloaded model predictions do not match OOF artifact: {job_id}/fold{f+1}")
                    metrics = runner._metric_payload(table["y"][test], decoded)
                    old_metrics = payload["folds"][f]["metrics"]
                    for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
                        if abs(metrics[metric] - old_metrics[metric]) > 1e-12:
                            raise ValueError(f"fold metric mismatch: {job_id}/fold{f+1}/{metric}")
                    fold_summaries.append(metrics)
                pooled_y = np.asarray([CLASSES[int(i)] for i in y_true], dtype=object)
                pooled_p = np.asarray([CLASSES[int(i)] for i in y_pred], dtype=object)
                pooled = runner._metric_payload(pooled_y, pooled_p)
                results.append({"stride": stride, "policy": policy, "feature": feature, "model": "RF100",
                    "population_n": int(len(table["y"])), "recording_count": int(len(set(table["group"]))),
                    "class_counts": {c: int(np.sum(table["y"] == c)) for c in CLASSES},
                    "fold_mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_summaries])),
                    "fold_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in fold_summaries])),
                    "pooled_oof": pooled, "folds_validated": 5, "model_hashes_validated": 5,
                    "source_rows_validated": True, "stored_predictions_reproduced_from_models": True})
    # Sensitivity differences are descriptive because source rows differ by stride.
    contrasts = []
    lookup = {(r["stride"], r["policy"], r["feature"]): r for r in results}
    for policy in sampling.POLICIES:
        for feature in sampling.FEATURES:
            ref = lookup[(10, policy, feature)] if (10, policy, feature) in lookup else None
            # Stride-10 is not refit or emitted here; its canonical A value is loaded below.
            a_path = PILOT / "runs" / "pilot_v2" / f"{policy}_group.{feature}.checkpoint.json"
            old = json.loads(a_path.read_text(encoding="utf-8"))
            y_all, p_all = [], []
            for fold in old["folds"]:
                y_all.extend(x["true"] for x in fold["predictions"])
                p_all.extend(x["pred"] for x in fold["predictions"])
            old_oof = runner._metric_payload(np.asarray(y_all, dtype=object), np.asarray(p_all, dtype=object))
            for stride in (5, 20):
                current = lookup[(stride, policy, feature)]
                contrasts.append({"policy": policy, "feature": feature, "stride": stride, "reference_stride": 10,
                    "fold_mean_macro_f1_difference_vs_stride10": current["fold_mean_macro_f1"] - float(np.mean([f["macro_f1"] for f in old["folds"]])),
                    "pooled_oof_macro_f1_difference_vs_stride10": current["pooled_oof"]["macro_f1"] - old_oof["macro_f1"],
                    "interpretation": "descriptive difference across different sampled row populations; not a paired estimate or causal effect"})
    report = {"status": "complete_validated", "run_id": run["run_id"], "jobs_validated": len(results),
        "folds_validated": sum(r["folds_validated"] for r in results), "failures": run["failures"],
        "estimator_warning_count": sum(j["warning_count"] for j in
            [json.loads(p.read_text(encoding="utf-8")) for p in (run_dir / "jobs").glob("stride*.json")]),
        "recording_population": {"total": 101, "normal_only_sources": 5, "mixed_sources": 96},
        "analysis_script_sha256": sha(Path(__file__).resolve()),
        "methods": results, "stride10_reference_contrasts": contrasts,
        "limitations": ["stride-specific populations differ; contrasts are descriptive, not same-row paired", 
            "101 source CSVs are operational records, not verified independent trips/routes", 
            "stride density does not increase independent acquisition count"]}
    (run_dir / "independent_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Sampling-stride sensitivity: independent validation", "",
        "All reported cells are RF100, recording-heldout seed42. Fold means and pooled OOF are separate estimands.",
        "Stride-10 values are the existing A reference; only strides 5 and 20 were newly fitted.",
        "Between-stride differences are descriptive because row populations differ.", "",
        "| Stride | Policy | Feature | Rows | Recordings | Fold-mean macro-F1 | Pooled OOF macro-F1 | Fold-mean BA |", 
        "|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['stride']} | {r['policy']} | {r['feature']} | {r['population_n']} | {r['recording_count']} | {r['fold_mean_macro_f1']:.4f} | {r['pooled_oof']['macro_f1']:.4f} | {r['fold_mean_balanced_accuracy']:.4f} |")
    lines += ["", "## Differences from existing stride-10 reference", "", 
        "| Policy | Feature | Stride | Δ fold mean macro-F1 | Δ pooled OOF macro-F1 |", 
        "|---|---|---:|---:|---:|"]
    for r in contrasts:
        lines.append(f"| {r['policy']} | {r['feature']} | {r['stride']} | {r['fold_mean_macro_f1_difference_vs_stride10']:+.4f} | {r['pooled_oof_macro_f1_difference_vs_stride10']:+.4f} |")
    lines += ["", f"All 40 model-fold artifacts, source-row identities, test labels, stored predictions, reloaded model predictions, and model hashes were checked; estimator warnings: {report['estimator_warning_count']}. The population contains five Normal-only and 96 mixed source recordings. No causal interpretation is attached to cross-stride score differences."]
    (run_dir / "independent_validation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    report = validate(args.run_dir.resolve())
    print(json.dumps({"status": report["status"], "jobs": report["jobs_validated"], "folds": report["folds_validated"]}))


if __name__ == "__main__":
    main()
