"""Read-only validation and concise reporting for the phase-A expansion outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ("Normal", "Pothole", "Bump")
POLICIES = ("primary", "blank_normal")
SPLITS = ("random42", "record42", "timeblock42")
FEATURES = ("full_numeric", "missing_only", "context_only", "imu_only")
MODELS = ("RF100", "ExtraTrees100", "HistGB", "LogisticRegression")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    present = [i for i in range(3) if np.any(y_true == i)]
    return {"n": int(len(y_true)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(3)), average="macro", zero_division=0)),
            "observed_class_macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "class_support": {c: int(np.sum(y_true == i)) for i, c in enumerate(CLASSES)},
            "class_recall": {c: float(v) for c, v in zip(CLASSES, recall_score(y_true, y_pred, labels=list(range(3)), average=None, zero_division=0))},
            "confusion": confusion_matrix(y_true, y_pred, labels=list(range(3))).tolist()}


def analyze(run_dir: Path, preflight_dir: Path, output_dir: Path | None = None) -> dict:
    run_dir, preflight_dir = run_dir.resolve(), preflight_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    preflight_path = preflight_dir / "preflight_manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    errors, missing = [], []
    if run_manifest.get("phase") != "A": errors.append("run manifest is not phase A")
    if run_manifest.get("status") != "phase_complete": errors.append("run manifest is not marked phase_complete")
    if run_manifest.get("failure_count") != 0: errors.append("run manifest records nonzero fit failures")
    if run_manifest.get("runtime", {}).get("scikit_learn") != "1.9.0": errors.append("run used an unexpected sklearn runtime")
    if run_manifest.get("cache_sha256") != preflight.get("cache_sha256"): errors.append("cache hash differs from preflight")
    if run_manifest.get("preflight_manifest_sha256") != sha256(preflight_path): errors.append("preflight manifest hash differs")
    if run_manifest.get("driver_sha256") != preflight.get("driver_sha256", run_manifest.get("driver_sha256")):
        # The preflight's per-index metadata has no runner hash in legacy snapshots.
        errors.append("runner source hash mismatch")
    catalog = {(x["policy"], x["split_id"]): x for x in preflight["split_catalog"]}
    row_tables = {}
    for policy in POLICIES:
        path = preflight_dir / "row_indices" / f"{policy}.npz"
        if not path.is_file(): errors.append(f"missing row identity file: {policy}"); continue
        with np.load(path, allow_pickle=False) as a:
            row_tables[policy] = {k: a[k].copy() for k in ("y", "group", "source_row", "time", "owner", "local_row")}
    job_rows, total_folds, complete_folds = [], 0, 0
    rf_reproduction = {"folds_checked": 0, "prediction_mismatches": 0, "row_identity_mismatches": 0}
    for policy in POLICIES:
      if policy not in row_tables: continue
      identity = row_tables[policy]
      for split_id in SPLITS:
        if (policy, split_id) not in catalog:
            errors.append(f"missing preflight split catalog entry {policy}/{split_id}"); continue
        cat = catalog[(policy, split_id)]
        index_path = preflight_dir / cat["index_file"]
        if not index_path.is_file() or sha256(index_path) != cat["index_file_sha256"]:
            errors.append(f"split index file/hash invalid: {policy}/{split_id}"); continue
        with np.load(index_path, allow_pickle=False) as ix:
            splits = [(ix[f"train_{f}"].copy(), ix[f"test_{f}"].copy()) for f in range(5)]
        for fold, (tr, te) in enumerate(splits):
            fold_meta = cat["metadata"]["folds"][fold]
            for key, array in (("train_indices_sha256", tr), ("test_indices_sha256", te)):
                digest = hashlib.sha256(json.dumps(array.tolist(), separators=(",", ":")).encode()).hexdigest()
                # Runner uses canonical JSON SHA for split index identity.
                if digest != fold_meta[key]: errors.append(f"{key} differs from preflight split {policy}/{split_id}/fold{fold+1}")
        for feature in FEATURES:
          for model in MODELS:
            total_folds += 5
            stem = f"{feature}__{model}"
            job_dir = run_dir / "predictions" / policy / split_id
            summary_file, pred_file = job_dir / f"{stem}.json", job_dir / f"{stem}.npz"
            if not summary_file.is_file() or not pred_file.is_file():
                missing.append({"policy": policy, "split": split_id, "feature": feature, "model": model})
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            try:
                with np.load(pred_file, allow_pickle=False) as p:
                    row = p["test_row"].copy(); y_true = p["y_true"].copy(); y_pred = p["y_pred"].copy()
            except Exception as exc:
                errors.append(f"cannot load predictions {policy}/{split_id}/{stem}: {exc}"); continue
            expected_row = np.concatenate([te for _, te in splits])
            if not np.array_equal(row, expected_row): errors.append(f"OOF test-row order differs from frozen split: {policy}/{split_id}/{stem}")
            if split_id.startswith(("random", "record")) and (len(row) != len(identity["y"]) or len(np.unique(row)) != len(identity["y"])):
                errors.append(f"5-fold test sets do not partition the scoreable rows: {policy}/{split_id}/{stem}")
            if len(row) != len(y_true) or len(row) != len(y_pred): errors.append(f"OOF row/prediction length mismatch: {policy}/{split_id}/{stem}")
            if np.any(y_true != identity["y"][row]): errors.append(f"stored y_true differs from frozen row labels: {policy}/{split_id}/{stem}")
            if np.any((y_pred < 0) | (y_pred >= len(CLASSES))): errors.append(f"invalid predicted class id: {policy}/{split_id}/{stem}")
            if model == "RF100":
                if split_id == "timeblock42":
                    old_path = Path(__file__).resolve().parents[1] / "runs" / "timeblock_v1" / f"{policy}_timeblock.{feature}.checkpoint.json"
                else:
                    old_split = "random" if split_id == "random42" else "group"
                    old_path = Path(__file__).resolve().parents[1] / "runs" / "pilot_v2" / f"{policy}_{old_split}.{feature}.checkpoint.json"
                old_folds = json.loads(old_path.read_text(encoding="utf-8"))["folds"]
                cursor_old = 0
                for fold, ((tr, te), old_fold) in enumerate(zip(splits, old_folds)):
                    n = len(te)
                    expected_rows = [(int(i), str(identity["group"][i]), int(identity["source_row"][i])) for i in te]
                    old_rows = [(int(x["row"]), str(x["session"]), int(x["source_row"])) for x in old_fold["test_rows"]]
                    if old_rows != expected_rows:
                        rf_reproduction["row_identity_mismatches"] += 1
                        errors.append(f"historical RF test-row identity differs: {policy}/{split_id}/{feature}/fold{fold+1}")
                    old_pred = np.asarray([CLASSES.index(str(x["pred"])) for x in old_fold["predictions"]], dtype=np.int8)
                    new_pred = y_pred[cursor_old:cursor_old+n]
                    if len(old_pred) != n or not np.array_equal(old_pred, new_pred):
                        rf_reproduction["prediction_mismatches"] += 1
                        errors.append(f"historical RF predictions differ: {policy}/{split_id}/{feature}/fold{fold+1}")
                    rf_reproduction["folds_checked"] += 1
                    cursor_old += n
            recomputed = _scores(y_true, y_pred)
            pool_stored = summary.get("pooled_oof") or {}
            for metric in ("n", "macro_f1", "observed_class_macro_f1", "balanced_accuracy", "class_support", "class_recall", "confusion"):
                a, b = recomputed.get(metric), pool_stored.get(metric)
                if isinstance(a, float) and isinstance(b, (int, float)):
                    if abs(a-b) > 1e-12: errors.append(f"pooled metric differs: {policy}/{split_id}/{stem}/{metric}")
                elif a != b: errors.append(f"pooled metric differs: {policy}/{split_id}/{stem}/{metric}")
            folds = summary.get("folds", [])
            if len(folds) != 5 or summary.get("status") != "complete":
                errors.append(f"job not complete: {policy}/{split_id}/{stem}")
            computed_fold_f1 = []
            cursor = 0
            for fold, ((tr, te), fmeta) in enumerate(zip(splits, folds)):
                total_folds += 0
                if np.intersect1d(tr, te).size: errors.append(f"train/test row overlap {policy}/{split_id}/fold{fold+1}")
                ytest = identity["y"][te]
                if any(not np.any(ytest == i) for i in range(3)): errors.append(f"missing class in test {policy}/{split_id}/fold{fold+1}")
                ytrain = identity["y"][tr]
                if any(not np.any(ytrain == i) for i in range(3)): errors.append(f"missing class in train {policy}/{split_id}/fold{fold+1}")
                count = len(te)
                observed = _scores(y_true[cursor:cursor+count], y_pred[cursor:cursor+count])
                cursor += count
                if fmeta.get("status") == "failed" or "macro_f1" not in fmeta:
                    errors.append(f"fold failure {policy}/{split_id}/{stem}/fold{fold+1}"); continue
                computed_fold_f1.append(observed["macro_f1"])
                if fmeta.get("class_support") != observed["class_support"]:
                    errors.append(f"fold support differs {policy}/{split_id}/{stem}/fold{fold+1}")
                if abs(float(fmeta.get("macro_f1", -1))-observed["macro_f1"]) > 1e-12:
                    errors.append(f"fold macro-F1 differs {policy}/{split_id}/{stem}/fold{fold+1}")
                if split_id == "record42" and set(identity["group"][tr]) & set(identity["group"][te]):
                    errors.append(f"recording overlap {policy}/{stem}/fold{fold+1}")
                artifact_rel = fmeta.get("model_artifact")
                if not artifact_rel:
                    errors.append(f"missing fold model artifact reference {policy}/{split_id}/{stem}/fold{fold+1}")
                else:
                    artifact = run_dir / artifact_rel
                    if not artifact.is_file() or sha256(artifact) != fmeta.get("model_sha256"):
                        errors.append(f"model artifact missing/hash mismatch: {policy}/{split_id}/{stem}/fold{fold+1}")
                complete_folds += 1
            if computed_fold_f1 and abs(np.mean(computed_fold_f1)-summary.get("fold_mean_macro_f1", -1)) > 1e-12:
                errors.append(f"fold-mean macro-F1 differs: {policy}/{split_id}/{stem}")
            job_rows.append({"policy": policy, "split": split_id, "feature": feature, "model": model,
                "status": summary.get("status"), "pooled_oof": recomputed,
                "fold_mean_macro_f1": summary.get("fold_mean_macro_f1"), "scored_rows": len(row),
                "estimator_warning_count": sum(int(f.get("warning_count", 0)) for f in folds)})
    status = "complete_validated" if not errors and not missing and len(job_rows)==96 and complete_folds==480 else "partial_or_invalid"
    result = {"status": status, "phase": "A", "model_fit_or_scores": True,
        "expected_jobs": 96, "present_jobs": len(job_rows), "expected_model_folds": 480,
        "complete_model_folds": complete_folds, "missing_jobs": missing,
        "historical_seed42_RF_reproduction": rf_reproduction,
        "errors": errors, "run_manifest_sha256": sha256(manifest_path),
        "preflight_manifest_sha256": sha256(preflight_path), "jobs": job_rows}
    report_dir = output_dir or (run_dir / "analysis")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "main_A_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = [f"# Phase A analysis: {status}", "", f"Validated {len(job_rows)}/96 model jobs and {complete_folds}/480 folds.", "",
             f"Missing jobs: {len(missing)}. Validation errors: {len(errors)}.", ""]
    if errors:
        lines += ["## Validation issues", ""] + [f"- {e}" for e in errors[:30]] + ([f"- ... {len(errors)-30} more"] if len(errors)>30 else []) + [""]
    lines += ["## Pooled OOF macro-F1 / balanced accuracy", "", "| Policy | Split | Feature | Model | Rows | Macro-F1 | BA |", "|---|---|---|---|---:|---:|---:|"]
    for j in job_rows:
        m = j["pooled_oof"]
        lines.append(f"| {j['policy']} | {j['split']} | {j['feature']} | {j['model']} | {j['scored_rows']} | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} |")
    (report_dir / "main_A_analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--preflight-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    result = analyze(args.run_dir, args.preflight_dir, args.output_dir)
    print(json.dumps({k: result[k] for k in ("status", "present_jobs", "expected_jobs", "complete_model_folds", "expected_model_folds", "missing_jobs", "errors")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
