from __future__ import annotations

"""Independent, no-fit validation for the T temporal-context experiment."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PILOT = EXP.parent
OUT = HERE / "T_temporal_v1"
PRE = OUT / "preflight_v1"
AXIS_RUN = PILOT / "runs" / "axis_window_v1"
POLICIES = ("primary", "blank_normal")
SPLITS = ("record42", "timeblock42")
FEATURES = ("imu_only", "axis_window")
MODELS = ("RF100", "HistGB")
CLASSES = ("Normal", "Pothole", "Bump")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_axis_builder():
    source = PILOT / "run_axis_window_control.py"
    spec = importlib.util.spec_from_file_location("roadsens_axis_builder_T_analysis", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen axis-window feature builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def window_members(session):
    times = np.asarray(session["times_full"], dtype=np.float64)
    source_rows = np.asarray(session["source_row"], dtype=np.int64)
    order = np.flatnonzero(np.isfinite(times))
    order = order[np.argsort(times[order], kind="stable")]
    ordered_times = times[order]
    position = np.full(len(times), -1, dtype=np.int64)
    position[order] = np.arange(len(order), dtype=np.int64)
    result = {}
    for row in source_rows:
        row = int(row)
        if row < 0 or row >= len(times) or position[row] < 0:
            continue
        left = int(np.searchsorted(ordered_times, times[row] - 1.0, side="left"))
        right = int(position[row]) + 1
        result[row] = frozenset(int(v) for v in order[left:right])
    return result


def metric(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    to_id = {c: i for i, c in enumerate(CLASSES)}
    yt = np.asarray([to_id[str(v)] for v in y_true], dtype=np.int8)
    yp = np.asarray([to_id[str(v)] for v in y_pred], dtype=np.int8)
    labels = np.arange(len(CLASSES))
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "class_order": list(CLASSES),
        "class_support": {c: int(np.sum(yt == i)) for i, c in enumerate(CLASSES)},
        "class_recall": {c: float(v) for c, v in zip(CLASSES, recall_score(yt, yp, labels=labels, average=None, zero_division=0))},
        "confusion": confusion_matrix(yt, yp, labels=labels).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", choices=("fit_gate_v1", "fit_remaining_v1", "all"), default="all")
    ap.add_argument("--output", type=Path, default=OUT / "independent_validation_v1.json")
    args = ap.parse_args()

    sys.path.insert(0, str(EXP))
    import runner as r
    import sklearn
    if Path(sys.executable).resolve() != Path(r"C:\Users\Administrator\miniconda3\python.exe").resolve() or sklearn.__version__ != "1.9.0":
        raise RuntimeError("locked Miniconda/sklearn runtime required for validation")
    cache, frozen, split_qa = r.validate_immutable_inputs()
    specs, rows = r.build_all_splits(cache["sessions"], frozen, split_qa)
    pre = json.loads((PRE / "preflight_manifest.json").read_text(encoding="utf-8"))
    membership = PRE / "split_membership_v1.npz"
    if sha(membership) != pre.get("split_membership_sha256"):
        raise ValueError("T split membership checksum mismatch")
    if pre.get("cache_sha256") != sha(r.CACHE) or pre.get("shared_runner_sha256") != sha(EXP / "runner.py"):
        raise ValueError("T frozen input/runtime code changed")
    bindings = {
        "parent_runner_sha256": PILOT / "run_pilot.py",
        "frozen_map_sha256": PILOT / "session_fold_map_v1.json",
        "timeblock_support_sha256": PILOT / "timeblock_split_support_v3.json",
        "identity_certificate_sha256": EXP / "shared_identity_audit_v1.json",
        "key_protocol_sha256": HERE / "protocol_key_v1.md",
        "T_protocol_sha256": HERE / "T_temporal_protocol_v1.md",
        "driver_sha256": HERE / "run_temporal_benchmark.py",
    }
    for field, path in bindings.items():
        if pre.get(field) != sha(path):
            raise ValueError(f"T preflight binding changed: {field}")
    for relative, expected_sha in pre.get("axis_window_reference_sha256", {}).items():
        if sha(AXIS_RUN / relative) != expected_sha:
            raise ValueError(f"historical axis-window reference changed: {relative}")
    review = json.loads((OUT / "review_T_v1.json").read_text(encoding="utf-8"))
    if review.get("axis_builder_sha256") != sha(PILOT / "run_axis_window_control.py"):
        raise ValueError("axis feature builder changed")
    idx = np.load(membership)

    axis = load_axis_builder()
    built, names = axis.build_axis_features(cache["sessions"], r._load_modules()[0])
    if names != pre.get("feature_schema"):
        raise ValueError("Axis feature schema differs from preflight")
    matrices = {}
    windows = {}
    for policy in POLICIES:
        table = rows[policy]
        x_imu, _ = r.feature_matrix(cache["sessions"], table, "imu_only")
        x_axis = np.concatenate([
            built[int(owner)]["axis_window"][int(local)][None, :]
            for owner, local in zip(table.owners, table.local_rows)
        ], axis=0)
        matrices[(policy, "imu_only")] = (table, x_imu)
        matrices[(policy, "axis_window")] = (table, x_axis)
    for session in cache["sessions"]:
        windows[session["session_key"]] = window_members(session)

    if args.fit_dir == "all":
        dirs = [OUT / "fit_gate_v1", OUT / "fit_remaining_v1"]
    else:
        dirs = [OUT / args.fit_dir]
    records = []
    errors = []
    gates = []
    for run_dir in dirs:
        manifest_path = run_dir / ("gate_manifest.json" if run_dir.name == "fit_gate_v1" else "run_manifest.json")
        if not manifest_path.is_file():
            if args.fit_dir == "all" and run_dir.name == "fit_remaining_v1":
                continue
            errors.append(f"missing fit manifest: {manifest_path}")
            continue
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("preflight_manifest_sha256") != sha(PRE / "preflight_manifest.json"):
            errors.append(f"preflight hash mismatch: {manifest_path}")
        if run_manifest.get("failure_count") or run_manifest.get("failures"):
            errors.append(f"recorded fit failures: {manifest_path}")
        if run_manifest.get("fit_driver_sha256") != sha(HERE / "fit_temporal_benchmark.py"):
            errors.append(f"fit driver hash mismatch: {manifest_path}")
        for fold_path in sorted((run_dir / "folds").glob("*.json")):
            rec = json.loads(fold_path.read_text(encoding="utf-8"))
            policy, split, feature, model, fold = (rec[k] for k in ("policy", "split", "feature", "model", "fold"))
            table, x = matrices[(policy, feature)]
            key = f"{policy}__{split}__fold{fold}"
            fit = idx[key + "__fit"].astype(np.int64)
            test = idx[key + "__test"].astype(np.int64)
            fit_identity = [{"row": int(i), "session": str(table.groups[i]), "source_row": int(table.source_rows[i])} for i in fit]
            test_identity = [{"row": int(i), "session": str(table.groups[i]), "source_row": int(table.source_rows[i])} for i in test]
            checks = {
                "fit_rows_match_preflight": rec.get("fit_rows") == fit_identity,
                "test_rows_match_preflight": rec.get("test_rows") == test_identity,
                "fit_hash_matches": rec.get("fit_rows_sha256") == r.object_hash(fit_identity),
                "test_hash_matches": rec.get("test_rows_sha256") == r.object_hash(test_identity),
                "test_source_rows_unique": len(set((str(table.groups[i]), int(table.source_rows[i])) for i in test)) == len(test),
            }
            model_path = OUT / rec["model_path"]
            pred_path = OUT / rec["prediction_path"]
            checks["model_sha256_matches"] = model_path.is_file() and sha(model_path) == rec.get("model_sha256")
            checks["prediction_sha256_matches"] = pred_path.is_file() and sha(pred_path) == rec.get("prediction_sha256")
            if not all(checks.values()):
                errors.append(f"artifact or row identity check failed: {fold_path.name}: {checks}")
                continue

            pred_data = np.load(pred_path)
            checks["prediction_ids_match_test"] = np.array_equal(pred_data["row"], test)
            checks["prediction_source_rows_match_test"] = np.array_equal(pred_data["source_row"], table.source_rows[test])
            checks["prediction_sessions_match_test"] = np.array_equal(pred_data["session"].astype(str), table.groups[test].astype(str))
            checks["prediction_truth_matches_cache"] = np.array_equal(pred_data["y_true"].astype(str), table.y[test].astype(str))
            checks["class_order_is_explicit"] = tuple(pred_data["class_order"].astype(str).tolist()) == CLASSES
            estimator = joblib.load(model_path)
            xtr, xte, kept, medians, prep = r.preprocess_train_test(x, fit, test, feature)
            if kept != rec.get("kept_feature_indices") or medians != rec.get("training_imputation_medians"):
                errors.append(f"preprocessing provenance mismatch: {fold_path.name}")
            rerun = np.asarray([str(v) for v in estimator.predict(xte)], dtype=object)
            saved = pred_data["y_pred"].astype(str)
            checks["reloaded_model_predictions_match"] = np.array_equal(rerun.astype(str), saved)
            independent_metrics = metric(table.y[test], rerun)
            checks["metrics_match"] = all(
                abs(independent_metrics[k] - rec["metrics"][k]) < 1e-12
                for k in ("accuracy", "balanced_accuracy", "macro_f1")
            ) and independent_metrics["confusion"] == rec["metrics"]["confusion"]
            checks["class_metrics_match"] = (
                independent_metrics["n"] == rec["metrics"]["n"]
                and independent_metrics["class_order"] == rec["metrics"]["class_order"]
                and independent_metrics["class_support"] == rec["metrics"]["class_support"]
                and all(abs(independent_metrics["class_recall"][c] - rec["metrics"]["class_recall"][c]) < 1e-12 for c in CLASSES)
            )

            if split == "timeblock42":
                test_raw = {}
                for i in test:
                    group = str(table.groups[i]); sr = int(table.source_rows[i])
                    test_raw.setdefault(group, set()).update(windows[group][sr])
                train_raw = {}
                for i in fit:
                    group = str(table.groups[i]); sr = int(table.source_rows[i])
                    train_raw.setdefault(group, set()).update(windows[group][sr])
                checks["raw_window_source_rows_disjoint"] = all(
                    not (test_raw.get(group, set()) & raw) for group, raw in train_raw.items()
                )
            else:
                checks["recording_units_disjoint"] = not (set(table.groups[fit]) & set(table.groups[test]))

            if split == "record42" and feature == "axis_window" and model == "RF100":
                old = json.loads((AXIS_RUN / f"{policy}_group.axis_window.fold{fold}.checkpoint.json").read_text(encoding="utf-8"))
                old_fit = old["fit_train_rows"]
                old_test = [{k: row[k] for k in ("row", "session", "source_row")} for row in old["test_rows"]]
                checks["historical_gate_fit_rows_match"] = old_fit == fit_identity
                checks["historical_gate_test_rows_match"] = old_test == test_identity
                checks["historical_gate_predictions_match"] = saved.tolist() == [str(row["pred"]) for row in old["test_rows"]]
                gates.append({"policy": policy, "fold": fold, "checks": {k: checks[k] for k in checks if k.startswith("historical_gate_")}})

            failed = [k for k, v in checks.items() if not v]
            if failed:
                errors.append(f"independent checks failed {fold_path.name}: {failed}")
            records.append({"fit_dir": run_dir.name, "policy": policy, "split": split, "feature": feature, "model": model, "fold": fold,
                            "fit_n": len(fit), "test_n": len(test), "metrics": independent_metrics, "checks": checks})

    expected = 10 if args.fit_dir == "fit_gate_v1" else (70 if args.fit_dir == "fit_remaining_v1" else 80 if (OUT / "fit_remaining_v1" / "run_manifest.json").exists() else 10)
    complete = len(records) == expected
    status = "passed" if complete and not errors else "failed_or_incomplete"
    output = {"status": status, "fit_dir": args.fit_dir, "expected_folds": expected, "validated_folds": len(records), "error_count": len(errors),
              "errors": errors, "gate_fold_checks": gates, "records": records,
              "interpretation": "Within-dataset diagnostics only; folds and sampled rows are not independent roads, routes, or events."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{k: output[k] for k in ("status", "expected_folds", "validated_folds", "error_count")}, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
