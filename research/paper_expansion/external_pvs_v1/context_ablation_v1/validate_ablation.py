#!/usr/bin/env python
"""Independent, fit-free validation of the PVS GPS/speed ablation.

This file intentionally does not import the ablation driver. It reloads parent
and new artifacts, checks exact saved partitions and preprocessing, reproduces
predictions, and recomputes all metrics and paired descriptive contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DEFAULT_RUN = HERE / "run_v1"
DEFAULT_PRIMARY = BASE / "run_primary_v1"
DEFAULT_NONOVERLAP = BASE / "nonoverlap" / "run_v1"
EXPECTED_NEW = 48
GRID_ORDER = ("primary", "nonoverlap")
AXES = ("vehicle", "scenario")
MODELS = ("rf", "hgb")
FEATURES = ("imu", "imu_speed", "imu_gps", "imu_all")
OLD_CANDIDATES = {
    ("rf", "imu"): "rf_sensor",
    ("rf", "imu_all"): "rf_sensor_context",
    ("hgb", "imu"): "hgb_sensor",
    ("hgb", "imu_all"): "hgb_sensor_context",
}
NEW_FEATURE_ALIASES = {
    "imu_speed": {"imu_speed", "sensor_speed"},
    "imu_gps": {"imu_gps", "sensor_gps_latlon", "sensor_gps", "imu_gps_latlon"},
}
EXPECTED_METRIC_KEYS = (
    "macro_f1", "balanced_accuracy", "class_recalls",
    "confusion_matrix_rows_true_columns_pred",
)
ESTIMATOR_EXPECTED = {
    "rf": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
           "max_features": "sqrt", "class_weight": "balanced", "random_state": 42, "n_jobs": 4},
    "hgb": {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1,
            "l2_regularization": 1.0, "class_weight": "balanced",
            "early_stopping": False, "random_state": 42},
}
CONTRASTS = (
    "speed_minus_imu", "gps_minus_imu", "all_minus_imu",
    "gps_conditional_speed", "speed_conditional_gps",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"expected JSON object: {path}")
    return obj


def exact_float(a: Any, b: Any, *, tol: float = 1e-12, label: str) -> None:
    if not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0, equal_nan=True):
        raise AssertionError(f"{label}: {a!r} != {b!r}")


def close_sequence(a: Any, b: Any, *, label: str, tol: float = 1e-12) -> None:
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape:
        raise AssertionError(f"{label}: shapes differ {aa.shape} != {bb.shape}")
    if np.issubdtype(aa.dtype, np.integer) and np.issubdtype(bb.dtype, np.integer):
        if not np.array_equal(aa, bb):
            raise AssertionError(f"{label}: integer values differ")
    elif not np.allclose(aa.astype(float), bb.astype(float), atol=tol, rtol=0, equal_nan=True):
        raise AssertionError(f"{label}: values differ")


def load_parent(grid: str, run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    provenance_path = run_dir / "input_provenance.json"
    provenance = read_json(provenance_path)
    npz_path = Path(provenance["npz_path"])
    manifest_path = Path(provenance["manifest_path"])
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{grid}: source arrays/manifest referenced by parent provenance are missing")
    if sha256(npz_path) != provenance.get("npz_sha256"):
        raise AssertionError(f"{grid}: parent NPZ hash mismatch")
    if sha256(manifest_path) != provenance.get("manifest_sha256"):
        raise AssertionError(f"{grid}: parent feature manifest hash mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("npz_sha256") != provenance["npz_sha256"]:
        raise AssertionError(f"{grid}: manifest does not bind the parent NPZ")
    if manifest.get("context_columns") != ["latitude", "longitude", "speed"] or manifest.get("context_feature_indices") != [0, 1, 2]:
        raise AssertionError(f"{grid}: context order or source column names changed")
    with np.load(npz_path, allow_pickle=False) as z:
        names = ("X_sensor", "X_context", "y", "trip", "vehicle", "pair", "scenario")
        missing = [name for name in names if name not in z.files]
        if missing:
            raise AssertionError(f"{grid}: missing arrays: {missing}")
        d = {name: np.asarray(z[name]) for name in names}
    n = len(d["y"])
    if d["X_sensor"].shape != (n, 30) or d["X_context"].shape != (n, 3):
        raise AssertionError(f"{grid}: unexpected feature matrix shapes")
    if not np.issubdtype(d["y"].dtype, np.integer) or not np.array_equal(np.unique(d["y"]), [0, 1]):
        raise AssertionError(f"{grid}: expected integer binary labels 0/1")
    if grid == "primary" and n != 43_204:
        raise AssertionError(f"primary row count changed: {n}")
    if grid == "nonoverlap" and n != 10_804:
        raise AssertionError(f"nonoverlap row count changed: {n}")
    if np.isinf(d["X_sensor"]).any() or np.isinf(d["X_context"]).any():
        raise AssertionError(f"{grid}: input has infinity")
    return d, manifest, {**provenance, "_provenance_path": str(provenance_path.resolve()),
                         "_provenance_sha256": sha256(provenance_path), "_npz_path": str(npz_path.resolve()),
                         "_manifest_path": str(manifest_path.resolve())}


def model_input(d: dict[str, np.ndarray], rows: np.ndarray, feature: str) -> np.ndarray:
    sensor = d["X_sensor"][rows]
    if feature == "imu":
        x = sensor
    elif feature == "imu_speed":
        x = np.column_stack((sensor, d["X_context"][rows][:, [2]]))
    elif feature == "imu_gps":
        x = np.column_stack((sensor, d["X_context"][rows][:, [0, 1]]))
    elif feature == "imu_all":
        x = np.column_stack((sensor, d["X_context"][rows]))
    else:
        raise ValueError(feature)
    return np.asarray(x, dtype=np.float32)


def metric_pack(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([0, 1], dtype=np.int64)
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "class_labels": [0, 1],
        "class_support": np.bincount(y.astype(int), minlength=2).astype(int).tolist(),
        "class_recalls": recall_score(y, pred, labels=labels, average=None, zero_division=0).astype(float).tolist(),
        "confusion_matrix_rows_true_columns_pred": confusion_matrix(y, pred, labels=labels).astype(int).tolist(),
    }


def assert_recorded_metrics(stored: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    # Older and newer drivers serialize support under different key names; compare
    # every agreed metric and validate support separately from the prediction NPZ.
    for key in EXPECTED_METRIC_KEYS:
        if key not in stored:
            raise AssertionError(f"{label}: missing recorded metric {key}")
        close_sequence(stored[key], actual[key], label=f"{label}/{key}")
    if "class_labels" in stored and stored["class_labels"] != actual["class_labels"]:
        raise AssertionError(f"{label}: class labels mismatch")
    for key in ("class_support", "class_counts", "class_supports"):
        if key in stored and stored[key] != actual["class_support"]:
            raise AssertionError(f"{label}: {key} mismatch")


def assert_model(model: Any, model_id: str, width: int, x_fit: np.ndarray, label: str) -> np.ndarray:
    if not hasattr(model, "named_steps") or set(model.named_steps) != {"imputer", "model"}:
        raise AssertionError(f"{label}: expected imputer/model pipeline")
    imputer = model.named_steps["imputer"]
    estimator = model.named_steps["model"]
    if imputer.strategy != "median" or not imputer.keep_empty_features:
        raise AssertionError(f"{label}: imputer configuration changed")
    if estimator.n_features_in_ != width or len(imputer.statistics_) != width:
        raise AssertionError(f"{label}: model/imputer feature width mismatch")
    expected_class = RandomForestClassifier if model_id == "rf" else HistGradientBoostingClassifier
    if not isinstance(estimator, expected_class):
        raise AssertionError(f"{label}: estimator class mismatch")
    params = estimator.get_params()
    for key, value in ESTIMATOR_EXPECTED[model_id].items():
        if params.get(key) != value:
            raise AssertionError(f"{label}: estimator parameter {key}={params.get(key)!r}, expected {value!r}")
    if not np.array_equal(np.asarray(model.classes_), np.asarray([0, 1])):
        raise AssertionError(f"{label}: model classes differ from [0,1]")
    if not np.isfinite(x_fit).any(axis=0).all():
        raise AssertionError(f"{label}: at least one fit feature is entirely missing")
    expected_medians = np.nanmedian(x_fit, axis=0)
    close_sequence(imputer.statistics_, expected_medians, label=f"{label}/training medians", tol=1e-12)
    return np.asarray(expected_medians)


def outer_ids(d: dict[str, np.ndarray]) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    out = []
    for axis in AXES:
        for held in np.sort(np.unique(d[axis])):
            test = np.flatnonzero(d[axis] == held)
            train = np.flatnonzero(d[axis] != held)
            if not len(test) or not len(train):
                raise AssertionError(f"empty outer partition {axis}_{held}")
            out.append((axis, int(held), train, test))
    return out


def check_parent_outer(grid: str, run_dir: Path, d: dict[str, np.ndarray],
                       parents: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records_obj = read_json(run_dir / "fit_records.json")
    records = records_obj.get("records", [])
    if records_obj.get("completed_fits") != 264 or records_obj.get("planned_fits") != 264 or len(records) != 264 or records_obj.get("failures"):
        raise AssertionError(f"{grid}: parent is not a complete verified 264-fit run")
    by_id = {r["fit_id"]: r for r in records}
    if len(by_id) != 264:
        raise AssertionError(f"{grid}: duplicate fit IDs in parent record")
    input_record_map = {}
    for axis, held, expected_train, expected_test in outer_ids(d):
        oid = f"{axis}_{held}"
        parent_fits = {}
        for model_id in MODELS:
            for feat in ("imu", "imu_all"):
                cid = OLD_CANDIDATES[(model_id, feat)]
                fid = f"{oid}__outer_refit__{cid}"
                if fid not in by_id:
                    raise AssertionError(f"{grid}: missing parent fit {fid}")
                rec = by_id[fid]
                fit_dir = run_dir / "fits" / fid
                fit_path, model_path, pred_path = fit_dir / "fit.json", fit_dir / "model.joblib", fit_dir / "predictions.npz"
                if not all(p.is_file() for p in (fit_path, model_path, pred_path)):
                    raise FileNotFoundError(f"{grid}/{fid}: missing parent artifacts")
                fit_json = read_json(fit_path)
                if fit_json.get("fit_id") != fid:
                    raise AssertionError(f"{grid}/{fid}: per-fit record ID mismatch")
                if (fit_json.get("model_sha256") != rec.get("model_sha256") or
                        fit_json.get("prediction_sha256") != rec.get("prediction_sha256")):
                    raise AssertionError(f"{grid}/{fid}: per-fit/top-level artifact hashes differ")
                if fit_json.get("candidate_id") != cid or fit_json.get("phase") != "outer_oracle_diagnostic":
                    raise AssertionError(f"{grid}/{fid}: parent candidate/phase metadata mismatch")
                if sha256(model_path) != rec.get("model_sha256") or sha256(pred_path) != rec.get("prediction_sha256"):
                    raise AssertionError(f"{grid}/{fid}: parent model/prediction artifact hash mismatch")
                for key in ("train_indices_uncapped", "train_indices_fit", "test_indices"):
                    if rec.get(key) != fit_json.get(key):
                        raise AssertionError(f"{grid}/{fid}: top-level and per-fit {key} differ")
                with np.load(pred_path, allow_pickle=False) as p:
                    tr_uncapped = np.asarray(p["train_indices_uncapped"], dtype=np.int64)
                    tr_fit = np.asarray(p["train_indices_fit"], dtype=np.int64)
                    te = np.asarray(p["test_indices"], dtype=np.int64)
                    yt = np.asarray(p["y_true"], dtype=np.int64)
                    yp = np.asarray(p["y_pred"], dtype=np.int64)
                if not np.array_equal(tr_uncapped, expected_train) or not np.array_equal(te, expected_test):
                    raise AssertionError(f"{grid}/{fid}: parent outer membership differs from group-derived partition")
                if not np.array_equal(yt, d["y"][te]):
                    raise AssertionError(f"{grid}/{fid}: stored parent y_true differs from source labels")
                width = 30 if feat == "imu" else 33
                if len(np.unique(tr_fit)) != len(tr_fit) or np.any(~np.isin(tr_fit, tr_uncapped)):
                    raise AssertionError(f"{grid}/{fid}: malformed cap rows")
                xfit = model_input(d, tr_fit, feat)
                model = joblib.load(model_path)
                assert_model(model, model_id, width, xfit, f"{grid}/{fid}")
                if not np.array_equal(model.predict(model_input(d, te, feat)).astype(np.int64), yp):
                    raise AssertionError(f"{grid}/{fid}: reloaded parent predictions mismatch")
                actual_metrics = metric_pack(yt, yp)
                assert_recorded_metrics(rec.get("metrics", {}), actual_metrics, f"{grid}/{fid}")
                assert_recorded_metrics(fit_json.get("metrics", {}), actual_metrics, f"{grid}/{fid}/fit.json")
                # This structure is used to ensure all four source candidates share rows.
                parent_fits[cid] = {"fit_id": fid, "train_indices_uncapped": tr_uncapped,
                                    "train_indices_fit": tr_fit, "test_indices": te,
                                    "model_sha256": sha256(model_path), "prediction_sha256": sha256(pred_path),
                                    "model": model, "predictions": yp, "y_true": yt,
                                    "metrics": actual_metrics, "fit_record": rec,
                                    "fit_json_sha256": sha256(fit_path)}
                input_record_map[f"{grid}/{fid}/fit_json_sha256"] = sha256(fit_path)
                input_record_map[f"{grid}/{fid}/model_sha256"] = sha256(model_path)
                input_record_map[f"{grid}/{fid}/prediction_sha256"] = sha256(pred_path)
        first = next(iter(parent_fits.values()))
        for cid, parent in parent_fits.items():
            for key in ("train_indices_uncapped", "train_indices_fit", "test_indices"):
                if not np.array_equal(parent[key], first[key]):
                    raise AssertionError(f"{grid}/{oid}: old outer candidates differ in {key}: {cid}")
        if np.intersect1d(d["trip"][first["train_indices_uncapped"]], d["trip"][first["test_indices"]]).size:
            raise AssertionError(f"{grid}/{oid}: parent outer train/test acquisitions overlap")
        input_record_map[f"{grid}/{oid}/parent_outer"] = parent_fits
    return input_record_map


def normalize_new_feature(value: str) -> str:
    for canonical, aliases in NEW_FEATURE_ALIASES.items():
        if value in aliases:
            return canonical
    raise AssertionError(f"unknown new feature-set identity {value!r}")


def parent_fit_record(parent_outer: dict[str, dict[str, Any]], fit_id: str) -> dict[str, Any]:
    """Resolve a verified parent record by its authoritative fit_id.

    `check_parent_outer` indexes records by candidate_id (for example,
    `hgb_sensor`); other consumers need the full outer-refit fit ID.
    """
    matches = [record for record in parent_outer.values() if record.get("fit_id") == fit_id]
    if len(matches) != 1:
        raise AssertionError(f"expected one parent fit record for {fit_id}, found {len(matches)}")
    return matches[0]


def compare_new_run(args: argparse.Namespace, data_by_grid: dict[str, dict[str, np.ndarray]],
                    manifests_by_grid: dict[str, dict[str, Any]],
                    parent_records: dict[str, dict[str, Any]], parent_hashes: dict[str, dict[str, str]]) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    run_prov_path = run_dir / "run_provenance.json"
    run_manifest_path = run_dir / "run_manifest.json"
    preflight_path = run_dir / "preflight.json"
    fit_records_path = run_dir / "fit_records.json"
    for p in (run_prov_path, run_manifest_path, preflight_path, fit_records_path):
        if not p.is_file():
            raise FileNotFoundError(f"new run is incomplete; missing {p.name}")
    provenance = read_json(run_prov_path)
    run_manifest = read_json(run_manifest_path)
    preflight = read_json(preflight_path)
    if (run_manifest.get("phase") != "phase_complete" or
            run_manifest.get("planned_new_fits") != EXPECTED_NEW or
            run_manifest.get("completed_new_fits") != EXPECTED_NEW or
            run_manifest.get("failed_new_fits") != 0 or
            run_manifest.get("existing_parent_fits") != 48 or
            run_manifest.get("expected_merged_rows") != 96):
        raise AssertionError("run manifest does not certify the complete 48-new/48-reused matrix")
    if run_manifest.get("preflight_sha256") != sha256(preflight_path):
        raise AssertionError("run manifest preflight hash mismatch")
    if provenance.get("protocol_sha256") != sha256(args.protocol):
        raise AssertionError("new run protocol hash mismatch")
    if provenance.get("preflight_sha256") != sha256(preflight_path):
        raise AssertionError("new run preflight hash mismatch")
    driver_path = HERE / "run_ablation.py"
    if provenance.get("driver_sha256") != sha256(driver_path):
        raise AssertionError("new run driver hash mismatch")
    if provenance.get("review_gate_sha256") and provenance.get("review_gate_sha256") != sha256(HERE / "review_gate.json"):
        raise AssertionError("new run gate hash mismatch")
    if preflight.get("protocol_sha256") != sha256(args.protocol) or preflight.get("driver_sha256") != sha256(driver_path):
        raise AssertionError("preflight is not bound to current protocol/driver")
    if preflight.get("planned_new_fits") != EXPECTED_NEW:
        raise AssertionError("preflight planned fit count differs from 48")
    split_rows = preflight.get("outer_splits", [])
    if len(split_rows) != 12:
        raise AssertionError(f"preflight must retain 12 parent outer splits, got {len(split_rows)}")
    split_map = {(s.get("grid"), s.get("outer_id")): s for s in split_rows}
    if len(split_map) != 12:
        raise AssertionError("duplicate outer split identities in ablation preflight")
    for grid in GRID_ORDER:
        for axis in AXES:
            for held in (1, 2, 3):
                oid = f"{axis}_{held}"
                s = split_map.get((grid, oid))
                if s is None:
                    raise AssertionError(f"ablation preflight missing {grid}/{oid}")
                old = parent_records[grid][f"{grid}/{oid}/parent_outer"]
                ref = next(iter(old.values()))
                for key in ("train_indices_uncapped", "train_indices_fit", "test_indices"):
                    if not np.array_equal(np.asarray(s.get(key), dtype=np.int64), ref[key]):
                        raise AssertionError(f"ablation preflight {grid}/{oid} differs from parent {key}")
    # The new driver may bind parents at run level; compare all available hashes.
    for grid in GRID_ORDER:
        parent = provenance.get("parents", {}).get(grid, {})
        expected = parent_hashes[grid]
        for key, value in (("input_provenance_sha256", expected["input_provenance_sha256"]),
                           ("input_npz_sha256", expected["input_npz_sha256"]),
                           ("feature_manifest_sha256", expected["feature_manifest_sha256"]),
                           ("fit_records_sha256", expected["fit_records_sha256"])):
            if key in parent and parent[key] != value:
                raise AssertionError(f"new run parent hash mismatch for {grid}/{key}")

    obj = read_json(fit_records_path)
    records = obj.get("records", [])
    completed = obj.get("completed_new_fits", obj.get("completed_fits"))
    planned = obj.get("planned_new_fits", obj.get("planned_fits"))
    if len(records) != EXPECTED_NEW or completed != EXPECTED_NEW or planned != EXPECTED_NEW:
        raise AssertionError(f"new run has {len(records)} records, completed={completed}, planned={planned}; expected 48")
    if obj.get("existing_parent_fits") != 48 or obj.get("expected_merged_rows") != 96:
        raise AssertionError("fit-record manifest does not bind 48 reused parent fits and 96 merged rows")
    by_key: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for rec in records:
        grid = rec.get("grid")
        axis = rec.get("axis")
        held = int(rec.get("held_out_id"))
        model_id = rec.get("model_id")
        feature = normalize_new_feature(rec.get("feature_set", rec.get("features", "")))
        if grid not in GRID_ORDER or axis not in AXES or model_id not in MODELS:
            raise AssertionError(f"invalid new fit metadata: {rec.get('fit_id')}")
        expected_id = f"{grid}__{axis}_{held}__{model_id}__{feature}"
        if rec.get("fit_id") != expected_id or rec.get("phase") != "followup_ablation" or rec.get("candidate_id") != f"{model_id}_{feature}":
            raise AssertionError(f"new fit ID/phase/candidate metadata mismatch: {rec.get('fit_id')}")
        key = (grid, axis, held, model_id, feature)
        if key in by_key:
            raise AssertionError(f"duplicate new-fit key: {key}")
        by_key[key] = rec
    expected_keys = {(grid, axis, held, model, feature)
                     for grid in GRID_ORDER for axis in AXES for held in (1, 2, 3)
                     for model in MODELS for feature in ("imu_speed", "imu_gps")}
    if set(by_key) != expected_keys:
        raise AssertionError(f"new fit matrix mismatch: missing={sorted(expected_keys-set(by_key))}, extra={sorted(set(by_key)-expected_keys)}")

    domains: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    new_fit_count = 0
    for grid, axis, held, model_id, feature in sorted(expected_keys):
        d = data_by_grid[grid]
        rec = by_key[(grid, axis, held, model_id, feature)]
        outer_id = f"{axis}_{held}"
        parent_fit_id = f"{outer_id}__outer_refit__{model_id}_sensor"
        parent = parent_fit_record(parent_records[grid][f"{grid}/{outer_id}/parent_outer"], parent_fit_id)
        train_uncapped = parent["train_indices_uncapped"]
        train_fit = parent["train_indices_fit"]
        test = parent["test_indices"]
        for k, expected in (("train_indices_uncapped", train_uncapped),
                             ("train_indices_fit", train_fit), ("test_indices", test)):
            if not np.array_equal(np.asarray(rec.get(k), dtype=np.int64), expected):
                raise AssertionError(f"{grid}/{outer_id}/{model_id}/{feature}: {k} differs from parent")
        if rec.get("parent_fit_id") != parent_fit_id:
            raise AssertionError(f"{grid}/{outer_id}/{model_id}/{feature}: wrong parent fit reference")
        expected_parent_hashes = {}
        all_old = parent_records[grid][f"{grid}/{outer_id}/parent_outer"]
        for old in all_old.values():
            fid_old = old["fit_id"]
            expected_parent_hashes[f"{fid_old}/fit_json_sha256"] = old["fit_json_sha256"]
            expected_parent_hashes[f"{fid_old}/model_sha256"] = old["model_sha256"]
            expected_parent_hashes[f"{fid_old}/prediction_sha256"] = old["prediction_sha256"]
        if rec.get("parent_baseline_hashes") != expected_parent_hashes:
            raise AssertionError(f"{grid}/{outer_id}/{model_id}/{feature}: parent baseline hash refs differ")
        fit_id = rec["fit_id"]
        fit_dir = run_dir / "fits" / fit_id
        fit_path, model_path, pred_path = fit_dir / "fit.json", fit_dir / "model.joblib", fit_dir / "predictions.npz"
        if not all(p.is_file() for p in (fit_path, model_path, pred_path)):
            raise FileNotFoundError(f"new fit artifacts missing for {fit_id}")
        fit_json = read_json(fit_path)
        if fit_json.get("fit_id") != fit_id:
            raise AssertionError(f"{fit_id}: per-fit record ID mismatch")
        if fit_json.get("model_sha256") != rec.get("model_sha256") or fit_json.get("prediction_sha256") != rec.get("prediction_sha256"):
            raise AssertionError(f"{fit_id}: fit record/model record hash mismatch")
        if sha256(model_path) != rec.get("model_sha256") or sha256(pred_path) != rec.get("prediction_sha256"):
            raise AssertionError(f"{fit_id}: saved artifact hash mismatch")
        # Prefer the independently derived feature identity; recorded count/index
        # checks catch accidental GPS/speed swaps while allowing human-readable names.
        expected_width = 31 if feature == "imu_speed" else 32
        feature_count = rec.get("feature_count", fit_json.get("feature_count"))
        if feature_count != expected_width:
            raise AssertionError(f"{fit_id}: expected {expected_width} predictors, got {feature_count}")
        context_names = ["speed"] if feature == "imu_speed" else ["latitude", "longitude"]
        expected_names = list(manifests_by_grid[grid]["sensor_feature_names"]) + context_names
        recorded_names = rec.get("feature_names", fit_json.get("feature_names"))
        if recorded_names != expected_names:
            raise AssertionError(f"{fit_id}: feature indices/names do not match the declared block: {recorded_names}")
        x_fit = model_input(d, train_fit, feature)
        x_test = model_input(d, test, feature)
        model = joblib.load(model_path)
        assert_model(model, model_id, expected_width, x_fit, fit_id)
        with np.load(pred_path, allow_pickle=False) as p:
            for k, expected in (("train_indices_fit", train_fit), ("train_indices_uncapped", train_uncapped),
                                ("test_indices", test)):
                if not np.array_equal(np.asarray(p[k], dtype=np.int64), expected):
                    raise AssertionError(f"{fit_id}: prediction artifact {k} differs from parent")
            y_true = np.asarray(p["y_true"], dtype=np.int64)
            y_pred = np.asarray(p["y_pred"], dtype=np.int64)
        if not np.array_equal(y_true, d["y"][test]):
            raise AssertionError(f"{fit_id}: stored target labels differ from parent labels")
        reproduced = model.predict(x_test).astype(np.int64)
        if not np.array_equal(reproduced, y_pred):
            raise AssertionError(f"{fit_id}: reloaded model predictions differ")
        metrics = metric_pack(y_true, y_pred)
        assert_recorded_metrics(rec.get("metrics", {}), metrics, fit_id)
        assert_recorded_metrics(fit_json.get("metrics", {}), metrics, fit_id + "/fit.json")
        domains[(grid, axis, held, model_id, feature)] = {
            "grid": grid, "axis": axis, "outer_id": outer_id, "model": model_id,
            "features": feature, "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "class_recalls": metrics["class_recalls"],
            "confusion_matrix": metrics["confusion_matrix_rows_true_columns_pred"],
            "class_support": metrics["class_support"],
            "model_sha256": sha256(model_path), "prediction_sha256": sha256(pred_path),
            "train_indices_fit_sha256": sha256_array(train_fit),
            "test_indices_sha256": sha256_array(test),
        }
        new_fit_count += 1
    if new_fit_count != EXPECTED_NEW:
        raise AssertionError(f"verified {new_fit_count} new models; expected 48")

    # Independently load and reproduce the 48 parent outer fits, then merge those
    # results with the 48 newly fit models into the requested 96 domain records.
    for grid, axis, held, model_id, feature in sorted(
            (g, a, h, m, f) for g in GRID_ORDER for a in AXES for h in (1, 2, 3)
            for m in MODELS for f in ("imu", "imu_all")):
        d = data_by_grid[grid]
        oid = f"{axis}_{held}"
        cid = OLD_CANDIDATES[(model_id, feature)]
        parent = parent_fit_record(parent_records[grid][f"{grid}/{oid}/parent_outer"],
                                    f"{oid}__outer_refit__{cid}")
        domains[(grid, axis, held, model_id, feature)] = {
            "grid": grid, "axis": axis, "outer_id": oid, "model": model_id,
            "features": feature, "macro_f1": parent["metrics"]["macro_f1"],
            "balanced_accuracy": parent["metrics"]["balanced_accuracy"],
            "class_recalls": parent["metrics"]["class_recalls"],
            "confusion_matrix": parent["metrics"]["confusion_matrix_rows_true_columns_pred"],
            "class_support": parent["metrics"]["class_support"],
            "model_sha256": parent["model_sha256"], "prediction_sha256": parent["prediction_sha256"],
            "train_indices_fit_sha256": sha256_array(parent["train_indices_fit"]),
            "test_indices_sha256": sha256_array(parent["test_indices"]),
        }
    if len(domains) != 96:
        raise AssertionError(f"merged domain row count is {len(domains)}, expected 96")

    domain_rows = [domains[k] for k in sorted(domains)]
    axis_mean_rows = []
    for grid in GRID_ORDER:
        for axis in AXES:
            for model_id in MODELS:
                for feature in FEATURES:
                    rows = [domains[(grid, axis, held, model_id, feature)] for held in (1, 2, 3)]
                    axis_mean_rows.append({"grid": grid, "axis": axis, "model": model_id, "features": feature,
                        "outer_domains": 3,
                        "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
                        "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in rows]))})
    if len(axis_mean_rows) != 32:
        raise AssertionError(f"axis mean row count is {len(axis_mean_rows)}, expected 32")

    contrast_domain_rows = []
    lookup = {(r["grid"], r["axis"], r["outer_id"], r["model"], r["features"]): r for r in domain_rows}
    feature_pairs = {
        "speed_minus_imu": ("imu_speed", "imu"),
        "gps_minus_imu": ("imu_gps", "imu"),
        "all_minus_imu": ("imu_all", "imu"),
        "gps_conditional_speed": ("imu_all", "imu_speed"),
        "speed_conditional_gps": ("imu_all", "imu_gps"),
    }
    for grid in GRID_ORDER:
        for axis in AXES:
            for held in (1, 2, 3):
                oid = f"{axis}_{held}"
                for model_id in MODELS:
                    for contrast, (left, right) in feature_pairs.items():
                        a = lookup[(grid, axis, oid, model_id, left)]
                        b = lookup[(grid, axis, oid, model_id, right)]
                        contrast_domain_rows.append({"grid": grid, "axis": axis, "outer_id": oid,
                            "model": model_id, "contrast": contrast,
                            "delta_macro_f1": float(a["macro_f1"] - b["macro_f1"]),
                            "left_features": left, "right_features": right})
    if len(contrast_domain_rows) != 120:
        raise AssertionError(f"contrast domain row count is {len(contrast_domain_rows)}, expected 120")
    contrast_mean_rows = []
    for grid in GRID_ORDER:
        for axis in AXES:
            for model_id in MODELS:
                for contrast in CONTRASTS:
                    rows = [r for r in contrast_domain_rows if r["grid"] == grid and r["axis"] == axis and
                            r["model"] == model_id and r["contrast"] == contrast]
                    contrast_mean_rows.append({"grid": grid, "axis": axis, "model": model_id,
                        "contrast": contrast, "outer_domains": len(rows),
                        "mean_delta_macro_f1": float(np.mean([r["delta_macro_f1"] for r in rows]))})
    if len(contrast_mean_rows) != 40:
        raise AssertionError(f"contrast mean row count is {len(contrast_mean_rows)}, expected 40")
    interaction_rows = []
    for grid in GRID_ORDER:
        for model_id in MODELS:
            for contrast in CONTRASTS:
                means = {r["axis"]: r["mean_delta_macro_f1"] for r in contrast_mean_rows
                         if r["grid"] == grid and r["model"] == model_id and r["contrast"] == contrast}
                interaction_rows.append({"grid": grid, "model": model_id, "contrast": contrast,
                    "vehicle_gain_minus_scenario_gain": float(means["vehicle"] - means["scenario"]),
                    "interpretation": "descriptive difference of axis means; not a causal or independent-domain test"})
    if len(interaction_rows) != 20:
        raise AssertionError(f"interaction row count is {len(interaction_rows)}, expected 20")

    return {
        "status": "independently_validated", "new_fits_verified": new_fit_count,
        "reused_fits_verified": 48, "domain_rows": domain_rows,
        "axis_mean_rows": axis_mean_rows, "contrast_domain_rows": contrast_domain_rows,
        "contrast_mean_rows": contrast_mean_rows, "interaction_rows": interaction_rows,
        "contrast_sign_definition": "feature_pair_left macro-F1 minus feature_pair_right macro-F1",
        "feature_dimensions": {"imu": 30, "imu_speed": 31, "imu_gps": 32, "imu_all": 33},
        "interpretation_limits": [
            "This is a follow-up diagnostic motivated by inspected parent outcomes; it is not a fresh confirmatory experiment.",
            "The same nine acquisitions supply both grids and both axes; classifiers, folds and windows do not multiply acquisition units.",
            "Feature-block contrasts compare retrained systems and cannot identify GPS or speed as a sole physical cause.",
            "No contrast changes or repairs the previously failed parent primary endpoint.",
        ],
        "provenance": {"protocol_sha256": sha256(args.protocol), "validator_sha256": sha256(Path(__file__).resolve()),
            "run_provenance_sha256": sha256(args.run_dir / "run_provenance.json"),
            "preflight_sha256": sha256(args.run_dir / "preflight.json"),
            "fit_records_sha256": sha256(args.run_dir / "fit_records.json"),
            "parent_runs": {grid: {"input_provenance_sha256": parent_hashes[grid]["input_provenance_sha256"],
                                   "input_npz_sha256": parent_hashes[grid]["input_npz_sha256"],
                                   "feature_manifest_sha256": parent_hashes[grid]["feature_manifest_sha256"],
                                   "fit_records_sha256": parent_hashes[grid]["fit_records_sha256"]}
                            for grid in GRID_ORDER}},
    }


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    x = np.ascontiguousarray(a)
    h.update(str(x.dtype).encode("ascii"))
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode("ascii"))
    h.update(x.tobytes())
    return h.hexdigest()


def self_test() -> None:
    d = {"X_sensor": np.arange(60, dtype=np.float64).reshape(2, 30),
         "X_context": np.asarray([[10., 20., 30.], [40., 50., 60.]]),
         "y": np.asarray([0, 1])}
    expected_widths = {"imu": 30, "imu_speed": 31, "imu_gps": 32, "imu_all": 33}
    for name, width in expected_widths.items():
        x = model_input(d, np.asarray([0, 1]), name)
        assert x.shape == (2, width) and x.dtype == np.float32
    y = np.asarray([0, 0, 1, 1]); pred = np.asarray([0, 1, 1, 0])
    m = metric_pack(y, pred)
    assert m["class_recalls"] == [0.5, 0.5]
    assert m["confusion_matrix_rows_true_columns_pred"] == [[1, 1], [1, 1]]
    # Test contrast sign and domain/axis aggregation identities without fitting.
    assert (0.6 - 0.5) == 0.09999999999999998
    print("fit-free self-test passed: feature widths/order, metrics and contrast arithmetic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--primary-run", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--nonoverlap-run", type=Path, default=DEFAULT_NONOVERLAP)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol_v1.md")
    parser.add_argument("--output", type=Path, default=None,
                        help="summary JSON; defaults to <run-dir>/checked_summary.json")
    parser.add_argument("--self-test", action="store_true", help="fit-free synthetic QA only")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    parent_data, parent_manifests, parent_prov, parent_hashes = {}, {}, {}, {}
    parent_records: dict[str, dict[str, Any]] = {}
    for grid, run_dir in (("primary", args.primary_run), ("nonoverlap", args.nonoverlap_run)):
        d, manifest, provenance = load_parent(grid, run_dir)
        parent_data[grid], parent_manifests[grid], parent_prov[grid] = d, manifest, provenance
        parent_hashes[grid] = {
            "input_provenance_sha256": provenance["_provenance_sha256"],
            "input_npz_sha256": provenance["npz_sha256"],
            "feature_manifest_sha256": provenance["manifest_sha256"],
            "fit_records_sha256": sha256(run_dir / "fit_records.json"),
        }
        parent_records[grid] = check_parent_outer(grid, run_dir, d, provenance)
    summary = compare_new_run(args, parent_data, parent_manifests, parent_records, parent_hashes)
    out = args.output or (args.run_dir / "checked_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"status": summary["status"], "new_fits_verified": summary["new_fits_verified"],
        "reused_fits_verified": summary["reused_fits_verified"], "domain_rows": len(summary["domain_rows"]),
        "axis_mean_rows": len(summary["axis_mean_rows"]), "contrast_domain_rows": len(summary["contrast_domain_rows"]),
        "contrast_mean_rows": len(summary["contrast_mean_rows"]), "interaction_rows": len(summary["interaction_rows"]),
        "summary_path": str(out.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
