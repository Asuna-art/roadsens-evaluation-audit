#!/usr/bin/env python
"""Fit-free-first PVS speed-vs-GPS context ablation.

The primary and nonoverlap parent splits/caps are read verbatim from their
saved outer-refit artifacts. This script never rebuilds a split or cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROTOCOL_PATH = HERE / "protocol_v1.md"
GATE_PATH = HERE / "review_gate.json"
DEFAULT_OUT = HERE / "run_v1"
GRID_RUNS = {
    "primary": BASE / "run_primary_v1",
    "nonoverlap": BASE / "nonoverlap" / "run_v1",
}
MODEL_IDS = ("rf", "hgb")
FEATURE_SETS = (
    {"id": "imu_speed", "context_indices": [2], "context_names": ["speed"]},
    {"id": "imu_gps", "context_indices": [0, 1], "context_names": ["latitude", "longitude"]},
)
BASELINE_IDS = ("rf_sensor", "rf_sensor_context", "hgb_sensor", "hgb_sensor_context")
OUTER_AXES = ("vehicle", "scenario")
EXPECTED_NEW_FITS = 48


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    x = np.ascontiguousarray(a)
    h.update(str(x.dtype).encode("ascii"))
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode("ascii"))
    h.update(x.tobytes())
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_grid(grid: str, run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, str]]:
    provenance = read_json(run_dir / "input_provenance.json")
    npz_path = Path(provenance["npz_path"])
    manifest_path = Path(provenance["manifest_path"])
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{grid}: parent input files in provenance are not available")
    npz_hash, manifest_hash = sha256(npz_path), sha256(manifest_path)
    if npz_hash != provenance["npz_sha256"] or manifest_hash != provenance["manifest_sha256"]:
        raise ValueError(f"{grid}: parent input hashes do not match input_provenance.json")
    manifest = read_json(manifest_path)
    if manifest.get("npz_sha256") != npz_hash:
        raise ValueError(f"{grid}: feature manifest references a different NPZ")
    with np.load(npz_path, allow_pickle=False) as z:
        required = ("X_sensor", "X_context", "y", "trip", "pair", "vehicle", "scenario")
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(f"{grid}: NPZ is missing arrays {missing}")
        data = {k: np.asarray(z[k]) for k in required}
    n = len(data["y"])
    if data["X_sensor"].shape != (n, 30):
        raise ValueError(f"{grid}: X_sensor must be N×30")
    if data["X_context"].shape != (n, 3):
        raise ValueError(f"{grid}: X_context must be N×3")
    if len(np.unique(data["y"])) != 2 or not np.issubdtype(data["y"].dtype, np.integer):
        raise ValueError(f"{grid}: y must contain exactly two integer classes")
    if np.isinf(data["X_sensor"]).any() or np.isinf(data["X_context"]).any():
        raise ValueError(f"{grid}: input feature matrices contain infinity")
    names = manifest.get("context_feature_names", [])
    indices = manifest.get("context_feature_indices", [])
    if names != ["latitude", "longitude", "speed"] or indices != [0, 1, 2]:
        raise ValueError(f"{grid}: context column order is not the frozen latitude/longitude/speed order")
    parent_hashes = {
        "run_dir": str(run_dir.resolve()),
        "input_provenance_sha256": sha256(run_dir / "input_provenance.json"),
        "input_npz_path": str(npz_path.resolve()),
        "input_npz_sha256": npz_hash,
        "feature_manifest_path": str(manifest_path.resolve()),
        "feature_manifest_sha256": manifest_hash,
        "fit_records_sha256": sha256(run_dir / "fit_records.json"),
    }
    checked_summary = run_dir.parent / "checked_summary.json"
    if not checked_summary.is_file():
        raise FileNotFoundError(f"{grid}: independently checked parent summary is missing: {checked_summary}")
    parent_hashes["checked_summary_path"] = str(checked_summary.resolve())
    parent_hashes["checked_summary_sha256"] = sha256(checked_summary)
    return data, manifest, provenance, parent_hashes


def validate_parent_refit(grid: str, run_dir: Path, data: dict[str, np.ndarray],
                          baseline_hashes: dict[str, str]) -> list[dict[str, Any]]:
    entries = []
    parent_plan = read_json(run_dir / "preflight.json")
    plan_folds = {fold["outer_id"]: fold for fold in parent_plan.get("folds", [])}
    for axis in OUTER_AXES:
        values = np.sort(np.unique(data[axis]))
        for held in values:
            outer_id = f"{axis}_{int(held)}"
            fit_ids = [f"{outer_id}__outer_refit__{cid}" for cid in BASELINE_IDS]
            refs = []
            local_baseline_hashes: dict[str, str] = {}
            for fit_id in fit_ids:
                fit_dir = run_dir / "fits" / fit_id
                fit_path, model_path, pred_path = fit_dir / "fit.json", fit_dir / "model.joblib", fit_dir / "predictions.npz"
                if not all(p.is_file() for p in (fit_path, model_path, pred_path)):
                    raise FileNotFoundError(f"{grid}: missing parent refit artifact {fit_id}")
                record = read_json(fit_path)
                if record.get("fit_id") != fit_id or record.get("phase") != "outer_oracle_diagnostic":
                    raise ValueError(f"{grid}: unexpected parent fit record {fit_id}")
                model_hash, pred_hash = sha256(model_path), sha256(pred_path)
                if model_hash != record.get("model_sha256") or pred_hash != record.get("prediction_sha256"):
                    raise ValueError(f"{grid}: parent model/prediction hash mismatch {fit_id}")
                with np.load(pred_path, allow_pickle=False) as p:
                    for key in ("train_indices_fit", "train_indices_uncapped", "test_indices"):
                        if not np.array_equal(p[key], np.asarray(record[key], dtype=np.int64)):
                            raise ValueError(f"{grid}: parent prediction file index mismatch {fit_id}/{key}")
                    if not np.array_equal(p["y_true"], data["y"][p["test_indices"]]):
                        raise ValueError(f"{grid}: parent prediction labels mismatch {fit_id}")
                refs.append(record)
                local_baseline_hashes[f"{fit_id}/fit_json_sha256"] = sha256(fit_path)
                local_baseline_hashes[f"{fit_id}/model_sha256"] = model_hash
                local_baseline_hashes[f"{fit_id}/prediction_sha256"] = pred_hash
                baseline_hashes[f"{grid}/{fit_id}/fit_json_sha256"] = sha256(fit_path)
                baseline_hashes[f"{grid}/{fit_id}/model_sha256"] = model_hash
                baseline_hashes[f"{grid}/{fit_id}/prediction_sha256"] = pred_hash
            ref = refs[0]
            keys = ("train_indices_fit", "train_indices_uncapped", "test_indices")
            for other, fit_id in zip(refs[1:], fit_ids[1:]):
                for key in keys:
                    if ref[key] != other[key]:
                        raise ValueError(f"{grid}/{outer_id}: old candidates do not share exact {key} ({fit_id})")
            tr_uncapped = np.asarray(ref["train_indices_uncapped"], dtype=np.int64)
            tr_fit = np.asarray(ref["train_indices_fit"], dtype=np.int64)
            te = np.asarray(ref["test_indices"], dtype=np.int64)
            if np.intersect1d(data["trip"][tr_uncapped], data["trip"][te]).size:
                raise ValueError(f"{grid}/{outer_id}: train/test acquisition groups overlap")
            split_col = "pair" if axis == "vehicle" else "scenario"
            if np.intersect1d(data[split_col][tr_uncapped], data[split_col][te]).size:
                raise ValueError(f"{grid}/{outer_id}: train/test held-axis groups overlap ({split_col})")
            if len(np.unique(data["y"][tr_fit])) != 2 or len(np.unique(data["y"][te])) != 2:
                raise ValueError(f"{grid}/{outer_id}: fit or test rows do not contain both labels")
            if outer_id not in plan_folds:
                raise ValueError(f"{grid}/{outer_id}: parent split is missing from saved preflight plan")
            parent_fold = plan_folds[outer_id]
            parent_cap = parent_fold.get("outer_refit_cap", {})
            if (not np.array_equal(np.asarray(parent_fold.get("train_indices", []), dtype=np.int64), tr_uncapped) or
                    not np.array_equal(np.asarray(parent_fold.get("test_indices", []), dtype=np.int64), te) or
                    not np.array_equal(np.asarray(parent_cap.get("train_indices_uncapped", []), dtype=np.int64), tr_uncapped) or
                    not np.array_equal(np.asarray(parent_cap.get("train_indices_fit", []), dtype=np.int64), tr_fit) or
                    not np.array_equal(np.asarray(parent_cap.get("test_indices", []), dtype=np.int64), te)):
                raise ValueError(f"{grid}/{outer_id}: baseline refit memberships disagree with the parent saved fold plan")
            x_s = data["X_sensor"][tr_fit]
            x_gps = data["X_context"][tr_fit][:, [0, 1]]
            x_speed = data["X_context"][tr_fit][:, [2]]
            feature_support = {}
            for feature_id, block in (("imu_speed", np.column_stack((x_s, x_speed))),
                                      ("imu_gps", np.column_stack((x_s, x_gps)))):
                with np.errstate(over="ignore", invalid="ignore"):
                    model_block = np.asarray(block, dtype=np.float32)
                if np.isinf(model_block).any():
                    raise ValueError(f"{grid}/{outer_id}/{feature_id}: float32 model input overflows to infinity")
                if not np.isfinite(model_block).any(axis=0).all():
                    raise ValueError(f"{grid}/{outer_id}/{feature_id}: a feature is entirely missing in capped float32 fit rows")
                feature_support[feature_id] = {"all_features_have_finite_fit_value": True,
                                               "feature_count": int(block.shape[1])}
            entries.append({
                "grid": grid, "outer_id": outer_id, "axis": axis, "held_out_id": int(held),
                "parent_fit_id": fit_ids[0],
                "train_rows_uncapped": int(len(tr_uncapped)), "train_rows_fit": int(len(tr_fit)),
                "test_rows": int(len(te)), "class_counts_fit": np.bincount(data["y"][tr_fit].astype(int), minlength=2).tolist(),
                "class_counts_test": np.bincount(data["y"][te].astype(int), minlength=2).tolist(),
                "train_indices_uncapped": tr_uncapped.tolist(), "train_indices_fit": tr_fit.tolist(),
                "test_indices": te.tolist(),
                "index_hashes": {"train_indices_uncapped": sha256_array(tr_uncapped),
                                 "train_indices_fit": sha256_array(tr_fit), "test_indices": sha256_array(te)},
                "feature_support": feature_support,
                "baseline_parent_fit_ids": fit_ids,
                "parent_baseline_hashes": local_baseline_hashes,
            })
    return entries


def build_preflight(primary_run: Path, nonoverlap_run: Path) -> dict[str, Any]:
    parents = {}
    hashes: dict[str, str] = {}
    splits = []
    for grid, run_dir in (("primary", primary_run), ("nonoverlap", nonoverlap_run)):
        data, manifest, provenance, p_hashes = load_grid(grid, run_dir)
        parents[grid] = {**p_hashes, "protocol_id": provenance.get("protocol_id"),
                         "parent_manifest_npz_sha256": manifest.get("npz_sha256"),
                         "row_count": int(len(data["y"])),
                         "label_counts": np.bincount(data["y"].astype(int), minlength=2).tolist()}
        splits.extend(validate_parent_refit(grid, run_dir, data, hashes))
    if len(splits) != 12:
        raise AssertionError(f"expected 12 saved outer splits, found {len(splits)}")
    planned_fits = []
    for split in splits:
        for model_id in MODEL_IDS:
            for feature_set in FEATURE_SETS:
                planned_fits.append({
                    "fit_id": f"{split['grid']}__{split['outer_id']}__{model_id}__{feature_set['id']}",
                    "grid": split["grid"], "outer_id": split["outer_id"], "axis": split["axis"],
                    "held_out_id": split["held_out_id"], "model_id": model_id,
                    "feature_set": feature_set["id"], "feature_count": 30 + len(feature_set["context_indices"]),
                    "parent_fit_id": f"{split['outer_id']}__outer_refit__{model_id}_sensor",
                })
    if len(planned_fits) != EXPECTED_NEW_FITS:
        raise AssertionError(f"expected {EXPECTED_NEW_FITS} new fits, found {len(planned_fits)}")
    protocol_hash = sha256(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None
    return {
        "protocol_id": "pvs_context_ablation_v1", "mode": "fit_free_preflight",
        "planned_new_fits": EXPECTED_NEW_FITS, "existing_parent_fits": 48,
        "expected_merged_rows": 96,
        "outer_splits": splits, "planned_fits": planned_fits,
        "parents": parents, "parent_baseline_artifact_hashes": hashes,
        "protocol_path": str(PROTOCOL_PATH.resolve()), "protocol_sha256": protocol_hash,
        "driver_path": str(Path(__file__).resolve()), "driver_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "feature_sets": list(FEATURE_SETS),
        "estimators": {
            "rf": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                   "max_features": "sqrt", "class_weight": "balanced", "random_state": 42, "n_jobs": 4},
            "hgb": {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1,
                    "l2_regularization": 1.0, "class_weight": "balanced", "early_stopping": False,
                    "random_state": 42},
        },
        "imputation": "training-fit-index median per feature; keep all columns; no scaling",
        "split_cap_policy": "reuse parent refit train_indices_fit, train_indices_uncapped, test_indices exactly; do not regenerate maps or cap RNG",
    }


def make_model(model_id: str) -> Pipeline:
    if model_id == "rf":
        estimator = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=4)
    elif model_id == "hgb":
        estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31,
            learning_rate=0.1, l2_regularization=1.0, class_weight="balanced",
            early_stopping=False, random_state=42)
    else:
        raise ValueError(model_id)
    return Pipeline((
        ("imputer", SimpleImputer(strategy="median", fill_value=0, keep_empty_features=True)),
        ("model", estimator),
    ))


def model_input(data: dict[str, np.ndarray], rows: np.ndarray, feature_set: dict[str, Any]) -> np.ndarray:
    sensor = data["X_sensor"][rows]
    context = data["X_context"][rows][:, feature_set["context_indices"]]
    return np.asarray(np.column_stack((sensor, context)), dtype=np.float32)


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = np.unique(y_true)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_labels": labels.astype(int).tolist(),
        "class_recalls": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).astype(float).tolist(),
        "confusion_matrix_rows_true_columns_pred": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def verify_run_gate(preflight_path: Path, gate_path: Path) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"run mode requires frozen protocol: {PROTOCOL_PATH}")
    if not gate_path.is_file():
        raise FileNotFoundError(f"run mode requires root review gate: {gate_path}")
    gate = read_json(gate_path)
    expected = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "preflight_sha256": sha256(preflight_path),
    }
    if gate.get("approved") is not True:
        raise PermissionError("review_gate.json does not set approved=true")
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError(f"review gate {key} mismatch: expected {value}, got {gate.get(key)}")
    return {"review_gate_path": str(gate_path.resolve()), "review_gate_sha256": sha256(gate_path), **expected}


def run_grid(grid: str, run_dir: Path, output: Path, splits: list[dict[str, Any]],
             completed: int) -> tuple[list[dict[str, Any]], int]:
    data, manifest, _, _ = load_grid(grid, run_dir)
    fit_records = []
    for split in (s for s in splits if s["grid"] == grid):
        tr_fit = np.asarray(split["train_indices_fit"], dtype=np.int64)
        tr_uncapped = np.asarray(split["train_indices_uncapped"], dtype=np.int64)
        te = np.asarray(split["test_indices"], dtype=np.int64)
        for model_id in MODEL_IDS:
            for feature_set in FEATURE_SETS:
                fit_id = f"{grid}__{split['outer_id']}__{model_id}__{feature_set['id']}"
                fit_dir = output / "fits" / fit_id
                fit_dir.mkdir(parents=True, exist_ok=True)
                X_fit, X_test = model_input(data, tr_fit, feature_set), model_input(data, te, feature_set)
                if not np.isfinite(X_fit).any(axis=0).all():
                    raise ValueError(f"{fit_id}: a feature is entirely missing in capped parent fit rows")
                model = make_model(model_id)
                model.fit(X_fit, data["y"][tr_fit])
                y_pred = model.predict(X_test).astype(np.int64)
                model_path = fit_dir / "model.joblib"
                pred_path = fit_dir / "predictions.npz"
                joblib.dump(model, model_path, compress=3)
                np.savez_compressed(pred_path, train_indices_fit=tr_fit, train_indices_uncapped=tr_uncapped,
                    test_indices=te, y_true=data["y"][te].astype(np.int64), y_pred=y_pred)
                record = {
                    "fit_id": fit_id, "phase": "followup_ablation",
                    "candidate_id": f"{model_id}_{feature_set['id']}",
                    "grid": grid, "outer_id": split["outer_id"],
                    "axis": split["axis"], "held_out_id": split["held_out_id"],
                    "model_id": model_id, "feature_set": feature_set["id"],
                    "feature_names": [*manifest["sensor_feature_names"], *feature_set["context_names"]],
                    "feature_count": int(X_fit.shape[1]),
                    "train_rows_uncapped": int(len(tr_uncapped)), "train_rows_fit": int(len(tr_fit)),
                    "test_rows": int(len(te)), "train_indices_fit": tr_fit.tolist(),
                    "train_indices_uncapped": tr_uncapped.tolist(), "test_indices": te.tolist(),
                    "class_counts_fit": np.bincount(data["y"][tr_fit].astype(int), minlength=2).tolist(),
                    "class_counts_test": np.bincount(data["y"][te].astype(int), minlength=2).tolist(),
                    "imputer_strategy": "median", "imputer_statistics": model.named_steps["imputer"].statistics_.astype(float).tolist(),
                    "scaled": False, "estimator_params": model.named_steps["model"].get_params(),
                    "metrics": metric_pack(data["y"][te], y_pred),
                    "parent_run_dir": str(run_dir.resolve()),
                    "parent_fit_id": f"{split['outer_id']}__outer_refit__{model_id}_sensor",
                    "parent_baseline_hashes": split["parent_baseline_hashes"],
                    "model_sha256": sha256(model_path), "prediction_sha256": sha256(pred_path),
                }
                write_json(fit_dir / "fit.json", record)
                fit_records.append(record)
                completed += 1
                print(json.dumps({"completed": completed, "planned": EXPECTED_NEW_FITS,
                                  "fit_id": fit_id}, ensure_ascii=False), flush=True)
    return fit_records, completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--primary-run", type=Path, default=GRID_RUNS["primary"])
    parser.add_argument("--nonoverlap-run", type=Path, default=GRID_RUNS["nonoverlap"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.rglob("model.joblib")):
        raise FileExistsError(f"refusing to overwrite existing models under {args.output_dir}")
    preflight = build_preflight(args.primary_run, args.nonoverlap_run)
    preflight_path = args.output_dir / "preflight.json"
    write_json(preflight_path, preflight)
    print(json.dumps({"preflight": str(preflight_path.resolve()), "sha256": sha256(preflight_path),
                      "planned_new_fits": EXPECTED_NEW_FITS}, ensure_ascii=False), flush=True)
    if args.mode == "preflight":
        write_json(args.output_dir / "run_manifest.json", {
            "phase": "fit_free_preflight_complete", "planned_new_fits": EXPECTED_NEW_FITS,
            "completed_new_fits": 0, "failed_new_fits": 0,
            "preflight_sha256": sha256(preflight_path),
        })
        return
    gate_info = verify_run_gate(preflight_path, GATE_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_provenance.json", {
        "protocol_sha256": sha256(PROTOCOL_PATH), "driver_sha256": sha256(Path(__file__).resolve()),
        "preflight_sha256": sha256(preflight_path), **gate_info,
        "parents": preflight["parents"], "parent_baseline_artifact_hashes": preflight["parent_baseline_artifact_hashes"],
    })
    write_json(args.output_dir / "run_manifest.json", {
        "phase": "running", "planned_new_fits": EXPECTED_NEW_FITS,
        "completed_new_fits": 0, "failed_new_fits": 0,
        "preflight_sha256": sha256(preflight_path), "review_gate_sha256": gate_info["review_gate_sha256"],
    })
    all_records: list[dict[str, Any]] = []
    completed = 0
    for grid, run_dir in (("primary", args.primary_run), ("nonoverlap", args.nonoverlap_run)):
        records, completed = run_grid(grid, run_dir, args.output_dir, preflight["outer_splits"], completed)
        all_records.extend(records)
    if completed != EXPECTED_NEW_FITS:
        raise AssertionError(f"expected {EXPECTED_NEW_FITS} new fits; got {completed}")
    write_json(args.output_dir / "fit_records.json", {
        "planned_new_fits": EXPECTED_NEW_FITS, "completed_new_fits": completed,
        "existing_parent_fits": 48, "expected_merged_rows": 96, "records": all_records})
    write_json(args.output_dir / "run_manifest.json", {
        "phase": "phase_complete", "planned_new_fits": EXPECTED_NEW_FITS,
        "completed_new_fits": completed, "failed_new_fits": 0,
        "existing_parent_fits": 48, "expected_merged_rows": 96,
        "preflight_sha256": sha256(preflight_path), "review_gate_sha256": gate_info["review_gate_sha256"],
    })


if __name__ == "__main__":
    main()
