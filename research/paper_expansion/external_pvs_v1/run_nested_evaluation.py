#!/usr/bin/env python
"""Nested evaluation draft for the external PVS dataset.

This script is intentionally opt-in: without --run it only validates the input
schema and writes a split/preflight report. It does not download data. The
primary result is the inner-selected candidate evaluated on the outer fold;
outer results for all candidates are retained as pre-specified oracle/regret
diagnostics and must not be used to select the reported model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


PROTOCOL_ID = "external_pvs_nested_v1"
SEED = 42
CAP_LIMIT = 20_000
CAP_SEED_BASE = 42_000
MAX_JOBS = 4
CANDIDATES = (
    {"id": "rf_sensor", "estimator": "rf", "features": "sensor"},
    {"id": "rf_sensor_context", "estimator": "rf", "features": "sensor_context"},
    {"id": "hgb_sensor", "estimator": "hgb", "features": "sensor"},
    {"id": "hgb_sensor_context", "estimator": "hgb", "features": "sensor_context"},
)
INNER_SCHEMES = ("row_random", "leave_one_trip", "matched_factor")
AXES = ("vehicle", "scenario")
FIT_COUNT = 6 * len(CANDIDATES) * (2 + 6 + 2 + 1)
PROTOCOL_PATH = Path(__file__).with_name("protocol_v1.md")
FITS_COMPLETED = 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("feature_manifest.json must contain a JSON object")
    return obj


def manifest_indices(manifest: dict[str, Any], key: str, width: int, default: list[int]) -> np.ndarray:
    """Read a fixed column list; support a few explicit manifest layouts."""
    value: Any = manifest.get(key)
    if value is None and isinstance(manifest.get("feature_indices"), dict):
        value = manifest["feature_indices"].get(key)
    if value is None:
        value = default
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0 or np.any(arr < 0) or np.any(arr >= width):
        raise ValueError(f"manifest field {key!r} must be a nonempty list of valid column indices")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"manifest field {key!r} contains duplicate feature indices")
    return arr


def load_inputs(npz_path: Path, manifest_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    with np.load(npz_path, allow_pickle=False) as z:
        required = ("X_sensor", "X_context", "y", "trip", "vehicle", "scenario",
                    "source_start", "source_end", "anchor_time")
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(f"features.npz missing required arrays: {missing}")
        data = {k: np.asarray(z[k]) for k in required}
    n = len(data["y"])
    for k, a in data.items():
        if a.ndim != 1 and k not in ("X_sensor", "X_context"):
            raise ValueError(f"{k} must be one-dimensional")
        if len(a) != n:
            raise ValueError(f"{k} row count {len(a)} does not match y row count {n}")
    if data["X_sensor"].ndim != 2 or data["X_sensor"].shape[1] != 30:
        raise ValueError("X_sensor must be a two-dimensional N×30 matrix")
    if data["X_sensor"].dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("X_sensor storage must be float32 or float64; matrix_for casts model inputs to float32")
    if data["X_context"].ndim != 2 or data["X_context"].shape[1] != 3:
        raise ValueError("X_context must be an N×3 matrix (latitude, longitude, speed)")
    for key in ("X_sensor", "X_context"):
        if np.isinf(data[key]).any():
            raise ValueError(f"{key} contains ±infinity; only finite values and NaN are allowed")
    if not np.issubdtype(data["y"].dtype, np.integer):
        raise ValueError("y must use an integer dtype")
    labels = np.unique(data["y"])
    if labels.size != 2:
        raise ValueError(f"expected binary labels, observed {labels.tolist()}")
    for key in ("trip", "vehicle", "scenario", "source_start", "source_end"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"{key} must use an integer dtype")
    if np.any(data["source_end"] < data["source_start"]):
        raise ValueError("source_end must be >= source_start (inclusive raw-row intervals)")
    if not np.all(np.isfinite(data["anchor_time"])):
        raise ValueError("anchor_time contains nonfinite values")
    trips = np.unique(data["trip"])
    vehicles, scenarios = np.unique(data["vehicle"]), np.unique(data["scenario"])
    if (len(trips), len(vehicles), len(scenarios)) != (9, 3, 3):
        raise ValueError(f"expected 9 trips and 3 vehicles × 3 scenarios; got {len(trips)}, {len(vehicles)}, {len(scenarios)}")
    cells = set()
    for t in trips:
        ix = data["trip"] == t
        pairs = np.unique(np.column_stack((data["vehicle"][ix], data["scenario"][ix])), axis=0)
        if pairs.shape[0] != 1:
            raise ValueError(f"trip {int(t)} maps to multiple vehicle/scenario pairs")
        pair = tuple(map(int, pairs[0]))
        if pair in cells:
            raise ValueError(f"multiple trips map to vehicle/scenario cell {pair}")
        cells.add(pair)
    if len(cells) != 9:
        raise ValueError("the trip table is not a complete 3×3 vehicle/scenario crossing")
    sensor_ix = manifest_indices(manifest, "sensor_feature_indices", 30, list(range(30)))
    context_ix = manifest_indices(manifest, "context_feature_indices", 3, [0, 1, 2])
    if len(sensor_ix) != 30 or len(context_ix) != 3:
        raise ValueError("the frozen representation requires all 30 sensor and all 3 context columns")
    # If the upstream manifest records the feature/window recipe, reject a
    # mismatch here rather than treating a different representation as PVS.
    for field, expected in (("window_rows", 100), ("window_stride_rows", 25)):
        if field in manifest and int(manifest[field]) != expected:
            raise ValueError(f"manifest {field}={manifest[field]!r}; protocol requires {expected}")
    data["_sensor_indices"] = sensor_ix
    data["_context_indices"] = context_ix
    return data, manifest


def raw_overlap(train_idx: np.ndarray, test_idx: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, int]:
    """Count sample windows in either split intersecting a window in the other.

    Inclusive raw-row intervals are compared only within trip. The reported
    row-span count is the sum of intersection lengths over train/test pairs;
    it can count a raw row more than once where windows overlap repeatedly.
    """
    tr_hits: set[int] = set()
    te_hits: set[int] = set()
    unique_overlap_rows = 0
    trips = np.intersect1d(np.unique(data["trip"][train_idx]), np.unique(data["trip"][test_idx]))

    def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        intervals.sort()
        out = [intervals[0]]
        for lo, hi in intervals[1:]:
            prev_lo, prev_hi = out[-1]
            if lo <= prev_hi + 1:
                out[-1] = (prev_lo, max(prev_hi, hi))
            else:
                out.append((lo, hi))
        return out

    for trip in trips:
        tr = train_idx[data["trip"][train_idx] == trip]
        te = test_idx[data["trip"][test_idx] == trip]
        test_intervals = merged([(int(data["source_start"][j]), int(data["source_end"][j])) for j in te])
        train_intervals = merged([(int(data["source_start"][j]), int(data["source_end"][j])) for j in tr])
        # Every train/test sample interval is checked against the opposite
        # split's sorted union, avoiding quadratic pairwise comparisons.
        starts = np.asarray([x[0] for x in test_intervals], dtype=np.int64)
        ends = np.asarray([x[1] for x in test_intervals], dtype=np.int64)
        for ti in tr:
            lo, hi = int(data["source_start"][ti]), int(data["source_end"][ti])
            upto = int(np.searchsorted(starts, hi, side="right"))
            if upto and int(np.max(ends[:upto])) >= lo:
                tr_hits.add(int(ti))
        starts_tr = np.asarray([x[0] for x in train_intervals], dtype=np.int64)
        ends_tr = np.asarray([x[1] for x in train_intervals], dtype=np.int64)
        for tj in te:
            lo, hi = int(data["source_start"][tj]), int(data["source_end"][tj])
            upto = int(np.searchsorted(starts_tr, hi, side="right"))
            if upto and int(np.max(ends_tr[:upto])) >= lo:
                te_hits.add(int(tj))
        # Count raw rows in the intersection of the two merged interval unions.
        i = j = 0
        while i < len(train_intervals) and j < len(test_intervals):
            a0, a1 = train_intervals[i]
            b0, b1 = test_intervals[j]
            unique_overlap_rows += max(0, min(a1, b1) - max(a0, b0) + 1)
            if a1 < b1:
                i += 1
            else:
                j += 1
    return {"train_samples_with_overlap": len(tr_hits), "test_samples_with_overlap": len(te_hits),
            "unique_overlapping_raw_rows": int(unique_overlap_rows)}


def assert_binary_support(name: str, train: np.ndarray, test: np.ndarray, y: np.ndarray) -> None:
    if np.unique(y[train]).size != 2 or np.unique(y[test]).size != 2:
        raise ValueError(f"{name}: train and test must both contain both classes")


def inner_splits(outer_train: np.ndarray, data: dict[str, np.ndarray], outer_axis: str,
                 scheme: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    y = data["y"]
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    if scheme == "row_random":
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        for fold, (tr_rel, te_rel) in enumerate(cv.split(np.zeros(len(outer_train)), y[outer_train])):
            tr, te = outer_train[tr_rel], outer_train[te_rel]
            result.append((f"random_{fold + 1}", tr, te))
    elif scheme == "leave_one_trip":
        for trip in np.sort(np.unique(data["trip"][outer_train])):
            te = outer_train[data["trip"][outer_train] == trip]
            tr = outer_train[data["trip"][outer_train] != trip]
            result.append((f"trip_{int(trip)}", tr, te))
    elif scheme == "matched_factor":
        factor = "vehicle" if outer_axis == "vehicle" else "scenario"
        for value in np.sort(np.unique(data[factor][outer_train])):
            te = outer_train[data[factor][outer_train] == value]
            tr = outer_train[data[factor][outer_train] != value]
            result.append((f"{factor}_{int(value)}", tr, te))
    else:
        raise ValueError(scheme)
    covered = np.concatenate([te for _, _, te in result])
    if len(covered) != len(outer_train) or not np.array_equal(np.sort(covered), np.sort(outer_train)):
        raise AssertionError(f"{scheme} test folds do not partition the outer training rows")
    for name, tr, te in result:
        assert_binary_support(f"{scheme}/{name}/train", tr, te, y)
        if scheme != "row_random":
            if np.intersect1d(data["trip"][tr], data["trip"][te]).size:
                raise AssertionError(f"{scheme}/{name} has a shared trip")
            if raw_overlap(tr, te, data)["train_samples_with_overlap"]:
                raise AssertionError(f"{scheme}/{name} has overlapping raw input windows")
    return result


def proportional_cap(train: np.ndarray, y: np.ndarray, cap: int, rng_seed: int) -> np.ndarray:
    if len(train) <= cap:
        return np.sort(train)
    classes, counts = np.unique(y[train], return_counts=True)
    if len(classes) != 2 or np.any(counts < 1):
        raise ValueError("class-proportional cap requires both classes")
    exact = counts * (cap / counts.sum())
    quota = np.floor(exact).astype(int)
    left = cap - int(quota.sum())
    order = np.argsort(-(exact - quota), kind="stable")
    quota[order[:left]] += 1
    quota = np.minimum(quota, counts)
    # Reallocate any slots constrained by a class count, in stable remainder order.
    while int(quota.sum()) < cap:
        candidates = [i for i in range(len(classes)) if quota[i] < counts[i]]
        if not candidates:
            break
        for i in candidates:
            if quota.sum() == cap:
                break
            quota[i] += 1
    rng = np.random.default_rng(rng_seed)
    chosen = [rng.choice(train[y[train] == c], size=int(q), replace=False) for c, q in zip(classes, quota)]
    return np.sort(np.concatenate(chosen).astype(np.int64))


def matrix_for(candidate: dict[str, str], rows: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    sensor = data["X_sensor"][rows][:, data["_sensor_indices"]]
    if candidate["features"] == "sensor":
        return np.asarray(sensor, dtype=np.float32)
    context = data["X_context"][rows][:, data["_context_indices"]]
    return np.asarray(np.column_stack((sensor, context)), dtype=np.float32)


def make_model(candidate: dict[str, str]) -> Pipeline:
    if candidate["estimator"] == "rf":
        estimator = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=MAX_JOBS)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31,
            learning_rate=0.1, l2_regularization=1.0, class_weight="balanced",
            random_state=SEED, early_stopping=False)
    # Median imputation is fitted only on the uncapped? No: preprocessing is
    # learned on exactly the capped model-fit rows to preserve train-only logic.
    return Pipeline((
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True, fill_value=0)),
        ("model", estimator),
    ))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_labels": classes.astype(int).tolist(),
        "class_recalls": recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0).astype(float).tolist(),
        "confusion_matrix_rows_true_columns_pred": cm.astype(int).tolist(),
    }
    return result  # type: ignore[return-value]


def stable_fit_index(outer_index: int, inner_position: int) -> int:
    # Ten inner folds (2 + 6 + 2), followed by the outer refit per outer fold.
    return outer_index * 11 + inner_position


def describe_fold(name: str, tr: np.ndarray, te: np.ndarray, data: dict[str, np.ndarray],
                  scheme: str, seed_index: int) -> dict[str, Any]:
    assert_binary_support(name, tr, te, data["y"])
    cap = proportional_cap(tr, data["y"], CAP_LIMIT, CAP_SEED_BASE + seed_index)
    if np.unique(data["y"][cap]).size != 2:
        raise ValueError(f"{name}: cap removed class support")
    features = np.column_stack((data["X_sensor"][cap][:, data["_sensor_indices"]],
                                data["X_context"][cap][:, data["_context_indices"]]))
    if np.isinf(features).any():
        raise ValueError(f"{name}: selected features contain infinity")
    if not np.isfinite(features).any(axis=0).all():
        # Every declared feature must be observed at least once in the exact
        # capped fit rows; empty-column imputation cannot mask it.
        raise ValueError(f"{name}: at least one declared feature has no finite training value")
    return {
        "fold_id": name,
        "scheme": scheme,
        "train_rows_uncapped": int(len(tr)),
        "train_rows_fit": int(len(cap)),
        "test_rows": int(len(te)),
        "train_indices_uncapped": tr.astype(int).tolist(),
        "train_indices_fit": cap.astype(int).tolist(),
        "test_indices": te.astype(int).tolist(),
        "cap_seed": int(CAP_SEED_BASE + seed_index),
        "raw_window_overlap": raw_overlap(cap, te, data),
        "raw_window_overlap_uncapped": raw_overlap(tr, te, data),
    }


def preflight(data: dict[str, np.ndarray]) -> dict[str, Any]:
    folds = []
    outer_idx = 0
    for axis in AXES:
        factor = "vehicle" if axis == "vehicle" else "scenario"
        for held in np.sort(np.unique(data[factor])):
            test = np.flatnonzero(data[factor] == held)
            train = np.flatnonzero(data[factor] != held)
            assert_binary_support(f"outer/{axis}/{int(held)}", train, test, data["y"])
            outer = {"outer_id": f"{axis}_{int(held)}", "axis": axis, "held_out_id": int(held),
                     "train_rows": int(len(train)), "test_rows": int(len(test)),
                     "train_indices": train.astype(int).tolist(), "test_indices": test.astype(int).tolist(),
                     "raw_window_overlap": raw_overlap(train, test, data), "inner": {}}
            if outer["raw_window_overlap"]["train_samples_with_overlap"]:
                raise AssertionError(f"outer {outer['outer_id']} has raw-window overlap")
            for scheme in INNER_SCHEMES:
                entries = []
                for fold_pos, (fold_name, tr, te) in enumerate(inner_splits(train, data, axis, scheme)):
                    # Index order is fixed: random(2), leave-trip(6), matched(2).
                    position = {"row_random": 0, "leave_one_trip": 2, "matched_factor": 8}[scheme] + fold_pos
                    entries.append(describe_fold(f"{outer['outer_id']}/{scheme}/{fold_name}", tr, te,
                                                 data, scheme, stable_fit_index(outer_idx, position)))
                outer["inner"][scheme] = entries
            outer["outer_refit_cap"] = describe_fold(f"{outer['outer_id']}/outer_refit", train, test,
                data, "outer_refit", stable_fit_index(outer_idx, 10))
            folds.append(outer)
            outer_idx += 1
    if FIT_COUNT != 264:
        raise AssertionError(f"protocol fit count drifted: {FIT_COUNT}")
    return {"protocol_id": PROTOCOL_ID, "fit_count_planned": FIT_COUNT,
            "fit_matrix": {"outer_folds": 6, "candidates": [x["id"] for x in CANDIDATES],
                           "inner_folds_per_outer_candidate": {"row_random": 2, "leave_one_trip": 6, "matched_factor": 2},
                           "outer_refits_per_outer_candidate": 1},
            "primary_rule": "For each outer axis, MAE_selector = mean over its 3 outer domains of abs(selected candidate pooled inner OOF macro-F1 estimate - that same selected candidate outer-test macro-F1). Improvement = MAE_row_random - MAE_matched_factor. Support requires improvement >=0.02 on both vehicle and scenario axes; otherwise report partial support/null. Outer test scores are secondary and need not improve.",
            "outer_candidate_results_role": "pre-specified oracle/regret diagnostics; never select the reported candidate from outer test results",
            "folds": folds}


def run_one_fit(candidate: dict[str, str], train: np.ndarray, test: np.ndarray,
                data: dict[str, np.ndarray], out: Path, fit_id: str,
                fit_index: int, phase: str) -> dict[str, Any]:
    fit_rows = proportional_cap(train, data["y"], CAP_LIMIT, CAP_SEED_BASE + fit_index)
    model = make_model(candidate)
    model.fit(matrix_for(candidate, fit_rows, data), data["y"][fit_rows])
    pred = model.predict(matrix_for(candidate, test, data)).astype(np.int64)
    fit_dir = out / "fits" / fit_id
    fit_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, fit_dir / "model.joblib", compress=3)
    np.savez_compressed(fit_dir / "predictions.npz", test_indices=test.astype(np.int64),
        y_true=data["y"][test].astype(np.int64), y_pred=pred,
        train_indices_uncapped=train.astype(np.int64), train_indices_fit=fit_rows.astype(np.int64))
    global FITS_COMPLETED
    FITS_COMPLETED += 1
    record = {"fit_id": fit_id, "fit_index": int(fit_index),
        "cap_seed": int(CAP_SEED_BASE + fit_index), "phase": phase, "candidate_id": candidate["id"],
        "candidate": candidate, "train_rows_uncapped": int(len(train)), "train_rows_fit": int(len(fit_rows)),
        "test_rows": int(len(test)), "train_indices_uncapped": train.astype(int).tolist(),
        "train_indices_fit": fit_rows.astype(int).tolist(), "test_indices": test.astype(int).tolist(),
        "metrics": metrics(data["y"][test], pred),
        "raw_window_overlap": raw_overlap(fit_rows, test, data),
        "raw_window_overlap_uncapped": raw_overlap(train, test, data),
        "model_sha256": sha256_file(fit_dir / "model.joblib"),
        "prediction_sha256": sha256_file(fit_dir / "predictions.npz")}
    canonical_json(fit_dir / "fit.json", record)
    print(json.dumps({"completed_fits": FITS_COMPLETED, "planned_fits": FIT_COUNT,
                      "fit_id": fit_id, "candidate_id": candidate["id"],
                      "phase": phase, "test_rows": len(test)}, ensure_ascii=False), flush=True)
    return {**record, "_pred": pred}


def fit_all(data: dict[str, np.ndarray], manifest: dict[str, Any], npz_path: Path,
            manifest_path: Path, out: Path) -> None:
    if out.exists() and any(out.rglob("model.joblib")):
        raise FileExistsError(f"refusing to overwrite existing models under {out}")
    plan = preflight(data)
    out.mkdir(parents=True, exist_ok=True)
    canonical_json(out / "preflight.json", plan)
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"frozen protocol file is missing: {PROTOCOL_PATH}")
    canonical_json(out / "input_provenance.json", {
        "protocol_id": PROTOCOL_ID, "npz_path": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path), "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path), "manifest": manifest,
        "protocol_path": str(PROTOCOL_PATH.resolve()), "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "script_sha256": sha256_file(Path(__file__).resolve())})
    summary: list[dict[str, Any]] = []
    outer_index = 0
    for outer in plan["folds"]:
        outer_train = np.asarray(outer["train_indices"], dtype=np.int64)
        outer_test = np.asarray(outer["test_indices"], dtype=np.int64)
        selectors: dict[str, dict[str, Any]] = {}
        inner_position = 0
        # Fit all inner candidate/fold models first. Store predictions in memory
        # only for this outer fold; no outer test predictions have occurred yet.
        inner_records: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
            scheme: {c["id"]: [] for c in CANDIDATES} for scheme in INNER_SCHEMES}
        for scheme in INNER_SCHEMES:
            for fold_name, tr_list, te_list in inner_splits(outer_train, data, outer["axis"], scheme):
                tr, te = np.asarray(tr_list), np.asarray(te_list)
                for candidate in CANDIDATES:
                    fit_index = stable_fit_index(outer_index, inner_position)
                    fit_id = f"{outer['outer_id']}__{scheme}__{fold_name}__{candidate['id']}"
                    rec = run_one_fit(candidate, tr, te, data, out, fit_id, fit_index, "inner")
                    inner_records[scheme][candidate["id"]].append((te.copy(), rec.pop("_pred")))
                    summary.append(rec)
                inner_position += 1
        for scheme in INNER_SCHEMES:
            scores = {}
            for candidate in CANDIDATES:
                parts = inner_records[scheme][candidate["id"]]
                ix = np.concatenate([x[0] for x in parts])
                pred = np.concatenate([x[1] for x in parts])
                order = np.argsort(ix, kind="stable")
                if not np.array_equal(ix[order], np.sort(outer_train)):
                    raise AssertionError(f"{outer['outer_id']}/{scheme}/{candidate['id']} OOF rows incomplete")
                scores[candidate["id"]] = metrics(data["y"][ix[order]], pred[order])["macro_f1"]
            selected = max((c["id"] for c in CANDIDATES), key=lambda cid: scores[cid])
            selectors[scheme] = {"pooled_inner_oof_macro_f1": scores, "selected_candidate_id": selected,
                                 "tie_break_order": [c["id"] for c in CANDIDATES]}
        # Freeze all selectors on disk before making any outer test prediction.
        selection_path = out / "outer_folds" / outer["outer_id"] / "selection.json"
        selection = {"outer_id": outer["outer_id"], "axis": outer["axis"], "held_out_id": outer["held_out_id"],
                     "outer_train_rows": len(outer_train), "outer_test_rows": len(outer_test),
                     "selectors": selectors, "outer_predictions_not_yet_generated": True,
                     "selection_rule": "max pooled inner OOF macro-F1; fixed candidate order resolves ties"}
        canonical_json(selection_path, selection)
        frozen_selection_sha256 = sha256_file(selection_path)
        # All four outer refits are required by the frozen oracle/regret plan.
        for candidate in CANDIDATES:
            fit_id = f"{outer['outer_id']}__outer_refit__{candidate['id']}"
            rec = run_one_fit(candidate, outer_train, outer_test, data, out, fit_id,
                              stable_fit_index(outer_index, 10), "outer_oracle_diagnostic")
            rec.pop("_pred", None)
            summary.append(rec)
            # Attach each selector's chosen model's outer result to support the
            # pre-registered estimation-error comparison. This only joins the
            # already frozen selections to diagnostics; it never changes a pick.
        outer_metrics = {r["candidate_id"]: r["metrics"] for r in summary
                         if r.get("phase") == "outer_oracle_diagnostic" and
                         r.get("fit_id", "").startswith(outer["outer_id"] + "__outer_refit__")}
        for scheme, selector in selectors.items():
            chosen = selector["selected_candidate_id"]
            selector["selected_candidate_inner_macro_f1"] = selector["pooled_inner_oof_macro_f1"][chosen]
            selector["selected_candidate_outer_test_metrics"] = outer_metrics[chosen]
            selector["signed_estimation_error_macro_f1"] = (
                outer_metrics[chosen]["macro_f1"] - selector["selected_candidate_inner_macro_f1"])
            selector["absolute_estimation_error_macro_f1"] = abs(
                selector["selected_candidate_inner_macro_f1"] - outer_metrics[chosen]["macro_f1"])
        canonical_json(selection_path.parent / "selection_results.json", {
            "outer_id": outer["outer_id"],
            "frozen_selection_path": str(selection_path.resolve()),
            "frozen_selection_sha256": frozen_selection_sha256,
            "selectors": selectors,
            "outer_candidate_results_role": "pre-specified oracle/regret diagnostics"})
        outer_index += 1
    canonical_json(out / "fit_records.json", {"planned_fits": FIT_COUNT, "completed_fits": len(summary),
        "failures": [], "records": summary})
    if len(summary) != FIT_COUNT:
        raise AssertionError(f"expected {FIT_COUNT} fits, completed {len(summary)}")
    axis_errors: dict[str, dict[str, list[float]]] = {axis: {s: [] for s in ("row_random", "matched_factor")} for axis in AXES}
    for outer in plan["folds"]:
        selection_path = out / "outer_folds" / outer["outer_id"] / "selection.json"
        selection = json.loads((out / "outer_folds" / outer["outer_id"] / "selection_results.json").read_text(encoding="utf-8"))
        for scheme in axis_errors[outer["axis"]]:
            axis_errors[outer["axis"]][scheme].append(float(
                selection["selectors"][scheme]["absolute_estimation_error_macro_f1"]))
    axis_summary = {}
    for axis, schemes in axis_errors.items():
        random_mae = float(np.mean(schemes["row_random"]))
        matched_mae = float(np.mean(schemes["matched_factor"]))
        improvement = random_mae - matched_mae
        axis_summary[axis] = {"row_random_MAE": random_mae, "matched_factor_MAE": matched_mae,
                              "improvement_random_minus_matched": improvement,
                              "supported_at_0_02": bool(improvement >= 0.02),
                              "outer_domains": 3}
    supported_both = all(v["supported_at_0_02"] for v in axis_summary.values())
    replication: dict[str, Any] = {}
    for axis in AXES:
        domains = []
        axis_folds = [f for f in plan["folds"] if f["axis"] == axis]
        for outer in axis_folds:
            selection = json.loads((out / "outer_folds" / outer["outer_id"] / "selection_results.json").read_text(encoding="utf-8"))
            for candidate in CANDIDATES:
                candidate_id = candidate["id"]
                model_rec = next(r for r in summary if r.get("phase") == "outer_oracle_diagnostic" and
                    r.get("fit_id") == f"{outer['outer_id']}__outer_refit__{candidate_id}")
                random_inner = selection["selectors"]["row_random"]["pooled_inner_oof_macro_f1"][candidate_id]
                domains.append({"outer_id": outer["outer_id"], "candidate_id": candidate_id,
                    "inner_row_random_macro_f1": random_inner,
                    "outer_macro_f1": model_rec["metrics"]["macro_f1"],
                    "random_minus_outer_macro_f1": random_inner - model_rec["metrics"]["macro_f1"],
                    "outer_balanced_accuracy": model_rec["metrics"]["balanced_accuracy"],
                    "outer_class_recalls": model_rec["metrics"]["class_recalls"],
                    "outer_confusion_matrix": model_rec["metrics"]["confusion_matrix_rows_true_columns_pred"]})
        by_candidate = {}
        for candidate in CANDIDATES:
            values = [d["random_minus_outer_macro_f1"] for d in domains if d["candidate_id"] == candidate["id"]]
            by_candidate[candidate["id"]] = {"domain_values": values, "mean_random_minus_outer_macro_f1": float(np.mean(values))}
        replication[axis] = {"domains": domains, "by_candidate": by_candidate}
    canonical_json(out / "primary_estimation_error.json", {
        "definition": "MAE is mean across 3 outer domains of abs(selected candidate's pooled inner OOF macro-F1 estimate - same candidate outer-test macro-F1); improvement is row-random MAE minus matched-factor MAE.",
        "axis_results": axis_summary, "supported_both_axes": supported_both,
        "interpretation": "supports both vehicle and scenario domains" if supported_both else "partial support/null; at least one axis is below the preregistered 0.02 threshold",
        "estimation_errors_by_outer_domain": {f["outer_id"]: json.loads((out / "outer_folds" / f["outer_id"] / "selection_results.json").read_text(encoding="utf-8"))["selectors"] for f in plan["folds"]},
        "replication_endpoint_random_minus_outer": replication})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True,
                        help="separate destination for fit-free split/schema validation")
    parser.add_argument("--output-dir", type=Path,
                        help="separate destination for model fits; required with --run")
    parser.add_argument("--run", action="store_true", help="fit the frozen 264-model matrix; default is fit-free preflight")
    args = parser.parse_args()
    if args.run and args.output_dir is None:
        parser.error("--output-dir is required with --run")
    if args.run and args.preflight_dir.resolve() == args.output_dir.resolve():
        parser.error("--preflight-dir and --output-dir must be different")
    data, manifest = load_inputs(args.input_npz, args.feature_manifest)
    plan = preflight(data)
    args.preflight_dir.mkdir(parents=True, exist_ok=True)
    canonical_json(args.preflight_dir / "preflight.json", plan)
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"frozen protocol file is missing: {PROTOCOL_PATH}")
    provenance = {
        "protocol_id": PROTOCOL_ID, "npz_path": str(args.input_npz.resolve()),
        "npz_sha256": sha256_file(args.input_npz), "manifest_path": str(args.feature_manifest.resolve()),
        "manifest_sha256": sha256_file(args.feature_manifest), "manifest": manifest,
        "protocol_path": str(PROTOCOL_PATH.resolve()), "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()), "mode": "fit" if args.run else "fit_free_preflight"}
    canonical_json(args.preflight_dir / "input_provenance.json", provenance)
    if args.run:
        fit_all(data, manifest, args.input_npz, args.feature_manifest, args.output_dir)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "mode": "fit" if args.run else "fit_free_preflight",
                      "planned_fits": FIT_COUNT, "outer_folds": len(plan["folds"]),
                      "preflight": str(args.preflight_dir / "preflight.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
