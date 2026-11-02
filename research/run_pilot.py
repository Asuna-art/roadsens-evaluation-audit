from __future__ import annotations

"""Frozen RoadSens first-pass runner. Formal mode is deliberately opt-in."""
import argparse
from collections import deque
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "inner"
CLASSES = ["Normal", "Pothole", "Bump"]
SENSORS = ("accelerometer_", "gyroscope_", "gravity_", "totalAcceleration_")
FORBIDDEN = {"seconds_elapsed", "annotation_text", "annotation_millisecond_press_duration"}
RF_CONFIG = dict(n_estimators=100, max_depth=20, min_samples_leaf=5,
                 max_features="sqrt", class_weight="balanced", random_state=42,
                 n_jobs=4)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    out = np.full((len(df), len(columns)), np.nan, dtype=np.float64)
    for j, col in enumerate(columns):
        if col in df:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            v[~np.isfinite(v)] = np.nan
            out[:, j] = v
    return out


def numeric_columns(headers: set[str]) -> list[str]:
    # Schema is taken from sensor fields only; labels, timing, and sparse press actions never enter.
    return sorted(c for c in headers if c not in FORBIDDEN and c != "session_id")


def _gravity_and_motion(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def vec(prefix: str) -> np.ndarray:
        cols = [f"{prefix}{a}" for a in "xyz"]
        return _finite_matrix(df, cols)
    acc, grav, gyro = vec("accelerometer_"), vec("gravity_"), vec("gyroscope_")
    gnorm = np.linalg.norm(grav, axis=1)
    unit = np.divide(grav, gnorm[:, None], out=np.zeros_like(grav), where=gnorm[:, None] > 1e-12)
    accel_mag = np.linalg.norm(acc, axis=1)
    projection = np.einsum("ij,ij->i", acc, unit)
    horizontal = np.sqrt(np.maximum(accel_mag * accel_mag - projection * projection, 0.0))
    gyro_mag = np.linalg.norm(gyro, axis=1)
    raw = np.column_stack([projection, horizontal, accel_mag, gyro_mag])
    raw[(~np.isfinite(raw)).any(axis=1) | (~np.isfinite(gnorm)) | (gnorm <= 1e-12)] = np.nan
    return raw, projection, unit


def causal_window_features(times: np.ndarray, values: np.ndarray, window_seconds: float = 1.0) -> np.ndarray:
    """Past-inclusive time window on stable (time, original-row) order; never crosses sessions."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n, d = values.shape
    result = np.full((n, d * 5), np.nan, dtype=np.float64)
    order = np.flatnonzero(np.isfinite(times))
    # Stable time sort. Insert one row at a time, so later rows with the same timestamp
    # cannot leak into earlier rows at that timestamp.
    order = order[np.argsort(times[order], kind="stable")]
    queue_min = [deque() for _ in range(d)]
    queue_max = [deque() for _ in range(d)]
    queue_abs = [deque() for _ in range(d)]
    sums = np.zeros(d, dtype=np.float64); sums2 = np.zeros(d, dtype=np.float64)
    counts = np.zeros(d, dtype=np.int64)
    left = 0
    for k, original_i in enumerate(order):
        now = times[original_i]
        while left <= k and times[order[left]] < now - window_seconds:
            old_i = order[left]
            for j in range(d):
                old = values[old_i, j]
                if np.isfinite(old):
                    sums[j] -= old; sums2[j] -= old * old; counts[j] -= 1
                    if queue_min[j] and queue_min[j][0][0] == left: queue_min[j].popleft()
                    if queue_max[j] and queue_max[j][0][0] == left: queue_max[j].popleft()
                    if queue_abs[j] and queue_abs[j][0][0] == left: queue_abs[j].popleft()
            left += 1
        for j in range(d):
            value = values[original_i, j]
            if np.isfinite(value):
                sums[j] += value; sums2[j] += value * value; counts[j] += 1
                while queue_min[j] and queue_min[j][-1][1] >= value: queue_min[j].pop()
                queue_min[j].append((k, value))
                while queue_max[j] and queue_max[j][-1][1] <= value: queue_max[j].pop()
                queue_max[j].append((k, value))
                av = abs(value)
                while queue_abs[j] and queue_abs[j][-1][1] <= av: queue_abs[j].pop()
                queue_abs[j].append((k, av))
            if counts[j]:
                mean = sums[j] / counts[j]
                variance = max(0.0, sums2[j] / counts[j] - mean * mean)
                result[original_i, j * 5:(j + 1) * 5] = (
                    mean, math.sqrt(variance), math.sqrt(max(0.0, sums2[j] / counts[j])),
                    queue_max[j][0][1] - queue_min[j][0][1], queue_abs[j][0][1]
                )
    return result


def label_rows(df: pd.DataFrame, group: str, blank_normal: bool) -> np.ndarray:
    if group == "normal":
        return np.full(len(df), "Normal", dtype=object)
    if "annotation_text" not in df:
        return np.full(len(df), None, dtype=object)
    raw = df["annotation_text"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
    out = np.full(len(df), None, dtype=object)
    for cls in ("Pothole", "Bump"):
        out[raw == cls] = cls
    if blank_normal:
        out[raw == ""] = "Normal"
    return out


def _session_group(path: Path) -> str:
    return "normal" if "combined_normal" in str(path).lower() else "anomaly_recording_labeled"


def read_sessions(blank_normal: bool = False) -> tuple[list[dict], list[str]]:
    paths = sorted((DATA / "combined_normal").glob("*.csv")) + sorted(
        p for p in (DATA / "combined_anomalies").glob("*.csv") if p.name != "99.csv")
    sessions = []
    # Establish one schema before materializing any feature matrix. Absent sensor fields
    # become NaN columns, keeping train/test and sessions dimensionally aligned.
    headers = set()
    for p in paths:
        if "seconds_elapsed" in pd.read_csv(p, nrows=0).columns:
            headers.update(pd.read_csv(p, nrows=0).columns)
    all_numeric = numeric_columns(headers)
    imu_cols = [c for c in all_numeric if any(c.startswith(p) for p in SENSORS)
                and "Uncalibrated" not in c]
    context_cols = [c for c in all_numeric if c.startswith("location_") or c == "GIS_Elevation"
                    or c in {"Temperature (°C)", "Humidity (%)", "Cloud Cover (%)", "Wind Speed (km/h)"}]
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        if "seconds_elapsed" not in df:
            continue
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        times = pd.to_numeric(df["seconds_elapsed"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        times[~np.isfinite(times)] = np.nan
        labels = label_rows(df, _session_group(path), blank_normal=False)
        labels_blank_normal = label_rows(df, _session_group(path), blank_normal=True)

        # Features and labels are derived at the original cadence, then every tenth source row is kept.
        numeric = _finite_matrix(df, all_numeric)
        raw_physics, _, _ = _gravity_and_motion(df)
        window = causal_window_features(times, raw_physics)
        keep_idx = np.arange(0, len(df), 10, dtype=np.int64)
        sessions.append({
            "path": str(path), "sha256": sha256_file(path), "session": path.name,
            "session_key": path.relative_to(DATA).as_posix(),
            "group": _session_group(path), "times": times[keep_idx], "labels": labels[keep_idx],
            "labels_blank_normal": labels_blank_normal[keep_idx],
            "numeric": numeric[keep_idx], "numeric_columns": all_numeric,
            "imu": _finite_matrix(df, imu_cols)[keep_idx], "imu_columns": imu_cols,
            "context": _finite_matrix(df, context_cols)[keep_idx], "context_columns": context_cols,
            "missing": np.isfinite(numeric)[keep_idx].astype(np.float32),
            "physics_raw": raw_physics[keep_idx], "physics_window": window[keep_idx],
            "source_row": keep_idx, "all_rows": len(df),
            "times_full": times, "physics_raw_full": raw_physics,
            "numeric_full": numeric,
        })
    return sessions, sorted(headers)


def _concat(sessions: list[dict], field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([s[field] for s in sessions], axis=0)
    y = np.concatenate([s["labels"] for s in sessions])
    groups = np.concatenate([np.full(len(s["labels"]), s["session_key"], dtype=object) for s in sessions])
    owner = np.concatenate([np.full(len(s["labels"]), i, dtype=np.int32) for i, s in enumerate(sessions)])
    all_times = np.concatenate([s["times"] for s in sessions])
    valid = np.isin(y, CLASSES) & np.isfinite(all_times)
    x, y, groups, owner = x[valid], y[valid], groups[valid], owner[valid]
    return x, y, groups, owner


def _impute_train_test(xtrain: np.ndarray, xtest: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
    keep = np.any(np.isfinite(xtrain), axis=0)
    kept = np.flatnonzero(keep).tolist()
    a, b = xtrain[:, keep].copy(), xtest[:, keep].copy()
    if not kept:
        return np.zeros((len(a), 1)), np.zeros((len(b), 1)), [], []
    med = np.nanmedian(a, axis=0)
    med[~np.isfinite(med)] = 0.0
    ai, bi = np.where(np.isfinite(a), a, med), np.where(np.isfinite(b), b, med)
    return ai, bi, kept, med.astype(float).tolist()


def _physics_calibrated_features(sessions: list[dict], train_ids: np.ndarray, test_ids: np.ndarray,
                                 all_y: np.ndarray, all_owner: np.ndarray, all_times: np.ndarray,
                                 all_raw: np.ndarray, all_source_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Fold defaults are estimated only from fit rows. Prefix parameters use unlabeled session history,
    # and are applied only after the first three elapsed seconds have actually passed.
    train_rows = np.asarray(train_ids, dtype=np.int64)
    fit = all_raw[train_rows]
    center = np.nanmedian(fit, axis=0)
    mad = np.nanmedian(np.abs(fit - center), axis=0)
    center[~np.isfinite(center)] = 0.0
    mad[~np.isfinite(mad)] = 0.0
    floor = np.maximum(0.1 * mad, 1e-6)
    global_scale = np.maximum(mad, floor)
    transformed = np.full((len(all_raw), 20), np.nan, dtype=np.float64)
    for sid, s in enumerate(sessions):
        rows = np.flatnonzero(all_owner == sid)
        if not len(rows):
            continue
        source = all_source_rows[rows]
        full_t = s["times_full"]
        full_raw = s["physics_raw_full"]
        finite_t = np.isfinite(full_t)
        start = np.min(full_t[finite_t]) if np.any(finite_t) else 0.0
        prefix = (full_t >= start) & (full_t < start + 3.0) & finite_t
        pc = np.nanmedian(full_raw[prefix], axis=0) if np.any(prefix) else center.copy()
        pm = np.nanmedian(np.abs(full_raw[prefix] - pc), axis=0) if np.any(prefix) else mad.copy()
        pc[~np.isfinite(pc)] = center[~np.isfinite(pc)]
        pm[~np.isfinite(pm)] = 0.0
        ps = np.maximum(pm, floor)
        full_cal = np.full_like(full_raw, np.nan)
        before = finite_t & (full_t < start + 3.0)
        after = finite_t & (full_t >= start + 3.0)
        full_cal[before] = (full_raw[before] - center) / global_scale
        full_cal[after] = (full_raw[after] - pc) / ps
        full_window = causal_window_features(full_t, full_cal)
        transformed[rows] = full_window[source]
    out = transformed
    return out[train_ids], out[test_ids], {"default_center": center.tolist(),
                                           "default_train_mad": mad.tolist(),
                                           "default_scale": global_scale.tolist(),
                                           "prefix_scale_floor": floor.tolist()}


def _build_folds(y: np.ndarray, groups: np.ndarray, split_kind: str,
                 group_fold_map: dict[str, int] | None = None):
    from sklearn.model_selection import StratifiedKFold
    if split_kind == "random":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return list(splitter.split(np.zeros(len(y)), y))
    if group_fold_map is None or set(np.unique(groups)) - set(group_fold_map):
        raise ValueError("group split requires a complete frozen session_fold_map_v1.json")
    folds = []
    for fold_id in range(1, 6):
        test = np.flatnonzero(np.fromiter((group_fold_map[g] == fold_id for g in groups), dtype=bool, count=len(groups)))
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        folds.append((train, test))
    return folds


def load_frozen_fold_map(path: Path | None = None) -> dict:
    path = path or (ROOT / "session_fold_map_v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("session_count") != 101:
        raise ValueError("Unexpected frozen session-fold map version or session count")
    fmap = payload.get("session_fold", {})
    if set(fmap.values()) != {1, 2, 3, 4, 5} or len(fmap) != 101:
        raise ValueError("Fold map must assign all 101 sessions exactly among folds 1..5")
    if set(payload.get("source_sha256", {})) != set(fmap):
        raise ValueError("Fold-map hashes and session keys differ")
    return payload


def validate_fold_map_labels(payload: dict) -> dict:
    """Independent runner-side label/count check; reads no feature values and fits no model."""
    fmap = payload["session_fold"]
    expected_hash = payload["source_sha256"]
    paths = sorted((DATA / "combined_normal").glob("*.csv")) + sorted(
        p for p in (DATA / "combined_anomalies").glob("*.csv") if p.name != "99.csv")
    primary_by_fold = {i: {c: 0 for c in CLASSES} for i in range(1, 6)}
    sensitive_by_fold = {i: {c: 0 for c in CLASSES} for i in range(1, 6)}
    primary_total = {c: 0 for c in CLASSES}; sensitive_total = {c: 0 for c in CLASSES}
    actual_hash = {}
    for path in paths:
        key = path.relative_to(DATA).as_posix()
        if key not in fmap:
            raise ValueError(f"source CSV absent from frozen fold map: {key}")
        digest = sha256_file(path); actual_hash[key] = digest
        if digest != expected_hash[key]:
            raise ValueError(f"source changed after fold map was frozen: {key}")
        df = pd.read_csv(path, usecols=lambda c: c in {"seconds_elapsed", "annotation_text"}, low_memory=False)
        times = pd.to_numeric(df["seconds_elapsed"], errors="coerce").to_numpy(dtype=np.float64)
        keep = np.arange(0, len(df), 10, dtype=np.int64)
        finite = np.isfinite(times[keep]); fold = int(fmap[key])
        if path.parent.name == "combined_normal":
            primary = np.full(len(keep), "Normal", dtype=object)
            sensitivity = primary.copy()
        else:
            labels = df["annotation_text"].fillna("").astype(str).str.strip().to_numpy(dtype=object) if "annotation_text" in df else np.full(len(df), "", dtype=object)
            labels = labels[keep]
            primary = np.where(np.isin(labels, ["Pothole", "Bump"]), labels, None)
            sensitivity = primary.copy()
            sensitivity[labels == ""] = "Normal"
        for label, counts in ((primary, primary_by_fold[fold]), (sensitivity, sensitive_by_fold[fold])):
            valid = finite & np.isin(label, CLASSES)
            for c in CLASSES:
                amount = int(np.sum(label[valid] == c)); counts[c] += amount
                (primary_total if counts is primary_by_fold[fold] else sensitive_total)[c] += amount
    if set(actual_hash) != set(fmap):
        raise ValueError("Local source session keys do not equal the frozen fold-map keys")
    test_counts = {"primary": primary_by_fold, "blank_normal_sensitivity": sensitive_by_fold}
    total_counts = {"primary": primary_total, "blank_normal_sensitivity": sensitive_total}
    for policy, counts_by_fold in test_counts.items():
        declared = payload["revised_counts"][policy]
        for fold in range(1, 6):
            expected = declared[str(fold)]["test"]["class_rows"]
            if counts_by_fold[fold] != expected:
                raise ValueError(f"fold-map label count mismatch: {policy}, fold {fold}: {counts_by_fold[fold]} != {expected}")
            if any(counts_by_fold[fold][c] <= 0 for c in CLASSES):
                raise ValueError(f"frozen fold {fold} has an empty class under {policy}")
            expected_train = declared[str(fold)]["train"]["class_rows"]
            got_train = {c: total_counts[policy][c] - counts_by_fold[fold][c] for c in CLASSES}
            if got_train != expected_train or any(got_train[c] <= 0 for c in CLASSES):
                raise ValueError(f"fold-map train count mismatch: {policy}, fold {fold}: {got_train} != {expected_train}")
    actual_counts = {p: {str(k): {"train": {c: total_counts[p][c] - v[c] for c in CLASSES}, "test": v}
                         for k, v in folds.items()} for p, folds in test_counts.items()}
    return {"passed": True, "session_count": len(actual_hash), "fold_count": 5,
            "class_counts": actual_counts,
            "source_hashes_match": True, "map_sha256": sha256_file(ROOT / "session_fold_map_v1.json"),
            "model_fit_or_scores": False}


def _cap_train(train: np.ndarray, y: np.ndarray, fold: int, cap: int = 20000) -> np.ndarray:
    if len(train) <= cap:
        return train
    rng = np.random.RandomState(42000 + fold)
    chosen = []
    classes = np.unique(y[train])
    counts = {c: int(np.sum(y[train] == c)) for c in classes}
    total = sum(counts.values())
    quota = {c: min(counts[c], max(1, int(cap * counts[c] / total))) for c in classes}
    while sum(quota.values()) > cap:
        c = max(quota, key=quota.get)
        quota[c] -= 1
    for c in classes:
        ix = train[y[train] == c]
        chosen.extend(rng.choice(ix, quota[c], replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    present = [c for c in CLASSES if np.any(y_true == c)]
    return {"macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
            "observed_class_macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
            "class_support": {c: int(np.sum(y_true == c)) for c in CLASSES},
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "recall_by_class": {c: float(v) for c, v in zip(CLASSES, recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0))},
            "confusion_matrix_labels": CLASSES,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()}


def _run_policy(sessions: list[dict], policy: str, split_kind: str, output: Path,
                group_fold_map: dict[str, int], methods: list[str] | None = None) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    methods = methods or ["full_numeric", "missing_only", "context_only", "imu_only", "physics_window", "physics_calibrated"]
    # Re-load labels according to policy; arrays were constructed before label filtering, with valid rows only.
    blank_normal = policy == "blank_normal"
    # Use the same row features and frozen session assignment for both label policies.
    if blank_normal:
        sessions = [{**s, "labels": s["labels_blank_normal"]} for s in sessions]
    fields = {m: ("numeric" if m == "full_numeric" else "missing" if m == "missing_only" else
                  "context" if m == "context_only" else "imu" if m == "imu_only" else
                  "physics_window" if m == "physics_window" else "physics_raw") for m in methods}
    all_results = {}
    for method in methods:
        print(f"START {policy}/{split_kind}/{method}", flush=True)
        x, y, groups, owner = _concat(sessions, fields[method])
        times = np.concatenate([s["times"] for s in sessions])
        raw = np.concatenate([s["physics_raw"] for s in sessions])
        source_rows = np.concatenate([s["source_row"] for s in sessions])
        all_labels = np.concatenate([s["labels"] for s in sessions])
        valid = np.isin(all_labels, CLASSES) & np.isfinite(np.concatenate([s["times"] for s in sessions]))
        times, raw, source_rows = times[valid], raw[valid], source_rows[valid]
        folds = _build_folds(y, groups, split_kind, group_fold_map)
        method_results = []
        for fold, (tr0, te) in enumerate(folds):
            print(f"START {policy}/{split_kind}/{method} fold={fold + 1}/5 train={len(tr0)} test={len(te)}", flush=True)
            if set(np.unique(y[tr0])) != set(CLASSES) or set(np.unique(y[te])) != set(CLASSES):
                raise RuntimeError(f"Fold {fold} lacks one or more classes; frozen protocol requires stop, no reseed")
            tr = _cap_train(tr0, y, fold)
            if method == "physics_calibrated":
                rawtr, rawte, calibration_meta = _physics_calibrated_features(sessions, tr, te, y, owner, times, raw, source_rows)
                xtr, xte, kept, imputation_medians = _impute_train_test(rawtr, rawte)
            elif method == "missing_only":
                # The mask itself is the feature. Remove only fields with no observed training value.
                raw_numeric, _, _, _ = _concat(sessions, "numeric")
                keep_mask = np.any(np.isfinite(raw_numeric[tr]), axis=0)
                kept = np.flatnonzero(keep_mask).tolist()
                imputation_medians = []
                calibration_meta = None
                xtr, xte = x[tr][:, keep_mask], x[te][:, keep_mask]
                if xtr.shape[1] == 0:
                    xtr, xte = np.zeros((len(tr), 1)), np.zeros((len(te), 1))
            else:
                xtr, xte, kept, imputation_medians = _impute_train_test(x[tr], x[te])
                calibration_meta = None
            model = RandomForestClassifier(**RF_CONFIG)
            t0 = time.perf_counter(); model.fit(xtr, y[tr]); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = model.predict(xte); predict_s = time.perf_counter() - t0
            model_path = output.with_name(f"{output.stem}.{method}.fold{fold + 1}.joblib")
            if model_path.exists():
                raise FileExistsError(f"Refusing to overwrite fitted model {model_path}")
            import joblib
            t0 = time.perf_counter(); joblib.dump(model, model_path, compress=3); save_s = time.perf_counter() - t0
            model_hash = sha256_file(model_path)
            result = _metrics(y[te], pred)
            session_perf = {}
            for sess in np.unique(groups[te]):
                ix = groups[te] == sess
                session_perf[str(sess)] = {"n": int(ix.sum()), **_metrics(y[te][ix], pred[ix])}
            fold_record = {"fold": fold, "train_n": int(len(tr)), "train_n_before_cap": int(len(tr0)),
                           "test_n": int(len(te)), "train_class_counts": {c: int(np.sum(y[tr] == c)) for c in CLASSES},
                           "test_class_counts": {c: int(np.sum(y[te] == c)) for c in CLASSES},
                           **result, "per_session": session_perf, "fit_seconds": fit_s, "predict_seconds": predict_s,
                           "model_artifact": model_path.name, "model_sha256": model_hash, "model_save_seconds": save_s,
                           "kept_feature_indices": kept,
                           "training_imputation_medians": imputation_medians,
                           "calibration_default_parameters": calibration_meta,
                           "split_train_rows": [{"row": int(i), "session": str(groups[i]), "source_row": int(source_rows[i])} for i in tr0],
                           "fit_train_rows": [{"row": int(i), "session": str(groups[i]), "source_row": int(source_rows[i])} for i in tr],
                           "test_rows": [{"row": int(i), "session": str(groups[i]), "source_row": int(source_rows[i])} for i in te],
                           "predictions": [{"row": int(i), "source_row": int(source_rows[i]),
                                            "session": str(groups[i]), "true": str(y[i]), "pred": str(p)}
                                          for i, p in zip(te, pred)]}
            method_results.append(fold_record)
            print(f"DONE {policy}/{split_kind}/{method} fold={fold + 1}/5 macro_f1={result['macro_f1']:.4f} fit_s={fit_s:.1f} predict_s={predict_s:.1f}", flush=True)
        all_results[method] = method_results
        checkpoint_path = output.with_name(output.stem + f".{method}.checkpoint.json")
        checkpoint_path.write_text(json.dumps({"policy": policy, "split": split_kind, "method": method,
                                               "rf_config": RF_CONFIG, "folds": method_results},
                                              indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"DONE {policy}/{split_kind}/{method} checkpoint={checkpoint_path.name}", flush=True)
    payload = {"policy": policy, "split": split_kind, "methods": all_results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def formal_run(output_dir: Path, sessions_cache_path: Path | None = None) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    fold_payload = load_frozen_fold_map()
    label_validation = validate_fold_map_labels(fold_payload)
    output_dir.mkdir(parents=True, exist_ok=False)
    if sessions_cache_path is None:
        sessions, headers = read_sessions(blank_normal=False)
    else:
        import joblib
        cache = joblib.load(sessions_cache_path)
        if cache.get("runner_sha256") != sha256_file(ROOT / "run_pilot.py"):
            raise ValueError("precomputed session cache was built from a different runner version")
        if cache.get("source_sha256") != fold_payload["source_sha256"]:
            raise ValueError("precomputed session cache source hashes differ from frozen map")
        sessions, headers = cache["sessions"], cache["headers"]
    if set(s["session_key"] for s in sessions) != set(fold_payload["session_fold"]):
        raise ValueError("loaded sessions differ from frozen fold map")
    if any(s["sha256"] != fold_payload["source_sha256"][s["session_key"]] for s in sessions):
        raise ValueError("loaded source hashes differ from frozen fold map")
    (output_dir / "session_split_validation.json").write_text(
        json.dumps(label_validation, indent=2, ensure_ascii=False), encoding="utf-8")
    schemas = {
        "full_numeric": numeric_columns(set(headers)),
        "missing_only": ["finite:" + c for c in numeric_columns(set(headers))],
        "context_only": sessions[0]["context_columns"],
        "imu_only": sessions[0]["imu_columns"],
        "physics_window": [f"{v}_{s}" for v in ("gravity_projection", "horizontal_linear_mag", "linear_acc_mag", "gyro_mag")
                           for s in ("mean", "std", "rms", "range", "absmax")],
        "physics_calibrated": [f"{v}_{s}_prefix3s" for v in ("gravity_projection", "horizontal_linear_mag", "linear_acc_mag", "gyro_mag")
                               for s in ("mean", "std", "rms", "range", "absmax")],
    }
    files = [ROOT / "pilot_protocol.md", ROOT / "pre_run_clarifications.md",
             ROOT / "fold_feasibility_amendment.md", ROOT / "session_fold_map_v1.json", ROOT / "run_pilot.py"]
    inv = ROOT / "session_inventory_v2.csv"
    if inv.exists(): files.append(inv)
    manifest = {"created_epoch": time.time(), "seed": 42, "rf_config": RF_CONFIG, "threads": 4,
                "source_files": [{"path": str(p), "sha256": sha256_file(p)} for p in files],
                "sessions": [{"path": s["path"], "sha256": s["sha256"], "session": s["session"],
                              "session_key": s["session_key"], "group": s["group"],
                              "retained_rows": int(len(s["labels"])), "source_rows": s["all_rows"]} for s in sessions],
                "schema": schemas, "frozen_session_fold": fold_payload["session_fold"],
                "session_fold_map_sha256": sha256_file(ROOT / "session_fold_map_v1.json"),
                "formal_metrics_started": True}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # No post-hoc tuning. One directory per policy/split; exclusive creation prevents accidental overwrites.
    for policy in ("primary", "blank_normal"):
        for split_kind in ("random", "group"):
            target = output_dir / f"{policy}_{split_kind}.json"
            _run_policy(sessions, policy, split_kind, target, fold_payload["session_fold"])


def run_synthetic_qa() -> dict:
    # Rolling cannot cross sessions: a huge prior value from session A must not enter B.
    t = np.array([0., .5, 1., 0., .5, 1.])
    v = np.array([[1.], [1.], [1.], [1000.], [2.], [2.]])
    a = causal_window_features(t[:3], v[:3])
    b = causal_window_features(t[3:], v[3:])
    assert np.allclose(a[0, 0], 1.) and np.allclose(b[0, 0], 1000.)
    assert not np.allclose(b[1, 0], np.mean(np.r_[v[:3, 0], v[4, 0]]))
    # A future impulse must not change an earlier feature.
    t2 = np.array([0., .5, 1., 1.5]); z = np.array([[0.], [0.], [0.], [999.]])
    before = causal_window_features(t2, z)
    z[-1] = -999.
    after = causal_window_features(t2, z)
    assert np.allclose(before[:3], after[:3], equal_nan=True)
    # Independent direct reference for unordered source times, duplicates, NaNs, and empty gaps.
    rt = np.array([1., 0., 1., 2.1, 5., np.nan])
    rv = np.array([[2., np.nan], [1., 3.], [4., 5.], [np.nan, 8.], [9., 10.], [100., 100.]])
    got = causal_window_features(rt, rv)
    want = np.full_like(got, np.nan)
    order = np.flatnonzero(np.isfinite(rt)); order = order[np.argsort(rt[order], kind="stable")]
    seen = []
    for idx in order:
        seen.append(idx)
        rows = [j for j in seen if rt[j] >= rt[idx] - 1.0]
        for dim in range(rv.shape[1]):
            xx = rv[rows, dim]; xx = xx[np.isfinite(xx)]
            if xx.size:
                want[idx, dim * 5:(dim + 1) * 5] = [xx.mean(), xx.std(ddof=0),
                    np.sqrt(np.mean(xx * xx)), xx.max() - xx.min(), np.max(np.abs(xx))]
    assert np.allclose(got, want, equal_nan=True, atol=1e-12)
    # A press-duration or label-like numeric must not be in candidate features.
    cols = numeric_columns({"seconds_elapsed", "annotation_text", "annotation_millisecond_press_duration",
                            "accelerometer_x", "location_latitude"})
    assert cols == ["accelerometer_x", "location_latitude"]
    # Label-policy split: mixed blank unknown in primary, Normal in sensitivity; explicit labels identical.
    toy = pd.DataFrame({"annotation_text": ["Pothole", "", "Bump"]})
    primary = label_rows(toy, "anomaly_recording_labeled", False)
    blank = label_rows(toy, "anomaly_recording_labeled", True)
    assert primary.tolist() == ["Pothole", None, "Bump"]
    assert blank.tolist() == ["Pothole", "Normal", "Bump"]
    # Missing schema is numeric-only and label/duration-like fields cannot enter the mask.
    left = _finite_matrix(pd.DataFrame({"accelerometer_x": [1.0], "location_latitude": [np.nan]}),
                         ["accelerometer_x", "location_latitude"])
    right = _finite_matrix(pd.DataFrame({"accelerometer_x": [2.0]}),
                           ["accelerometer_x", "location_latitude"])
    assert left.shape == right.shape == (1, 2) and np.isnan(right[0, 1])
    assert np.array_equal(np.isfinite(left).astype(np.float32), [[1., 0.]])
    # Calibration consumes the full unlabeled source prefix, is session-local, and does not read y.
    ts = np.arange(9, dtype=float) * 0.5
    raw_a = np.column_stack([ts + np.arange(9), ts * 0 + 1, ts + 2, ts + 3])
    raw_b = raw_a + np.array([100., 50., 25., 10.])
    fake_sessions = [{"times_full": ts, "physics_raw_full": raw_a},
                     {"times_full": ts, "physics_raw_full": raw_b}]
    all_raw = np.concatenate([raw_a, raw_b]); all_times = np.tile(ts, 2)
    all_owner = np.repeat([0, 1], 9); all_src = np.tile(np.arange(9), 2)
    all_y = np.array(["Normal"] * 9 + ["Bump"] * 9, dtype=object)
    f1 = _physics_calibrated_features(fake_sessions, np.arange(9), np.arange(9, 18), all_y,
                                      all_owner, all_times, all_raw, all_src)[1]
    f2 = _physics_calibrated_features(fake_sessions, np.arange(9), np.arange(9, 18), all_y[::-1],
                                      all_owner, all_times, all_raw, all_src)[1]
    assert np.allclose(f1, f2, equal_nan=True)
    raw_b_late = raw_b.copy(); raw_b_late[-1] += 1e9
    altered_sessions = [fake_sessions[0], {"times_full": ts, "physics_raw_full": raw_b_late}]
    altered_raw = np.concatenate([raw_a, raw_b_late])
    f3 = _physics_calibrated_features(altered_sessions, np.arange(9), np.arange(9, 18), all_y,
                                      all_owner, all_times, altered_raw, all_src)[1]
    assert np.allclose(f1[:8], f3[:8], equal_nan=True)
    # Projection is along normalized gravity; horizontal magnitude is orthogonal component.
    toy_axes = pd.DataFrame({"accelerometer_x": [3.0], "accelerometer_y": [4.0], "accelerometer_z": [0.0],
                             "gravity_x": [0.0], "gravity_y": [0.0], "gravity_z": [2.0],
                             "gyroscope_x": [3.0], "gyroscope_y": [4.0], "gyroscope_z": [0.0]})
    toy_raw, _, _ = _gravity_and_motion(toy_axes)
    assert np.allclose(toy_raw[0], [0., 5., 5., 5.])
    toy_axes.loc[0, "gravity_z"] = 0.0
    bad_gravity, _, _ = _gravity_and_motion(toy_axes)
    assert np.isnan(bad_gravity).all()
    return {"passed": True, "checks": ["session-local rolling isolation", "past-only window and future perturbation",
                                        "forbidden columns excluded", "primary vs blank-normal labels",
                                        "global schema alignment and finite masks", "calibration full-prefix/no test-label access",
                                        "future raw extremes do not alter prior calibrated features",
                                        "gravity-invalid rows become missing", "naive rolling parity with duplicate times, NaNs, and gaps"],
            "note": "Synthetic-only QA; no RoadSens model fitting or outcome scoring was performed."}


def run_split_qa() -> dict:
    payload = load_frozen_fold_map()
    result = validate_fold_map_labels(payload)
    out = ROOT / "session_fold_map_runner_validation.json"
    version = 2
    while out.exists():
        out = ROOT / f"session_fold_map_runner_validation_v{version}.json"
        version += 1
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**result, "saved_to": str(out)}


def build_session_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache {cache_path}")
    fold_payload = load_frozen_fold_map()
    label_validation = validate_fold_map_labels(fold_payload)
    sessions, headers = read_sessions(blank_normal=False)
    if set(s["session_key"] for s in sessions) != set(fold_payload["session_fold"]):
        raise ValueError("precomputed sessions differ from frozen session-fold map")
    if any(s["sha256"] != fold_payload["source_sha256"][s["session_key"]] for s in sessions):
        raise ValueError("precomputed source hashes differ from frozen fold map")
    import joblib
    payload = {"runner_sha256": sha256_file(ROOT / "run_pilot.py"),
               "source_sha256": fold_payload["source_sha256"], "headers": headers,
               "sessions": sessions, "label_validation": label_validation,
               "model_fit_or_scores": False}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=1)
    return {"passed": True, "cache_path": str(cache_path), "cache_sha256": sha256_file(cache_path),
            "session_count": len(sessions), "retained_rows": int(sum(len(s["labels"]) for s in sessions)),
            "feature_columns": {"numeric": len(sessions[0]["numeric_columns"]),
                                "context": len(sessions[0]["context_columns"]),
                                "imu": len(sessions[0]["imu_columns"]), "physics": 20},
            "model_fit_or_scores": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-qa", action="store_true")
    ap.add_argument("--split-qa", action="store_true", help="Validate frozen session map and labels only; no model fitting.")
    ap.add_argument("--formal", action="store_true", help="Run frozen real-data pilot (requires separate authorization).")
    ap.add_argument("--precompute-cache", type=Path, help="Build/cache protocol features only; no model fit or scoring.")
    ap.add_argument("--sessions-cache", type=Path, help="Reuse a validated precomputed feature cache for --formal.")
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    if args.synthetic_qa:
        result = run_synthetic_qa()
        qa_path = ROOT / "synthetic_qa.json"
        version = 2
        while qa_path.exists():
            qa_path = ROOT / f"synthetic_qa_v{version}.json"
            version += 1
        qa_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.split_qa:
        print(json.dumps(run_split_qa(), indent=2, ensure_ascii=False))
    elif args.precompute_cache is not None:
        print(json.dumps(build_session_cache(args.precompute_cache), indent=2, ensure_ascii=False))
    elif args.formal:
        if args.output_dir is None:
            ap.error("--formal requires --output-dir")
        formal_run(args.output_dir, args.sessions_cache)
    else:
        ap.error("Choose --synthetic-qa or --formal")


if __name__ == "__main__":
    main()
