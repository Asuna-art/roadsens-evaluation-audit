"""Frozen 17-record STRIDE recording-type supplement. No fitting unless --run is explicit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "stride_road_data.zip"
PROTOCOL = HERE / "stride_supplement_protocol_v1.md"
EXPECTED_ARCHIVE_SHA256 = "475853a7aaf9a15068dc9c17014f3e46197acc4b705cd3851b84692cdd82f851"
EXPECTED_PYTHON = Path(r"C:\Users\Administrator\miniconda3\python.exe")
EXPECTED_SKLEARN = "1.9.0"
CLASSES = ("Bump", "Pothole")
FEATURES = ("sensor_norm_statistics10", "duration_only1")
SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _member_stream(zf: zipfile.ZipFile, directory: str, filename: str) -> tuple[str, bytes]:
    expected = f"Road Data/Road Anomalies/{directory}/{filename}"
    with zf.open(expected) as f:
        data = f.read()
    return expected, data


def _csv_axes(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import pandas as pd
    from io import BytesIO
    frame = pd.read_csv(BytesIO(blob))
    if not {"seconds_elapsed", "x", "y", "z"}.issubset(frame.columns):
        raise ValueError(f"sensor CSV missing required columns: {frame.columns.tolist()}")
    elapsed = frame["seconds_elapsed"].to_numpy(dtype=np.float64)
    axes = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
    time_finite = np.isfinite(elapsed)
    if not np.any(time_finite):
        raise ValueError("sensor stream has no finite seconds_elapsed values")
    duration = float(np.max(elapsed[time_finite]) - np.min(elapsed[time_finite]))
    valid = np.all(np.isfinite(axes), axis=1)
    if not np.any(valid):
        raise ValueError("sensor stream has no rows with all three axes finite")
    # Sensor summaries use only finite axes. Timestamp values are used solely for
    # the duration feature, never to select sensor observations or as a predictor.
    return axes[valid, 0], axes[valid, 1], axes[valid, 2], duration


def extract_population() -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    if sha256_file(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("STRIDE archive SHA256 differs from frozen protocol")
    rows, feature_by_id, duration_by_id, member_hashes = [], {}, {}, {}
    with zipfile.ZipFile(ARCHIVE) as zf:
        for info in zf.infolist():
            match = re.fullmatch(r"Road Data/Road Anomalies/(\d+)\. (Bump|Pothole)/Metadata\.csv", info.filename)
            if not match:
                continue
            record_id, label = int(match.group(1)), match.group(2)
            folder = f"{record_id}. {label}"
            metadata_path, metadata_bytes = _member_stream(zf, folder, "Metadata.csv")
            import pandas as pd
            from io import BytesIO
            metadata = pd.read_csv(BytesIO(metadata_bytes)).iloc[0].to_dict()
            acc_path, acc_blob = _member_stream(zf, folder, "Accelerometer.csv")
            gyr_path, gyr_blob = _member_stream(zf, folder, "Gyroscope.csv")
            member_hashes[metadata_path] = bytes_sha256(metadata_bytes)
            member_hashes[acc_path] = bytes_sha256(acc_blob)
            member_hashes[gyr_path] = bytes_sha256(gyr_blob)
            ax, ay, az, duration = _csv_axes(acc_blob)
            gx, gy, gz, _ = _csv_axes(gyr_blob)
            stats = []
            for stream, axes in (("accel", (ax, ay, az)), ("gyro", (gx, gy, gz))):
                norm = np.sqrt(axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2)
                stats.extend([float(np.mean(norm)), float(np.std(norm, ddof=0)),
                              float(np.sqrt(np.mean(norm ** 2))), float(np.max(norm) - np.min(norm)),
                              float(np.max(np.abs(norm)))])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid accelerometer elapsed duration for {folder}")
            rows.append({"record_id": record_id, "record": folder, "label": label,
                         "metadata_device": str(metadata.get("device name", "")),
                         "metadata_device_id": str(metadata.get("device id", "")),
                         "metadata_recording_time": str(metadata.get("recording time", "")),
                         "accel_finite_rows": int(len(ax)), "gyro_finite_rows": int(len(gx)),
                         "duration_seconds": duration, "accel_member": acc_path, "gyro_member": gyr_path,
                         "metadata_member": metadata_path})
            feature_by_id[record_id] = stats
            duration_by_id[record_id] = [duration]
    rows.sort(key=lambda r: r["record_id"])
    if len(rows) != 17 or {r["label"] for r in rows} != set(CLASSES):
        raise ValueError(f"expected 17 anomaly recordings with Bump/Pothole only, got {len(rows)}")
    record_ids = [r["record_id"] for r in rows]
    x_sensor = np.asarray([feature_by_id[i] for i in record_ids], dtype=np.float64)
    x_duration = np.asarray([duration_by_id[i] for i in record_ids], dtype=np.float64)
    metadata = {"archive_sha256": sha256_file(ARCHIVE), "member_sha256": member_hashes,
                "record_count": len(rows), "class_counts": {c: sum(r["label"] == c for r in rows) for c in CLASSES},
                "sensor_feature_columns": [f"{stream}_{stat}" for stream in ("accel_norm", "gyro_norm")
                    for stat in ("mean", "std_ddof0", "rms", "range", "absmax")],
                "sensor_feature_units": {"accel_norm_*": "native Accelerometer.csv source units; no conversion",
                                         "gyro_norm_*": "native Gyroscope.csv source units; no conversion"},
                "duration_feature_column": ["accelerometer_seconds_elapsed_range"]}
    return rows, x_sensor, x_duration, metadata


def _splits(y: np.ndarray, seed: int):
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in cv.split(np.zeros(len(y)), y)]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    return {"n": int(len(y_true)), "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "class_support": {c: int(np.sum(y_true == c)) for c in CLASSES},
            "class_recall": {c: float(v) for c, v in zip(CLASSES, recall_score(y_true, y_pred, labels=list(CLASSES), average=None, zero_division=0))},
            "confusion": confusion_matrix(y_true, y_pred, labels=list(CLASSES)).tolist()}


def qa() -> dict:
    rows, xs, xd, meta = extract_population()
    y = np.asarray([r["label"] for r in rows], dtype=object)
    if meta["class_counts"] != {"Bump": 9, "Pothole": 8}:
        raise ValueError(f"frozen expected class counts differ: {meta['class_counts']}")
    if any(r["metadata_device"] != "POCO X2" for r in rows):
        raise ValueError("device-model QA failed")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(xd)):
        raise ValueError("feature table contains non-finite values")
    # Feature construction is independent of labels and folder identifiers: it accepts only raw CSV bytes.
    for seed in SEEDS:
        folds = _splits(y, seed)
        seen = []
        for tr, te in folds:
            if set(np.unique(y[tr])) != set(CLASSES) or set(np.unique(y[te])) != set(CLASSES):
                raise ValueError(f"both classes not represented for seed {seed}")
            if set(tr) & set(te):
                raise AssertionError("train/test row overlap")
            seen.extend(te.tolist())
        if sorted(seen) != list(range(17)):
            raise AssertionError(f"seed {seed} test folds do not partition records exactly once")
    # Check that fit-only transformations do not inspect held-out extrema.
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    sample = np.asarray([[1.0, np.nan], [3.0, 4.0], [1000.0, -1000.0]])
    changed = sample.copy(); changed[2] = [-1e8, 1e8]
    def stats(a):
        imp = SimpleImputer(strategy="median").fit(a[:2])
        scaled = StandardScaler().fit(imp.transform(a[:2]))
        return imp.statistics_, scaled.mean_, scaled.scale_
    if any(not np.array_equal(a, b) for a, b in zip(stats(sample), stats(changed))):
        raise AssertionError("held-out values affected imputation/scaling")
    return {"passed": True, "model_fit_or_scores": False, "record_count": len(rows),
            "class_counts": meta["class_counts"], "sensor_feature_shape": list(xs.shape),
            "duration_feature_shape": list(xd.shape), "seeds": list(SEEDS),
            "archive_sha256": meta["archive_sha256"], "member_hash_count": len(meta["member_sha256"]),
            "checks": ["17 folder-level anomaly records, 9 Bump/8 Pothole only",
                       "feature set uses only whole-record accelerometer/gyroscope norm statistics",
                       "three fixed stratified seed maps partition every record with both classes in all folds",
                       "train-only median/scaler fixture ignores held-out values"]}


def run(run_id: str, expected_protocol_sha256: str) -> Path:
    import sklearn
    if str(Path(sys.executable).resolve()).casefold() != str(EXPECTED_PYTHON.resolve()).casefold():
        raise RuntimeError(f"STRIDE run requires locked interpreter: {EXPECTED_PYTHON}")
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(f"STRIDE run requires scikit-learn {EXPECTED_SKLEARN}")
    if expected_protocol_sha256.casefold() != sha256_file(PROTOCOL).casefold():
        raise ValueError("explicit protocol hash differs from frozen supplement protocol")
    rows, xs, xd, source = extract_population()
    out = HERE / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    source["protocol_sha256"] = sha256_file(PROTOCOL)
    source["protocol_acknowledged_sha256"] = expected_protocol_sha256
    source["driver_sha256"] = sha256_file(Path(__file__).resolve())
    import sklearn, pandas
    source["runtime"] = {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
                         "platform": platform.platform(), "numpy": np.__version__,
                         "scikit_learn": sklearn.__version__, "pandas": pandas.__version__}
    source["model_configs"] = {
        "RF100_depth5_leaf2": {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 2,
            "max_features": "sqrt", "class_weight": "balanced", "random_state": "split seed", "n_jobs": 4},
        "LogisticRegression_scaled": {"imputer": "training median", "scaler": "StandardScaler fit on training rows",
            "C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4,
            "class_weight": "balanced", "random_state": "split seed"}}
    source["features"] = {FEATURES[0]: source["sensor_feature_columns"],
                          FEATURES[1]: source["duration_feature_column"]}
    source["seeds"] = list(SEEDS)
    source["split_config"] = "StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)"
    (out / "manifest.json").write_text(json.dumps({"status": "running", "expected_model_folds": 60, **source}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    labels = np.asarray([r["label"] for r in rows], dtype=object)
    records = np.asarray([r["record"] for r in rows], dtype=str)
    feature_matrices = {FEATURES[0]: xs, FEATURES[1]: xd}
    (out / "source_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (out / "features.npz").open("wb") as f:
        np.savez_compressed(f, sensor_norm_statistics10=xs, duration_only1=xd,
                            y=np.asarray([CLASSES.index(str(v)) for v in labels], dtype=np.int8), record=records)
    split_dir = out / "splits"; split_dir.mkdir()
    prediction_dir = out / "predictions"; prediction_dir.mkdir()
    model_dir = out / "models"; model_dir.mkdir()
    completion, failures = [], []
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    for seed in SEEDS:
        folds = _splits(labels, seed)
        split_arrays = {}
        for fold, (tr, te) in enumerate(folds):
            split_arrays[f"train_{fold}"] = tr; split_arrays[f"test_{fold}"] = te
        np.savez_compressed(split_dir / f"seed{seed}.npz", **split_arrays)
        for feature, matrix_x in feature_matrices.items():
            for model_name in ("RF100_depth5_leaf2", "LogisticRegression_scaled"):
                pred_oof = np.full(len(labels), -1, dtype=np.int8)
                fold_results = []
                for fold, (tr, te) in enumerate(folds):
                    started = time.perf_counter()
                    try:
                        xtr0, xte0 = matrix_x[tr], matrix_x[te]
                        imputer = SimpleImputer(strategy="median")
                        xtr = imputer.fit_transform(xtr0); xte = imputer.transform(xte0)
                        estimator = (RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2,
                            max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=4)
                            if model_name.startswith("RF") else None)
                        scaler = None
                        if model_name.startswith("Logistic"):
                            scaler = StandardScaler().fit(xtr)
                            xtr, xte = scaler.transform(xtr), scaler.transform(xte)
                            estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                tol=1e-4, class_weight="balanced", random_state=seed)
                        with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            estimator.fit(xtr, labels[tr])
                            pred = estimator.predict(xte)
                        pred_oof[te] = np.asarray([CLASSES.index(str(v)) for v in pred], dtype=np.int8)
                        art = model_dir / f"seed{seed}_{feature}_{model_name}_fold{fold+1}.joblib"
                        joblib.dump({"estimator": estimator, "imputer": imputer, "scaler": scaler}, art, compress=3)
                        fold_results.append({"fold": fold+1, **_metrics(labels[te], pred),
                            "train_record_ids": records[tr].tolist(), "test_record_ids": records[te].tolist(),
                            "fit_wall_seconds": time.perf_counter()-started,
                            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
                            "n_iter": getattr(estimator, "n_iter_", None).tolist() if hasattr(getattr(estimator, "n_iter_", None), "tolist") else getattr(estimator, "n_iter_", None),
                            "artifact": art.name, "artifact_sha256": sha256_file(art),
                            "fit_row_ids_sha256": hashlib.sha256("\n".join(records[tr]).encode()).hexdigest(),
                            "test_row_ids_sha256": hashlib.sha256("\n".join(records[te]).encode()).hexdigest()})
                    except Exception as exc:
                        fail = {"seed": seed, "feature": feature, "model": model_name,
                                "fold": fold+1, "error_type": type(exc).__name__, "error": str(exc)}
                        failures.append(fail); fold_results.append({"fold": fold+1, "status": "failed", **fail})
                job_complete = len(fold_results) == 5 and not any(f.get("status") == "failed" for f in fold_results)
                np.savez_compressed(prediction_dir / f"seed{seed}_{feature}_{model_name}.npz",
                    y_true=np.asarray([CLASSES.index(str(v)) for v in labels], dtype=np.int8),
                    y_pred=pred_oof, record=records)
                result = {"seed": seed, "feature": feature, "model": model_name,
                    "status": "complete" if job_complete else "partial_failure",
                    "folds": fold_results,
                    "fold_mean_macro_f1": float(np.mean([f["macro_f1"] for f in fold_results if "macro_f1" in f])) if job_complete else None,
                    "pooled_oof": _metrics(labels, np.asarray([CLASSES[i] for i in pred_oof], dtype=object)) if job_complete else None}
                result_path = out / f"seed{seed}_{feature}_{model_name}.json"
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                completion.append({"seed": seed, "feature": feature, "model": model_name, "status": result["status"], "result_file": result_path.name})
                (out / "completion.json").write_text(json.dumps({"completion": completion, "failures": failures}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    final = {"status": "complete_with_failures" if failures else "complete", **source,
             "expected_model_folds": 60, "completed_jobs": completion, "failures": failures}
    (out / "manifest.json").write_text(json.dumps(final, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--qa", action="store_true", help="read archive, extract features/splits, and run fit-free checks")
    modes.add_argument("--run", action="store_true", help="fit the frozen 60-fold supplement")
    parser.add_argument("--run-id", help="exclusive directory name under paper_expansion/runs")
    parser.add_argument("--protocol-sha256", help="exact frozen protocol SHA256; required for --run")
    args = parser.parse_args()
    if args.qa:
        print(json.dumps(qa(), ensure_ascii=False, indent=2)); return
    if not args.run_id or not args.protocol_sha256:
        parser.error("--run requires --run-id and --protocol-sha256")
    print(str(run(args.run_id, args.protocol_sha256)))


if __name__ == "__main__":
    main()
