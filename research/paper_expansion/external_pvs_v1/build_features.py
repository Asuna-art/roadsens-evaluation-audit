"""Build fit-free PVS left-dashboard window features from the preserved source CSVs."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "features.npz"
MANIFEST = ROOT / "feature_manifest.json"
WINDOW_ROWS = 100
STRIDE = 25
AXES = [
    "acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard",
    "gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard",
]
CONTEXT = ["latitude", "longitude", "speed"]
LABEL_NEG = "no_speed_bump"
LABEL_POS = ["speed_bump_asphalt", "speed_bump_cobblestone"]
STAT_NAMES = ["mean", "std_pop", "rms", "range", "max_abs"]
PAIR_BY_TRIP = {**{i: 1 for i in (1, 2, 3)}, **{i: 2 for i in (4, 5, 6)}, **{i: 3 for i in (7, 8, 9)}}
SCENARIO_BY_TRIP = {i: ((i - 1) % 3) + 1 for i in range(1, 10)}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def feature_names():
    return [f"{axis}__{stat}" for axis in AXES for stat in STAT_NAMES]


def main():
    records = []
    arrays = {"X_sensor": [], "X_context": [], "y": [], "trip": [], "vehicle": [], "pair": [], "scenario": [],
              "source_start": [], "source_end": [], "anchor_row": [], "timestamp_start": [], "anchor_time": [],
              "window_span_seconds": []}
    global_excluded = {"candidate_windows": 0, "nonfinite_sensor_window": 0, "nonfinite_timestamp_window": 0,
                       "ambiguous_anchor_label": 0}
    for trip_id in range(1, 10):
        trip = f"PVS {trip_id}"
        data_dir = ROOT / "data" / f"PVS_{trip_id}"
        sensor_path = data_dir / "dataset_gps_mpu_left.csv"
        label_path = data_dir / "dataset_labels.csv"
        if not sensor_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Required source missing for {trip}: {sensor_path} or {label_path}")
        sensor = pd.read_csv(sensor_path, low_memory=False)
        labels = pd.read_csv(label_path, low_memory=False)
        required = ["timestamp", *AXES, *CONTEXT]
        missing = sorted(set(required) - set(sensor.columns))
        missing_labels = sorted(set([LABEL_NEG, *LABEL_POS]) - set(labels.columns))
        if missing or missing_labels:
            raise ValueError(f"{trip}: missing sensor columns={missing}, label columns={missing_labels}")
        if len(sensor) != len(labels):
            raise ValueError(f"{trip}: sensor/label row count mismatch {len(sensor)} != {len(labels)}; refusing to join")
        ts = pd.to_numeric(sensor["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(ts).all():
            raise ValueError(f"{trip}: nonfinite timestamp found; chronological span cannot be verified")
        if np.any(np.diff(ts) < 0):
            bad = int(np.flatnonzero(np.diff(ts) < 0)[0])
            raise ValueError(f"{trip}: timestamp decreases between source rows {bad} and {bad+1}; stopping as required")
        values = sensor[AXES].to_numpy(dtype=np.float64)
        context = sensor[CONTEXT].to_numpy(dtype=np.float64)
        label_values = labels[[LABEL_NEG, *LABEL_POS]].to_numpy(dtype=np.float64)
        # Source labels are binary indicators. Reject unexpected values before mapping.
        if not np.isin(label_values, [0.0, 1.0]).all():
            raise ValueError(f"{trip}: target indicator contains values outside 0/1")
        sensor_features = []
        context_features = []
        ys = []
        starts = []
        anchors = []
        time_starts = []
        time_anchors = []
        spans = []
        ex = {"candidate_windows": 0, "nonfinite_sensor_window": 0, "nonfinite_timestamp_window": 0,
              "ambiguous_anchor_label": 0}
        for anchor in range(WINDOW_ROWS - 1, len(sensor), STRIDE):
            start = anchor - WINDOW_ROWS + 1
            ex["candidate_windows"] += 1
            block = values[start:anchor + 1]
            if block.shape != (WINDOW_ROWS, len(AXES)):
                raise AssertionError("window length changed unexpectedly")
            if not np.isfinite(block).all():
                ex["nonfinite_sensor_window"] += 1
                continue
            window_ts = ts[start:anchor + 1]
            if not np.isfinite(window_ts).all():
                ex["nonfinite_timestamp_window"] += 1
                continue
            indicator = label_values[anchor]
            if int(indicator.sum()) != 1:
                ex["ambiguous_anchor_label"] += 1
                continue
            y = 0 if indicator[0] == 1 else 1
            feats = []
            for col in range(len(AXES)):
                x = block[:, col]
                feats.extend((float(np.mean(x)), float(np.std(x, ddof=0)),
                              float(np.sqrt(np.mean(np.square(x)))), float(np.max(x) - np.min(x)),
                              float(np.max(np.abs(x)))))
            sensor_features.append(feats)
            context_features.append(context[anchor].tolist())
            ys.append(y)
            starts.append(start)
            anchors.append(anchor)
            time_starts.append(float(window_ts[0]))
            time_anchors.append(float(window_ts[-1]))
            spans.append(float(window_ts[-1] - window_ts[0]))
        for key in global_excluded:
            global_excluded[key] += ex[key]
        if not ys:
            raise ValueError(f"{trip}: no retained windows")
        sensor_features = np.asarray(sensor_features, dtype=np.float64)
        context_features = np.asarray(context_features, dtype=np.float64)
        ys_arr = np.asarray(ys, dtype=np.int8)
        arrays["X_sensor"].append(sensor_features)
        arrays["X_context"].append(context_features)
        arrays["y"].append(ys_arr)
        arrays["trip"].append(np.repeat(trip_id, len(ys_arr)))
        arrays["vehicle"].append(np.repeat(PAIR_BY_TRIP[trip_id], len(ys_arr)))
        arrays["pair"].append(np.repeat(PAIR_BY_TRIP[trip_id], len(ys_arr)))
        arrays["scenario"].append(np.repeat(SCENARIO_BY_TRIP[trip_id], len(ys_arr)))
        for name, vals in (("source_start", starts), ("source_end", anchors), ("anchor_row", anchors),
                           ("timestamp_start", time_starts), ("anchor_time", time_anchors),
                           ("window_span_seconds", spans)):
            arrays[name].append(np.asarray(vals))
        counts = {str(c): int(np.sum(ys_arr == c)) for c in (0, 1)}
        span_arr = np.asarray(spans, dtype=np.float64)
        records.append({
            "trip": trip_id, "trip_name": trip, "vehicle": PAIR_BY_TRIP[trip_id], "driver": PAIR_BY_TRIP[trip_id],
            "vehicle_driver_pair": PAIR_BY_TRIP[trip_id], "scenario": SCENARIO_BY_TRIP[trip_id],
            "raw_rows": int(len(sensor)), "retained_windows": int(len(ys_arr)), "class_counts": counts,
            "window_span_seconds": {"min": float(np.min(span_arr)), "median": float(np.median(span_arr)), "max": float(np.max(span_arr))},
            "sensor_source": {"path": str(sensor_path.relative_to(ROOT)), "sha256": sha256_file(sensor_path), "bytes": sensor_path.stat().st_size},
            "label_source": {"path": str(label_path.relative_to(ROOT)), "sha256": sha256_file(label_path), "bytes": label_path.stat().st_size},
            "sensor_columns_in_source": sensor.columns.tolist(), "label_columns_in_source": labels.columns.tolist(),
            "excluded_counts": ex,
        })
        print(f"{trip}: rows={len(sensor)} retained={len(ys_arr)} classes={counts} spans={records[-1]['window_span_seconds']}", flush=True)
    combined = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    np.savez_compressed(OUT, **combined)
    manifest = {
        "created_by": str(pathlib.Path(__file__).name), "python": sys.version,
        "dataset_source": "Kaggle individual source-file API; official UFSC/GitHub release linked in source_probe.json",
        "label_policy": {"task": "NoSpeedBump-vs-SpeedBump", "class_map": {"0": "no_speed_bump", "1": "speed_bump_asphalt OR speed_bump_cobblestone"},
                         "required_indicator_columns": [LABEL_NEG, *LABEL_POS], "anchor_rule": "exactly one of the three indicators must equal 1; no remapping to RoadSens labels"},
        "window_rows": WINDOW_ROWS, "stride": STRIDE, "stride25": STRIDE, "anchor_rows_zero_based": "99,124,149,...; label is indicator at final anchor",
        "sensor_columns": AXES, "sensor_feature_names": feature_names(), "context_columns": CONTEXT,
        "context_feature_names": CONTEXT, "sensor_feature_indices": list(range(30)), "context_feature_indices": [0, 1, 2],
        "excluded_counts": global_excluded, "trips": records,
        "combined_rows": int(len(combined["y"])), "combined_class_counts": {str(c): int(np.sum(combined["y"] == c)) for c in (0, 1)},
        "npz_path": str(OUT.relative_to(ROOT)), "npz_sha256": sha256_file(OUT), "npz_bytes": OUT.stat().st_size,
        "feature_semantics": {"sensor": "per-axis mean, population std, RMS, range, maximum absolute value over trailing 100 rows", "context": "same 30 features plus anchor latitude, longitude, speed", "timestamps_and_ids_excluded_from_predictors": True},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} with {len(combined['y'])} windows; manifest {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
