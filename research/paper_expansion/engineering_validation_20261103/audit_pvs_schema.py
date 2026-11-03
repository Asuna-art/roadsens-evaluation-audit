"""Read-only schema and label-quality audit for the nine preserved PVS trips."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "external_pvs_v1" / "data"
OUT = Path(__file__).with_name("pvs_schema_audit.json")
POS_COLS = ["speed_bump_asphalt", "speed_bump_cobblestone"]
NEG_COL = "no_speed_bump"
LABEL_COLS = [NEG_COL, *POS_COLS]
PRED = next((BASE / "external_pvs_v1" / "run_primary_v1" / "fits").glob("*/predictions.npz"))


def quantiles(values: np.ndarray) -> dict:
    x = values[np.isfinite(values)]
    if not len(x):
        return {"n": 0, "min": None, "p01": None, "p50": None, "p99": None, "max": None}
    q = np.quantile(x, [0, .01, .5, .99, 1])
    return {"n": int(len(x)), "min": float(q[0]), "p01": float(q[1]), "p50": float(q[2]),
            "p99": float(q[3]), "max": float(q[4])}


def main() -> None:
    trips = []
    for i in range(1, 10):
        d = DATA / f"PVS_{i}"
        sensor = pd.read_csv(d / "dataset_gps_mpu_left.csv", low_memory=False)
        labels = pd.read_csv(d / "dataset_labels.csv", low_memory=False)
        if len(sensor) != len(labels):
            raise ValueError(f"PVS {i}: sensor/label row mismatch")
        raw = labels[LABEL_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        exactly_one = np.isfinite(raw).all(axis=1) & (raw.sum(axis=1) == 1) & np.isin(raw, [0, 1]).all(axis=1)
        neg = exactly_one & (raw[:, 0] == 1)
        pos = exactly_one & (raw[:, 1:].sum(axis=1) == 1)
        ambiguous = ~(neg | pos)
        ts = pd.to_numeric(sensor["timestamp"], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(ts)
        # Annotation runs are row-contiguous and split at gaps greater than 100 ms.
        indices = np.flatnonzero(pos)
        segments = []
        if len(indices):
            cuts = np.flatnonzero((np.diff(indices) != 1) | (dt[np.minimum(indices[:-1], len(dt)-1)] > .1)) + 1
            for group in np.split(indices, cuts):
                a, b = int(group[0]), int(group[-1])
                segments.append({"start_row_zero_based": a, "end_row_zero_based": b,
                                 "start_timestamp": float(ts[a]), "end_timestamp": float(ts[b]),
                                 "duration_seconds": float(ts[b] - ts[a])})
        gps = sensor[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        gps_finite = np.isfinite(gps).all(axis=1)
        speed = pd.to_numeric(sensor["speed"], errors="coerce").to_numpy(dtype=float)
        trips.append({"trip": i, "sensor_path": str((d / "dataset_gps_mpu_left.csv").relative_to(BASE)),
                      "label_path": str((d / "dataset_labels.csv").relative_to(BASE)),
                      "sensor_columns": sensor.columns.tolist(), "label_columns": labels.columns.tolist(),
                      "rows": int(len(sensor)), "positive_rows": int(pos.sum()), "negative_rows": int(neg.sum()),
                      "ambiguous_rows": int(ambiguous.sum()), "timestamp": {"unit": "source numeric; verify from documentation",
                      "finite_n": int(np.isfinite(ts).sum()), "min": float(np.nanmin(ts)), "max": float(np.nanmax(ts)),
                      "diff_seconds_assumed": quantiles(dt)}, "positive_annotation_segments_gap_le_0.1s": segments,
                      "gps": {"finite_both_n": int(gps_finite.sum()), "finite_both_fraction": float(gps_finite.mean()),
                              "zero_latitude_fraction": float(np.mean(gps[:, 0] == 0)),
                              "zero_longitude_fraction": float(np.mean(gps[:, 1] == 0))},
                      "speed": quantiles(speed)})
    def npz_info(path: Path) -> dict:
        with np.load(path, allow_pickle=False) as z:
            return {"path": str(path.relative_to(BASE)), "arrays": {k: list(z[k].shape) for k in z.files}}
    report = {"audit": "read-only raw schema and quality diagnostics; no model fitting or split generation",
              "label_rule": {"positive": "exactly one of no_speed_bump/speed_bump_asphalt/speed_bump_cobblestone is 1, and a positive indicator is 1",
                             "negative": "exactly one indicator is 1 and it is no_speed_bump", "otherwise": "ambiguous"},
              "positive_segment_rule": "positive rows contiguous in raw row order; split when timestamp gap > 0.1 seconds",
              "timestamp_note": "raw numeric values and differences reported; seconds are assumed for gap threshold",
              "trips": trips, "features_npz": npz_info(BASE / "external_pvs_v1" / "features.npz"),
              "prediction_npz": npz_info(PRED)}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
