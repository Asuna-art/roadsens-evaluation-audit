#!/usr/bin/env python3
"""Build the predeclared KIT trailing-window feature set; no fitting or splitting."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE = Path(r"D:\三区灌水专用\research_20261015\candidate_roadsens\paper_expansion")
AUDIT = BASE / "engineering_validation_20261103" / "kit_audit"
ARRAYS = AUDIT / "raw_arrays_verified.npz"
GROUP_AUDIT = AUDIT / "group_audit_verified.json"
OUT = BASE / "engineering_validation_20261103" / "features_kit_v1.npz"
MANIFEST = BASE / "engineering_validation_20261103" / "features_kit_v1_manifest.json"
DATES = {"2016-05-13": 0, "2016-06-20": 1, "2016-07-05": 2}
FEATURE_NAMES = [f"axis{a}_{s}" for a in range(6) for s in ("mean", "std_ddof0", "rms", "ptp", "maxabs")]
SCHEMA = {
    "version": "features_kit_v1",
    "sensor_features": FEATURE_NAMES,
    "sensor_order": "axis-major; axes6 input order; mean,std(ddof=0),rms,ptp,maxabs",
    "context_features": ["lat", "lon", "speed"],
    "label_mapping": {"1": 0, "3": 1, "other": "excluded"},
    "decision_times": "t[0]+1.0+j*0.25 through t[-1], inclusive where available",
    "window": "closed [decision_time-1.0, decision_time]; anchor=last timestamp<=decision",
    "constraints": {"min_rows": 50, "all_axes_finite": True, "max_timestamp_gap_seconds": 0.1,
                    "anchor_label": [1, 3], "anchor_lat_range": [48, 50], "anchor_lon_range": [7, 10]},
    "day_id": DATES,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gps_dates(gps_ts: np.ndarray) -> set[str]:
    vals = gps_ts[np.isfinite(gps_ts)]
    if not vals.size:
        return set()
    # GPS timestamps in the verified extraction are Unix-like seconds; use them only to
    # establish calendar date. The numeric sensor timestamp remains the timing axis.
    return {datetime.fromtimestamp(float(v), tz=timezone.utc).date().isoformat() for v in vals}


def features(w: np.ndarray) -> np.ndarray:
    return np.column_stack((w.mean(axis=0), w.std(axis=0, ddof=0),
                            np.sqrt(np.mean(w * w, axis=0)), np.ptp(w, axis=0),
                            np.max(np.abs(w), axis=0))).reshape(-1)


def main() -> None:
    if OUT.exists() or MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT} or {MANIFEST}")
    audit = json.loads(GROUP_AUDIT.read_text(encoding="utf-8"))
    records = audit["files"]
    Xs, Xc, ys, fids, days, anchors, starts, ends, atimes, dtimes = ([] for _ in range(10))
    per_file = []
    z = np.load(ARRAYS, allow_pickle=False)
    for rec in records:
        fid = int(rec["file_id"][1:])
        prefix = rec["file_id"]
        dates = gps_dates(z[f"{prefix}_gps_timestamp"])
        info = {"file_id": fid, "source_path": rec["path"], "gps_dates": sorted(dates),
                "date_basis": "GPS clock calendar date; numeric timestamp used only for within-file timing",
                "candidate_windows": 0, "retained_windows": 0, "retained_by_class": {"0": 0, "1": 0},
                "exclusion_reasons": {}}
        if len(dates) != 1:
            info["exclusion_reasons"]["gps_date_not_single"] = 1
            per_file.append(info)
            continue
        date = next(iter(dates))
        if date not in DATES:
            info["exclusion_reasons"]["gps_date_not_selected"] = 1
            per_file.append(info)
            continue
        ts = np.asarray(z[f"{prefix}_timestamp"], dtype=np.float64)
        axes = np.asarray(z[f"{prefix}_axes6"], dtype=np.float64)
        labels = np.asarray(z[f"{prefix}_label"])
        lat, lon, speed = (np.asarray(z[f"{prefix}_{k}"], dtype=np.float64) for k in ("lat", "lon", "speed"))
        if not (np.all(np.isfinite(ts)) and np.all(np.diff(ts) >= 0)):
            info["exclusion_reasons"]["invalid_or_decreasing_numeric_timestamp"] = 1
            per_file.append(info)
            continue
        decisions = np.arange(ts[0] + 1.0, ts[-1] + 1e-10, 0.25)
        reasons = Counter()
        for decision in decisions:
            info["candidate_windows"] += 1
            anchor = int(np.searchsorted(ts, decision, side="right") - 1)
            lo = int(np.searchsorted(ts, decision - 1.0, side="left"))
            hi = anchor + 1
            w = axes[lo:hi]
            if len(w) < 50:
                reasons["fewer_than_50_rows"] += 1; continue
            if not np.isfinite(w).all():
                reasons["nonfinite_sensor_axis"] += 1; continue
            if len(w) > 1 and np.max(np.diff(ts[lo:hi])) > 0.1 + 1e-12:
                reasons["timestamp_gap_over_0.1s"] += 1; continue
            if labels[anchor] not in (1, 3):
                reasons["anchor_label_not_1_or_3"] += 1; continue
            if not (np.isfinite(lat[anchor]) and np.isfinite(lon[anchor]) and 48 <= lat[anchor] <= 50 and 7 <= lon[anchor] <= 10):
                reasons["anchor_coordinates_invalid_or_outside_bounds"] += 1; continue
            y = 0 if labels[anchor] == 1 else 1
            Xs.append(features(w)); Xc.append([lat[anchor], lon[anchor], speed[anchor]])
            ys.append(y); fids.append(fid); days.append(DATES[date]); anchors.append(anchor)
            starts.append(lo); ends.append(anchor); atimes.append(ts[anchor]); dtimes.append(decision)
            info["retained_windows"] += 1; info["retained_by_class"][str(y)] += 1
        info["exclusion_reasons"] = dict(sorted(reasons.items()))
        per_file.append(info)

    if not ys:
        raise RuntimeError("No qualifying windows; refusing to write an empty feature artifact")
    np.savez_compressed(OUT, X_sensor=np.asarray(Xs, dtype=np.float64), X_context=np.asarray(Xc, dtype=np.float64),
                        y=np.asarray(ys, dtype=np.int8), file_id=np.asarray(fids, dtype=np.int16),
                        day_id=np.asarray(days, dtype=np.int8), anchor_row=np.asarray(anchors, dtype=np.int32),
                        source_start=np.asarray(starts, dtype=np.int32), source_end=np.asarray(ends, dtype=np.int32),
                        anchor_time=np.asarray(atimes, dtype=np.float64), decision_time=np.asarray(dtimes, dtype=np.float64))
    schema_json = json.dumps(SCHEMA, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest = {"artifact": OUT.name, "rows": len(ys), "schema": SCHEMA,
                "schema_sha256": hashlib.sha256(schema_json.encode("utf-8")).hexdigest(),
                "input_sha256": {ARRAYS.name: sha256(ARRAYS), GROUP_AUDIT.name: sha256(GROUP_AUDIT)},
                "script_sha256": sha256(Path(__file__)), "files": per_file,
                "class_totals": {"0": int(sum(y == 0 for y in ys)), "1": int(sum(y == 1 for y in ys))},
                "scope": "feature construction only; no split, fitting, or model training",
                "unit_note": "file_id identifies source files; 53 source files must not be described as 53 trips"}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
