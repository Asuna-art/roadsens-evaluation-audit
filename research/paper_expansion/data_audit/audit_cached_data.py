"""Read-only audit of the immutable RoadSens v2 pilot cache and its source CSVs."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
CACHE_PATH = BASE / "session_feature_cache_v1.joblib"
FOLD_MAP_PATH = BASE / "session_fold_map_v1.json"
OUT = Path(__file__).resolve().parent
CLASSES = ("Normal", "Pothole", "Bump")
POLICIES = {"primary": "labels", "blank_normal_sensitivity": "labels_blank_normal"}
EXPECTED_RF = {"n_estimators": 300, "max_depth": 25, "max_features": "sqrt",
               "class_weight": "balanced", "random_state": 42}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canon_signature(row: np.ndarray) -> str:
    x = np.asarray(row, dtype="<f8").copy()
    x[np.isnan(x)] = np.nan
    x[x == 0] = 0.0
    return hashlib.sha256(x.tobytes()).hexdigest()


def count_table(labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=object)
    return {c: int(np.sum(labels == c)) for c in CLASSES} | {
        "unlabeled_or_unrecognized": int(np.sum(~np.isin(labels, CLASSES)))}


def labels_from_source(df: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    if key.startswith("combined_normal/"):
        a = np.full(len(df), "Normal", dtype=object)
        return a, a.copy()
    if "annotation_text" in df:
        raw = df["annotation_text"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
    else:
        raw = np.full(len(df), "", dtype=object)
    primary = np.full(len(df), None, dtype=object)
    for c in ("Pothole", "Bump"):
        primary[raw == c] = c
    sensitivity = primary.copy()
    sensitivity[raw == ""] = "Normal"
    return primary, sensitivity


def pair_grid_overlap(sessions: list[dict], lat_i: int, lon_i: int, columns: list[str], meters: int) -> dict:
    lats, lons = [], []
    for s in sessions:
        lat = s["numeric"][:, lat_i]
        lon = s["numeric"][:, lon_i]
        ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        lats.extend(lat[ok].tolist()); lons.extend(lon[ok].tolist())
    if not lats:
        return {"valid_gps_sample_rows": 0, "grid_cell_m": meters, "unique_cells": 0,
                "cells_shared_by_multiple_recordings": 0, "recording_pairs_sharing_any_cell": 0}
    lat_ref = float(np.median(lats))
    cos_ref = math.cos(math.radians(lat_ref))
    min_x = min(lons) * 111320.0 * cos_ref
    min_y = min(lats) * 110574.0
    cells: dict[tuple[int, int], set[str]] = defaultdict(set)
    per_record: dict[str, set[tuple[int, int]]] = {}
    gps_record_rows = {}
    for s in sessions:
        lat = np.asarray(s["numeric"][:, lat_i], dtype=np.float64)
        lon = np.asarray(s["numeric"][:, lon_i], dtype=np.float64)
        ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        rec = s["session_key"]
        gps_record_rows[rec] = int(ok.sum())
        x = lon[ok] * 111320.0 * cos_ref
        y = lat[ok] * 110574.0
        ix = np.floor((x - min_x) / meters).astype(np.int64)
        iy = np.floor((y - min_y) / meters).astype(np.int64)
        rec_cells = set(zip(ix.tolist(), iy.tolist()))
        per_record[rec] = rec_cells
        for cell in rec_cells:
            cells[cell].add(rec)
    pair_cells = defaultdict(int)
    cell_intersections = 0
    for recs in cells.values():
        if len(recs) >= 2:
            records = sorted(recs)
            cell_intersections += len(records) * (len(records) - 1) // 2
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    pair_cells[(records[i], records[j])] += 1
    return {
        "grid_cell_m": meters, "grid_method": "local equirectangular meters; global median-latitude scale and origin; sampled rows only",
        "latitude_reference_used_internally_only": True,
        "valid_gps_sample_rows": int(sum(gps_record_rows.values())),
        "recordings_with_any_valid_gps": int(sum(n > 0 for n in gps_record_rows.values())),
        "unique_cells": len(cells), "cells_shared_by_multiple_recordings": int(sum(len(v) >= 2 for v in cells.values())),
        "recording_pairs_sharing_any_cell": len(pair_cells),
        "recording_pair_cell_intersections": int(cell_intersections),
        "max_shared_cells_for_any_recording_pair": int(max(pair_cells.values(), default=0)),
        "raw_coordinates_exported": False,
    }


def main() -> None:
    cache = joblib.load(CACHE_PATH)
    fold_map = json.loads(FOLD_MAP_PATH.read_text(encoding="utf-8"))
    sessions = sorted(cache["sessions"], key=lambda s: s["session_key"])
    if len(sessions) != 101 or set(cache["source_sha256"]) != {s["session_key"] for s in sessions}:
        raise RuntimeError("Cached sessions/source hashes are not the expected 101-record set")
    if cache["source_sha256"] != fold_map["source_sha256"]:
        raise RuntimeError("Cache source hashes differ from frozen split map")

    columns = list(sessions[0]["numeric_columns"])
    context_columns = list(sessions[0]["context_columns"])
    imu_columns = list(sessions[0]["imu_columns"])
    if any(s["numeric_columns"] != columns or s["context_columns"] != context_columns or s["imu_columns"] != imu_columns for s in sessions):
        raise RuntimeError("Feature schema differs between cached recordings")
    col_i = {c: i for i, c in enumerate(columns)}
    raw_label_counts_by_source = {}
    raw_label_totals = {p: {c: 0 for c in (*CLASSES, "unlabeled_or_unrecognized")} for p in POLICIES}
    session_details, availability = [], {p: {} for p in POLICIES}
    source_hash_seen, source_csv_duplicate_groups = defaultdict(list), defaultdict(list)
    sig12, sigfull = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    feature_constant_records = defaultdict(list)
    signature_rows = 0
    source_dir_totals = defaultdict(lambda: {"source_rows": 0, "retained_rows": 0,
                                             "primary": {c: 0 for c in (*CLASSES, "unlabeled_or_unrecognized")},
                                             "blank_normal_sensitivity": {c: 0 for c in (*CLASSES, "unlabeled_or_unrecognized")}})
    gps_class_rows = defaultdict(lambda: {p: {c: {"rows": 0, "valid_gps": 0} for c in CLASSES} for p in POLICIES})

    for s in sessions:
        key = s["session_key"]
        path = Path(s["path"])
        digest = sha256_file(path)
        if digest != s["sha256"] or digest != cache["source_sha256"][key]:
            raise RuntimeError(f"Live source hash mismatch for {key}")
        source_hash_seen[digest].append(key)
        usecols = lambda c: c in {"seconds_elapsed", "annotation_text"}
        df_small = pd.read_csv(path, usecols=usecols, low_memory=False)
        raw_primary, raw_sensitivity = labels_from_source(df_small, key)
        source_dir = key.split("/", 1)[0]
        source_dir_totals[source_dir]["source_rows"] += len(df_small)
        raw_counts = {"primary": count_table(raw_primary),
                      "blank_normal_sensitivity": count_table(raw_sensitivity)}
        raw_label_counts_by_source[key] = {"source_rows": len(df_small), **raw_counts}
        for p in POLICIES:
            for c, v in raw_counts[p].items():
                raw_label_totals[p][c] += v

        keep_idx = np.asarray(s["source_row"], dtype=np.int64)
        labels_sampled = {p: np.asarray(s[k], dtype=object) for p, k in POLICIES.items()}
        source_dir_totals[source_dir]["retained_rows"] += len(keep_idx)
        for p in POLICIES:
            counts = count_table(labels_sampled[p])
            for c, v in counts.items():
                source_dir_totals[source_dir][p][c] += v

        times = np.asarray(s["times_full"], dtype=np.float64)
        finite_t = times[np.isfinite(times)]
        sorted_t = np.sort(finite_t, kind="stable")
        diffs = np.diff(sorted_t)
        positive = diffs[diffs > 0]
        duration = float(sorted_t[-1] - sorted_t[0]) if len(sorted_t) else None
        sample = np.asarray(s["numeric"], dtype=np.float64)
        session_details.append({
            "session_key": key, "source_dir": source_dir, "source_rows": len(df_small),
            "retained_every_tenth_rows": len(keep_idx), "sha256": digest,
            "time_finite_rows_full": len(finite_t), "time_nonfinite_rows_full": int(len(times) - len(finite_t)),
            "time_min_seconds": float(sorted_t[0]) if len(sorted_t) else None,
            "time_max_seconds": float(sorted_t[-1]) if len(sorted_t) else None,
            "time_span_seconds": duration,
            "positive_time_delta_count": int(len(positive)),
            "positive_dt_median_seconds": float(np.median(positive)) if len(positive) else None,
            "positive_dt_p90_seconds": float(np.quantile(positive, .9)) if len(positive) else None,
            "positive_dt_min_seconds": float(np.min(positive)) if len(positive) else None,
            "positive_dt_max_seconds": float(np.max(positive)) if len(positive) else None,
            "rows_per_elapsed_second": float(len(df_small) / duration) if duration and duration > 0 else None,
            "retained_time_duplicate_rows": int(len(s["times"]) - len(np.unique(s["times"][np.isfinite(s["times"])]))),
            "labels_raw_source": raw_counts,
            "labels_retained_cache": {p: count_table(labels_sampled[p]) for p in POLICIES},
        })

        # Feature availability by recording × label policy × class; denominators are retained rows.
        for policy, labels in labels_sampled.items():
            availability[policy][key] = {}
            for cls in CLASSES:
                mask = labels == cls
                vals = sample[mask]
                finite_n = np.isfinite(vals).sum(axis=0).astype(int)
                availability[policy][key][cls] = {
                    "rows": int(mask.sum()),
                    "finite_count_by_feature": {c: int(n) for c, n in zip(columns, finite_n)},
                    "finite_rate_by_feature": {c: (float(n / mask.sum()) if mask.sum() else None)
                                               for c, n in zip(columns, finite_n)},
                }
                if cls in CLASSES:
                    gps_class_rows[key][policy][cls]["rows"] = int(mask.sum())
                    gps_mask = np.isfinite(sample[:, col_i["location_latitude"]]) & np.isfinite(sample[:, col_i["location_longitude"]])
                    gps_class_rows[key][policy][cls]["valid_gps"] = int(np.sum(mask & gps_mask))
        for j, col in enumerate(columns):
            vals = sample[:, j]
            finite_vals = vals[np.isfinite(vals)]
            if not len(finite_vals):
                feature_constant_records[col].append({"session_key": key, "status": "all_missing"})
            elif len(np.unique(finite_vals)) <= 1:
                feature_constant_records[col].append({"session_key": key, "status": "constant",
                                                      "finite_rows": int(len(finite_vals))})

        # Per-row exact signatures over sampled rows, canonicalizing NaN and signed zero.
        imu_idx = [col_i[c] for c in imu_columns]
        for row in range(len(sample)):
            sig12[canon_signature(sample[row, imu_idx])][key] += 1
            sigfull[canon_signature(sample[row, :])][key] += 1
            signature_rows += 1

    def duplicate_summary(index: dict) -> dict:
        cross = [(sig, file_counts) for sig, file_counts in index.items() if len(file_counts) >= 2]
        return {
            "cross_record_duplicate_signature_count": len(cross),
            "rows_involved_across_records": int(sum(sum(fc.values()) for _, fc in cross)),
            "excess_occurrences_beyond_one_per_record": int(sum(sum(max(0, n - 1) for n in fc.values()) for _, fc in cross)),
            "recording_pairs_with_any_exact_duplicate_signature": int(len({tuple(pair) for _, fc in cross for pair in __import__("itertools").combinations(sorted(fc), 2)})),
            "examples": [{"signature_sha256": sig, "recordings": sorted(fc), "occurrences_by_recording": fc}
                         for sig, fc in sorted(cross, key=lambda item: sum(item[1].values()), reverse=True)[:20]],
            "coordinates_exported": False,
        }

    duplicate_csv_groups = [sorted(v) for v in source_hash_seen.values() if len(v) > 1]
    lat_i, lon_i = col_i["location_latitude"], col_i["location_longitude"]
    class_counts = {}
    for policy, key in POLICIES.items():
        class_counts[policy] = {c: int(sum(np.sum(s[key] == c) for s in sessions)) for c in CLASSES}
    report = {
        "audit_type": "read_only_cached_feature_and_source_integrity_audit",
        "created_local_date": "2026-10-16", "model_fit_or_score_performed": False,
        "cache_path": str(CACHE_PATH), "cache_sha256": sha256_file(CACHE_PATH),
        "feature_cache_runner_sha256": cache["runner_sha256"], "source_count": len(sessions),
        "source_hashes_revalidated_live": True, "fold_map_sha256": sha256_file(FOLD_MAP_PATH),
        "retained_row_count": int(sum(len(s["labels"]) for s in sessions)),
        "total_source_rows": int(sum(x["source_rows"] for x in session_details)),
        "feature_lists": {"full_numeric_42": columns, "context_8": context_columns, "imu_12": imu_columns},
        "policy_class_counts_retained": class_counts,
        "policy_class_counts_source_all_rows": raw_label_totals,
        "counts_by_source_directory": source_dir_totals,
        "source_csv_duplicate_sha256_groups": duplicate_csv_groups,
        "duplicate_source_file_groups_count": len(duplicate_csv_groups),
        "recording_time_extent_and_label_summaries": session_details,
        "feature_availability_by_recording_policy_class": availability,
        "feature_constant_or_all_missing_by_recording": dict(feature_constant_records),
        "gps_coverage_by_recording_policy_class": gps_class_rows,
        "gps_grid_overlap": {
            "scope": "GPS-complete sampled rows only; Normal recordings with missing GPS remain separate and are not imputed",
            "100m": pair_grid_overlap(sessions, lat_i, lon_i, columns, 100),
            "500m": pair_grid_overlap(sessions, lat_i, lon_i, columns, 500),
        },
        "exact_cross_record_duplicates_sampled_rows": {
            "denominator_rows": signature_rows, "sampling": "every tenth source row from immutable cache",
            "12_axis_imu_vector": duplicate_summary(sig12),
            "all_42_union_numeric_features": duplicate_summary(sigfull),
        },
        "paper_baseline_check_and_full_training_plan": {
            "paper_reported_rf": EXPECTED_RF,
            "paper_reported_split": "80/20 stratified random rows over eight Combined CSV files; paper reports 37 sensor-derived features",
            "current_pilot_recipe": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                                      "max_features": "sqrt", "class_weight": "balanced", "random_state": 42,
                                      "training_cap": 20000},
            "parameter_match": "Current pilot matches max_features/class_weight/random_state but differs in trees, depth, leaf constraint and cap; paper details do not specify min_samples_leaf in the text retrieved.",
            "reproduction_caveat": "Paper's 37-feature engineering and eight-file random split are not reproduced by the current 42-column cache or recording/time-block diagnostics.",
            "planned_no_fit_checks": ["match the paper's stated 300-tree/depth-25 baseline before comparing scores",
                                      "compare capped training with full available training at the same fixed split, seed and imputation",
                                      "record source-row train/test counts, runtime/memory, and whether uncapped training changes predictions/metrics",
                                      "do not use an uncapped model or paper-reported 98.23% as a validation claim without matching feature schema and split"],
            "models_fit_in_this_audit": False,
        },
        "limitations": ["All duplicate checks use 61,730 retained every-tenth sampled rows, not every source row.",
                        "CSV-level source identifiers are not verified trip/route identities.",
                        "GPS grid overlap is coarse location co-occurrence, not route matching or independent leakage evidence.",
                        "The audit does not establish that context features causally explain prior classification differences."]
    }
    out_json = OUT / "cached_data_audit_v1.json"
    if out_json.exists():
        raise FileExistsError(out_json)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_json), "sources": len(sessions), "retained_rows": report["retained_row_count"],
                      "source_hashes_revalidated_live": True, "model_fit_or_scores": False,
                      "class_counts_retained": class_counts,
                      "gps_overlap_100m": report["gps_grid_overlap"]["100m"],
                      "gps_overlap_500m": report["gps_grid_overlap"]["500m"],
                      "duplicate_12axis": report["exact_cross_record_duplicates_sampled_rows"]["12_axis_imu_vector"]["cross_record_duplicate_signature_count"],
                      "duplicate_full42": report["exact_cross_record_duplicates_sampled_rows"]["all_42_union_numeric_features"]["cross_record_duplicate_signature_count"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
