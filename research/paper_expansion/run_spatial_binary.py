"""Frozen observed-coordinate spatial-cell binary diagnostic for RoadSens.

The default action is fit-free preflight. Model fitting requires an explicit
--run plus the exact successful preflight directory. Outputs are versioned and
never overwrite prior evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
RUNS = HERE / "runs"
PROTOCOL = HERE / "spatial_binary_protocol_v1.md"
EXPECTED_PYTHON = Path(r"C:\Users\Administrator\miniconda3\python.exe")
EXPECTED_SKLEARN = "1.9.0"
GPS_NAMES = ("location_latitude", "location_longitude")
LABELS = ("Pothole", "Bump")
GRIDS = (100, 500)
FEATURES = ("full_numeric", "context_only", "imu_only")
MODELS = ("RF100", "HistGB")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def runtime_guard():
    import sklearn
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"wrong interpreter: {sys.executable}")
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(f"wrong sklearn: {sklearn.__version__}")
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "sklearn": sklearn.__version__, "numpy": np.__version__}


def import_runner():
    sys.path.insert(0, str(HERE))
    import runner
    return runner


def build_population(cache):
    sessions = cache["sessions"]
    lat_col = list(sessions[0]["numeric_columns"]).index(GPS_NAMES[0])
    lon_col = list(sessions[0]["numeric_columns"]).index(GPS_NAMES[1])
    # Keep only explicit Bump/Pothole labels in mixed anomaly recordings.
    labels, groups, sources, owners, locals_, lats, lons = [], [], [], [], [], [], []
    rejected_bad_range = 0
    excluded_nonmixed_or_other = 0
    candidate_pb = 0
    for owner, s in enumerate(sessions):
        if not str(s["session_key"]).replace("\\", "/").startswith("combined_anomalies/"):
            excluded_nonmixed_or_other += int(np.sum(np.isin(s["labels"], LABELS)))
            continue
        y = np.asarray(s["labels"], dtype=object)
        num = np.asarray(s["numeric"], dtype=np.float64)
        sr = np.asarray(s["source_row"], dtype=np.int64)
        tm = np.asarray(s["times"], dtype=np.float64)
        ix = np.flatnonzero(np.isin(y, LABELS) & np.isfinite(tm))
        candidate_pb += len(ix)
        lat = num[ix, lat_col]
        lon = num[ix, lon_col]
        finite = np.isfinite(lat) & np.isfinite(lon)
        valid_range = (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
        rejected_bad_range += int(np.sum(finite & ~valid_range))
        keep = finite & valid_range
        ix = ix[keep]
        labels.append(y[ix].astype(str))
        groups.append(np.full(len(ix), s["session_key"], dtype=object))
        sources.append(sr[ix])
        owners.append(np.full(len(ix), owner, dtype=np.int32))
        locals_.append(ix.astype(np.int32))
        lats.append(num[ix, lat_col])
        lons.append(num[ix, lon_col])
    out = {"y": np.concatenate(labels), "group": np.concatenate(groups),
           "source_row": np.concatenate(sources), "owner": np.concatenate(owners),
           "local_row": np.concatenate(locals_), "lat": np.concatenate(lats),
           "lon": np.concatenate(lons)}
    out["global_row"] = np.arange(len(out["y"]), dtype=np.int64)
    return out, {"explicit_pb_finite_time_before_gps": int(candidate_pb),
                 "excluded_pb_rows_outside_mixed_anomaly_paths": int(excluded_nonmixed_or_other),
                 "rejected_finite_out_of_range_coordinates": int(rejected_bad_range),
                 "retained": int(len(out["y"])),
                 "class_counts": {c: int(np.sum(out["y"] == c)) for c in LABELS},
                 "recording_count": int(len(np.unique(out["group"]))) }


def projected_xy(pop):
    radius = 6371008.8
    lat0 = float(np.mean(pop["lat"]))
    x = radius * np.cos(np.deg2rad(lat0)) * np.deg2rad(pop["lon"])
    y = radius * np.deg2rad(pop["lat"])
    return np.column_stack([x, y]), lat0


def make_grid_splits(pop, xy, width):
    from sklearn.model_selection import StratifiedGroupKFold
    cell_xy = np.floor(xy / float(width)).astype(np.int64)
    # String key avoids object tuple compatibility differences across sklearn.
    cell_id = np.asarray([f"{a}:{b}" for a, b in cell_xy], dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    from scipy.spatial import cKDTree
    class_support_ok = True
    for fold, (tr0, te) in enumerate(cv.split(np.zeros(len(pop["y"])), pop["y"], cell_id)):
        tree = cKDTree(xy[te])
        distances, _ = tree.query(xy[tr0], k=1, eps=0)
        retained = tr0[distances > 100.0]
        removed = tr0[distances <= 100.0]
        fit = retained
        counts = {side: {c: int(np.sum(pop["y"][ix] == c)) for c in LABELS}
                  for side, ix in (("candidate_train", tr0), ("test", te),
                                   ("post_buffer_train", retained))}
        feasible = all(counts[side][c] > 0 for side in counts for c in LABELS)
        class_support_ok &= feasible
        folds.append({"fold": fold, "candidate_train": tr0.astype(np.int64),
                      "removed": removed.astype(np.int64), "post_buffer_train": retained.astype(np.int64),
                      "test": te.astype(np.int64), "cell_id": cell_id,
                      "nearest_test_distance_min": float(np.min(distances[distances > 100.0])) if np.any(distances > 100.0) else None,
                      "removed_distance_max": float(np.max(distances[distances <= 100.0])) if np.any(distances <= 100.0) else None,
                      "counts": counts, "feasible": bool(feasible)})
    # Fit cap is applied strictly after spatial buffering.
    runner = import_runner()
    cap_removed = 0
    for f in folds:
        fit = runner.cap_train_indices(f["post_buffer_train"], pop["y"], f["fold"], 42, cap=20000)
        f["fit"] = fit
        f["cap_removed_n"] = int(len(f["post_buffer_train"]) - len(fit))
        cap_removed += f["cap_removed_n"]
        f["counts"]["capped_fit"] = {c: int(np.sum(pop["y"][fit] == c)) for c in LABELS}
        f["feasible"] = bool(f["feasible"] and all(f["counts"]["capped_fit"][c] > 0 for c in LABELS))
        class_support_ok &= f["feasible"]
        if len(set(f["cell_id"][fit]) & set(f["cell_id"][f["test"]])):
            raise AssertionError("test cell appears in fit rows")
        dist, _ = cKDTree(xy[f["test"]]).query(xy[fit], k=1, eps=0)
        if len(dist) and float(np.min(dist)) <= 100.0:
            raise AssertionError("buffered fit row is at most 100m from test")
        f["nearest_test_distance_fit_min"] = float(np.min(dist)) if len(dist) else None
    return folds, cell_xy, cell_id, class_support_ok


def save_preflight(out: Path):
    runtime = runtime_guard()
    runner = import_runner()
    cache, _, _ = runner.validate_immutable_inputs()
    pop, cohort = build_population(cache)
    if cohort["retained"] != 28478:
        raise RuntimeError(f"frozen cohort count mismatch: expected 28478, observed {cohort['retained']}")
    if cohort["rejected_finite_out_of_range_coordinates"] != 0:
        raise RuntimeError("unexpected out-of-range finite coordinates; preserve and stop for review")
    xy, lat0 = projected_xy(pop)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "population_rows.npz", global_row=pop["global_row"],
                        group=pop["group"], source_row=pop["source_row"], y=pop["y"],
                        lat=pop["lat"], lon=pop["lon"], x=xy[:, 0], y_projected=xy[:, 1],
                        owner=pop["owner"], local_row=pop["local_row"])
    summary = {"status": "preflight_complete", "fit_performed": False,
        "created_epoch": time.time(), "cohort": cohort, "projection": {
            "radius_m": 6371008.8, "lat0_mean_degrees": lat0,
            "projection": "x=R*cos(lat0)*radians(lon); y=R*radians(lat)"},
        "source_sha256": cache["source_sha256"], "cache_sha256": sha(PILOT / "session_feature_cache_v1.joblib"),
        "cache_parent_runner_sha256": cache["runner_sha256"],
        "immutable_runner_sha256": sha(HERE / "runner.py"), "protocol_sha256": sha(PROTOCOL),
        "spatial_driver_sha256": sha(Path(__file__)), "runtime": runtime,
        "grids": {}, "models": list(MODELS), "features": list(FEATURES)}
    for width in GRIDS:
        folds, cell_xy, cell_id, feasible = make_grid_splits(pop, xy, width)
        grid_id = f"grid{width}m"
        unique_cells, cell_frequency = np.unique(cell_id, return_counts=True)
        np.savez_compressed(out / f"{grid_id}_splits.npz",
            cell_xy=cell_xy, cell_id=cell_id, unique_cell_id=unique_cells,
            cell_frequency=cell_frequency,
            **{f"fold{i}_{name}": f[name] for i, f in enumerate(folds)
               for name in ("candidate_train", "removed", "post_buffer_train", "fit", "test")})
        counts = [{k: v for k, v in f.items() if k not in ("candidate_train", "removed", "post_buffer_train", "fit", "test", "cell_id")}
                  for f in folds]
        summary["grids"][grid_id] = {"cell_count": int(len(np.unique(cell_id))),
            "class_support_all_folds": bool(feasible), "folds": counts,
            "will_fit": bool(feasible)}
    summary["status"] = "preflight_complete" if all(g["will_fit"] for g in summary["grids"].values()) else "preflight_infeasible"
    membership_files = ["population_rows.npz", "grid100m_splits.npz", "grid500m_splits.npz"]
    summary["membership_files_sha256"] = {name: sha(out / name) for name in membership_files}
    dump_json(out / "preflight.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="Build and validate splits; performs no model fit.")
    ap.add_argument("--run", action="store_true", help="Explicitly run only after reviewed successful preflight.")
    ap.add_argument("--out", type=Path, help="New output directory under paper_expansion/runs.")
    ap.add_argument("--preflight-dir", type=Path, help="Reviewed spatial preflight directory for --run.")
    args = ap.parse_args()
    if args.preflight == args.run or args.out is None:
        ap.error("choose exactly one of --preflight/--run and provide --out")
    out = args.out.resolve()
    if not out.is_relative_to(RUNS.resolve()):
        ap.error("output must be under paper_expansion/runs")
    if args.preflight:
        print(json.dumps(save_preflight(out), ensure_ascii=False, indent=2))
        return
    run_models(out, args.preflight_dir.resolve() if args.preflight_dir else None)


def binary_metrics(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    return {"n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "class_support": {c: int(np.sum(y_true == c)) for c in LABELS},
        "class_recall": {c: float(v) for c, v in zip(LABELS,
            recall_score(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0))},
        "confusion_labels": list(LABELS),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()}


def run_models(out: Path, preflight_dir: Path | None):
    if preflight_dir is None or not (preflight_dir / "preflight.json").is_file():
        raise ValueError("--run requires --preflight-dir with a successful reviewed preflight")
    runtime = runtime_guard()
    pf = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    if pf.get("status") != "preflight_complete" or pf.get("fit_performed") is not False:
        raise ValueError("preflight is not successful and fit-free")
    expected = {"source_sha256": pf["source_sha256"],
                "cache_sha256": sha(PILOT / "session_feature_cache_v1.joblib"),
                "cache_parent_runner_sha256": sha(PILOT / "run_pilot.py"),
                "immutable_runner_sha256": sha(HERE / "runner.py"),
                "protocol_sha256": sha(PROTOCOL),
                "spatial_driver_sha256": sha(Path(__file__))}
    for k, v in expected.items():
        if pf.get(k) != v:
            raise ValueError(f"preflight provenance mismatch for {k}")
    for name, digest in pf.get("membership_files_sha256", {}).items():
        path = preflight_dir / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"preflight membership artifact hash mismatch: {name}")
    if set(pf.get("membership_files_sha256", {})) != {
            "population_rows.npz", "grid100m_splits.npz", "grid500m_splits.npz"}:
        raise ValueError("preflight is missing a frozen membership artifact hash")
    if not all(v["will_fit"] for v in pf["grids"].values()):
        raise ValueError("one or more frozen grid conditions is infeasible; no partial model runs permitted")
    if out.exists():
        raise FileExistsError(out)
    runner = import_runner()
    cache, _, _ = runner.validate_immutable_inputs()
    pop_npz = np.load(preflight_dir / "population_rows.npz", allow_pickle=True)
    pop = {k: pop_npz[k] for k in pop_npz.files}
    if len(pop["y"]) != pf["cohort"]["retained"]:
        raise ValueError("preflight population row count changed")
    live_pop, _ = build_population(cache)
    live_xy, _ = projected_xy(live_pop)
    checks = {"global_row": live_pop["global_row"], "group": live_pop["group"],
              "source_row": live_pop["source_row"], "y": live_pop["y"],
              "lat": live_pop["lat"], "lon": live_pop["lon"],
              "x": live_xy[:, 0], "y_projected": live_xy[:, 1],
              "owner": live_pop["owner"], "local_row": live_pop["local_row"]}
    for key, live_value in checks.items():
        if not np.array_equal(pop[key], live_value):
            raise ValueError(f"preflight population differs from reconstructed live source/cache: {key}")
    from types import SimpleNamespace
    rows = SimpleNamespace(y=pop["y"].astype(str), owners=pop["owner"].astype(np.int32),
                           local_rows=pop["local_row"].astype(np.int32))
    y_ids = {c: i for i, c in enumerate(LABELS)}
    cfg = {"status": "running", "fit_started": True, "created_epoch": time.time(),
        "run_id": out.name, "preflight_dir": str(preflight_dir),
        "preflight_sha256": sha(preflight_dir / "preflight.json"),
        **expected, "runtime": runtime, "grids": list(pf["grids"]),
        "features": list(FEATURES), "models": list(MODELS),
        "folds": 5, "seed": 42, "buffer_m": 100, "cap": 20000,
        "fit_count_expected": 60}
    out.mkdir(parents=True)
    dump_json(out / "run_manifest.json", cfg)
    results, failures = [], []
    try:
        from threadpoolctl import threadpool_limits
        limit_ctx = threadpool_limits(limits=4)
    except Exception:
        from contextlib import nullcontext
        limit_ctx = nullcontext()
    try:
        with limit_ctx:
            for width in GRIDS:
                grid_id = f"grid{width}m"
                if not pf["grids"][grid_id]["will_fit"]:
                    continue
                split_npz = np.load(preflight_dir / f"{grid_id}_splits.npz", allow_pickle=True)
                for feature in FEATURES:
                    x, feature_columns = runner.feature_matrix(cache["sessions"], rows, feature)
                    for model_name in MODELS:
                        preds_rows, preds_true, preds_pred = [], [], []
                        for fold in range(5):
                            fit = split_npz[f"fold{fold}_fit"].astype(np.int64)
                            te = split_npz[f"fold{fold}_test"].astype(np.int64)
                            if set(np.unique(rows.y[fit])) != set(LABELS) or set(np.unique(rows.y[te])) != set(LABELS):
                                raise RuntimeError(f"frozen class support changed at {grid_id}/{feature}/{model_name}/fold{fold+1}")
                            xtr, xte, kept, medians, prep = runner.preprocess_train_test(x, fit, te, feature)
                            estimator = runner._make_estimator(model_name)
                            fit_start = time.perf_counter()
                            rss_before = None
                            try:
                                import psutil
                                rss_before = int(psutil.Process().memory_info().rss)
                            except Exception:
                                pass
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                estimator.fit(xtr, rows.y[fit])
                                fit_seconds = time.perf_counter() - fit_start
                                pred_start = time.perf_counter()
                                pred = np.asarray(estimator.predict(xte), dtype=str)
                                predict_seconds = time.perf_counter() - pred_start
                            if set(np.unique(pred)) - set(LABELS):
                                raise RuntimeError("estimator returned a label outside frozen binary classes")
                            artifact_dir = out / "models" / grid_id / feature
                            artifact_dir.mkdir(parents=True, exist_ok=True)
                            artifact = artifact_dir / f"{model_name}.fold{fold+1}.joblib"
                            joblib.dump({"estimator": estimator, "feature_columns": feature_columns,
                                         "kept_columns": kept, "imputation_medians": medians,
                                         "preprocessing": prep}, artifact, compress=3)
                            y_true = rows.y[te]
                            item = {"grid": grid_id, "feature": feature, "model": model_name,
                                "fold": fold + 1, **binary_metrics(y_true, pred),
                                "train_n": int(len(fit)), "test_n": int(len(te)),
                                "fit_wall_seconds": fit_seconds, "predict_wall_seconds": predict_seconds,
                                "rss_before_fit_bytes": rss_before,
                                "warning_count": len(caught), "warnings": [
                                    {"category": w.category.__name__, "message": str(w.message)} for w in caught],
                                "fit_rows_sha256": hashlib.sha256(fit.tobytes()).hexdigest(),
                                "test_rows_sha256": hashlib.sha256(te.tobytes()).hexdigest(),
                                "model_artifact": artifact.relative_to(out).as_posix(), "model_sha256": sha(artifact)}
                            results.append(item)
                            preds_rows.append(te)
                            preds_true.append(np.asarray([y_ids[str(v)] for v in y_true], dtype=np.int8))
                            preds_pred.append(np.asarray([y_ids[str(v)] for v in pred], dtype=np.int8))
                            print(f"DONE {grid_id}/{feature}/{model_name} fold={fold+1}/5", flush=True)
                        pred_dir = out / "predictions" / grid_id
                        pred_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(pred_dir / f"{feature}__{model_name}.npz",
                            test_row=np.concatenate(preds_rows), y_true=np.concatenate(preds_true),
                            y_pred=np.concatenate(preds_pred))
                        dump_json(out / "metrics.json", {"status": "running", "folds": results})
    except Exception as exc:
        failures.append({"error_type": type(exc).__name__, "error": str(exc)})
        dump_json(out / "failures.json", failures)
        cfg["status"] = "failed_preserved"
        cfg["failure_count"] = len(failures)
        dump_json(out / "run_manifest.json", cfg)
        raise
    cfg["status"] = "phase_complete" if len(results) == cfg["fit_count_expected"] else "incomplete"
    cfg["fit_count_completed"] = len(results)
    dump_json(out / "run_manifest.json", cfg)
    dump_json(out / "metrics.json", {"status": cfg["status"], "folds": results})


if __name__ == "__main__":
    main()
