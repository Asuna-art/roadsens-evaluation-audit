from __future__ import annotations

"""Strong temporal-context control for the frozen RoadSens pilot.

Default execution is synthetic QA only. Real model fitting requires explicit --run
and a sessions cache produced by run_pilot.py; this script never changes the parent.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNNER_PATH = ROOT / "run_pilot.py"
OUT_DEFAULT = ROOT / "runs" / "axis_window_v1"
WINDOW_SECONDS = 1.0
STATS = ("mean", "std0", "rms", "range", "absmax")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_parent():
    spec = importlib.util.spec_from_file_location("roadsens_run_pilot_axis_control", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen parent runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def axis_columns(numeric_columns: list[str]) -> list[str]:
    expected = {f"{prefix}{axis}" for prefix in
                ("accelerometer_", "gyroscope_", "gravity_", "totalAcceleration_")
                for axis in "xyz"}
    missing = sorted(expected - set(numeric_columns))
    if missing:
        raise ValueError(f"Frozen numeric schema lacks required raw IMU axis columns: {missing}")
    selected = [c for c in numeric_columns if c in expected]
    if len(selected) != 12:
        raise ValueError(f"Expected exactly 12 raw IMU axes; found {len(selected)}")
    return selected


def build_axis_features(sessions: list[dict], parent) -> tuple[list[dict], list[str]]:
    """Calculate 12-axis past-inclusive 1 s statistics at source cadence, then decimate."""
    output = []
    names = [f"{col}_{stat}" for col in axis_columns(sessions[0]["numeric_columns"])
             for stat in STATS]
    for s in sessions:
        cols = axis_columns(s["numeric_columns"])
        if list(s["imu_columns"]) != cols:
            raise ValueError(f"axis_window columns do not exactly match imu_only: {s['session_key']}")
        index = {c: i for i, c in enumerate(s["numeric_columns"])}
        raw = np.asarray(s["numeric_full"][:, [index[c] for c in cols]], dtype=np.float64)
        raw[~np.isfinite(raw)] = np.nan
        # Parent implementation defines order, inclusive [t-1s, t], duplicate timestamp,
        # and non-finite-value semantics. It inserts rows stably before computing features.
        full = parent.causal_window_features(s["times_full"], raw, window_seconds=WINDOW_SECONDS)
        source_rows = np.asarray(s["source_row"], dtype=np.int64)
        if len(source_rows) and (source_rows.min() < 0 or source_rows.max() >= len(full)):
            raise ValueError(f"source_row outside full-cadence feature array: {s['session_key']}")
        output.append({**s, "axis_window": full[source_rows]})
    return output, names


def _naive_reference(times: np.ndarray, values: np.ndarray, window_seconds: float) -> np.ndarray:
    """Deliberately simple row-by-row reference for synthetic cases only."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full((len(times), values.shape[1] * 5), np.nan)
    order = np.flatnonzero(np.isfinite(times))
    order = order[np.argsort(times[order], kind="stable")]
    seen: list[int] = []
    for i in order:
        seen.append(int(i))
        rows = [j for j in seen if times[j] >= times[i] - window_seconds]
        for d in range(values.shape[1]):
            x = values[rows, d]
            x = x[np.isfinite(x)]
            if x.size:
                out[i, d * 5:(d + 1) * 5] = (
                    x.mean(), x.std(ddof=0), np.sqrt(np.mean(x * x)),
                    x.max() - x.min(), np.max(np.abs(x)))
    return out


def run_synthetic_qa() -> dict:
    parent = load_parent()
    # Duplicate times are inserted in source-row order. NaNs are ignored per-axis;
    # an observation exactly one second old is included; older values and NaN times are not.
    t = np.array([0.0, 0.5, 1.0, 1.0, 2.0000001, 3.2, np.nan])
    x = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0], [7.0, np.nan],
                  [9.0, 8.0], [11.0, 10.0], [999.0, 999.0]])
    fast = parent.causal_window_features(t, x, window_seconds=WINDOW_SECONDS)
    naive = _naive_reference(t, x, WINDOW_SECONDS)
    assert np.allclose(fast, naive, equal_nan=True, atol=1e-12)
    # Explicit boundary checks at t=1.0 include t=0.0 and both equal-time rows;
    # a causal duplicate tie sees only rows through itself in stable source order.
    assert np.isclose(fast[2, 0], (1.0 + 3.0 + 5.0) / 3.0)
    assert np.isclose(fast[3, 0], (1.0 + 3.0 + 5.0 + 7.0) / 4.0)
    assert np.isclose(fast[4, 0], 9.0)  # t=0.0 is outside [1.0000001, 2.0000001]
    assert np.isnan(fast[-1]).all()     # non-finite timestamp has no window features

    # Session boundaries are handled by separate calls, as in the parent per-session path.
    session_a = parent.causal_window_features(np.array([0.0, 0.5]), np.array([[1000.0], [1000.0]]))
    session_b = parent.causal_window_features(np.array([0.0, 0.5]), np.array([[2.0], [2.0]]))
    assert session_b[0, 0] == 2.0 and session_b[1, 0] == 2.0

    expected = axis_columns([
        f"{p}{a}" for p in ("accelerometer_", "gyroscope_", "gravity_", "totalAcceleration_")
        for a in "xyz"])
    assert len(expected) == 12
    names = [f"{c}_{stat}" for c in expected for stat in STATS]
    assert len(names) == 60 and len(set(names)) == 60
    toy_numeric = np.arange(5 * 12, dtype=float).reshape(5, 12)
    toy_session = {"numeric_columns": expected, "imu_columns": expected,
                   "numeric_full": toy_numeric, "times_full": np.array([0., .5, 1., 1.5, 2.]),
                   "source_row": np.array([0, 2, 4]), "session_key": "synthetic/session.csv"}
    built, built_names = build_axis_features([toy_session], parent)
    direct_full = parent.causal_window_features(toy_session["times_full"], toy_numeric)
    assert built_names == names
    assert built[0]["axis_window"].shape == (3, 60)
    assert np.allclose(built[0]["axis_window"], direct_full[[0, 2, 4]], equal_nan=True)
    return {
        "passed": True,
        "scope": "synthetic-only; no RoadSens data loaded; no model fit or scores",
        "window_seconds": WINDOW_SECONDS,
        "window_interval": "[current_time - 1.0, current_time], past-inclusive; stable source-row order at duplicate timestamps",
        "boundary_float_rule": "timestamps are float64 seconds_elapsed; include iff timestamp >= current_time - 1.0 (no epsilon/tolerance)",
        "feature_shape_per_row": [12, 5, 60],
        "checks": ["optimized parent rolling equals naive rowwise reference with duplicates/NaN/gaps",
                   "left boundary inclusive and older-than-window excluded",
                   "duplicate timestamp rows do not see later source rows",
                   "non-finite timestamps produce missing features",
                   "window state resets at session boundary",
                   "exactly 12 raw IMU axes and 60 named statistics",
                   "source-cadence features are computed before source_row decimation"]
    }


def run_formal(cache_path: Path, output_dir: Path) -> dict:
    parent = load_parent()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if not cache_path.is_file():
        raise FileNotFoundError(f"Validated run_pilot sessions cache required: {cache_path}")
    import joblib
    cache = joblib.load(cache_path)
    parent_hash = sha256_file(RUNNER_PATH)
    fold_payload = parent.load_frozen_fold_map()
    if cache.get("runner_sha256") != parent_hash:
        raise ValueError("session cache was built from a different run_pilot.py version")
    if cache.get("source_sha256") != fold_payload["source_sha256"]:
        raise ValueError("session cache hashes do not match frozen session fold map")
    sessions, headers = cache["sessions"], cache["headers"]
    if set(s["session_key"] for s in sessions) != set(fold_payload["session_fold"]):
        raise ValueError("session cache does not match frozen fold-map sessions")
    if set(cache.get("source_sha256", {})) != set(fold_payload["source_sha256"]):
        raise ValueError("session cache source-hash keys differ from frozen fold map")
    for s in sessions:
        key = s["session_key"]
        expected_hash = fold_payload["source_sha256"][key]
        if s["sha256"] != expected_hash or cache["source_sha256"].get(key) != expected_hash:
            raise ValueError(f"session source hash mismatch: {s['session_key']}")
        live_path = parent.DATA / Path(key)
        if not live_path.is_file() or sha256_file(live_path) != expected_hash:
            raise ValueError(f"live source CSV hash mismatch: {key}")
    # Verify row identity using the cheap cached imu population before computing windows.
    fmap = fold_payload["session_fold"]

    # Verify every row identity against the already-fitted pilot_v2 IMU checkpoint
    # before creating output or fitting anything. This catches population, order,
    # fold, and deterministic train-cap drift.
    pilot_dir = ROOT / "runs" / "pilot_v2"
    validated_splits: dict[str, list[dict]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for policy in ("primary", "blank_normal"):
        policy_sessions = sessions
        if policy == "blank_normal":
            policy_sessions = [{**s, "labels": s["labels_blank_normal"]} for s in sessions]
        _, y, groups, owner = parent._concat(policy_sessions, "imu")
        policy_folds = parent._build_folds(y, groups, "group", fmap)
        all_labels = np.concatenate([s["labels"] for s in policy_sessions])
        all_times = np.concatenate([s["times"] for s in policy_sessions])
        all_source_rows = np.concatenate([s["source_row"] for s in policy_sessions])
        valid = np.isin(all_labels, parent.CLASSES) & np.isfinite(all_times)
        source_rows = all_source_rows[valid]
        if len(source_rows) != len(y):
            raise AssertionError(f"{policy}: source-row alignment differs from imu_only population")
        checkpoint_path = pilot_dir / f"{policy}_group.imu_only.checkpoint.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Pilot IMU checkpoint required for row identity verification: {checkpoint_path}")
        checkpoint_hashes[policy] = sha256_file(checkpoint_path)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (checkpoint.get("policy") != policy or checkpoint.get("split") != "group"
                or checkpoint.get("method") != "imu_only" or checkpoint.get("rf_config") != parent.RF_CONFIG
                or len(checkpoint.get("folds", [])) != 5):
            raise ValueError(f"Unexpected corresponding IMU checkpoint metadata: {checkpoint_path.name}")
        rows_for_policy = []
        for fold0, (tr0, te) in enumerate(policy_folds):
            if set(np.unique(y[tr0])) != set(parent.CLASSES) or set(np.unique(y[te])) != set(parent.CLASSES):
                raise RuntimeError(f"Frozen {policy} fold {fold0 + 1} lacks a class; stop without reseeding")
            tr = parent._cap_train(tr0, y, fold0)
            expected = checkpoint["folds"][fold0]
            split_rows = [{"row": int(i), "session": str(groups[i]), "source_row": int(source_rows[i])}
                          for i in tr0]
            fit_rows = [{"row": int(i), "session": str(groups[i]), "source_row": int(source_rows[i])}
                        for i in tr]
            test_rows = [{"row": int(i), "session": str(groups[i]), "source_row": int(source_rows[i])}
                         for i in te]
            for field, got in (("split_train_rows", split_rows), ("fit_train_rows", fit_rows)):
                ref = [{k: row[k] for k in ("row", "session", "source_row")}
                       for row in expected.get(field, [])]
                if got != ref:
                    raise AssertionError(f"{policy} fold {fold0 + 1}: {field} differ from pilot_v2 imu_only")
            ref_test = [{k: row[k] for k in ("row", "session", "source_row")}
                        for row in expected.get("test_rows", [])]
            if test_rows != ref_test:
                raise AssertionError(f"{policy} fold {fold0 + 1}: test_rows differ from pilot_v2 imu_only")
            if len(tr) != expected.get("train_n") or len(tr0) != expected.get("train_n_before_cap"):
                raise AssertionError(f"{policy} fold {fold0 + 1}: train counts differ from pilot_v2 imu_only")
            rows_for_policy.append({"train": tr, "train_before_cap": tr0, "test": te,
                                    "source_rows": source_rows, "groups": groups, "y": y,
                                    "owner": owner})
        validated_splits[policy] = rows_for_policy

    sessions, features = build_axis_features(sessions, parent)
    for policy in ("primary", "blank_normal"):
        policy_sessions = sessions
        if policy == "blank_normal":
            policy_sessions = [{**s, "labels": s["labels_blank_normal"]} for s in sessions]
        x, y_axis, groups_axis, owner_axis = parent._concat(policy_sessions, "axis_window")
        first = validated_splits[policy][0]
        if not (np.array_equal(y_axis, first["y"]) and np.array_equal(groups_axis, first["groups"])
                and np.array_equal(owner_axis, first["owner"])):
            raise AssertionError(f"{policy}: axis_window row population/order differs from imu_only")
        for split in validated_splits[policy]:
            split["x"] = x

    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "method": "axis_window",
        "policy_results": {},
        "summary": {},
        "schema": features,
        "feature_count": len(features),
        "raw_axis_columns": axis_columns(sessions[0]["numeric_columns"]),
        "window_seconds": WINDOW_SECONDS,
        "window_interval": "[current_time - 1.0, current_time], stable source-row order at duplicate timestamps",
        "boundary_float_rule": "float64 seconds_elapsed; exact comparison timestamp >= now - 1.0; no epsilon/tolerance",
        "statistics": list(STATS),
        "split": "frozen session fold map v1, group only; same population/order as imu_only",
        "rf_config": parent.RF_CONFIG,
        "session_cache_sha256": sha256_file(cache_path),
        "run_pilot_sha256": parent_hash,
        "axis_control_code_sha256": sha256_file(Path(__file__).resolve()),
        "session_fold_map_sha256": sha256_file(ROOT / "session_fold_map_v1.json"),
        "pilot_v2_imu_checkpoint_sha256": checkpoint_hashes,
        "row_identity_match_vs_pilot_v2_imu_only": True,
        "live_101_source_hashes_validated": True,
        "formal_fit_started": True,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({k: v for k, v in result.items()
                                         if k not in {"policy_results", "formal_fit_started"}},
                                        indent=2, ensure_ascii=False), encoding="utf-8")

    from sklearn.ensemble import RandomForestClassifier
    for policy in ("primary", "blank_normal"):
        policy_records = []
        records = []
        for fold0, split in enumerate(validated_splits[policy]):
            tr0, tr, te = split["train_before_cap"], split["train"], split["test"]
            y, groups = split["y"], split["groups"]
            source_rows = split["source_rows"]
            x = split["x"]
            xtr, xte, kept, medians = parent._impute_train_test(x[tr], x[te])
            model = RandomForestClassifier(**parent.RF_CONFIG)
            t0 = time.perf_counter(); model.fit(xtr, y[tr]); fit_seconds = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = model.predict(xte); predict_seconds = time.perf_counter() - t0
            model_path = output_dir / f"{policy}_group.axis_window.fold{fold0 + 1}.joblib"
            joblib.dump(model, model_path, compress=3)
            from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
            metric = parent._metrics(y[te], pred)
            record = {
                "fold": fold0 + 1,
                "train_n_before_cap": int(len(tr0)), "train_n": int(len(tr)), "test_n": int(len(te)),
                "train_class_counts": {c: int(np.sum(y[tr] == c)) for c in parent.CLASSES},
                "test_class_counts": {c: int(np.sum(y[te] == c)) for c in parent.CLASSES},
                **metric,
                "per_session": {},
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "model_artifact": model_path.name, "model_sha256": sha256_file(model_path),
                "kept_feature_indices": kept, "training_imputation_medians": medians,
                "split_train_rows": [{"row": int(i), "session": str(groups[i]),
                                      "source_row": int(source_rows[i])} for i in tr0],
                "fit_train_rows": [{"row": int(i), "session": str(groups[i]),
                                    "source_row": int(source_rows[i])} for i in tr],
                "test_rows": [{"row": int(i), "session": str(groups[i]),
                               "source_row": int(source_rows[i]), "true": str(y[i]), "pred": str(p)}
                              for i, p in zip(te, pred)],
            }
            for sess in np.unique(groups[te]):
                ix_sess = groups[te] == sess
                record["per_session"][str(sess)] = {"n": int(ix_sess.sum()),
                                                       **parent._metrics(y[te][ix_sess], pred[ix_sess])}
            records.append(record)
            policy_records.append({"fold": fold0 + 1,
                                   "macro_f1": metric["macro_f1"],
                                   "balanced_accuracy": metric["balanced_accuracy"],
                                   "recall_by_class": metric["recall_by_class"],
                                   "train_n": int(len(tr)), "test_n": int(len(te))})
            (output_dir / f"{policy}_group.axis_window.fold{fold0 + 1}.checkpoint.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"DONE {policy}/group/axis_window fold={fold0 + 1}/5 "
                  f"macro_f1={metric['macro_f1']:.4f}", flush=True)
        result["policy_results"][policy] = records
        result["summary"][policy] = policy_records
        (output_dir / f"{policy}_group.axis_window.json").write_text(
            json.dumps({"policy": policy, "split": "group", "method": "axis_window",
                        "rf_config": parent.RF_CONFIG, "folds": records}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-qa", action="store_true", help="Synthetic equivalence tests only; no data or model fit.")
    ap.add_argument("--run", action="store_true", help="Explicitly fit the formal group-heldout axis_window control.")
    ap.add_argument("--sessions-cache", type=Path, help="Cache created by run_pilot.py --precompute-cache.")
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    if args.synthetic_qa:
        if args.run:
            ap.error("Use either --synthetic-qa or --run, not both")
        result = run_synthetic_qa()
        path = ROOT / "synthetic_axis_window_qa.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.run:
        if args.sessions_cache is None:
            ap.error("--run requires --sessions-cache; create it separately with run_pilot.py --precompute-cache")
        result = run_formal(args.sessions_cache, args.output_dir)
        print(json.dumps({"completed": True, "output_dir": str(args.output_dir),
                          "formal_fit_started": result["formal_fit_started"]}, indent=2))
    else:
        ap.error("Choose --synthetic-qa or explicitly authorized --run")


if __name__ == "__main__":
    main()
