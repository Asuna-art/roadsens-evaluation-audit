"""Fit-free preflight and approved sampling-stride sensitivity experiment.

The --preflight command reads the immutable full-cadence feature cache and live
source labels, reconstructs strides 5/10/20, checks the existing stride-10 A
identity and frozen recording map, then writes compact row/split manifests.
The --run command is intentionally blocked until an explicit root approval file
is supplied. Outputs are confined to paper_expansion/key_experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PILOT = EXP.parent
ROOT = PILOT.parent
CACHE = PILOT / "session_feature_cache_v1.joblib"
MAP42 = PILOT / "session_fold_map_v1.json"
OUTROOT = HERE / "runs"
PYTHON = Path(r"C:\Users\Administrator\miniconda3\python.exe")
SKLEARN_VERSION = "1.9.0"
POLICIES = ("primary", "blank_normal")
STRIDES = (5, 10, 20)
FIT_STRIDES = (5, 20)
FEATURES = ("full_numeric", "imu_only")
CLASSES = ("Normal", "Pothole", "Bump")
CAP = 20000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def ndarray_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def object_rows_hash(groups, source_rows, labels) -> str:
    return canonical_hash([[str(g), int(r), str(y)] for g, r, y in zip(groups, source_rows, labels)])


def cap_rows(train: np.ndarray, y: np.ndarray, fold_zero: int, seed: int = 42) -> np.ndarray:
    train = np.asarray(train, dtype=np.int64)
    if len(train) <= CAP:
        return train.copy()
    rng = np.random.RandomState(seed * 1000 + fold_zero)
    classes = np.unique(y[train])
    counts = {c: int(np.sum(y[train] == c)) for c in classes}
    total = sum(counts.values())
    quota = {c: min(counts[c], max(1, int(CAP * counts[c] / total))) for c in classes}
    while sum(quota.values()) > CAP:
        c = max(quota, key=quota.get)
        quota[c] -= 1
    chosen = []
    for c in classes:
        ix = train[y[train] == c]
        chosen.extend(rng.choice(ix, quota[c], replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def live_labels(path: Path, normal_directory: bool, expected_n: int) -> np.ndarray:
    if normal_directory:
        return np.full(expected_n, "Normal", dtype=object)
    header = pd.read_csv(path, nrows=0).columns
    if "annotation_text" not in header:
        return np.full(expected_n, "", dtype=object)
    df = pd.read_csv(path, usecols=["annotation_text"], low_memory=False)
    if "annotation_text" not in df:
        raw = np.full(expected_n, "", dtype=object)
    else:
        raw = df["annotation_text"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
    out = np.full(len(raw), "", dtype=object)
    out[np.isin(raw, ("Pothole", "Bump"))] = raw[np.isin(raw, ("Pothole", "Bump"))]
    return out


def read_locked_inputs():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use locked interpreter {PYTHON}; got {sys.executable}")
    import sklearn
    if sklearn.__version__ != SKLEARN_VERSION:
        raise RuntimeError(f"Expected scikit-learn {SKLEARN_VERSION}, got {sklearn.__version__}")
    cache = joblib.load(CACHE)
    frozen = json.loads(MAP42.read_text(encoding="utf-8"))
    parent = PILOT / "run_pilot.py"
    if cache.get("runner_sha256") != sha(parent):
        raise ValueError("immutable parent runner SHA differs from cache")
    if cache.get("source_sha256") != frozen.get("source_sha256"):
        raise ValueError("cache and frozen seed-42 map source hashes differ")
    sessions = cache["sessions"]
    if len(sessions) != 101 or set(frozen["session_fold"].values()) != {1, 2, 3, 4, 5}:
        raise ValueError("unexpected recording count or frozen map folds")
    for s in sessions:
        key = s["session_key"]
        src = PILOT / "data" / "inner" / Path(key)
        if sha(src) != s["sha256"] or s["sha256"] != frozen["source_sha256"].get(key):
            raise ValueError(f"live source SHA mismatch: {key}")
        n = int(s["all_rows"])
        if np.asarray(s["numeric_full"]).shape != (n, len(s["numeric_columns"])):
            raise ValueError(f"full numeric cache shape mismatch: {key}")
        if np.asarray(s["times_full"]).shape != (n,):
            raise ValueError(f"full time cache shape mismatch: {key}")
        if s["numeric_columns"] != sessions[0]["numeric_columns"] or s["imu_columns"] != sessions[0]["imu_columns"]:
            raise ValueError(f"feature schema/order mismatch: {key}")
    return cache, frozen


def reconstruct(cache: dict, frozen: dict, stride: int, policy: str):
    groups, source_rows, labels, times, owners, local_rows, xfull = [], [], [], [], [], [], []
    sessions = cache["sessions"]
    for owner, s in enumerate(sessions):
        key = s["session_key"]
        source = PILOT / "data" / "inner" / Path(key)
        full_labels = live_labels(source, s["group"] == "normal", int(s["all_rows"]))
        if len(full_labels) != int(s["all_rows"]):
            raise ValueError(f"raw label rows do not match cached rows: {key}")
        take = np.arange(0, len(full_labels), stride, dtype=np.int64)
        y = full_labels[take].copy()
        if policy == "blank_normal" and s["group"] != "normal":
            y[y == ""] = "Normal"
        t = np.asarray(s["times_full"], dtype=np.float64)[take]
        valid = np.isfinite(t) & np.isin(y, CLASSES)
        take, y, t = take[valid], y[valid], t[valid]
        if stride == 10:
            cache_rows = np.asarray(s["source_row"], dtype=np.int64)
            cache_labels = np.asarray(s["labels"] if policy == "primary" else s["labels_blank_normal"], dtype=object)
            cache_times = np.asarray(s["times"], dtype=np.float64)
            cache_valid = np.isfinite(cache_times) & np.isin(cache_labels, CLASSES)
            if not np.array_equal(take, cache_rows[cache_valid]):
                raise ValueError(f"stride-10 raw/cache source-row mismatch: {key}/{policy}")
            if not np.array_equal(y, cache_labels[cache_valid]) or not np.array_equal(t, cache_times[cache_valid]):
                raise ValueError(f"stride-10 raw/cache labels or times mismatch: {key}/{policy}")
            sampled = take // 10
            if not np.array_equal(np.asarray(s["numeric_full"], dtype=np.float64)[take],
                                  np.asarray(s["numeric"], dtype=np.float64)[sampled], equal_nan=True):
                raise ValueError(f"stride-10 numeric values differ from immutable cache: {key}")
        groups.extend([key] * len(take))
        source_rows.extend(take.tolist())
        labels.extend(y.tolist())
        times.extend(t.tolist())
        owners.extend([owner] * len(take))
        local_rows.extend(take.tolist())
        xfull.extend(np.asarray(s["numeric_full"], dtype=np.float64)[take].tolist())
    table = {
        "group": np.asarray(groups, dtype=str),
        "source_row": np.asarray(source_rows, dtype=np.int64),
        "y": np.asarray(labels, dtype=object),
        "time": np.asarray(times, dtype=np.float64),
        "owner": np.asarray(owners, dtype=np.int16),
        "local_row": np.asarray(local_rows, dtype=np.int64),
        "x_full": np.asarray(xfull, dtype=np.float64),
    }
    # Match the immutable paper runner's source-file order and fixed seed-42 map.
    fmap = frozen["session_fold"]
    folds = []
    yy = table["y"]
    gg = table["group"]
    for fold in range(1, 6):
        test = np.flatnonzero(np.fromiter((fmap[g] == fold for g in gg), dtype=bool, count=len(gg)))
        train = np.setdiff1d(np.arange(len(yy), dtype=np.int64), test, assume_unique=True)
        fit = cap_rows(train, yy, fold - 1)
        for side, ix in (("train", train), ("fit", fit), ("test", test)):
            counts = {c: int(np.sum(yy[ix] == c)) for c in CLASSES}
            if any(v == 0 for v in counts.values()):
                raise ValueError(f"missing class: stride={stride}/{policy}/fold={fold}/{side}/{counts}")
        if set(gg[train]) & set(gg[test]):
            raise AssertionError("recording overlap in fixed recording-heldout map")
        folds.append((train, fit, test))
    return table, folds


def verify_stride10_matches_A(cache, frozen, outputs: dict) -> dict:
    """Check stride-10 full-feature and IMU membership against immutable A checkpoints."""
    sys.path.insert(0, str(EXP))
    import runner
    result = {}
    for policy, exp_policy in (("primary", "primary"), ("blank_normal", "blank_normal")):
        table, folds = outputs[(10, policy)]
        base = PILOT / "runs" / "pilot_v2" / f"{exp_policy}_group"
        rr = runner.policy_rows(cache["sessions"], policy)
        if object_rows_hash(rr.groups, rr.source_rows, rr.y) != object_rows_hash(
                table["group"], table["source_row"], table["y"]):
            raise ValueError(f"stride-10 ordered row identity differs from A cache: {policy}")
        numeric_parts = []
        imu_parts = []
        for s in cache["sessions"]:
            ix = np.flatnonzero(rr.groups == s["session_key"])
            local = rr.source_rows[ix]
            numeric_parts.append(np.asarray(s["numeric_full"], dtype=np.float64)[local])
            colmap = {name: i for i, name in enumerate(s["numeric_columns"])}
            imu_idx = [colmap[name] for name in s["imu_columns"]]
            imu_parts.append(np.asarray(s["numeric_full"], dtype=np.float64)[local][:, imu_idx])
        x10_full = np.concatenate(numeric_parts)
        x10_imu = np.concatenate(imu_parts)
        if not np.array_equal(x10_full, table["x_full"], equal_nan=True):
            raise ValueError(f"stride-10 numeric feature rows differ from cache: {policy}")
        # Verify the original cached IMU representation as an additional column/order guard.
        x10_cached_imu = []
        for s in cache["sessions"]:
            ix = np.flatnonzero(rr.groups == s["session_key"])
            local_sample_ids = rr.local_rows[ix]
            x10_cached_imu.append(np.asarray(s["imu"], dtype=np.float64)[local_sample_ids])
        if not np.array_equal(x10_imu, np.concatenate(x10_cached_imu), equal_nan=True):
            raise ValueError(f"stride-10 IMU feature rows differ from cache: {policy}")
        result[f"{policy}_stride10_feature_rows"] = {
            "full_numeric_exact_cache_match": True,
            "imu_exact_cache_match": True,
            "full_numeric_sha256": ndarray_hash(x10_full),
            "imu_sha256": ndarray_hash(x10_imu),
        }
        verified = {}
        for feature in FEATURES:
            ckpt = json.loads(Path(str(base) + f".{feature}.checkpoint.json").read_text(encoding="utf-8"))
            for fold, (train, fit, test) in enumerate(folds):
                old = ckpt["folds"][fold]
                for field, idx in (("split_train_rows", train), ("fit_train_rows", fit), ("test_rows", test)):
                    got = [(int(a["row"]), str(a["session"]), int(a["source_row"])) for a in old[field]]
                    want = [(int(i), str(rr.groups[i]), int(rr.source_rows[i])) for i in idx]
                    if got != want:
                        raise ValueError(f"A row identity mismatch: {policy}/{feature}/fold{fold+1}/{field}")
            verified[feature] = "all five train/fit/test row sequences identical to frozen A checkpoint"
        result[policy] = verified
    return result


def create_preflight(run_id: str) -> Path:
    out = OUTROOT / run_id
    if out.exists():
        raise FileExistsError(out)
    cache, frozen = read_locked_inputs()
    outputs = {(stride, policy): reconstruct(cache, frozen, stride, policy)
               for stride in STRIDES for policy in POLICIES}
    identity = verify_stride10_matches_A(cache, frozen, outputs)
    out.mkdir(parents=True)
    (out / "row_indices").mkdir()
    (out / "split_indices").mkdir()
    catalog = []
    for stride in STRIDES:
        for policy in POLICIES:
            table, folds = outputs[(stride, policy)]
            row_path = out / "row_indices" / f"stride{stride}__{policy}.npz"
            np.savez_compressed(row_path, group=table["group"], source_row=table["source_row"],
                                y=np.asarray([CLASSES.index(str(y)) for y in table["y"]], dtype=np.int8),
                                time=table["time"], owner=table["owner"], local_row=table["local_row"])
            method_meta = []
            split_arrays = {}
            for fold, (train, fit, test) in enumerate(folds, start=1):
                split_arrays[f"train_{fold-1}"] = train
                split_arrays[f"fit_{fold-1}"] = fit
                split_arrays[f"test_{fold-1}"] = test
                counts = lambda ix: {c: int(np.sum(table["y"][ix] == c)) for c in CLASSES}
                method_meta.append({
                    "fold": fold, "train_n": int(len(train)), "fit_n": int(len(fit)), "test_n": int(len(test)),
                    "train_class_counts": counts(train), "fit_class_counts": counts(fit), "test_class_counts": counts(test),
                    "train_sessions": int(len(set(table["group"][train]))),
                    "fit_sessions": int(len(set(table["group"][fit]))),
                    "test_sessions": int(len(set(table["group"][test]))),
                    "train_test_session_intersection": sorted(set(table["group"][train]) & set(table["group"][test])),
                    "train_rows_sha256": canonical_hash(train.tolist()),
                    "fit_rows_sha256": canonical_hash(fit.tolist()),
                    "test_rows_sha256": canonical_hash(test.tolist()),
                })
            split_path = out / "split_indices" / f"stride{stride}__{policy}__record42.npz"
            np.savez_compressed(split_path, **split_arrays)
            catalog.append({
                "stride": stride, "policy": policy, "fit_status": "planned" if stride in FIT_STRIDES else "reference_only",
                "feature_sets": list(FEATURES),
                "model": {"name": "RandomForestClassifier", "n_estimators": 100, "max_depth": 20,
                          "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced",
                          "random_state": 42, "n_jobs": 4},
                "split": "frozen recording-heldout seed42; five folds; original source file assignments",
                "population_n": int(len(table["y"])), "recording_count": int(len(set(table["group"]))),
                "class_counts": {c: int(np.sum(table["y"] == c)) for c in CLASSES},
                "row_identity_sha256": object_rows_hash(table["group"], table["source_row"], table["y"]),
                "feature_data_sha256": {
                    "full_numeric": ndarray_hash(table["x_full"]),
                    "imu_only": ndarray_hash(table["x_full"][:, [cache["sessions"][0]["numeric_columns"].index(c)
                                                                  for c in cache["sessions"][0]["imu_columns"]]]),
                },
                "row_file": str(row_path.relative_to(out).as_posix()), "row_file_sha256": sha(row_path),
                "split_file": str(split_path.relative_to(out).as_posix()), "split_file_sha256": sha(split_path),
                "folds": method_meta,
            })
    import sklearn
    manifest = {
        "status": "preflight_complete_fit_free", "fit_performed": False, "scores_computed": False,
        "created_unix": time.time(), "run_id": run_id, "strides": list(STRIDES),
        "interpretation": "The original stride-10 population is the reference; stride-5/20 populations differ and comparisons are descriptive sampling sensitivity, not paired causal effects.",
        "sampling_rule": "Within each of 101 source CSVs, choose raw source row indices 0, stride, 2*stride, ... before policy filtering; derive labels from the live original annotation_text and immutable cached full-cadence numeric/time arrays.",
        "policies": list(POLICIES), "features": list(FEATURES), "seed": 42, "recording_count": 101,
        "class_order": list(CLASSES), "cap": CAP, "cap_sampler": "original class-proportional RandomState(seed*1000+zero_based_fold), then sorted indices",
        "source_sha256": cache["source_sha256"], "cache_sha256": sha(CACHE),
        "parent_runner_sha256": cache["runner_sha256"], "seed42_record_map_sha256": sha(MAP42),
        "protocol_sha256": sha(EXP / "protocol_v1.md"), "frozen_split_authority_sha256": sha(EXP / "frozen_split_authority_v1.md"),
        "key_experiments_protocol_sha256": sha(HERE / "protocol_key_v1.md"),
        "shared_runner_sha256": sha(EXP / "runner.py"),
        "feature_columns": {
            "full_numeric": cache["sessions"][0]["numeric_columns"],
            "full_numeric_columns_sha256": canonical_hash(cache["sessions"][0]["numeric_columns"]),
            "imu_only": cache["sessions"][0]["imu_columns"],
            "imu_columns_sha256": canonical_hash(cache["sessions"][0]["imu_columns"]),
        },
        "driver_sha256": sha(Path(__file__)),
        "runtime": {"executable": str(Path(sys.executable).resolve()), "python": sys.version,
                    "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "stride10_matches_frozen_A": identity,
        "fit_strides": list(FIT_STRIDES), "reference_stride": 10,
        "jobs_planned": len(FIT_STRIDES) * len(POLICIES) * len(FEATURES),
        "fits_planned": len(FIT_STRIDES) * len(POLICIES) * len(FEATURES) * 5,
        "jobs": catalog,
    }
    (out / "preflight_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Sampling-stride sensitivity preflight\n\n"
        "Fit-free reconstruction of stride-5, stride-10, and stride-20 populations. "
        "The stride-10 identities match the immutable A recording-heldout full_numeric and imu_only checkpoints. "
        "Stride populations differ, so between-stride score differences will be descriptive sensitivity, not paired causal effects.\n\n"
        "No model was fitted and no scores were computed. Root review is required before the explicit `--run` phase.\n",
        encoding="utf-8")
    return out


def run_models(run_id: str, preflight_dir: Path, approval_file: Path):
    if not approval_file.is_file():
        raise RuntimeError("--run requires an explicit root approval file")
    pre = json.loads((preflight_dir / "preflight_manifest.json").read_text(encoding="utf-8"))
    if pre.get("status") != "preflight_complete_fit_free" or pre.get("driver_sha256") != sha(Path(__file__)):
        raise ValueError("preflight is incomplete or driver hash changed")
    sys.path.insert(0, str(EXP))
    import runner
    expected_hashes = {
        "cache_sha256": sha(CACHE),
        "parent_runner_sha256": sha(PILOT / "run_pilot.py"),
        "seed42_record_map_sha256": sha(MAP42),
        "protocol_sha256": sha(EXP / "protocol_v1.md"),
        "frozen_split_authority_sha256": sha(EXP / "frozen_split_authority_v1.md"),
        "key_experiments_protocol_sha256": sha(HERE / "protocol_key_v1.md"),
        "shared_runner_sha256": sha(EXP / "runner.py"),
    }
    for key, expected in expected_hashes.items():
        if pre.get(key) != expected:
            raise ValueError(f"preflight/live provenance mismatch: {key}")
    approval = json.loads(approval_file.read_text(encoding="utf-8"))
    if approval.get("preflight_manifest_sha256") != sha(preflight_dir / "preflight_manifest.json"):
        raise ValueError("approval does not authorize this exact preflight manifest")
    if approval.get("approved_fits") != 40:
        raise ValueError("approval must explicitly authorize exactly 40 fits")
    if pre.get("fits_planned") != 40 or pre.get("fit_strides") != list(FIT_STRIDES):
        raise ValueError("unexpected planned fit count")
    out = OUTROOT / run_id
    if out.exists():
        raise FileExistsError(out)
    cache, frozen = read_locked_inputs()
    import sklearn
    sys.path.insert(0, str(EXP))
    import runner
    out.mkdir(parents=True)
    (out / "predictions").mkdir()
    (out / "models").mkdir()
    (out / "jobs").mkdir()
    run_manifest = {
        "status": "running", "run_id": run_id, "preflight_dir": str(preflight_dir.resolve()),
        "preflight_manifest_sha256": sha(preflight_dir / "preflight_manifest.json"),
        "approval_file_sha256": sha(approval_file), "approval": approval,
        "driver_sha256": sha(Path(__file__)), "cache_sha256": sha(CACHE),
        "parent_runner_sha256": sha(PILOT / "run_pilot.py"), "seed42_map_sha256": sha(MAP42),
        "source_sha256": cache["source_sha256"], "runtime": {"executable": str(Path(sys.executable).resolve()),
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "planned_fits": 40, "completed_fits": 0, "failures": [],
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pre_manifest = pre
    catalog = {(j["stride"], j["policy"]): j for j in pre_manifest["jobs"]}
    completed = []
    for stride in FIT_STRIDES:
        for policy in POLICIES:
            table, _ = reconstruct(cache, frozen, stride, policy)
            jobmeta = catalog[(stride, policy)]
            row_file = preflight_dir / jobmeta["row_file"]
            split_file = preflight_dir / jobmeta["split_file"]
            if sha(row_file) != jobmeta["row_file_sha256"] or sha(split_file) != jobmeta["split_file_sha256"]:
                raise ValueError(f"preflight membership hash mismatch: stride{stride}/{policy}")
            with np.load(split_file, allow_pickle=False) as z:
                folds = [(z[f"train_{f}"], z[f"fit_{f}"], z[f"test_{f}"]) for f in range(5)]
            with np.load(row_file, allow_pickle=False) as z:
                if not np.array_equal(z["source_row"], table["source_row"]) or not np.array_equal(z["group"], table["group"]):
                    raise ValueError(f"preflight row identities changed: stride{stride}/{policy}")
                stored_y = np.asarray([CLASSES[int(i)] for i in z["y"]], dtype=object)
                if not np.array_equal(stored_y, table["y"]):
                    raise ValueError(f"preflight labels changed: stride{stride}/{policy}")
            colmap = {name: i for i, name in enumerate(cache["sessions"][0]["numeric_columns"])}
            imu_indices = [colmap[name] for name in cache["sessions"][0]["imu_columns"]]
            x_by_feature = {"full_numeric": table["x_full"], "imu_only": table["x_full"][:, imu_indices]}
            for feature in FEATURES:
                job_id = f"stride{stride}__{policy}__{feature}__RF100"
                pred_rows, pred_true, pred_pred, pred_fold = [], [], [], []
                fold_results, artifact_hashes, warnings_by_fold = [], [], []
                x = x_by_feature[feature]
                for f0, (train, fit, test) in enumerate(folds):
                    joblib_path = out / "models" / f"{job_id}__fold{f0+1}.joblib"
                    try:
                        xtr, xte, kept, medians, prep = runner.preprocess_train_test(x, fit, test, feature)
                        model = runner._make_estimator("RF100")
                        fit_start = time.perf_counter()
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(xtr, table["y"][fit])
                        fit_seconds = time.perf_counter() - fit_start
                        predict_start = time.perf_counter()
                        pred = model.predict(xte)
                        predict_seconds = time.perf_counter() - predict_start
                        joblib.dump(model, joblib_path, compress=3)
                        model_sha = sha(joblib_path)
                        metrics = runner._metric_payload(table["y"][test], pred)
                        warnings_payload = [str(w.message) for w in caught]
                        warnings_by_fold.append(warnings_payload)
                        # Append only after model artifact persists successfully.
                        pred_rows.extend(test.tolist())
                        pred_true.extend(table["y"][test].tolist())
                        pred_pred.extend(np.asarray(pred, dtype=object).tolist())
                        pred_fold.extend([f0+1] * len(test))
                        fold_results.append({"fold": f0+1, "train_n": int(len(train)), "fit_n": int(len(fit)),
                            "test_n": int(len(test)), "train_sessions": int(len(set(table["group"][train]))),
                            "test_sessions": int(len(set(table["group"][test]))), "class_order": list(CLASSES),
                            "fit_class_counts": {c: int(np.sum(table["y"][fit] == c)) for c in CLASSES},
                            "test_class_counts": metrics["class_support"], "metrics": metrics,
                            "kept_columns": kept, "medians": medians, "preprocessing": prep,
                            "fit_wall_seconds": fit_seconds, "predict_wall_seconds": predict_seconds,
                            "model_path": str(joblib_path.relative_to(out).as_posix()), "model_sha256": model_sha,
                            "warnings": warnings_payload})
                        artifact_hashes.append(model_sha)
                    except Exception as exc:
                        fail = {"job": job_id, "fold": f0+1, "error": repr(exc)}
                        run_manifest["failures"].append(fail)
                        (out / "failures.json").write_text(json.dumps(run_manifest["failures"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                        (out / "jobs" / f"{job_id}.partial.json").write_text(json.dumps({
                            "job": job_id, "stride": stride, "policy": policy, "feature": feature,
                            "completed_fold_summaries": fold_results, "failed_fold": f0+1,
                            "error": repr(exc), "model_hashes_already_persisted": artifact_hashes,
                        }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                        break
                if len(fold_results) == 5:
                    pred_file = out / "predictions" / f"{job_id}.npz"
                    encoded_true = np.asarray([CLASSES.index(str(y)) for y in pred_true], dtype=np.int8)
                    encoded_pred = np.asarray([CLASSES.index(str(y)) for y in pred_pred], dtype=np.int8)
                    np.savez_compressed(pred_file, row=np.asarray(pred_rows, dtype=np.int64),
                                        fold=np.asarray(pred_fold, dtype=np.int8), y_true=encoded_true, y_pred=encoded_pred)
                    payload = {"job": job_id, "stride": stride, "policy": policy, "feature": feature,
                        "model": "RF100", "class_order": list(CLASSES), "source_row_hash": jobmeta["row_identity_sha256"],
                        "prediction_sha256": sha(pred_file), "model_artifact_sha256": artifact_hashes,
                        "folds": fold_results, "warning_count": sum(len(x) for x in warnings_by_fold)}
                    (out / "jobs" / f"{job_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                    completed.append(job_id)
                    run_manifest["completed_fits"] += 5
                    run_manifest["completed_jobs"] = completed
                    (out / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    run_manifest["status"] = "phase_complete" if run_manifest["completed_fits"] == 40 and not run_manifest["failures"] else "phase_incomplete"
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (out / "completion.json").write_text(json.dumps({"status": run_manifest["status"], "planned_fits": 40,
        "completed_fits": run_manifest["completed_fits"], "failures": run_manifest["failures"]}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--preflight-dir", type=Path)
    ap.add_argument("--approval-file", type=Path)
    args = ap.parse_args()
    if args.preflight == args.run:
        ap.error("choose exactly one of --preflight or --run")
    if args.preflight:
        out = create_preflight(args.run_id)
        print(out)
    else:
        if args.preflight_dir is None or args.approval_file is None:
            ap.error("--run requires --preflight-dir and --approval-file")
        run_models(args.run_id, args.preflight_dir, args.approval_file)


if __name__ == "__main__":
    main()
