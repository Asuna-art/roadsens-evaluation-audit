from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNNER_PATH = HERE / "runner.py"
RUNNER_SHA = "b4b240c7df89b8feb1ed9ff314566f2b642b165564a26816c378bca556c0d7a3"
FREEZE_JSON = HERE / "data_audit" / "content_component_freeze_v4.json"
PREFLIGHT_JSON = HERE / "content_overlap_preflight_v2.json"
PREFLIGHT_NPZ = HERE / "content_overlap_preflight_rows_v2.npz"
OUT = HERE / "runs" / "content_overlap_v2"
CLASSES = ("Normal", "Pothole", "Bump")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_runner():
    live = sha(RUNNER_PATH)
    if live != RUNNER_SHA:
        raise RuntimeError(f"Frozen runner hash mismatch: expected {RUNNER_SHA}, got {live}")
    spec = importlib.util.spec_from_file_location("roadsens_frozen_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _support(runner, rows, folds, split_id: str) -> list[dict]:
    y = np.asarray(rows.y, dtype=object)
    result = []
    for fi, (tr, te) in enumerate(folds):
        tr = np.asarray(tr, dtype=np.int64); te = np.asarray(te, dtype=np.int64)
        if len(np.intersect1d(tr, te)):
            raise RuntimeError(f"Train/test index overlap: {split_id} fold {fi+1}")
        if any(np.sum(y[tr] == c) == 0 or np.sum(y[te] == c) == 0 for c in CLASSES):
            raise RuntimeError(f"Missing class support: {split_id} fold {fi+1}")
        result.append({"fold": fi + 1, "train_indices": tr, "test_indices": te,
                       "train_class_counts": {c: int(np.sum(y[tr] == c)) for c in CLASSES},
                       "test_class_counts": {c: int(np.sum(y[te] == c)) for c in CLASSES},
                       "train_groups": set(rows.groups[tr].astype(str)),
                       "test_groups": set(rows.groups[te].astype(str))})
    return result


def prepare(runner):
    cache, frozen_map, timeblock_qa = runner.validate_immutable_inputs()
    freeze = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    if freeze["artifact_version"] != 4 or freeze["software"]["scikit_learn"] != runner.EXPECTED_SKLEARN:
        raise RuntimeError("Component graph/runtime mismatch")
    if sha(FREEZE_JSON) != preflight["component_freeze_sha256"]:
        raise RuntimeError("Preflight was generated from another component freeze")
    if sha(RUNNER_PATH) != RUNNER_SHA:
        raise RuntimeError("Runner changed during validation")
    components = {k: c["component_id"] for c in freeze["components"] for k in c["members"]}
    prepared = {"H1": {}, "H2": {}}
    # H1 seed42/43/44 component-heldout maps; same map across label policies.
    for seed in (42, 43, 44):
        cfold = freeze["fold_seeds"][str(seed)]["component_fold"]
        fmap = {key: cfold[components[key]] for key in components}
        for policy in runner.POLICIES:
            rows = runner.policy_rows(cache["sessions"], policy)
            folds = runner._record_splits(rows, fmap)
            checked = _support(runner, rows, folds, f"H1-component{seed}-{policy}")
            for item in checked:
                if item["train_groups"] & item["test_groups"]:
                    raise RuntimeError("H1 component map splits a recording")
                tr_comp = {components[g] for g in item["train_groups"]}
                te_comp = {components[g] for g in item["test_groups"]}
                if tr_comp & te_comp:
                    raise RuntimeError("H1 component is shared across train/test")
            prepared["H1"][(seed, policy)] = (rows, folds)

    # H2 deduplicated cohort, retaining byte-hash representative 46.csv.
    h2_sessions = [s for s in cache["sessions"] if s["session_key"] != "combined_anomalies/47.csv"]
    if len(h2_sessions) != 100:
        raise RuntimeError("H2 expected the frozen 100-record cohort")
    h2_record_map = preflight["H2"]["record_component_fold"]
    with np.load(PREFLIGHT_NPZ, allow_pickle=False) as npz:
        for policy in runner.POLICIES:
            rows = runner.policy_rows(h2_sessions, policy)
            key = "blank_normal" if policy == "blank_normal" else "primary"
            if not np.array_equal(rows.groups.astype(str), npz[f"H2_{key}_source_key"].astype(str)):
                raise RuntimeError(f"H2 row source sequence mismatch: {policy}")
            if not np.array_equal(rows.source_rows, npz[f"H2_{key}_source_row"]):
                raise RuntimeError(f"H2 source-row sequence mismatch: {policy}")
            random_folds = runner._random_splits(rows, 42)
            record_folds = runner._record_splits(rows, h2_record_map)
            for split_id, folds in (("random42", random_folds), ("record42", record_folds)):
                _support(runner, rows, folds, f"H2-{split_id}-{policy}")
            if not np.array_equal(rows.y.astype(str), npz[f"H2_{key}_label"].astype(str)):
                raise RuntimeError("H2 labels differ from fit-free preflight")
            _, _, block_ids, near_test = runner._load_modules()
            original_rows = runner.policy_rows(cache["sessions"], policy)
            original_folds = runner._timeblock_splits(cache["sessions"], original_rows, policy, timeblock_qa, block_ids, near_test)
            h2_lookup = {(str(g), int(sr), str(y)): i for i, (g, sr, y) in enumerate(zip(rows.groups, rows.source_rows, rows.y))}
            time_folds = []
            for tr0, te0 in original_folds:
                tr = np.asarray([h2_lookup[(str(original_rows.groups[i]), int(original_rows.source_rows[i]), str(original_rows.y[i]))]
                                 for i in tr0 if str(original_rows.groups[i]) != "combined_anomalies/47.csv"], dtype=np.int64)
                te = np.asarray([h2_lookup[(str(original_rows.groups[i]), int(original_rows.source_rows[i]), str(original_rows.y[i]))]
                                 for i in te0 if str(original_rows.groups[i]) != "combined_anomalies/47.csv"], dtype=np.int64)
                time_folds.append((tr, te))
            checked = _support(runner, rows, time_folds, f"H2-timeblock42-{policy}")
            frozen_desc = preflight["H2"]["policies"][policy]["timeblock42"]
            for item, saved in zip(checked, frozen_desc):
                def member_sha(ix):
                    vals = [(str(rows.groups[i]), int(rows.source_rows[i]), str(rows.y[i])) for i in ix]
                    raw = json.dumps(vals, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    return hashlib.sha256(raw).hexdigest()
                if member_sha(item["train_indices"]) != saved["train_row_membership_sha256"] or member_sha(item["test_indices"]) != saved["test_row_membership_sha256"]:
                    raise RuntimeError("H2 timeblock train/test row identities differ from v2 preflight")
            for item in checked:
                for group in item["test_groups"]:
                    local = np.flatnonzero(rows.groups.astype(str) == group)
                    tr_sources = set(rows.source_rows[np.intersect1d(local, item["train_indices"])].tolist())
                    te_sources = set(rows.source_rows[np.intersect1d(local, item["test_indices"])].tolist())
                    if tr_sources & te_sources:
                        raise RuntimeError(f"H2 timeblock source-row overlap: {group}")
            prepared["H2"][(policy, "random42")] = (rows, random_folds)
            prepared["H2"][(policy, "record42")] = (rows, record_folds)
            prepared["H2"][(policy, "timeblock42")] = (rows, time_folds)
    return cache, freeze, preflight, prepared


def model_specs():
    specs = []
    for policy in ("primary", "blank_normal"):
        for feature in ("full_numeric", "context_only", "imu_only"):
            for model in ("RF100", "HistGB"):
                specs.append(("H1", policy, "component42", feature, model, 42))
        for seed in (43, 44):
            for feature in ("full_numeric", "context_only"):
                specs.append(("H1", policy, f"component{seed}", feature, "RF100", seed))
        for split in ("random42", "record42", "timeblock42"):
            for feature in ("full_numeric", "context_only", "imu_only"):
                specs.append(("H2", policy, split, feature, "RF100", 42))
    return specs


def _folds_for(prepared, phase, policy, split):
    if phase == "H1":
        seed = int(split.replace("component", ""))
        return prepared["H1"][(seed, policy)]
    return prepared["H2"][(policy, split)]


def run_fit(runner, cache, prepared):
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    import sklearn
    from threadpoolctl import threadpool_limits
    OUT.mkdir(parents=True)
    artifact_dir = OUT / "models"; artifact_dir.mkdir()
    result_dir = OUT / "folds"; result_dir.mkdir()
    manifest = {"status": "running", "fit_started": True, "runner_sha256": sha(RUNNER_PATH),
                "component_freeze_sha256": sha(FREEZE_JSON), "preflight_sha256": sha(PREFLIGHT_JSON),
                "cache_sha256": sha(runner.CACHE), "python": __import__("sys").version,
                "scikit_learn": sklearn.__version__, "expected_fits": 190,
                "specs": [list(x) for x in model_specs()], "completed_folds": 0, "failures": []}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with threadpool_limits(limits=4):
        for phase, policy, split, feature, model, seed in model_specs():
            rows, folds = _folds_for(prepared, phase, policy, split)
            x, columns = runner.feature_matrix(cache["sessions"] if phase == "H1" else
                [s for s in cache["sessions"] if s["session_key"] != "combined_anomalies/47.csv"], rows, feature)
            if phase == "H2" and feature == "imu_only":
                pass
            config_id = f"{phase}_{policy}_{split}_{feature}_{model}_seed{seed}"
            for fold_i, (train0, test) in enumerate(folds):
                fit = runner.cap_train_indices(train0, rows.y, fold_i, seed, 20000)
                xtr, xte, kept, medians, prep = runner.preprocess_train_test(x, fit, test, feature)
                estimator = runner._make_estimator(model)
                estimator.set_params(random_state=seed)
                fit_kwargs = {}
                if model == "HistGB" and "class_weight" not in estimator.get_params():
                    from sklearn.utils.class_weight import compute_sample_weight
                    fit_kwargs["sample_weight"] = compute_sample_weight("balanced", rows.y[fit])
                t0 = time.perf_counter(); estimator.fit(xtr, rows.y[fit], **fit_kwargs); fit_seconds = time.perf_counter() - t0
                t0 = time.perf_counter(); pred = np.asarray(estimator.predict(xte), dtype=object); pred_seconds = time.perf_counter() - t0
                payload = {"phase": phase, "policy": policy, "split": split, "feature": feature,
                           "model": model, "seed": seed, "fold": fold_i + 1,
                           "metrics": runner._metric_payload(rows.y[test], pred),
                           "fit_n": int(len(fit)), "train_n_before_cap": int(len(train0)),
                           "fit_class_counts": {c: int(np.sum(rows.y[fit] == c)) for c in CLASSES},
                           "fit_seconds": fit_seconds, "predict_seconds": pred_seconds,
                           "kept_columns": kept, "feature_columns": columns, "medians": medians,
                           "preprocessing": prep,
                           "fit_rows_sha256": runner.object_hash(fit.tolist()),
                           "test_rows_sha256": runner.object_hash(np.asarray(test).tolist()),
                           "predictions": [{"row": int(i), "session": str(rows.groups[i]),
                                            "source_row": int(rows.source_rows[i]),
                                            "true": str(rows.y[i]), "pred": str(pred[j])}
                                           for j, i in enumerate(test)]}
                model_path = artifact_dir / f"{config_id}.fold{fold_i+1}.joblib"
                joblib.dump({"estimator": estimator, "columns": columns, "kept": kept,
                             "medians": medians, "preprocessing": prep}, model_path, compress=3)
                payload["model_path"] = str(model_path.relative_to(OUT).as_posix())
                payload["model_sha256"] = sha(model_path)
                out_file = result_dir / f"{config_id}.fold{fold_i+1}.json"
                out_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                manifest["completed_folds"] += 1
                print(f"DONE {config_id} fold {fold_i+1}/5", flush=True)
                (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = "complete"; (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-manifest", type=Path)
    args = parser.parse_args()
    if args.qa == args.run:
        parser.error("choose exactly one of --qa or --run")
    if args.run:
        if not args.review_manifest or not args.review_manifest.is_file():
            parser.error("--run requires a root-reviewed fit-slot manifest; no models are fit before review")
        review = json.loads(args.review_manifest.read_text(encoding="utf-8"))
        if not review.get("root_reviewed") or not review.get("fit_slot_authorized"):
            parser.error("review manifest must explicitly set root_reviewed=true and fit_slot_authorized=true")
    import sys, sklearn, pandas
    if Path(sys.executable).resolve() != Path(r"C:\Users\Administrator\miniconda3\python.exe").resolve():
        raise RuntimeError(f"Wrong Python executable: {sys.executable}")
    if sklearn.__version__ != "1.9.0" or np.__version__ != "2.5.2" or pandas.__version__ != "3.0.5":
        raise RuntimeError(f"Locked runtime mismatch: sklearn={sklearn.__version__}, numpy={np.__version__}, pandas={pandas.__version__}")
    runner = load_runner()
    cache, freeze, preflight, prepared = prepare(runner)
    if args.qa:
        print(json.dumps({"passed": True, "runner_sha256": sha(RUNNER_PATH), "sklearn_expected": runner.EXPECTED_SKLEARN,
                          "H1_fits": 100, "H2_fits": 90, "source_count_h1": 101,
                          "source_count_h2": 100, "support_preflight_sha256": sha(PREFLIGHT_JSON),
                          "fit_started": False}, ensure_ascii=False))
    else:
        run_fit(runner, cache, prepared)


if __name__ == "__main__":
    main()
