"""Prepare seed-43/44 RoadSens confirmation runs; training requires explicit --run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "session_feature_cache_v1.joblib"
PILOT_RUN = ROOT / "runs" / "pilot_v2"
POLICIES = ("primary", "blank_normal")
SPLITS = ("random", "group")
METHODS = ("full_numeric", "missing_only", "context_only", "imu_only")
CLASSES = ("Normal", "Pothole", "Bump")
SEEDS = (43, 44)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _relative_session_keys(sessions: list[dict]) -> list[str]:
    keys = [s["session_key"] for s in sessions]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate session_key in feature cache")
    return sorted(keys)


def _class_counts(sessions: list[dict], fold_map: dict[str, int], policy: str) -> dict:
    totals = {f: {side: {c: 0 for c in CLASSES} for side in ("train", "test")}
              for f in range(1, 6)}
    session_counts = {f: {side: 0 for side in ("train", "test")} for f in range(1, 6)}
    for session in sessions:
        key = session["session_key"]
        fold = fold_map[key]
        labels = session["labels"] if policy == "primary" else session["labels_blank_normal"]
        valid = np.isfinite(session["times"]) & np.isin(labels, CLASSES)
        test = {c: int(np.sum(labels[valid] == c)) for c in CLASSES}
        for f in range(1, 6):
            side = "test" if f == fold else "train"
            session_counts[f][side] += int(test["Normal"] + test["Pothole"] + test["Bump"] > 0)
            for c in CLASSES:
                totals[f][side][c] += test[c]
    return {str(f): {side: {"sessions": session_counts[f][side], "class_rows": totals[f][side]}
                     for side in ("train", "test")} for f in range(1, 6)}


def build_seed_fold_map(sessions: list[dict], source_hashes: dict[str, str], seed: int,
                        frozen_seed42: dict[str, int]) -> dict:
    keys = _relative_session_keys(sessions)
    if set(keys) != set(source_hashes):
        raise ValueError("feature cache session keys differ from source hash keys")
    by_key = {s["session_key"]: s for s in sessions}
    normal_keys = sorted(k for k in keys if k.startswith("combined_normal/"))
    anomaly_keys = sorted(k for k in keys if k.startswith("combined_anomalies/"))
    if len(normal_keys) != 5 or len(anomaly_keys) != 96:
        raise ValueError(f"expected 5 normal and 96 anomaly sessions; got {len(normal_keys)}, {len(anomaly_keys)}")

    ys, groups = [], []
    for key in anomaly_keys:
        session = by_key[key]
        labels = session["labels"]
        valid = np.isfinite(session["times"]) & np.isin(labels, ("Pothole", "Bump"))
        ys.extend(labels[valid].tolist())
        groups.extend([key] * int(valid.sum()))
    y = np.asarray(ys, dtype=object)
    g = np.asarray(groups, dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_map: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, g), 1):
        for key in set(g[test_idx]):
            key = str(key)
            if key in fold_map:
                raise AssertionError(f"anomaly session assigned more than once: {key}")
            fold_map[key] = fold
    if set(fold_map) != set(anomaly_keys):
        raise AssertionError("anomaly SGKF did not assign every anomaly session")

    normal_permutation = np.random.RandomState(seed).permutation(len(normal_keys)).tolist()
    normal_order = [normal_keys[i] for i in normal_permutation]
    for fold, key in enumerate(normal_order, 1):
        fold_map[key] = fold
    if set(fold_map) != set(keys) or len(fold_map) != 101:
        raise AssertionError("combined map does not cover exactly all 101 source sessions")

    if seed == 42 and fold_map != frozen_seed42:
        raise ValueError("seed-42 reconstructed fold map differs from frozen session_fold_map_v1.json")

    counts = {policy: _class_counts(sessions, fold_map, policy) for policy in POLICIES}
    for policy, folds in counts.items():
        for fold, sides in folds.items():
            for side, detail in sides.items():
                missing = [c for c in CLASSES if detail["class_rows"][c] <= 0]
                if missing:
                    raise ValueError(f"STOP seed {seed}: {policy} fold {fold} {side} lacks {missing}; no seed search")

    return {"version": 1, "seed": seed, "session_count": len(fold_map),
            "design": "Anomaly-only StratifiedGroupKFold on every tenth retained source row with finite seconds_elapsed and explicit primary Pothole/Bump; five sorted normal session paths assigned one per fold by RandomState(seed) permutation.",
            "session_fold": {k: fold_map[k] for k in keys}, "source_sha256": source_hashes,
            "normal_assignment_order": normal_order,
            "counts_by_policy": counts,
            "fold_class_support_passed": True, "model_fit_or_scores": False}


def _cap_train(train: np.ndarray, y: np.ndarray, fold: int, seed: int,
               cap: int = 20000) -> np.ndarray:
    if len(train) <= cap:
        return train
    rng = np.random.RandomState(seed * 1000 + fold)
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
    return np.asarray(sorted(chosen), dtype=np.int64)


def _random_folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def _load_parent_and_cache(cache_path: Path):
    sys.path.insert(0, str(ROOT))
    import run_pilot as parent
    import joblib
    cache = joblib.load(cache_path)
    current_parent_hash = sha256(ROOT / "run_pilot.py")
    if cache.get("runner_sha256") != current_parent_hash:
        raise ValueError("feature cache was built from a different run_pilot.py")
    frozen = json.loads((ROOT / "session_fold_map_v1.json").read_text(encoding="utf-8"))
    if cache.get("source_sha256") != frozen.get("source_sha256"):
        raise ValueError("cache source hashes differ from frozen seed-42 map")
    sessions = cache["sessions"]
    if {s["session_key"] for s in sessions} != set(frozen["session_fold"]):
        raise ValueError("cache session keys differ from frozen seed-42 map")
    if any(s["sha256"] != frozen["source_sha256"][s["session_key"]] for s in sessions):
        raise ValueError("cache session data hash differs from frozen seed-42 map")
    for key, expected in frozen["source_sha256"].items():
        path = ROOT / "data" / "inner" / Path(key)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"live source file hash mismatch: {key}")
    return parent, cache, frozen


def run_label_map_qa(cache_path: Path = DEFAULT_CACHE) -> dict:
    """Use cached labels only: validate seed maps/support/parameter determinism; no model fit."""
    parent, cache, frozen = _load_parent_and_cache(cache_path)
    results = {}
    for seed in (42, 43, 44):
        fmap = build_seed_fold_map(cache["sessions"], cache["source_sha256"], seed,
                                   frozen["session_fold"])
        # Parameter QA: fold construction is deterministic and uses the specified seed.
        sample_y = np.asarray(["Normal", "Pothole", "Bump"] * 20, dtype=object)
        folds1 = _random_folds(sample_y, seed)
        folds2 = _random_folds(sample_y, seed)
        if any(not np.array_equal(a[1], b[1]) for a, b in zip(folds1, folds2)):
            raise AssertionError(f"random split seed {seed} is not deterministic")
        fit1 = _cap_train(np.arange(len(sample_y)), sample_y, 0, seed, cap=12)
        fit2 = _cap_train(np.arange(len(sample_y)), sample_y, 0, seed, cap=12)
        if not np.array_equal(fit1, fit2):
            raise AssertionError(f"training sample seed {seed} is not deterministic")
        results[str(seed)] = {"group_map_sha256": hashlib.sha256(json.dumps(
            fmap["session_fold"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "normal_assignment_order": fmap["normal_assignment_order"],
            "fold_support": {policy: {fold: sides["test"]["class_rows"]
                                      for fold, sides in folds.items()}
                             for policy, folds in fmap["counts_by_policy"].items()},
            "all_fold_sides_three_class": fmap["fold_class_support_passed"]}
    if results["42"]["group_map_sha256"] != hashlib.sha256(json.dumps(
            frozen["session_fold"], sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise AssertionError("seed-42 map does not exactly reproduce frozen mapping")
    return {"passed": True, "checks": ["seed42 frozen map equivalence", "seed43/44 constrained group maps",
                                        "both label policies have three classes in every train/test fold",
                                        "random split parameter determinism", "training sample parameter determinism"],
            "results": results, "cache_sha256": sha256(cache_path),
            "parent_runner_sha256": cache["runner_sha256"], "model_fit_or_scores": False}


def _read_first_pass_gate() -> dict:
    path = PILOT_RUN / "summary.json"
    if not path.exists():
        raise ValueError("first-pass summary.json is missing; run the complete pilot analyzer first")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("first-pass summary is partial; confirmation is not authorized by the frozen plan")
    if summary.get("screening", {}).get("track_A", {}).get("status") != "meets_in_both_label_policies":
        raise ValueError("track A did not pass in both policies; do not start seed confirmation")
    return {"summary_sha256": sha256(path), "track_A_status": summary["screening"]["track_A"]["status"]}


def execute(seed: int, cache_path: Path, output_dir: Path) -> dict:
    if seed not in SEEDS:
        raise ValueError("confirmation seed must be 43 or 44")
    expected_name = f"confirmation_seed{seed}_v1"
    if output_dir.name != expected_name:
        raise ValueError(f"output directory must be named {expected_name}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    gate = _read_first_pass_gate()
    parent, cache, frozen = _load_parent_and_cache(cache_path)
    seed_map = build_seed_fold_map(cache["sessions"], cache["source_sha256"], seed,
                                   frozen["session_fold"])

    # Freeze and verify the seed map before any output directory or model is created.
    output_dir.mkdir(parents=True, exist_ok=False)
    map_path = output_dir / "session_fold_map.json"
    map_path.write_text(json.dumps(seed_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    original_rf = dict(parent.RF_CONFIG)
    original_build = parent._build_folds
    original_cap = parent._cap_train
    config = {**original_rf, "random_state": seed}
    run_manifest = {
        "status": "running", "seed": seed,
        "first_pass_gate": gate,
        "model_config": config,
        "split_config": {"random": "StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)",
                         "group": "seed-specific constrained map; anomaly-only SGKF and one permuted normal session per fold"},
        "training_sample_config": "Original class-proportional cap at 20000; RandomState(seed*1000+zero_based_fold).",
        "policies": list(POLICIES), "splits": list(SPLITS), "methods": list(METHODS),
        "session_fold_map_file": map_path.name, "session_fold_map_sha256": sha256(map_path),
        "source_sha256": cache["source_sha256"],
        "feature_cache_path": str(cache_path.resolve()), "feature_cache_sha256": sha256(cache_path),
        "parent_runner_path": str((ROOT / "run_pilot.py").resolve()),
        "parent_runner_sha256": cache["runner_sha256"],
        "confirmation_driver_path": str(Path(__file__).resolve()),
        "confirmation_driver_sha256": sha256(Path(__file__).resolve()),
        "confirmation_plan_path": str((ROOT / "confirmation_plan.md").resolve()),
        "confirmation_plan_sha256": sha256(ROOT / "confirmation_plan.md"),
        "model_fit_or_scores_started": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def build_folds(y, groups, split_kind, group_fold_map=None):
        if split_kind == "random":
            return _random_folds(y, seed)
        if split_kind == "group":
            return original_build(y, groups, split_kind, seed_map["session_fold"])
        raise ValueError(f"unknown split kind: {split_kind}")

    def cap_train(train, y, fold, cap=20000):
        return _cap_train(train, y, fold, seed, cap)

    try:
        parent.RF_CONFIG = config
        parent._build_folds = build_folds
        parent._cap_train = cap_train
        run_manifest["model_fit_or_scores_started"] = True
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for policy in POLICIES:
            for split in SPLITS:
                target = output_dir / f"{policy}_{split}.json"
                parent._run_policy(cache["sessions"], policy, split, target,
                                   seed_map["session_fold"], list(METHODS))
        run_manifest["status"] = "complete"
        run_manifest["completed_epoch"] = time.time()
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        run_manifest["status"] = "failed_or_partial"
        run_manifest["error_type"] = type(exc).__name__
        run_manifest["error"] = str(exc)
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        parent.RF_CONFIG = original_rf
        parent._build_folds = original_build
        parent._cap_train = original_cap
    return {"status": run_manifest["status"], "output_dir": str(output_dir), "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, help="confirmation seed (43 or 44)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-qa", action="store_true",
                        help="validate seed maps and deterministic parameters from cached labels; no model fit")
    parser.add_argument("--run", action="store_true", help="explicitly start the requested confirmation models")
    args = parser.parse_args()
    if args.synthetic_qa:
        print(json.dumps(run_label_map_qa(args.cache), ensure_ascii=False, indent=2))
        return
    if not args.run:
        parser.error("real confirmation requires explicit --run; use --synthetic-qa for label-map QA")
    if args.seed is None:
        parser.error("--run requires --seed 43 or 44")
    output_dir = args.output_dir or (ROOT / "runs" / f"confirmation_seed{args.seed}_v1")
    print(json.dumps(execute(args.seed, args.cache, output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
