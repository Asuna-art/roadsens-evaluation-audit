"""Run the approved time-block diagnostic through the unchanged RoadSens pilot runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_pilot as pilot  # noqa: E402
from qa_timeblock_splits import block_ids, _near_test, sha256_file  # noqa: E402

METHODS = ["full_numeric", "missing_only", "context_only", "imu_only"]
POLICIES = [("primary", "labels", "primary"),
            ("blank_normal", "labels_blank_normal", "blank_normal_sensitivity")]
CACHE = ROOT / "session_feature_cache_v1.joblib"
SPLIT_QA = ROOT / "timeblock_split_support_v3.json"
PROTOCOL = ROOT / "recording_timeblock_protocol_v2.md"
OUTPUT = ROOT / "runs" / "timeblock_v1"


def object_hash(value) -> str:
    b = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def exact_splits(sessions: list[dict], labels_key: str, support_policy: str,
                 split_qa: dict) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    expected_folds = split_qa["folds"][support_policy]
    train_full_by_fold = [[] for _ in range(5)]
    test_full_by_fold = [[] for _ in range(5)]
    labels_parts, groups_parts, source_rows_parts, valid_parts = [], [], [], []
    source_membership = {}
    offset = 0
    for s in sessions:  # same cached session order used by pilot._concat
        key = s["session_key"]
        t = np.asarray(s["times"], dtype=np.float64)
        y = np.asarray(s[labels_key], dtype=object)
        bins = block_ids(t, s["times_full"])
        valid = np.isfinite(t) & np.isin(y, pilot.CLASSES)
        labels_parts.append(y)
        groups_parts.append(np.full(len(t), key, dtype=object))
        source_rows_parts.append(np.asarray(s["source_row"], dtype=np.int64))
        valid_parts.append(valid)
        source_membership[key] = {}
        for fold in range(5):
            test_mask = (bins == fold) & np.isfinite(t)
            candidate_train = (bins >= 0) & (bins != fold) & np.isfinite(t)
            near = _near_test(t, t[test_mask], 1.0)
            train_mask = candidate_train & ~near
            train_score, test_score = train_mask & valid, test_mask & valid
            if np.any(train_score & test_score):
                raise AssertionError(f"Train/test overlap: {key}, fold={fold + 1}")
            if np.any(_near_test(t[train_mask], t[test_mask], 1.0)):
                raise AssertionError(f"Embargo violation: {key}, fold={fold + 1}")
            exp_list = expected_folds[str(fold + 1)]["session_membership"]
            exp = next(z for z in exp_list if z["session_key"] == key)
            got_train = np.asarray(s["source_row"])[train_mask].astype(int).tolist()
            got_test = np.asarray(s["source_row"])[test_mask].astype(int).tolist()
            if got_train != exp["train_source_rows"] or got_test != exp["test_source_rows"]:
                raise AssertionError(f"Cached source-row membership mismatch: {key}, fold={fold + 1}")
            source_membership[key][str(fold + 1)] = {"train": got_train, "test": got_test}
            train_full_by_fold[fold].append(np.flatnonzero(train_score) + offset)
            test_full_by_fold[fold].append(np.flatnonzero(test_score) + offset)
        offset += len(t)

    labels_all = np.concatenate(labels_parts)
    groups_all = np.concatenate(groups_parts)
    source_rows_all = np.concatenate(source_rows_parts)
    valid_all = np.concatenate(valid_parts)
    valid_positions = np.flatnonzero(valid_all)
    labels_scored = labels_all[valid_all]
    groups_scored = groups_all[valid_all]
    source_rows_scored = source_rows_all[valid_all]
    folds, compact = [], {}
    for fold in range(5):
        tr_full = np.concatenate(train_full_by_fold[fold])
        te_full = np.concatenate(test_full_by_fold[fold])
        tr = np.searchsorted(valid_positions, tr_full)
        te = np.searchsorted(valid_positions, te_full)
        if not np.array_equal(valid_positions[tr], tr_full) or not np.array_equal(valid_positions[te], te_full):
            raise AssertionError("Split includes a row removed by the runner's label/time filter")
        tr_counts = {c: int(np.sum(labels_scored[tr] == c)) for c in pilot.CLASSES}
        te_counts = {c: int(np.sum(labels_scored[te] == c)) for c in pilot.CLASSES}
        expected = expected_folds[str(fold + 1)]
        if tr_counts != expected["train_class_rows"] or te_counts != expected["test_class_rows"]:
            raise AssertionError(f"Global runner-row class support mismatch, fold={fold + 1}")
        if any(v <= 0 for v in (*tr_counts.values(), *te_counts.values())):
            raise AssertionError(f"A train/test fold lacks a class: fold={fold + 1}")
        capped = pilot._cap_train(tr, labels_scored, fold)
        capped_counts = {c: int(np.sum(labels_scored[capped] == c)) for c in pilot.CLASSES}
        if capped_counts != expected["capped_train_class_rows_expected"] or any(v <= 0 for v in capped_counts.values()):
            raise AssertionError(f"Capped training support mismatch: fold={fold + 1}")
        folds.append((tr, te))
        compact[str(fold + 1)] = {
            "train_class_rows": tr_counts, "test_class_rows": te_counts,
            "capped_train_class_rows": capped_counts,
            "train_n": int(len(tr)), "capped_train_n": int(len(capped)), "test_n": int(len(te)),
            "test_recordings": int(len(np.unique(groups_scored[te]))),
        }
    identity = [[str(g), int(r), str(y)] for g, r, y in zip(groups_scored, source_rows_scored, labels_scored)]
    result = {"source_rows_after_runner_filter": int(valid_all.sum()), "folds": compact,
              "global_scored_row_keys_sha256": object_hash(identity),
              "source_membership": source_membership}
    return folds, result


def validate() -> tuple[list[dict], dict, dict]:
    cache = joblib.load(CACHE)
    qa = json.loads(SPLIT_QA.read_text(encoding="utf-8"))
    if not qa.get("passed_split_integrity") or qa.get("fit_or_score_performed"):
        raise ValueError("Split QA is not a passing fit-free artifact")
    if qa["cache_sha256"] != sha256_file(CACHE):
        raise ValueError("Feature cache changed after split QA")
    if qa["protocol_sha256"] != sha256_file(PROTOCOL):
        raise ValueError("Protocol changed after split QA")
    if qa["script_sha256"] != sha256_file(ROOT / "qa_timeblock_splits.py"):
        raise ValueError("Split QA code changed after split QA")
    if cache.get("runner_sha256") != sha256_file(ROOT / "run_pilot.py"):
        raise ValueError("Feature cache was made by a different runner version")
    fold_map = pilot.load_frozen_fold_map()
    live = pilot.validate_fold_map_labels(fold_map)  # re-hash all 101 source CSVs before any fit
    if not live.get("passed") or not live.get("source_hashes_match"):
        raise ValueError("Live source hash/label validation failed")
    if cache.get("source_sha256") != fold_map["source_sha256"]:
        raise ValueError("Cache and frozen source hashes differ")
    sessions = cache["sessions"]
    if len(sessions) != 101 or {s["session_key"] for s in sessions} != set(fold_map["session_fold"]):
        raise ValueError("Expected the 101 frozen CSV recordings")
    if any(s["sha256"] != fold_map["source_sha256"][s["session_key"]] for s in sessions):
        raise ValueError("A cached recording hash differs from the frozen map")
    policy_checks = {}
    for policy, label_key, support_key in POLICIES:
        if not all(qa["folds"][support_key][str(k)]["all_three_classes"] for k in range(1, 6)):
            raise ValueError(f"Support QA marks {policy} infeasible")
        folds, support = exact_splits(sessions, label_key, support_key, qa)
        policy_checks[policy] = {"folds": [(tr.tolist(), te.tolist()) for tr, te in folds], "support": support}
    # Compare source-row membership, not compressed indices (the policies score different row counts).
    if policy_checks["primary"]["support"]["source_membership"] != policy_checks["blank_normal"]["support"]["source_membership"]:
        raise AssertionError("The two label policies do not share identical source-row split membership")
    return sessions, fold_map, {"source_validation": live, "policy_checks": policy_checks,
        "cache_sha256": sha256_file(CACHE), "split_support_sha256": sha256_file(SPLIT_QA),
        "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(ROOT / "run_pilot.py"),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "split_qa_builder_sha256": sha256_file(ROOT / "qa_timeblock_splits.py")}


def main() -> None:
    ap = argparse.ArgumentParser()
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true", help="Validate exact source-row splits; fit-free.")
    modes.add_argument("--formal", action="store_true", help="Run the approved 40 model-folds after validation.")
    args = ap.parse_args()
    sessions, _fold_map, validation = validate()
    print("Time-block row membership, source hashes, and class support passed; no fit has started.", flush=True)
    if args.validate_only:
        path = ROOT / "timeblock_runner_validation_v1.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {path}", flush=True)
        return
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    manifest = {
        "created_epoch": time.time(), "diagnostic": "within_record_timeblock_5fold",
        "interpretation": "within-record temporal-block interpolation; not independent trip, route, or external validation",
        "methods": METHODS, "policies": [p for p, _, _ in POLICIES],
        "rf_config": pilot.RF_CONFIG, "training_cap": 20000, "seed": 42,
        "class_cap_seed": "42000 + zero-based fold", "threads": 4,
        "source_hashes_revalidated_live": True,
        "source_files": [{"path": name, "sha256": sha256_file(ROOT / name)} for name in (
            "pilot_protocol.md", "pre_run_clarifications.md", "fold_feasibility_amendment.md",
            "recording_timeblock_protocol_v2.md", "session_fold_map_v1.json", "run_pilot.py",
            "qa_timeblock_splits.py", "run_timeblock_diagnostic.py", "timeblock_split_support_v3.json")],
        "cache_sha256": sha256_file(CACHE), "validation": validation,
        "model_fit_started": True,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # Patch only the imported function in this process; the original runner file is untouched.
    original_build_folds = pilot._build_folds
    try:
        for policy, _labels_key, _support_key in POLICIES:
            frozen = [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
                      for tr, te in validation["policy_checks"][policy]["folds"]]
            expected_n = validation["policy_checks"][policy]["support"]["source_rows_after_runner_filter"]
            def timeblock_folds(y, groups, split_kind, group_fold_map=None, _folds=frozen, _n=expected_n):
                if split_kind != "timeblock" or len(y) != _n:
                    raise ValueError("Unexpected split kind or runner row order/length")
                return [(a.copy(), b.copy()) for a, b in _folds]
            pilot._build_folds = timeblock_folds
            pilot._run_policy(sessions, policy, "timeblock", OUTPUT / f"{policy}_timeblock.json", {}, methods=METHODS)
    finally:
        pilot._build_folds = original_build_folds
    (OUTPUT / "complete.json").write_text(json.dumps({"complete": True, "fit_count": 40,
        "method_count": 4, "policy_count": 2, "folds_per_method": 5}, indent=2), encoding="utf-8")
    print(f"TIMEBLOCK COMPLETE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
