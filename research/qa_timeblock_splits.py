"""Construct frozen time-block splits and report label support only; never fits a model."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent
CLASSES = ("Normal", "Pothole", "Bump")
POLICIES = {"primary": "labels", "blank_normal_sensitivity": "labels_blank_normal"}
METHODS = ("full_numeric", "missing_only", "context_only", "imu_only")
EMBARGO_SECONDS = 1.0
TRAIN_CAP = 20000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def block_ids(times: np.ndarray, span_times: np.ndarray | None = None) -> np.ndarray:
    """Equal elapsed-time bins, [edge_i, edge_i+1), except the final closed bin."""
    t = np.asarray(times, dtype=np.float64)
    out = np.full(t.shape, -1, dtype=np.int8)
    good = np.isfinite(t)
    span = t if span_times is None else np.asarray(span_times, dtype=np.float64)
    span = span[np.isfinite(span)]
    if not np.any(good) or not len(span):
        return out
    lo, hi = float(np.min(span)), float(np.max(span))
    if hi == lo:
        out[good] = 4
        return out
    raw = np.floor((t[good] - lo) * (5.0 / (hi - lo))).astype(np.int64)
    out[good] = np.minimum(raw, 4).astype(np.int8)
    return out


def _near_test(timestamp: np.ndarray, test_times: np.ndarray, embargo: float) -> np.ndarray:
    """True for finite timestamp within <= embargo of nearest timestamp in test bin."""
    out = np.zeros(len(timestamp), dtype=bool)
    q = np.isfinite(timestamp)
    if not np.any(q) or not len(test_times):
        return out
    tt = np.sort(np.asarray(test_times, dtype=np.float64))
    vals = timestamp[q]
    right = np.searchsorted(tt, vals, side="left")
    dist = np.full(len(vals), np.inf)
    has_right = right < len(tt)
    dist[has_right] = np.abs(vals[has_right] - tt[right[has_right]])
    has_left = right > 0
    dist[has_left] = np.minimum(dist[has_left], np.abs(vals[has_left] - tt[right[has_left] - 1]))
    out[np.flatnonzero(q)] = dist <= embargo
    return out


def build(cache: dict) -> dict:
    sessions = sorted(cache["sessions"], key=lambda s: s["session_key"])
    if len(sessions) != 101:
        raise ValueError(f"Expected 101 source CSV recordings, got {len(sessions)}")
    assignments = {s["session_key"]: block_ids(s["times"], s["times_full"]) for s in sessions}
    folds_out = {p: {} for p in POLICIES}
    all_checks = []
    for fold in range(5):
        per_policy = {}
        for policy, label_key in POLICIES.items():
            train_counts = {c: 0 for c in CLASSES}
            test_counts = {c: 0 for c in CLASSES}
            train_n = test_n = embargo_removed = 0
            block_test_counts = [0] * 5
            session_membership = []
            for s in sessions:
                key = s["session_key"]
                t = np.asarray(s["times"], dtype=np.float64)
                b = assignments[key]
                labels = np.asarray(s[label_key], dtype=object)
                test_mask = (b == fold) & np.isfinite(t)
                candidate_train = (b >= 0) & (b != fold) & np.isfinite(t)
                remove = _near_test(t, t[test_mask], EMBARGO_SECONDS)
                train_mask = candidate_train & ~remove
                embargo_removed += int(np.sum(candidate_train & remove))
                # Row is scoreable only when finite-time and labeled under that policy.
                test_score = test_mask & np.isin(labels, CLASSES)
                train_score = train_mask & np.isin(labels, CLASSES)
                train_n += int(train_score.sum()); test_n += int(test_score.sum())
                for c in CLASSES:
                    train_counts[c] += int(np.sum(train_score & (labels == c)))
                    test_counts[c] += int(np.sum(test_score & (labels == c)))
                block_test_counts[fold] += int(test_mask.sum())
                # Keep split provenance compact but exact: each recording's selected original rows.
                session_membership.append({
                    "session_key": key,
                    "train_source_rows": np.asarray(s["source_row"])[train_mask].astype(int).tolist(),
                    "test_source_rows": np.asarray(s["source_row"])[test_mask].astype(int).tolist(),
                    "n_embargoed_candidate_train_rows": int(np.sum(candidate_train & remove)),
                })
            train_ok = all(train_counts[c] > 0 for c in CLASSES)
            test_ok = all(test_counts[c] > 0 for c in CLASSES)
            # Reproduce the pilot's stratified-cap quota arithmetic as a fit-free support check.
            capped_counts = dict(train_counts)
            if train_n > TRAIN_CAP:
                class_values = sorted(CLASSES)
                quota = {c: min(train_counts[c], max(1, int(TRAIN_CAP * train_counts[c] / train_n)))
                         for c in class_values}
                while sum(quota.values()) > TRAIN_CAP:
                    c = max(quota, key=quota.get)
                    quota[c] -= 1
                capped_counts = quota
            per_policy[policy] = {
                "train_n": train_n, "test_n": test_n,
                "train_class_rows": train_counts, "test_class_rows": test_counts,
                "capped_train_n": min(train_n, TRAIN_CAP),
                "capped_train_class_rows_expected": capped_counts,
                "capped_train_all_three_classes": all(capped_counts[c] > 0 for c in CLASSES),
                "train_all_three_classes": train_ok, "test_all_three_classes": test_ok,
                "all_three_classes": train_ok and test_ok,
                "embargo_removed_candidate_train_rows": embargo_removed,
                "test_finite_time_rows_all_labels": block_test_counts[fold],
                "sessions_with_train_and_test_rows": sum(
                    bool(m["train_source_rows"]) and bool(m["test_source_rows"]) for m in session_membership),
                "session_membership": session_membership,
            }
            folds_out[policy][str(fold + 1)] = per_policy[policy]
        all_checks.append({"fold": fold + 1, "policies": {
            p: {"train_class_rows": v["train_class_rows"], "test_class_rows": v["test_class_rows"],
                "all_three_classes": v["all_three_classes"]} for p, v in per_policy.items()}})
    feasible = all(v["all_three_classes"] for p in POLICIES for v in folds_out[p].values())
    return {"passed_split_integrity": True, "all_folds_feasible_for_later_fit": feasible,
            "fit_or_score_performed": False, "class_order": list(CLASSES),
            "methods_reserved_for_later_authorized_fit": list(METHODS),
            "folds": folds_out, "compact_support": all_checks,
            "row_assignment": {k: v.astype(int).tolist() for k, v in assignments.items()},
            "split_parameters": {"fold_count": 5, "equal_elapsed_time_bins": True,
                                 "bin_convention": "[left,right), final bin includes maximum",
                                 "embargo_seconds_inclusive": EMBARGO_SECONDS,
                                 "training_cap_for_support_check_only": TRAIN_CAP,
                                 "embargo_computed_using_all_finite_time_test_rows": True}}


def synthetic_checks() -> dict:
    # Equal bins, final endpoint, NaN handling, inclusive embargo and no cross-record leakage.
    ts = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., np.nan])
    b = block_ids(ts)
    assert b.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, -1]
    near = _near_test(np.array([0., 1., 1.00001, 8., np.nan]), np.array([0.]), 1.0)
    assert near.tolist() == [True, True, False, False, False]
    # Same timestamps in different recording calls are evaluated independently.
    assert _near_test(np.array([100.]), np.array([0.]), 1.0).tolist() == [False]
    # Every fold gets test rows, and remaining train rows are outside the 1s embargo.
    for k in range(5):
        test = b == k
        train = np.isfinite(ts) & (b >= 0) & (b != k)
        trt = ts[train & ~_near_test(ts, ts[test], 1.0)]
        assert not np.any(_near_test(trt, ts[test], 1.0))
    return {"passed": True, "fit_or_score_performed": False,
            "checks": ["half-open bins and closed final endpoint", "nonfinite timestamp exclusion",
                       "inclusive 1s same-record embargo", "no cross-record timestamp comparison",
                       "train/test assignment disjointness"]}


def main() -> None:
    cache_path = ROOT / "session_feature_cache_v1.joblib"
    cache = joblib.load(cache_path)
    out = build(cache)
    protocol = ROOT / "recording_timeblock_protocol_v2.md"
    clarifications = ROOT / "pre_run_clarifications.md"
    amendment = ROOT / "fold_feasibility_amendment.md"
    manifest = {
        "created_local": "2026-10-16",
        "synthetic_qa": synthetic_checks(),
        "input_cache": str(cache_path.name), "cache_sha256": sha256_file(cache_path),
        "cache_runner_sha256": cache.get("runner_sha256"),
        "source_recording_sha256": cache["source_sha256"],
        "protocol_sha256": sha256_file(protocol),
        "preliminary_protocol_sha256": sha256_file(ROOT / "recording_timeblock_protocol.md"),
        "original_protocol_sha256": sha256_file(ROOT / "pilot_protocol.md"),
        "clarifications_sha256": sha256_file(clarifications),
        "fold_feasibility_amendment_sha256": sha256_file(amendment),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        **out,
    }
    output = ROOT / "timeblock_split_support_v3.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "all_folds_feasible_for_later_fit": out["all_folds_feasible_for_later_fit"],
                      "compact_support": out["compact_support"], "fit_or_score_performed": False},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
