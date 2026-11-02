"""Build the frozen RoadSens session fold map without fitting a model.

Reads only seconds_elapsed, annotation_text, and source-file bytes. It mirrors
the pilot's every-tenth-original-row labels and records feasibility counts.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "inner"
OUT = ROOT / "session_fold_map_v1.json"
CLASSES = ("Normal", "Pothole", "Bump")
N_SPLITS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relpath(path: Path) -> str:
    return path.relative_to(DATA).as_posix()


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, usecols=lambda c: c in {"seconds_elapsed", "annotation_text"},
                     low_memory=False)
    if "seconds_elapsed" not in df:
        raise ValueError(f"missing seconds_elapsed: {relpath(path)}")
    times = pd.to_numeric(df["seconds_elapsed"], errors="coerce").to_numpy(dtype=float)
    finite_time = np.isfinite(times)
    labels = np.full(len(df), None, dtype=object)
    if path.parent.name == "combined_normal":
        labels[:] = "Normal"
    elif "annotation_text" in df:
        raw = df["annotation_text"].fillna("").astype(str).str.strip().to_numpy()
        labels[raw == "Pothole"] = "Pothole"
        labels[raw == "Bump"] = "Bump"
    # Preserve the pilot's source cadence: retain original indices 0, 10, 20, ...
    keep = np.arange(0, len(df), 10, dtype=np.int64)
    return labels[keep], finite_time[keep]


def fold_counts(files: list[str], labels_by_file: dict[str, np.ndarray],
                times_by_file: dict[str, np.ndarray], fold_by_file: dict[str, int],
                blank_normal: bool, sensitivity_labels: dict[str, np.ndarray]) -> dict:
    out = {}
    for fold in range(N_SPLITS):
        train_counts, test_counts = Counter(), Counter()
        train_files, test_files = [], []
        for key in files:
            labels, finite = labels_by_file[key], times_by_file[key]
            # Counts mirror the runner's scored-row mask: retained, labeled rows
            # with finite seconds_elapsed. This also matches its map label validator.
            valid = finite & np.isin(labels, CLASSES)
            if blank_normal and key.startswith("combined_anomalies/"):
                # Blank labels are intentionally represented as None in primary data;
                # recover them from the companion sensitivity labels below.
                labels = sensitivity_labels[key]
                valid = finite & np.isin(labels, CLASSES)
            if fold_by_file[key] == fold:
                test_files.append(key)
                test_counts.update(labels[valid].tolist())
            else:
                train_files.append(key)
                train_counts.update(labels[valid].tolist())
        out[str(fold + 1)] = {
            "train": {"sessions": len(train_files), "class_rows": {c: train_counts[c] for c in CLASSES}},
            "test": {"sessions": len(test_files), "class_rows": {c: test_counts[c] for c in CLASSES}},
        }
    return out


def main() -> None:
    normal_paths = sorted((DATA / "combined_normal").glob("*.csv"), key=relpath)
    anomaly_paths = sorted((DATA / "combined_anomalies").glob("*.csv"), key=relpath)
    anomaly_paths = [p for p in anomaly_paths if p.name != "99.csv"]
    all_paths = normal_paths + anomaly_paths
    labels_by_file: dict[str, np.ndarray] = {}
    sensitivity_labels: dict[str, np.ndarray] = {}
    times_by_file: dict[str, np.ndarray] = {}
    hashes = {}
    for path in all_paths:
        key = relpath(path)
        labels, finite = read_labels(path)
        labels_by_file[key], times_by_file[key] = labels, finite
        hashes[key] = sha256(path)
        if key.startswith("combined_anomalies/"):
            df = pd.read_csv(path, usecols=lambda c: c in {"seconds_elapsed", "annotation_text"},
                             low_memory=False)
            raw = df["annotation_text"].fillna("").astype(str).str.strip().to_numpy()
            sens = np.full(len(df), None, dtype=object)
            sens[raw == "Pothole"] = "Pothole"
            sens[raw == "Bump"] = "Bump"
            sens[raw == ""] = "Normal"
            sensitivity_labels[key] = sens[::10]

    # Independent anomaly-only SGKF: one row per labeled, time-finite retained sample.
    anomaly_keys = [relpath(p) for p in anomaly_paths]
    y_parts, group_parts = [], []
    for key in anomaly_keys:
        y = labels_by_file[key]
        mask = times_by_file[key] & np.isin(y, ("Pothole", "Bump"))
        y_parts.extend(y[mask].tolist())
        group_parts.extend([key] * int(mask.sum()))
    y = np.asarray(y_parts, dtype=object)
    groups = np.asarray(group_parts, dtype=object)
    if not len(y) or set(y) != {"Pothole", "Bump"} or len(set(groups)) != 96:
        raise AssertionError(f"unexpected anomaly labeled data: rows={len(y)}, sessions={len(set(groups))}")
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    anomaly_fold: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        for key in sorted(set(groups[test_idx])):
            if key in anomaly_fold:
                raise AssertionError(f"anomaly session assigned twice: {key}")
            anomaly_fold[key] = fold
    if set(anomaly_fold) != set(anomaly_keys):
        raise AssertionError("anomaly SGKF did not assign every anomaly session")

    # Sorted normal paths, then the frozen RandomState(42) permutation, one per fold.
    normal_keys = sorted(relpath(p) for p in normal_paths)
    if len(normal_keys) != N_SPLITS:
        raise AssertionError(f"expected 5 normal-only sessions, got {len(normal_keys)}")
    normal_order = np.random.RandomState(42).permutation(len(normal_keys)).tolist()
    normal_fold = {normal_keys[index]: fold for fold, index in enumerate(normal_order)}
    fold_by_file = {**anomaly_fold, **normal_fold}
    all_keys = sorted(labels_by_file)
    if len(fold_by_file) != 101 or set(fold_by_file) != set(all_keys):
        raise AssertionError("final map does not cover exactly 101 source-relative paths")

    # Explicitly recalculate the original all-session SGKF and preserve its missing-class counts.
    old_y, old_g = [], []
    for key in all_keys:
        yy = labels_by_file[key]
        valid = times_by_file[key] & np.isin(yy, CLASSES)
        old_y.extend(yy[valid].tolist())
        # Reproduce the original runner's all-session grouping key for the recorded
        # failure (bare basename); current inventory has no basename collisions.
        old_g.extend([Path(key).name] * int(valid.sum()))
    old_y, old_g = np.asarray(old_y, dtype=object), np.asarray(old_g, dtype=object)
    original_folds = {}
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(old_y)), old_y, old_g), 1):
        original_folds[str(fold)] = {
            "train": {c: int(np.sum(old_y[tr] == c)) for c in CLASSES},
            "test": {c: int(np.sum(old_y[te] == c)) for c in CLASSES},
        }

    primary = fold_counts(all_keys, labels_by_file, times_by_file, fold_by_file, False, sensitivity_labels)
    sensitive = fold_counts(all_keys, labels_by_file, times_by_file, fold_by_file, True, sensitivity_labels)
    for policy, folds in (("primary", primary), ("blank_normal", sensitive)):
        for fold, counts in folds.items():
            for side in ("train", "test"):
                absent = [c for c, n in counts[side]["class_rows"].items() if n == 0]
                if absent:
                    raise SystemExit(f"STOP: {policy} fold {fold} {side} missing classes {absent}")

    # Ensure there is no overlap and every source session occurs in exactly one test fold.
    test_sets = [{k for k, f in fold_by_file.items() if f == fold} for fold in range(N_SPLITS)]
    for i in range(N_SPLITS):
        for j in range(i + 1, N_SPLITS):
            assert not (test_sets[i] & test_sets[j]), f"fold overlap {i + 1}/{j + 1}"
    assert set.union(*test_sets) == set(all_keys)
    assert sum(len(s) for s in test_sets) == len(all_keys)

    payload = {
        "version": 1,
        "created_at": "2026-10-16",
        "design": "Five sorted normal-only paths permuted with RandomState(42), one per fold; 96 anomaly sessions assigned by anomaly-only StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on every tenth original row with finite seconds_elapsed and explicit Pothole/Bump label.",
        "data_root": "candidate_roadsens/data/inner",
        "class_order": list(CLASSES),
        "session_count": len(all_keys),
        "session_fold": {k: fold_by_file[k] + 1 for k in all_keys},
        "source_sha256": {k: hashes[k] for k in all_keys},
        "normal_assignment_order": [normal_keys[i] for i in normal_order],
        "original_unconstrained_all_session_sgkf": {
            "seed": 42, "n_splits": 5, "fold_counts": original_folds,
        },
        "revised_counts": {"primary": primary, "blank_normal_sensitivity": sensitive},
        "disjoint_test_fold_assertions": {"passed": True, "fold_count": 5, "sessions": len(all_keys)},
        "model_fit_or_scores": False,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.name}: {len(all_keys)} sessions, 5 folds; no model fit or scores.")


if __name__ == "__main__":
    main()
