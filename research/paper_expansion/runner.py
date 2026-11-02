"""Paper-expansion split, estimator, and artifact scaffold.

Nothing is fit by default. ``--run`` requires a root-reviewed frozen matrix and an
existing, hash-matched preflight directory. All artifacts stay under paper_expansion/runs.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
RUNS = HERE / "runs"
CACHE = PILOT / "session_feature_cache_v1.joblib"
EXPECTED_PYTHON = Path(r"C:\Users\Administrator\miniconda3\python.exe")
EXPECTED_SKLEARN = "1.9.0"
CLASSES = ("Normal", "Pothole", "Bump")
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
POLICIES = ("primary", "blank_normal")
SPLIT_DESIGNS = (
    {"id": f"random{seed}", "kind": "random", "seed": seed} for seed in (42, 43, 44)
)
SPLIT_DESIGNS = tuple(SPLIT_DESIGNS) + tuple(
    {"id": f"record{seed}", "kind": "record", "seed": seed} for seed in (42, 43, 44)
) + ({"id": "timeblock42", "kind": "timeblock", "seed": 42},)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_modules():
    sys.path.insert(0, str(PILOT))
    import run_pilot as pilot
    import confirm_seeds
    from qa_timeblock_splits import block_ids, _near_test
    return pilot, confirm_seeds, block_ids, _near_test


@dataclass
class PolicyRows:
    y: np.ndarray
    groups: np.ndarray
    source_rows: np.ndarray
    times: np.ndarray
    owners: np.ndarray
    local_rows: np.ndarray


def policy_rows(sessions: list[dict], policy: str) -> PolicyRows:
    labels_parts, group_parts, source_parts, time_parts, owner_parts, local_parts = [], [], [], [], [], []
    label_key = "labels" if policy == "primary" else "labels_blank_normal"
    for owner, session in enumerate(sessions):
        labels = np.asarray(session[label_key], dtype=object)
        times = np.asarray(session["times"], dtype=np.float64)
        keep = np.isfinite(times) & np.isin(labels, CLASSES)
        loc = np.flatnonzero(keep)
        labels_parts.append(labels[keep])
        group_parts.append(np.full(len(loc), session["session_key"], dtype=object))
        source_parts.append(np.asarray(session["source_row"], dtype=np.int64)[keep])
        time_parts.append(times[keep])
        owner_parts.append(np.full(len(loc), owner, dtype=np.int32))
        local_parts.append(loc.astype(np.int32))
    return PolicyRows(np.concatenate(labels_parts), np.concatenate(group_parts),
                      np.concatenate(source_parts), np.concatenate(time_parts),
                      np.concatenate(owner_parts), np.concatenate(local_parts))


def cap_train_indices(train: np.ndarray, y: np.ndarray, fold: int, seed: int,
                      cap: int = 20000) -> np.ndarray:
    """Original proportional quota sampler; seed*1000+fold preserves seed-42 rows."""
    train = np.asarray(train, dtype=np.int64)
    if len(train) <= cap:
        return train.copy()
    rng = np.random.RandomState(seed * 1000 + fold)
    classes = np.unique(y[train])
    counts = {c: int(np.sum(y[train] == c)) for c in classes}
    total = sum(counts.values())
    quota = {c: min(counts[c], max(1, int(cap * counts[c] / total))) for c in classes}
    while sum(quota.values()) > cap:
        c = max(quota, key=quota.get)
        quota[c] -= 1
    chosen = []
    for c in classes:
        rows = train[y[train] == c]
        chosen.extend(rng.choice(rows, quota[c], replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _fold_class_counts(y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    return {side: {"n": int(len(ix)), "classes": {c: int(np.sum(y[ix] == c)) for c in CLASSES}}
            for side, ix in (("train", tr), ("test", te))}


def _random_splits(rows: PolicyRows, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in splitter.split(np.zeros(len(rows.y)), rows.y)]


def _record_splits(rows: PolicyRows, fmap: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    if set(rows.groups) - set(fmap):
        raise ValueError("record map misses one or more scored source recordings")
    folds = []
    for fold in range(1, 6):
        te = np.flatnonzero(np.fromiter((fmap[k] == fold for k in rows.groups),
                                        dtype=bool, count=len(rows.groups)))
        tr = np.setdiff1d(np.arange(len(rows.y), dtype=np.int64), te, assume_unique=True)
        folds.append((tr, te))
    return folds


def _timeblock_splits(sessions: list[dict], rows: PolicyRows, policy: str,
                      split_qa: dict, block_ids, near_test):
    """Rebuild v2 diagnostic split exactly and verify against the frozen support QA."""
    policy_key = "primary" if policy == "primary" else "blank_normal_sensitivity"
    expected = split_qa["folds"][policy_key]
    train_source, test_source = [[] for _ in range(5)], [[] for _ in range(5)]
    for owner, session in enumerate(sessions):
        t = np.asarray(session["times"], dtype=np.float64)
        bins = block_ids(t, session["times_full"])
        key = session["session_key"]
        for fold in range(5):
            test_mask = (bins == fold) & np.isfinite(t)
            candidate = (bins >= 0) & (bins != fold) & np.isfinite(t)
            train_mask = candidate & ~near_test(t, t[test_mask], 1.0)
            exp = next(x for x in expected[str(fold + 1)]["session_membership"]
                       if x["session_key"] == key)
            if np.asarray(session["source_row"])[test_mask].astype(int).tolist() != exp["test_source_rows"]:
                raise ValueError(f"timeblock test source rows differ from v3 QA: {key}, fold {fold + 1}")
            if np.asarray(session["source_row"])[train_mask].astype(int).tolist() != exp["train_source_rows"]:
                raise ValueError(f"timeblock train source rows differ from v3 QA: {key}, fold {fold + 1}")
            # Use the shared policy-row ordering to translate source records to compact IDs.
            local_for_session = np.flatnonzero(rows.owners == owner)
            source_for_session = rows.source_rows[local_for_session]
            test_sr = set(np.asarray(session["source_row"])[test_mask].astype(int).tolist())
            train_sr = set(np.asarray(session["source_row"])[train_mask].astype(int).tolist())
            test_ids = local_for_session[np.isin(source_for_session, list(test_sr))]
            train_ids = local_for_session[np.isin(source_for_session, list(train_sr))]
            test_source[fold].extend(test_ids.tolist())
            train_source[fold].extend(train_ids.tolist())
    folds = []
    for fold in range(5):
        tr = np.asarray(sorted(train_source[fold]), dtype=np.int64)
        te = np.asarray(sorted(test_source[fold]), dtype=np.int64)
        support = expected[str(fold + 1)]
        got_train = {c: int(np.sum(rows.y[tr] == c)) for c in CLASSES}
        got_test = {c: int(np.sum(rows.y[te] == c)) for c in CLASSES}
        if got_train != support["train_class_rows"] or got_test != support["test_class_rows"]:
            raise ValueError(f"timeblock scored class rows differ from support QA, fold {fold + 1}")
        folds.append((tr, te))
    return folds


def build_all_splits(sessions: list[dict], frozen_map: dict, split_qa: dict) -> tuple[dict, dict]:
    pilot, confirm, block_ids, near_test = _load_modules()
    all_specs, row_tables = {}, {}
    for policy in POLICIES:
        rows = policy_rows(sessions, policy)
        row_tables[policy] = rows
        for spec in SPLIT_DESIGNS:
            design_id, kind, seed = spec["id"], spec["kind"], spec["seed"]
            if kind == "random":
                folds = _random_splits(rows, seed)
                fmap_payload = None
            elif kind == "record":
                if seed == 42:
                    # Preserve the original frozen map byte-for-byte across sklearn versions;
                    # rebuilding SGKF can change assignments and break the v2 RF identity gate.
                    fmap_payload = {"session_fold": dict(frozen_map["session_fold"])}
                else:
                    map_path = PILOT / "runs" / f"confirmation_seed{seed}_v1" / "session_fold_map.json"
                    fmap_payload = json.loads(map_path.read_text(encoding="utf-8"))
                    if (fmap_payload.get("seed") != seed or
                        fmap_payload.get("source_sha256") != frozen_map["source_sha256"] or
                        set(fmap_payload.get("session_fold", {})) != set(frozen_map["session_fold"])):
                        raise ValueError(f"frozen confirmation map/source mismatch: seed {seed}")
                folds = _record_splits(rows, fmap_payload["session_fold"])
            else:
                fmap_payload = None
                folds = _timeblock_splits(sessions, rows, policy, split_qa, block_ids, near_test)
            fold_meta = []
            test_seen = []
            for fold, (tr, te) in enumerate(folds):
                if len(np.intersect1d(tr, te)):
                    raise AssertionError(f"train/test index overlap: {policy}/{design_id}/fold{fold + 1}")
                if any(int(_fold_class_counts(rows.y, tr, te)[side]["classes"][c]) == 0
                       for side in ("train", "test") for c in CLASSES):
                    raise ValueError(f"infeasible classes: {policy}/{design_id}/fold{fold + 1}")
                if kind == "record":
                    if set(rows.groups[tr]) & set(rows.groups[te]):
                        raise AssertionError("source-record overlap in recording-heldout split")
                    if any(fmap_payload["session_fold"][key] != fold + 1 for key in set(rows.groups[te])):
                        raise AssertionError("record test rows disagree with generated fold map")
                if kind == "timeblock":
                    if set(rows.groups[tr]) != set(rows.groups[te]):
                        # A source may have no labeled rows in a block; require no surprise only.
                        if not set(rows.groups[tr]).issubset(set(s["session_key"] for s in sessions)):
                            raise AssertionError("timeblock contains unknown source recording")
                fit = cap_train_indices(tr, rows.y, fold, seed, cap=20000)
                if set(np.unique(rows.y[fit])) != set(CLASSES):
                    raise ValueError(f"fit cap removed a class: {policy}/{design_id}/fold{fold + 1}")
                fold_meta.append({"fold": fold + 1, **_fold_class_counts(rows.y, tr, te),
                                  "fit": {"n": len(fit), "classes": {c: int(np.sum(rows.y[fit] == c)) for c in CLASSES}},
                                  "train_indices_sha256": object_hash(tr.tolist()),
                                  "fit_indices_sha256": object_hash(fit.tolist()),
                                  "test_indices_sha256": object_hash(te.tolist())})
                test_seen.extend(te.tolist())
            if kind in ("random", "record"):
                joined = np.concatenate([te for _, te in folds])
                if len(joined) != len(rows.y) or len(np.unique(joined)) != len(rows.y):
                    raise AssertionError(f"5-fold tests do not partition all scoreable rows: {policy}/{design_id}")
            all_specs[(policy, design_id)] = {"spec": spec, "folds": folds,
                "metadata": {"folds": fold_meta, "scored_rows": len(rows.y),
                             "row_identity_sha256": object_hash([[str(g), int(sr), str(y)]
                                 for g, sr, y in zip(rows.groups, rows.source_rows, rows.y)]),
                             "session_fold_map": fmap_payload["session_fold"] if fmap_payload else None,
                             "session_fold_map_sha256": object_hash(fmap_payload["session_fold"])
                                 if fmap_payload else None}}
    return all_specs, row_tables


def validate_immutable_inputs() -> tuple[dict, dict, dict]:
    pilot, confirm, _, _ = _load_modules()
    cache = joblib.load(CACHE)
    parent_hash = sha256(PILOT / "run_pilot.py")
    if cache.get("runner_sha256") != parent_hash:
        raise ValueError("cache parent code hash differs from current immutable runner")
    frozen_map = json.loads((PILOT / "session_fold_map_v1.json").read_text(encoding="utf-8"))
    if cache.get("source_sha256") != frozen_map.get("source_sha256"):
        raise ValueError("cache source hashes differ from frozen original map")
    if len(cache.get("sessions", [])) != 101:
        raise ValueError("expected exactly 101 cached source recordings")
    for session in cache["sessions"]:
        key = session["session_key"]
        if session["sha256"] != frozen_map["source_sha256"].get(key):
            raise ValueError(f"cache/session source hash mismatch: {key}")
        source = PILOT / "data" / "inner" / Path(key)
        if not source.is_file() or sha256(source) != session["sha256"]:
            raise ValueError(f"live source recording hash mismatch: {key}")
    # Seed 42 is an immutable map artifact. Do not regenerate it with SGKF because
    # sklearn versions can alter assignments; exact train/test row identity is checked below.
    if set(frozen_map["session_fold"]) != {s["session_key"] for s in cache["sessions"]}:
        raise ValueError("frozen seed-42 recording map does not cover cached recordings")
    if set(frozen_map["session_fold"].values()) != {1, 2, 3, 4, 5}:
        raise ValueError("frozen seed-42 recording map has unexpected fold values")
    split_qa = json.loads((PILOT / "timeblock_split_support_v3.json").read_text(encoding="utf-8"))
    expected_qa = {
        "cache_sha256": sha256(CACHE),
        "cache_runner_sha256": parent_hash,
        "protocol_sha256": sha256(PILOT / "recording_timeblock_protocol_v2.md"),
        "script_sha256": sha256(PILOT / "qa_timeblock_splits.py"),
    }
    for key, value in expected_qa.items():
        if split_qa.get(key) != value:
            raise ValueError(f"timeblock v3 QA provenance mismatch: {key}")
    return cache, frozen_map, split_qa


def _compare_seed42_original(specs: dict, rows: dict) -> dict:
    """Check exact train/fit/test row sequence against immutable original pilot records."""
    report = {}
    for policy in POLICIES:
        for split_id in ("random42", "record42", "timeblock42"):
            if split_id == "timeblock42":
                base = PILOT / "runs" / "timeblock_v1" / f"{policy}_timeblock"
                split_key = "timeblock"
            else:
                split_key = "random" if split_id == "random42" else "group"
                base = PILOT / "runs" / "pilot_v2" / f"{policy}_{split_key}"
            folds = specs[(policy, split_id)]["folds"]
            rr = rows[policy]
            for feature in ("full_numeric", "missing_only", "context_only", "imu_only"):
                path = Path(str(base) + f".{feature}.checkpoint.json")
                payload = json.loads(path.read_text(encoding="utf-8"))
                for fold, (tr, te) in enumerate(folds):
                    fit = cap_train_indices(tr, rr.y, fold, 42, 20000)
                    record = payload["folds"][fold]
                    for name, indices in (("split_train_rows", tr), ("fit_train_rows", fit), ("test_rows", te)):
                        observed = record[name]
                        expected = [(int(i), str(rr.groups[i]), int(rr.source_rows[i])) for i in indices]
                        got = [(int(x["row"]), str(x["session"]), int(x["source_row"])) for x in observed]
                        if got != expected:
                            raise ValueError(f"seed-42 exact identity mismatch: {policy}/{split_key}/{feature}/fold{fold + 1}/{name}")
            checked = [{"fold": fold + 1, "rows_match": True} for fold in range(5)]
            report[f"{policy}_{split_key}"] = checked
    return report


def _compare_existing_confirmation_rows(specs: dict, rows: dict) -> dict:
    """Preserve the exact seed-43/44 random and group maps used by prior confirmation runs."""
    report = {}
    for seed in (43, 44):
        for policy in POLICIES:
            for split_id, split_name in ((f"random{seed}", "random"), (f"record{seed}", "group")):
                rr = rows[policy]
                folds = specs[(policy, split_id)]["folds"]
                base = PILOT / "runs" / f"confirmation_seed{seed}_v1" / f"{policy}_{split_name}"
                for feature in ("full_numeric", "missing_only", "context_only", "imu_only"):
                    payload = json.loads(Path(str(base) + f".{feature}.checkpoint.json").read_text(encoding="utf-8"))
                    for fold, (tr, te) in enumerate(folds):
                        fit = cap_train_indices(tr, rr.y, fold, seed, 20000)
                        record = payload["folds"][fold]
                        for name, indices in (("split_train_rows", tr), ("fit_train_rows", fit), ("test_rows", te)):
                            observed = record[name]
                            expected = [(int(i), str(rr.groups[i]), int(rr.source_rows[i])) for i in indices]
                            got = [(int(x["row"]), str(x["session"]), int(x["source_row"])) for x in observed]
                            if got != expected:
                                raise ValueError(f"seed-{seed} existing row identity mismatch: {policy}/{split_name}/{feature}/fold{fold+1}/{name}")
                report[f"seed{seed}_{policy}_{split_name}"] = [{"fold": f+1, "rows_match": True} for f in range(5)]
    return report


def _save_preflight(out: Path, cache: dict, frozen: dict, split_qa: dict,
                    specs: dict, rows: dict, exact_identity: dict) -> dict:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    row_dir = out / "row_indices"
    row_dir.mkdir()
    split_dir = out / "split_indices"
    split_dir.mkdir()
    # Persist each policy's ordered scoreable-row identity once.
    for policy, table in rows.items():
        np.savez_compressed(row_dir / f"{policy}.npz",
                            y=np.asarray([CLASS_TO_ID[str(y)] for y in table.y], dtype=np.int8),
                            group=table.groups.astype(str), source_row=table.source_rows,
                            time=table.times, owner=table.owners, local_row=table.local_rows)
    catalog = []
    for (policy, split_id), value in sorted(specs.items()):
        arrays = {}
        for fold, (tr, te) in enumerate(value["folds"]):
            arrays[f"train_{fold}"] = tr
            arrays[f"test_{fold}"] = te
        path = split_dir / f"{policy}__{split_id}.npz"
        np.savez_compressed(path, **arrays)
        catalog.append({"policy": policy, "split_id": split_id, **value["spec"],
                        "index_file": str(path.relative_to(out).as_posix()),
                        "index_file_sha256": sha256(path), "metadata": value["metadata"]})
    manifest = {
        "status": "preflight_complete_fit_free", "model_fit_or_scores": False,
        "cache_path": str(CACHE), "cache_sha256": sha256(CACHE),
        "parent_runner_path": str(PILOT / "run_pilot.py"),
        "parent_runner_sha256": cache["runner_sha256"],
        "frozen_seed42_record_map_sha256": sha256(PILOT / "session_fold_map_v1.json"),
        "timeblock_protocol_sha256": sha256(PILOT / "recording_timeblock_protocol_v2.md"),
        "timeblock_support_sha256": sha256(PILOT / "timeblock_split_support_v3.json"),
        "source_sha256": cache["source_sha256"], "recording_count": len(cache["sessions"]),
        "policies": list(POLICIES), "split_catalog": catalog,
        "seed42_exact_original_train_fit_test_rows": exact_identity,
        "storage_note": "One row identity NPZ per policy and one train/test index NPZ per policy/split. No per-method split copies or predictions exist during preflight.",
    }
    (out / "preflight_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    return manifest


def run_qa() -> dict:
    """Meaningful fit-free checks for split identity, label independence, and train-only preprocessing."""
    cache, frozen, split_qa = validate_immutable_inputs()
    specs, rows = build_all_splits(cache["sessions"], frozen, split_qa)
    identity = {"seed42": _compare_seed42_original(specs, rows),
                "existing_seed43_44": _compare_existing_confirmation_rows(specs, rows)}
    # Split assignments must not vary with target labels. Record/timeblock designs are label-free;
    # verify IDs from source metadata remain identical across policies.
    label_independent = True
    for split_id in ("record42", "record43", "record44"):
        a = specs[("primary", split_id)]["metadata"]["session_fold_map"]
        b = specs[("blank_normal", split_id)]["metadata"]["session_fold_map"]
        if a != b:
            label_independent = False
    # The timeblock source membership is independently checked against a single
    # frozen source-row QA artifact for both policies inside _timeblock_splits.
    if not label_independent:
        raise AssertionError("record/timeblock test source rows differ across label policies")
    # A perturbation of test values must not alter this runner's training-only
    # imputation or scaling values. Exercise the same function used by model fits.
    x_fixture = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0], [7.0, 6.0]])
    fit_ids, test_ids = np.array([0, 1, 2]), np.array([3])
    x_test_perturbed = np.array([[7000.0, -9000.0]])
    x_changed = x_fixture.copy(); x_changed[test_ids] = x_test_perturbed
    _, _, _, med_a, meta_a = preprocess_train_test(x_fixture, fit_ids, test_ids, "__logistic__")
    _, _, _, med_b, meta_b = preprocess_train_test(x_changed, fit_ids, test_ids, "__logistic__")
    if med_a != med_b or meta_a["scaler_mean"] != meta_b["scaler_mean"] or meta_a["scaler_scale"] != meta_b["scaler_scale"]:
        raise AssertionError("test features affected train-only preprocessing")
    if np.array_equal(x_fixture[test_ids], x_changed[test_ids]):
        raise AssertionError("QA fixture perturbation was ineffective")
    return {"passed": True, "checks": ["seed-42 original random/record/timeblock split and fit rows exact for all four feature methods",
                                        "all 42/43/44 random, record, and timeblock splits feasible",
                                        "label-policy-independent record/timeblock source membership",
                                        "train-only median imputation and scaling ignore test values",
                                        "live 101-source/cache/parent/timeblock provenance verified"],
            "original_and_confirmation_exact_rows": identity,
            "recording_count": len(cache["sessions"]),
            "split_count": len(specs), "model_fit_or_scores": False}


def preflight(run_id: str) -> Path:
    cache, frozen, split_qa = validate_immutable_inputs()
    specs, rows = build_all_splits(cache["sessions"], frozen, split_qa)
    identity = {"seed42": _compare_seed42_original(specs, rows),
                "existing_seed43_44": _compare_existing_confirmation_rows(specs, rows)}
    out = RUNS / run_id
    _save_preflight(out, cache, frozen, split_qa, specs, rows, identity)
    return out


def feature_matrix(sessions: list[dict], rows: PolicyRows, feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Materialize one aligned feature matrix from the immutable cached arrays."""
    field = {"full_numeric": "numeric", "missing_only": "missing",
             "context_only": "context", "imu_only": "imu"}[feature_set]
    columns_key = {"numeric": "numeric_columns", "missing": "numeric_columns",
                   "context": "context_columns", "imu": "imu_columns"}[field]
    width = len(sessions[0][columns_key])
    out = np.empty((len(rows.y), width), dtype=np.float64)
    for owner, session in enumerate(sessions):
        ix = np.flatnonzero(rows.owners == owner)
        if len(ix):
            out[ix] = np.asarray(session[field], dtype=np.float64)[rows.local_rows[ix]]
    cols = list(sessions[0][columns_key])
    if feature_set == "missing_only":
        cols = [f"isfinite({c})" for c in cols]
    return out, cols


def preprocess_train_test(x: np.ndarray, tr: np.ndarray, te: np.ndarray,
                          feature_set: str) -> tuple[np.ndarray, np.ndarray, list[int], list[float], dict]:
    """Match pilot training-only median behavior; scale only logistic-regression inputs."""
    xtr0, xte0 = np.asarray(x[tr], dtype=np.float64), np.asarray(x[te], dtype=np.float64)
    if feature_set == "missing_only":
        raw_numeric = None
        # Missing-only columns are retained only when the corresponding numeric field
        # had at least one finite value among this fold's fit rows.
        keep = np.any(np.isfinite(xtr0), axis=0)
        kept = np.flatnonzero(keep).tolist()
        xtr, xte = xtr0[:, keep], xte0[:, keep]
        medians = []
        if xtr.shape[1] == 0:
            xtr, xte = np.zeros((len(tr), 1)), np.zeros((len(te), 1))
            kept = [0]
        return xtr, xte, kept, medians, {"kind": "missing_mask_no_imputation"}
    finite = np.isfinite(xtr0)
    keep = finite.any(axis=0)
    kept = np.flatnonzero(keep).tolist()
    xtr0, xte0 = xtr0[:, keep], xte0[:, keep]
    if xtr0.shape[1] == 0:
        xtr0, xte0 = np.zeros((len(tr), 1)), np.zeros((len(te), 1))
        medians = [0.0]
        kept = [0]
    else:
        med = np.nanmedian(xtr0, axis=0)
        med[~np.isfinite(med)] = 0.0
        xtr0 = np.where(np.isfinite(xtr0), xtr0, med)
        xte0 = np.where(np.isfinite(xte0), xte0, med)
        medians = [float(v) for v in med]
    meta = {"kind": "median_train_only"}
    if feature_set == "__logistic__":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        xtr0 = scaler.fit_transform(xtr0)
        xte0 = scaler.transform(xte0)
        meta.update({"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()})
    return np.asarray(xtr0, dtype=np.float64), np.asarray(xte0, dtype=np.float64), kept, medians, meta


def _metric_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
    present = [c for c in CLASSES if np.any(y_true == c)]
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)),
            "observed_class_macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
            "class_support": {c: int(np.sum(y_true == c)) for c in CLASSES},
            "class_recall": {c: float(v) for c, v in zip(CLASSES, recall_score(
                y_true, y_pred, labels=list(CLASSES), average=None, zero_division=0))},
            "confusion": confusion_matrix(y_true, y_pred, labels=list(CLASSES)).tolist()}


def _make_estimator(name: str):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    if name == "RF100":
        return RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=4)
    if name == "ExtraTrees100":
        return ExtraTreesClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced", bootstrap=False, random_state=42, n_jobs=4)
    if name == "HistGB":
        import inspect
        params = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=42)
        if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
            params["class_weight"] = "balanced"
        return HistGradientBoostingClassifier(**params)
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced", tol=1e-4, random_state=42)
    raise ValueError(name)


def _prediction_baseline(policy: str, split_id: str, feature: str, fold_zero: int) -> np.ndarray:
    """Read old predictions from the existing per-method checkpoint, once per fold call."""
    if split_id == "timeblock42":
        path = PILOT / "runs" / "timeblock_v1" / f"{policy}_timeblock.{feature}.checkpoint.json"
    else:
        split = "random" if split_id.startswith("random") else "group"
        path = PILOT / "runs" / "pilot_v2" / f"{policy}_{split}.{feature}.checkpoint.json"
    return np.asarray(_baseline_predictions(str(path))[fold_zero], dtype=object)


@lru_cache(maxsize=128)
def _baseline_predictions(path: str) -> tuple[tuple[str, ...], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(tuple(str(x["pred"]) for x in fold["predictions"]) for fold in payload["folds"])


def run_main(run_id: str, preflight_dir: Path, matrix: dict) -> Path:
    """Run frozen phase A only, preserving every fold result or failure."""
    if not preflight_dir.is_dir():
        raise FileNotFoundError(f"matching preflight directory required: {preflight_dir}")
    manifest = json.loads((preflight_dir / "preflight_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "preflight_complete_fit_free" or manifest.get("model_fit_or_scores") is not False:
        raise ValueError("preflight is not a validated fit-free snapshot")
    if manifest["cache_sha256"] != sha256(CACHE) or manifest["parent_runner_sha256"] != sha256(PILOT / "run_pilot.py"):
        raise ValueError("preflight hashes no longer match live cache/parent")
    cache, frozen, split_qa = validate_immutable_inputs()
    specs, rows = build_all_splits(cache["sessions"], frozen, split_qa)
    exact = _compare_seed42_original(specs, rows)
    import sklearn, pandas
    out = RUNS / run_id
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "preflight_manifest.json").write_bytes((preflight_dir / "preflight_manifest.json").read_bytes())
    cfg = {"phase": "A", "matrix": matrix, "features": ["full_numeric", "missing_only", "context_only", "imu_only"],
           "models": ["RF100", "ExtraTrees100", "HistGB", "LogisticRegression"],
           "splits": ["random42", "record42", "timeblock42"], "policies": list(POLICIES),
           "folds": 5, "preflight_manifest_sha256": sha256(preflight_dir / "preflight_manifest.json"),
           "cache_sha256": sha256(CACHE), "parent_runner_sha256": sha256(PILOT / "run_pilot.py"),
           "protocol_sha256": sha256(HERE / "protocol_v1.md"), "driver_sha256": sha256(Path(__file__).resolve()),
           "known_content_overlap_amendment": "content_overlap_amendment_v1.md",
           "known_content_overlap_amendment_sha256": sha256(HERE / "content_overlap_amendment_v1.md"),
           "runtime_amendment": matrix.get("runtime_amendment"),
           "runtime_amendment_sha256": sha256(HERE / matrix["runtime_amendment"]) if matrix.get("runtime_amendment") else None,
           "matrix_path": str(Path(matrix.get("matrix_path", "phase_A_matrix_v2.json"))),
           "matrix_sha256": object_hash(matrix),
           "runtime": {"python": sys.version, "platform": platform.platform(),
                       "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                       "executable": str(Path(sys.executable).resolve()), "numpy": np.__version__,
                       "scikit_learn": sklearn.__version__, "pandas": pandas.__version__},
           "seed42_identity_check": exact, "fit_started": False}
    (out / "run_manifest.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import warnings
    from threadpoolctl import threadpool_limits
    failures = []
    # Complete all RF seed-42 fits first; the random/record cells reproduce old outputs
    # before any new estimator family is allowed to run.
    jobs = [(p, s, f, "RF100") for p in POLICIES for s in ("random42", "record42", "timeblock42")
            for f in ("full_numeric", "missing_only", "context_only", "imu_only")]
    jobs += [(p, s, f, m) for m in ("ExtraTrees100", "HistGB", "LogisticRegression")
             for p in POLICIES for s in ("random42", "record42", "timeblock42")
             for f in ("full_numeric", "missing_only", "context_only", "imu_only")]
    # The historical seed-42 RF is checked before any other estimator is fit.
    with threadpool_limits(limits=4):
      for job_index, (policy, split_id, feature, model_name) in enumerate(jobs):
        split = specs[(policy, split_id)]
        table = rows[policy]
        x, feature_columns = feature_matrix(cache["sessions"], table, feature)
        fold_predictions, fold_truth, fold_rows, fold_metrics = [], [], [], []
        for fold, (tr0, te) in enumerate(split["folds"]):
            fit = cap_train_indices(tr0, table.y, fold, 42, 20000)
            is_reproduction = model_name == "RF100" and feature in ("full_numeric", "missing_only", "context_only", "imu_only") and split_id in ("random42", "record42", "timeblock42")
            fit_feature = "missing_only" if feature == "missing_only" else feature
            xtr, xte, kept, medians, preprocess = preprocess_train_test(x, fit, te, fit_feature)
            # Original missing-only features use finite masks, but retain columns according to
            # the corresponding numeric value's fit-row availability.
            if feature == "missing_only":
                numeric, _ = feature_matrix(cache["sessions"], table, "full_numeric")
                mask = np.any(np.isfinite(numeric[fit]), axis=0)
                kept = np.flatnonzero(mask).tolist()
                xtr, xte = x[fit][:, mask], x[te][:, mask]
                if xtr.shape[1] == 0:
                    xtr, xte = np.zeros((len(fit), 1)), np.zeros((len(te), 1)); kept = [0]
                preprocess["kind"] = "pilot_missing_mask_numeric_train_availability"
            if model_name == "LogisticRegression":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                xtr = scaler.fit_transform(xtr)
                xte = scaler.transform(xte)
                preprocess["scaler_mean"] = scaler.mean_.tolist()
                preprocess["scaler_scale"] = scaler.scale_.tolist()
            warnings_seen = []
            try:
                fit_wall = None
                estimator = _make_estimator(model_name)
                # sklearn HistGB before class_weight support receives balanced train weights.
                fit_params = {}
                if model_name == "HistGB" and "class_weight" not in estimator.get_params():
                    from sklearn.utils.class_weight import compute_sample_weight
                    fit_params["sample_weight"] = compute_sample_weight("balanced", table.y[fit])
                fit_started = time.perf_counter()
                try:
                    import psutil
                    rss_before = int(psutil.Process().memory_info().rss)
                except Exception:
                    rss_before = None
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator.fit(xtr, table.y[fit], **fit_params)
                    fit_wall = time.perf_counter() - fit_started
                    predict_started = time.perf_counter()
                    pred_labels = estimator.predict(xte)
                    predict_wall = time.perf_counter() - predict_started
                    warnings_seen = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
                try:
                    rss_after = int(psutil.Process().memory_info().rss)
                except Exception:
                    rss_after = None
                pred_labels = np.asarray(pred_labels, dtype=object)
                if is_reproduction:
                    baseline = _prediction_baseline(policy, split_id, feature, fold)
                    if len(baseline) != len(pred_labels) or not np.array_equal(baseline, pred_labels):
                        raise RuntimeError(f"seed42 RF prediction mismatch {policy}/{split_id}/{feature}/fold{fold+1}")
                fold_result = {"fold": fold + 1, **_metric_payload(table.y[te], pred_labels),
                    "train_n": int(len(fit)), "train_before_cap_n": int(len(tr0)),
                    "train_class_counts": {c: int(np.sum(table.y[fit] == c)) for c in CLASSES},
                    "test_class_counts": {c: int(np.sum(table.y[te] == c)) for c in CLASSES},
                    "fit_wall_seconds": fit_wall, "predict_wall_seconds": predict_wall,
                    "rss_before_fit_bytes": rss_before, "rss_after_predict_bytes": rss_after,
                    "warning_count": len(warnings_seen),
                    "warnings": warnings_seen, "n_iter": (getattr(estimator, "n_iter_", None).tolist()
                        if hasattr(getattr(estimator, "n_iter_", None), "tolist") else
                        (int(estimator.n_iter_) if hasattr(estimator, "n_iter_") else None)),
                    "kept_columns": kept, "imputation_medians": medians, "preprocessing": preprocess,
                    "fit_rows_sha256": object_hash(fit.tolist()), "test_rows_sha256": object_hash(te.tolist())}
                model_path = out / "models" / policy / split_id / feature
                model_path.mkdir(parents=True, exist_ok=True)
                artifact = model_path / f"{model_name}.fold{fold+1}.joblib"
                if artifact.exists(): raise FileExistsError(artifact)
                save_started = time.perf_counter()
                joblib.dump({"estimator": estimator, "feature_columns": feature_columns,
                             "kept_columns": kept, "medians": medians, "preprocessing": preprocess}, artifact, compress=3)
                fold_result["save_wall_seconds"] = time.perf_counter() - save_started
                fold_result["model_artifact"] = str(artifact.relative_to(out).as_posix())
                fold_result["model_sha256"] = sha256(artifact)
                fold_metrics.append(fold_result)
                fold_predictions.append(np.asarray([CLASS_TO_ID[str(y)] for y in pred_labels], dtype=np.int8))
                fold_truth.append(np.asarray([CLASS_TO_ID[str(y)] for y in table.y[te]], dtype=np.int8))
                fold_rows.append(te.copy())
            except Exception as exc:
                fail = {"policy": policy, "split": split_id, "feature": feature, "model": model_name,
                        "fold": fold + 1, "error_type": type(exc).__name__, "error": str(exc),
                        "fit_wall_seconds": fit_wall}
                failures.append(fail)
                fold_metrics.append({"fold": fold + 1, "status": "failed", **fail})
                # Do not continue past a failed RF reproduction: later scores would be unvalidated.
                if is_reproduction:
                    (out / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
                    cfg["fit_started"] = True; cfg["status"] = "stopped_rf_reproduction_failure"
                    (out / "run_manifest.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
                    raise
            if len(fold_predictions) == fold + 1:
                print(f"DONE {policy}/{split_id}/{feature}/{model_name} fold={fold+1}/5", flush=True)
        method_dir = out / "predictions" / policy / split_id
        method_dir.mkdir(parents=True, exist_ok=True)
        if fold_predictions:
            row_ids = np.concatenate(fold_rows)
            truth_ids = np.concatenate(fold_truth)
            pred_ids = np.concatenate(fold_predictions)
            y_true_labels = np.asarray([CLASSES[i] for i in truth_ids], dtype=object)
            y_pred_labels = np.asarray([CLASSES[i] for i in pred_ids], dtype=object)
            per_record = {}
            groups = table.groups[row_ids]
            for group in np.unique(groups):
                ix = groups == group
                per_record[str(group)] = _metric_payload(y_true_labels[ix], y_pred_labels[ix])
            np.savez_compressed(method_dir / f"{feature}__{model_name}.npz",
                test_row=row_ids, y_true=truth_ids, y_pred=pred_ids)
        else:
            per_record = {}
        summary = {"policy": policy, "split": split_id, "feature": feature, "model": model_name,
                   "status": "complete" if len(fold_predictions) == 5 else "partial_failure",
                   "folds": fold_metrics, "pooled_oof": _metric_payload(
                       np.asarray([CLASSES[i] for i in np.concatenate(fold_truth)], dtype=object),
                       np.asarray([CLASSES[i] for i in np.concatenate(fold_predictions)], dtype=object)) if fold_predictions else None,
                   "fold_mean_macro_f1": float(np.mean([f["macro_f1"] for f in fold_metrics if "macro_f1" in f])) if any("macro_f1" in f for f in fold_metrics) else None,
                   "per_record": per_record}
        (method_dir / f"{feature}__{model_name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
        if failures:
            (out / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg["fit_started"] = True; cfg["completed_jobs"] = job_index + 1; cfg["failure_count"] = len(failures)
        (out / "run_manifest.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    cfg["status"] = "phase_complete_with_failures" if failures else "phase_complete"
    (out / "run_manifest.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--qa", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--run-id", type=str, help="exclusive output directory name under paper_expansion/runs")
    parser.add_argument("--matrix", type=Path, help="root-reviewed frozen experiment-matrix JSON")
    parser.add_argument("--preflight-dir", type=Path, help="validated preflight directory required for --run")
    args = parser.parse_args()
    import sklearn
    expected_exe = str(EXPECTED_PYTHON.resolve()).casefold()
    if str(Path(sys.executable).resolve()).casefold() != expected_exe:
        parser.error(f"runtime lock requires {EXPECTED_PYTHON}; got {sys.executable}")
    if sklearn.__version__ != EXPECTED_SKLEARN:
        parser.error(f"runtime lock requires scikit-learn {EXPECTED_SKLEARN}; got {sklearn.__version__}")
    if args.qa:
        print(json.dumps(run_qa(), ensure_ascii=False, indent=2))
        return
    if args.preflight:
        if not args.run_id:
            parser.error("--preflight requires --run-id")
        print(str(preflight(args.run_id)))
        return
    # Model fitting is gated on an explicit run, an externally reviewed matrix, and a preflight.
    if not args.run_id or not args.matrix or not args.preflight_dir:
        parser.error("--run requires --run-id, --matrix, and --preflight-dir")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("root_reviewed") is not True or matrix.get("frozen") is not True:
        parser.error("matrix must be frozen and marked root_reviewed=true before any model fit")
    if matrix.get("python_executable", "").casefold() != expected_exe or matrix.get("sklearn_version") != EXPECTED_SKLEARN:
        parser.error("matrix runtime lock does not match active interpreter and scikit-learn")
    if matrix.get("phase") != "A" or matrix.get("expected_model_folds") != 480:
        parser.error("phase-A matrix does not match the frozen 480-fold phase")
    print(str(run_main(args.run_id, args.preflight_dir, matrix)))


if __name__ == "__main__":
    main()
