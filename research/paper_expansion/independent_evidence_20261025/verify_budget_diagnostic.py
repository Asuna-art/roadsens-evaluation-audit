"""Independent, no-fit verification for the PVS common-budget diagnostic.

This script deliberately does not import the budget experiment driver. It uses
the immutable original run preflight as split authority, independently rebuilds
the balanced sampled training arrays, reloads each persisted estimator, and
recomputes every prediction-derived outcome. It writes
independent_verification.json only after every assertion passes.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


CANDIDATES = ("rf_sensor", "rf_sensor_context", "hgb_sensor", "hgb_sensor_context")
SCHEME_BY_POSITION = {
    0: "row_random", 1: "row_random",
    2: "leave_one_trip", 3: "leave_one_trip", 4: "leave_one_trip",
    5: "leave_one_trip", 6: "leave_one_trip", 7: "leave_one_trip",
    8: "matched_factor", 9: "matched_factor",
}
SCHEME_KEY = {"row_random": "random", "leave_one_trip": "trip", "matched_factor": "matched"}
TOL = 1e-12


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"required summary CSV is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_rows(actual: list[dict[str, Any]], expected: list[dict[str, Any]],
               keys: tuple[str, ...], numeric: tuple[str, ...], label: str) -> None:
    amap = {tuple(str(row[k]) for k in keys): row for row in actual}
    emap = {tuple(str(row[k]) for k in keys): row for row in expected}
    require(len(amap) == len(actual) and len(emap) == len(expected) and set(amap) == set(emap),
            f"{label}: row identity/inventory mismatch")
    for key in emap:
        for field in numeric:
            require(math.isclose(float(amap[key][field]), float(emap[key][field]), abs_tol=TOL, rel_tol=0.0),
                    f"{label}/{key}/{field}: saved summary differs from prediction-derived value")
        for field, value in emap[key].items():
            if field in numeric or isinstance(value, (dict, list, tuple)):
                continue
            actual_value = amap[key].get(field)
            if isinstance(value, bool):
                require(actual_value in (value, str(value)), f"{label}/{key}/{field}: saved summary metadata mismatch")
            elif isinstance(value, (int, float)):
                require(float(actual_value) == float(value), f"{label}/{key}/{field}: saved summary metadata mismatch")
            else:
                require(actual_value == value, f"{label}/{key}/{field}: saved summary metadata mismatch")


def equal_array(a: Any, b: Any, label: str) -> None:
    aa, bb = np.asarray(a), np.asarray(b)
    require(aa.shape == bb.shape and np.array_equal(aa, bb), f"array mismatch: {label}")


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    require(yt.shape == yp.shape, "y_true/y_pred shape mismatch")
    require(np.isin(yt, [0, 1]).all() and np.isin(yp, [0, 1]).all(), "non-binary labels")
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return {
        "macro_f1": float(f1_score(yt, yp, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix_labels_0_1": cm.astype(int).tolist(),
        "class_counts_true_0_1": np.bincount(yt, minlength=2).astype(int).tolist(),
        "class_counts_pred_0_1": np.bincount(yp, minlength=2).astype(int).tolist(),
    }


def matrices(data: dict[str, np.ndarray], manifest: dict[str, Any], candidate: str,
             rows: np.ndarray) -> np.ndarray:
    sensor_ix = np.asarray(manifest["sensor_feature_indices"], dtype=np.int64)
    ctx_ix = np.asarray(manifest["context_feature_indices"], dtype=np.int64)
    sensor = data["X_sensor"][rows][:, sensor_ix]
    if candidate.endswith("sensor"):
        return np.asarray(sensor, dtype=np.float32)
    context = data["X_context"][rows][:, ctx_ix]
    return np.asarray(np.column_stack((sensor, context)), dtype=np.float32)


def interval_overlap(a: np.ndarray, b: np.ndarray, data: dict[str, np.ndarray]) -> bool:
    """Return whether any inclusive source-row intervals intersect in a trip."""
    if not len(a) or not len(b):
        return False
    trip_a = data["trip"][a]
    trip_b = data["trip"][b]
    for trip in np.intersect1d(np.unique(trip_a), np.unique(trip_b)):
        ai = a[trip_a == trip]
        bi = b[trip_b == trip]
        # Two-pointer interval intersection; source intervals are inclusive.
        ai = ai[np.argsort(data["source_start"][ai], kind="stable")]
        bi = bi[np.argsort(data["source_start"][bi], kind="stable")]
        i = j = 0
        while i < len(ai) and j < len(bi):
            alo, ahi = int(data["source_start"][ai[i]]), int(data["source_end"][ai[i]])
            blo, bhi = int(data["source_start"][bi[j]]), int(data["source_end"][bi[j]])
            if alo <= bhi and blo <= ahi:
                return True
            if ahi < bhi:
                i += 1
            else:
                j += 1
    return False


def validate_model(model: Any, candidate_id: str, x_train: np.ndarray,
                   y_train: np.ndarray) -> dict[str, Any]:
    require(hasattr(model, "named_steps"), f"{candidate_id}: expected imputer/model pipeline")
    require(list(model.named_steps) == ["imputer", "model"], f"{candidate_id}: pipeline step mismatch")
    imp = model.named_steps["imputer"]
    est = model.named_steps["model"]
    require(getattr(imp, "strategy", None) == "median", f"{candidate_id}: imputer is not median")
    require(bool(getattr(imp, "keep_empty_features", False)), f"{candidate_id}: keep_empty_features mismatch")
    expected_medians = np.nanmedian(x_train, axis=0)
    # Historical feature checks require every declared feature to contain a
    # finite selected training value, so the medians should all be finite.
    require(np.isfinite(expected_medians).all(), f"{candidate_id}: selected rows have an all-missing feature")
    require(np.allclose(np.asarray(imp.statistics_, dtype=float), expected_medians,
                        rtol=0.0, atol=1e-7, equal_nan=False),
            f"{candidate_id}: fitted imputer statistics do not match selected training rows")
    params = est.get_params(deep=False)
    if candidate_id.startswith("rf_"):
        require(est.__class__.__name__ == "RandomForestClassifier", f"{candidate_id}: estimator type mismatch")
        expected = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                    "max_features": "sqrt", "class_weight": "balanced", "random_state": 42,
                    "n_jobs": 4}
    else:
        require(est.__class__.__name__ == "HistGradientBoostingClassifier", f"{candidate_id}: estimator type mismatch")
        expected = {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1,
                    "l2_regularization": 1.0, "class_weight": "balanced",
                    "random_state": 42, "early_stopping": False}
    for k, v in expected.items():
        require(params.get(k) == v, f"{candidate_id}: estimator setting {k}={params.get(k)!r}, expected {v!r}")
    return {"imputer_medians_match_selected_training_rows": True,
            "n_features_in": int(x_train.shape[1]), "estimator_class": est.__class__.__name__,
            "estimator_settings_checked": expected}


def flatten_fit_pools(old_outer: dict[str, Any]) -> list[dict[str, Any]]:
    old = []
    for pos in range(10):
        scheme = SCHEME_BY_POSITION[pos]
        scheme_list = old_outer["inner"][scheme]
        rank = pos if scheme == "row_random" else (pos - 2 if scheme == "leave_one_trip" else pos - 8)
        entry = scheme_list[rank]
        old.append({"position": pos, "scheme": scheme, "fold_id": entry["fold_id"],
                    "pool": np.asarray(entry["train_indices_fit"], dtype=np.int64),
                    "test": np.asarray(entry["test_indices"], dtype=np.int64)})
    entry = old_outer["outer_refit_cap"]
    old.append({"position": 10, "scheme": "outer_refit", "fold_id": entry["fold_id"],
                "pool": np.asarray(entry["train_indices_fit"], dtype=np.int64),
                "test": np.asarray(entry["test_indices"], dtype=np.int64)})
    return old


def expected_subset(pool: np.ndarray, y: np.ndarray, class_budget: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    picks = [rng.choice(np.sort(pool[y[pool] == c]), size=class_budget[c], replace=False) for c in (0, 1)]
    return np.sort(np.concatenate(picks).astype(np.int64))


def check_hash_inputs(preflight: dict[str, Any], base: Path, protocol: Path,
                      run_dir: Path) -> dict[str, str]:
    hashes = preflight.get("input_sha256")
    require(isinstance(hashes, dict) and len(hashes) >= 5,
            "preflight_manifest.input_sha256 must bind the five frozen inputs")
    # Accept either {logical_name: digest} with documented paths in a sibling
    # field, or {path: digest}; all values are still compared against bytes now.
    canonical = {}
    declared_paths = preflight.get("inputs", {})
    for key, value in hashes.items():
        if isinstance(value, dict):
            p = Path(value.get("path", key))
            digest = value.get("sha256")
        else:
            p = Path(key)
            digest = value
        name = p.name.lower()
        candidates: list[Path] = []
        if isinstance(declared_paths, dict) and key in declared_paths:
            candidates.append(Path(declared_paths[key]))
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([base / p, run_dir.parent / p, run_dir / p, protocol.parent / p])
        if name == "features.npz": candidates = [base / "features.npz"]
        elif name == "preflight.json": candidates = [base / "run_v1" / "preflight.json"]
        elif name == "checked_summary.json": candidates = [base / "checked_summary.json"]
        elif name == "budget_diagnostic_protocol_v1.md": candidates = [protocol]
        elif name.endswith(".py") and ("budget" in name or "diagnostic" in name): candidates.extend([
            run_dir.parent / p.name, run_dir / p.name, protocol.parent / p.name,
            base / p.name, base.parent / p.name])
        elif key.lower() in {"features", "feature_npz", "input_npz"}: candidates = [base / "features.npz"]
        elif key.lower() in {"historical_preflight", "original_preflight"}: candidates = [base / "run_v1" / "preflight.json"]
        elif key.lower() in {"historical_summary", "original_summary"}: candidates = [base / "checked_summary.json"]
        elif key.lower() in {"protocol", "diagnostic_protocol"}: candidates = [protocol]
        elif key.lower() in {"code", "driver", "runner", "helper"}:
            candidates.extend([p] if p.is_absolute() else [run_dir.parent / p, protocol.parent / p, base / p])
            if key.lower() in {"code", "driver", "runner"}:
                candidates.append(protocol.parent / "run_budget_diagnostic.py")
        matched = next((q for q in candidates if q.is_file()), None)
        require(matched is not None, f"cannot resolve hash-bound input {key!r}")
        actual = sha256(matched)
        require(str(digest) == actual, f"hash-bound input changed: {matched}")
        canonical[key] = actual
    require({"features", "original_preflight", "original_summary", "protocol", "code", "helper"}.issubset(
        {k.lower() for k in canonical}), "preflight hashes must bind features, original preflight/summary, protocol, code, and helper")
    return canonical


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    base = args.base_dir.resolve()
    protocol = args.protocol.resolve()
    original_preflight_path = base / "run_v1" / "preflight.json"
    original_summary_path = base / "checked_summary.json"
    old_pre = read_json(original_preflight_path)
    old_summary = read_json(original_summary_path)
    pre = read_json(run_dir / "preflight_manifest.json")
    input_hashes = check_hash_inputs(pre, base, protocol, run_dir)
    require(pre.get("planned_new_fits") == 264, "preflight planned fit count must be 264")
    require(len(pre.get("domains", [])) == 6, "expected six budget domains")
    require(len(old_pre.get("folds", [])) == 6, "original split authority must have six folds")

    manifest = read_json(base / "feature_manifest.json")
    with np.load(base / "features.npz", allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    require(len(data["y"]) == 10804, "feature row count changed")
    require(np.array_equal(np.unique(data["y"]), [0, 1]), "binary feature labels expected")

    # Input digests are independently recalculated, with the original frozen
    # split JSON remaining the sole authority for every membership below.
    folds_by_id = {x["outer_id"]: x for x in old_pre["folds"]}
    domains = {x["outer_id"]: x for x in pre["domains"]}
    require(set(domains) == set(folds_by_id), "budget preflight domains differ from frozen split authority")
    with np.load(run_dir / "selected_indices.npz", allow_pickle=False) as z:
        selected_arrays = {k: np.asarray(z[k], dtype=np.int64) for k in z.files}
    index_hash = sha256(run_dir / "selected_indices.npz")
    require(index_hash == pre.get("selected_indices_sha256"), "selected-index archive hash mismatch")

    fold_authority: dict[tuple[str, int], dict[str, Any]] = {}
    budget_info = {}
    for outer_position, old_outer in enumerate(old_pre["folds"]):
        outer_id = old_outer["outer_id"]
        domain = domains[outer_id]
        require(int(domain["outer_position"]) == outer_position, f"{outer_id}: outer order changed")
        require(domain["axis"] == old_outer["axis"] and int(domain["held_out_id"]) == int(old_outer["held_out_id"]), f"{outer_id}: identity mismatch")
        source_pools = flatten_fit_pools(old_outer)
        class_budget = tuple(min(int(np.sum(data["y"][x["pool"]] == c)) for x in source_pools) for c in (0, 1))
        require(min(class_budget) >= 2, f"{outer_id}: budget below two examples/class")
        declared_budget = domain["budget_per_class"]
        if isinstance(declared_budget, dict):
            declared_budget = (int(declared_budget.get("0", declared_budget.get(0))),
                               int(declared_budget.get("1", declared_budget.get(1))))
        else:
            declared_budget = tuple(int(v) for v in declared_budget)
        require(declared_budget == class_budget, f"{outer_id}: separate class budgets do not equal historical pool minima")
        domain_class_counts = []
        newfolds = domain.get("folds", [])
        require(len(newfolds) == 11, f"{outer_id}: expected 11 preflight subsets")
        method_local_positions = {"row_random": 0, "leave_one_trip": 0, "matched_factor": 0}
        for source in source_pools:
            pos = source["position"]
            row = newfolds[pos]
            require(row is not None, f"{outer_id}: missing budget preflight position {pos}")
            scheme = SCHEME_BY_POSITION[pos] if pos < 10 else "outer_refit"
            expected_kind = scheme if pos < 10 else "outer"
            require(row["kind"] == expected_kind, f"{outer_id}/{pos}: fold kind mismatch")
            require(row["fold_id"] == source["fold_id"], f"{outer_id}/{pos}: fold identity mismatch")
            require(row.get("fit_pool_sha256") == hashlib.sha256(np.sort(np.asarray(source["pool"], dtype=np.int64)).tobytes()).hexdigest(),
                    f"{outer_id}/{pos}: historical fit-pool hash mismatch")
            require(row.get("test_indices_sha256") == hashlib.sha256(np.sort(np.asarray(source["test"], dtype=np.int64)).tobytes()).hexdigest(),
                    f"{outer_id}/{pos}: original test-membership hash mismatch")
            if pos == 10:
                require(row.get("inner_position") is None, f"{outer_id}/outer: expected no inner position")
            else:
                require(row.get("inner_position") == pos, f"{outer_id}/{pos}: preflight global inner position mismatch")
                method_local_positions[scheme] += 1
            expected_seed = 20261025 + 100 * outer_position + pos
            require(int(row["seed"]) == expected_seed, f"{outer_id}/{pos}: seed mismatch")
            selected = expected_subset(source["pool"], data["y"], class_budget, expected_seed)
            equal_array(row["selected_indices"], selected, f"{outer_id}/{pos} balanced selected indices")
            key = row.get("selected_array")
            require(isinstance(key, str) and key, f"{outer_id}/{pos}: selected array key missing")
            require(key in selected_arrays, f"selected_indices.npz missing {key}")
            equal_array(selected_arrays[key], selected, f"selected archive {key}")
            counts = np.bincount(data["y"][selected].astype(int), minlength=2).astype(int).tolist()
            require(counts == list(class_budget), f"{outer_id}/{pos}: separate class budget mismatch")
            require(row.get("class_counts") == {"0": class_budget[0], "1": class_budget[1]} or row.get("class_counts") == list(class_budget), f"{outer_id}/{pos}: declared class counts disagree")
            require(not interval_overlap(selected, source["test"], data), f"{outer_id}/{pos}: raw source rows overlap")
            require(not interval_overlap(source["pool"], source["test"], data), f"{outer_id}/{pos}: historical pool/test raw overlap")
            x = np.column_stack((data["X_sensor"][selected][:, manifest["sensor_feature_indices"]],
                                 data["X_context"][selected][:, manifest["context_feature_indices"]]))
            require(not np.isinf(x).any() and np.isfinite(x).any(axis=0).all(), f"{outer_id}/{pos}: invalid training feature values")
            fold_authority[(outer_id, pos)] = {**source, "selected": selected, "budget": class_budget,
                                                "seed": expected_seed, "scheme": scheme,
                                                "fold_id": source["fold_id"]}
            domain_class_counts.append(counts)
        require(all(x == list(class_budget) for x in domain_class_counts), f"{outer_id}: class counts differ among 11 fits")
        budget_info[outer_id] = {"axis": old_outer["axis"], "held_out_id": int(old_outer["held_out_id"]),
                                 "class_budget": {"0": class_budget[0], "1": class_budget[1]},
                                 "total_per_fit": sum(class_budget),
                                 "all_11_fit_counts": domain_class_counts}

    fit_payload = read_json(run_dir / "fit_records.json")
    records = fit_payload.get("records", [])
    require(len(records) == 264 and fit_payload.get("record_count") == 264 and fit_payload.get("expected_new_fit_count") == 264,
            "fit record inventory is incomplete or reports failures")
    require(len({r["fit_id"] for r in records}) == 264, "duplicate fit_id")
    record_map = {r["fit_id"]: r for r in records}
    domain_oof: dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]] = {
        outer_id: {scheme: {candidate: [] for candidate in CANDIDATES}
                   for scheme in SCHEME_KEY} for outer_id in folds_by_id}
    outer_metric_map: dict[str, dict[str, dict[str, Any]]] = {k: {} for k in folds_by_id}
    fit_level_metrics: dict[str, dict[str, Any]] = {}
    model_checks = 0
    expected_fit_ids = set()
    for outer_id, old_outer in folds_by_id.items():
        for pos in range(11):
            authority = fold_authority[(outer_id, pos)]
            for candidate in CANDIDATES:
                if pos == 10:
                    fit_id = f"{outer_id}__budget_outer__{candidate}"
                    phase = "budget_outer"
                    expected_scheme = "shared_outer_refit"
                    expected_local_position = None
                else:
                    local_position = sum(1 for p in range(pos) if SCHEME_BY_POSITION[p] == authority["scheme"])
                    fit_id = f"{outer_id}__{SCHEME_KEY[authority['scheme']]}_{local_position + 1}__{candidate}"
                    phase = "inner"
                    expected_scheme = authority["scheme"]
                    expected_local_position = local_position
                expected_fit_ids.add(fit_id)
                rec = record_map.get(fit_id)
                require(rec is not None, f"missing fit record {fit_id}")
                require(rec["outer_id"] == outer_id and rec["candidate_id"] == candidate and rec["phase"] == phase and rec["scheme"] == expected_scheme,
                        f"{fit_id}: metadata mismatch")
                require(rec.get("inner_position") == expected_local_position, f"{fit_id}: fit position mismatch")
                equal_array(rec["train_indices_pool"], authority["pool"], f"{fit_id}: historical training pool")
                equal_array(rec["train_indices_fit"], authority["selected"], f"{fit_id}: selected training rows")
                equal_array(rec["test_indices"], authority["test"], f"{fit_id}: frozen validation/test memberships")
                require(not interval_overlap(np.asarray(rec["train_indices_fit"], dtype=np.int64), authority["test"], data), f"{fit_id}: raw-row overlap")
                fit_dir = run_dir / "fits" / fit_id
                model_path, pred_path = fit_dir / "model.joblib", fit_dir / "predictions.npz"
                require(model_path.is_file() and pred_path.is_file(), f"{fit_id}: model/prediction artifact missing")
                require(sha256(model_path) == rec["model_sha256"], f"{fit_id}: model file hash mismatch")
                require(sha256(pred_path) == rec["prediction_sha256"], f"{fit_id}: prediction file hash mismatch")
                with np.load(pred_path, allow_pickle=False) as z:
                    pred = {k: np.asarray(z[k]) for k in z.files}
                equal_array(pred["test_indices"], authority["test"], f"{fit_id}: prediction test rows")
                equal_array(pred["train_indices_fit"], authority["selected"], f"{fit_id}: prediction train rows")
                equal_array(pred["train_indices_pool"], authority["pool"], f"{fit_id}: prediction source pool")
                equal_array(pred["y_true"], data["y"][authority["test"]], f"{fit_id}: true labels")
                with model_path.open("rb") as f:
                    model = pickle.load(f)
                xtr = matrices(data, manifest, candidate, authority["selected"])
                xte = matrices(data, manifest, candidate, authority["test"])
                check = validate_model(model, candidate, xtr, data["y"][authority["selected"]])
                reproduced = np.asarray(model.predict(xte), dtype=np.int64)
                equal_array(reproduced, pred["y_pred"], f"{fit_id}: reloaded model predictions")
                m = macro_metrics(pred["y_true"], reproduced)
                fit_level_metrics[fit_id] = m
                if pos == 10:
                    outer_metric_map[outer_id][candidate] = m
                else:
                    domain_oof[outer_id][authority["scheme"]][candidate].append(
                        (authority["test"], np.asarray(pred["y_true"], dtype=np.int64), reproduced))
                model_checks += 1
    require(expected_fit_ids == set(record_map), "fit record IDs do not equal expected 264 matrix")

    # Rebuild OOF scores and selector decisions from exact validation arrays.
    frozen = read_json(run_dir / "frozen_selections.json")
    frozen_list = frozen.get("selections", [])
    require(len(frozen_list) == 18, "expected 18 frozen selector decisions")
    # The frozen selector payload must bind the same historical inputs and
    # exact sampled-index archive that were checked above.
    frozen_inputs = frozen.get("input_sha256", frozen.get("input_hashes"))
    require(frozen_inputs == pre.get("input_sha256"), "frozen selector manifest input hashes differ from preflight")
    frozen_index_sha = frozen.get("selected_indices_sha256", frozen.get("selected_index_sha256"))
    require(frozen_index_sha == index_hash, "frozen selector manifest does not bind selected-index archive")
    frozen_map = {(x["outer_id"], x["scheme"]): x for x in frozen_list}
    require(len(frozen_map) == 18, "duplicate frozen selector decision")
    candidate_scores: dict[tuple[str, str], dict[str, float]] = {}
    selectors: dict[str, dict[str, dict[str, Any]]] = {}
    for outer_id, old_outer in folds_by_id.items():
        selectors[outer_id] = {}
        outer_train = np.asarray(old_outer["train_indices"], dtype=np.int64)
        for scheme in SCHEME_KEY:
            scores = {}
            for candidate in CANDIDATES:
                parts = domain_oof[outer_id][scheme][candidate]
                ix = np.concatenate([p[0] for p in parts])
                yt = np.concatenate([p[1] for p in parts])
                yp = np.concatenate([p[2] for p in parts])
                order = np.argsort(ix, kind="stable")
                equal_array(ix[order], np.sort(outer_train), f"{outer_id}/{scheme}/{candidate}: OOF coverage")
                equal_array(yt[order], data["y"][outer_train][np.argsort(outer_train, kind="stable")], f"{outer_id}/{scheme}/{candidate}: OOF truth")
                scores[candidate] = macro_metrics(yt[order], yp[order])["macro_f1"]
            selected = max(CANDIDATES, key=lambda c: scores[c])
            row = frozen_map[(outer_id, scheme)]
            require(row.get("candidate_order") == list(CANDIDATES) or frozen.get("candidate_order") == list(CANDIDATES), f"{outer_id}/{scheme}: candidate tie order mismatch")
            recorded_scores = row.get("candidate_scores")
            require(isinstance(recorded_scores, dict), f"{outer_id}/{scheme}: missing candidate scores")
            for candidate in CANDIDATES:
                require(math.isclose(float(recorded_scores[candidate]), scores[candidate], abs_tol=TOL, rel_tol=0.0), f"{outer_id}/{scheme}/{candidate}: OOF score mismatch")
            require(row.get("selected_candidate") == selected, f"{outer_id}/{scheme}: selection/tie-break mismatch")
            require(math.isclose(float(row.get("inner_macro_f1", float("nan"))), scores[selected], abs_tol=TOL, rel_tol=0.0), f"{outer_id}/{scheme}: selected inner score mismatch")
            candidate_scores[(outer_id, scheme)] = scores
            selectors[outer_id][scheme] = {"candidate_scores": scores, "selected_candidate": selected,
                                           "inner_macro_f1": scores[selected],
                                           "budget_outer_metrics": outer_metric_map[outer_id]}

    # Check the independent frozen-selection digest and ordered event evidence.
    declared_frozen_sha = frozen.get("sha256")
    frozen_payload = {k: v for k, v in frozen.items() if k != "sha256"}
    require(declared_frozen_sha == canonical_sha(frozen_payload), "frozen_selections.json canonical payload hash mismatch")
    events = fit_payload.get("events", [])
    require(events, "ordered fit event ledger missing")
    require([int(e.get("sequence", -1)) for e in events] == list(range(len(events))),
            "event ledger sequence is incomplete or reordered")
    freeze_events = [i for i, e in enumerate(events) if e.get("event") == "selections_frozen"]
    outer_events = [i for i, e in enumerate(events) if e.get("event") == "budget_outer_fit_complete"]
    require(len(freeze_events) == 1 and len(outer_events) == 24 and max(freeze_events) < min(outer_events),
            "ordered evidence does not place all selector freezes before outer model work")
    for e in events:
        if "frozen_selections_sha256" in e:
            require(e.get("frozen_selections_sha256") == declared_frozen_sha,
                    "event ledger frozen-selection hash mismatch")

    # The existing original full-development outer predictions are reused and
    # evaluated independently; no new model is fitted here.
    historical = {}
    historical_hashes = {}
    for outer_id, old_outer in folds_by_id.items():
        historical[outer_id] = {}
        historical_hashes[outer_id] = {}
        test_expected = np.asarray(old_outer["test_indices"], dtype=np.int64)
        for candidate in CANDIDATES:
            p = base / "run_v1" / "fits" / f"{outer_id}__outer_refit__{candidate}" / "predictions.npz"
            require(p.is_file(), f"historical full-budget prediction missing: {p}")
            with np.load(p, allow_pickle=False) as z:
                h = {k: np.asarray(z[k]) for k in z.files}
            equal_array(h["test_indices"], test_expected, f"{outer_id}/{candidate}: historical test membership")
            equal_array(h["y_true"], data["y"][test_expected], f"{outer_id}/{candidate}: historical labels")
            historical[outer_id][candidate] = macro_metrics(h["y_true"], h["y_pred"])
            historical_hashes[outer_id][candidate] = sha256(p)

    outcomes: dict[str, Any] = {"domains": {}, "axis_means": {"vehicle": {}, "scenario": {}}}
    for outer_id, old_outer in folds_by_id.items():
        outcomes["domains"][outer_id] = {"axis": old_outer["axis"], "selectors": {}}
        for scheme in SCHEME_KEY:
            chosen = selectors[outer_id][scheme]["selected_candidate"]
            inner = selectors[outer_id][scheme]["inner_macro_f1"]
            budget_outer = outer_metric_map[outer_id][chosen]["macro_f1"]
            historical_outer = historical[outer_id][chosen]["macro_f1"]
            outcomes["domains"][outer_id]["selectors"][scheme] = {
                "selected_candidate": chosen, "inner_macro_f1": inner,
                "budget_outer_macro_f1": budget_outer,
                "budget_signed_error_outer_minus_inner": budget_outer - inner,
                "budget_absolute_error": abs(budget_outer - inner),
                "historical_full_budget_macro_f1_reused": historical_outer,
                "historical_signed_error_outer_minus_inner": historical_outer - inner,
                "historical_absolute_error": abs(historical_outer - inner),
                "candidate_level_budget_errors": {
                    c: {"inner_macro_f1": candidate_scores[(outer_id, scheme)][c],
                        "budget_outer_macro_f1": outer_metric_map[outer_id][c]["macro_f1"],
                        "signed_error_outer_minus_inner": outer_metric_map[outer_id][c]["macro_f1"] - candidate_scores[(outer_id, scheme)][c],
                        "absolute_error": abs(outer_metric_map[outer_id][c]["macro_f1"] - candidate_scores[(outer_id, scheme)][c])}
                    for c in CANDIDATES},
                "candidate_level_historical_errors": {
                    c: {"inner_macro_f1": candidate_scores[(outer_id, scheme)][c],
                        "historical_outer_macro_f1": historical[outer_id][c]["macro_f1"],
                        "signed_error_outer_minus_inner": historical[outer_id][c]["macro_f1"] - candidate_scores[(outer_id, scheme)][c],
                        "absolute_error": abs(historical[outer_id][c]["macro_f1"] - candidate_scores[(outer_id, scheme)][c]),
                        "historical_prediction_sha256": historical_hashes[outer_id][c]}
                    for c in CANDIDATES},
            }
    for axis in ("vehicle", "scenario"):
        axis_domains = [k for k, v in outcomes["domains"].items() if v["axis"] == axis]
        require(len(axis_domains) == 3, f"axis {axis}: expected three domains")
        for scheme in SCHEME_KEY:
            outcomes["axis_means"][axis][scheme] = {
                "budget_MAE": float(np.mean([outcomes["domains"][k]["selectors"][scheme]["budget_absolute_error"] for k in axis_domains])),
                "historical_full_budget_MAE": float(np.mean([outcomes["domains"][k]["selectors"][scheme]["historical_absolute_error"] for k in axis_domains])),
                "mean_inner_macro_f1": float(np.mean([outcomes["domains"][k]["selectors"][scheme]["inner_macro_f1"] for k in axis_domains])),
                "budget_outer_macro_f1_mean": float(np.mean([outcomes["domains"][k]["selectors"][scheme]["budget_outer_macro_f1"] for k in axis_domains])),
                "mean_signed_error": float(np.mean([outcomes["domains"][k]["selectors"][scheme]["budget_signed_error_outer_minus_inner"] for k in axis_domains])),
                "historical_outer_macro_f1_mean": float(np.mean([outcomes["domains"][k]["selectors"][scheme]["historical_full_budget_macro_f1_reused"] for k in axis_domains])),
            }
        outcomes["axis_means"][axis]["random_minus_matched_budget_MAE"] = (
            outcomes["axis_means"][axis]["row_random"]["budget_MAE"] -
            outcomes["axis_means"][axis]["matched_factor"]["budget_MAE"])
        outcomes["axis_means"][axis]["random_minus_matched_historical_MAE"] = (
            outcomes["axis_means"][axis]["row_random"]["historical_full_budget_MAE"] -
            outcomes["axis_means"][axis]["matched_factor"]["historical_full_budget_MAE"])

    # Reconcile saved CSV/JSON summaries against independently recomputed
    # predictions. This checks means and random-minus-matched contrasts too.
    summary_path = run_dir / "summary.json"
    saved_summary = read_json(summary_path)
    require(saved_summary.get("status") == "completed", "runner summary is not marked completed")
    require(saved_summary.get("planned_new_fits") == 264, "runner summary fit inventory mismatch")
    require(saved_summary.get("input_sha256") == pre.get("input_sha256"), "runner summary historical hashes differ")
    expected_budget_domains = []
    expected_budget_axes = []
    expected_historical_candidates = []
    expected_historical_axes = []
    for outer_id, old_outer in folds_by_id.items():
        for scheme in SCHEME_KEY:
            chosen = selectors[outer_id][scheme]["selected_candidate"]
            inner = candidate_scores[(outer_id, scheme)][chosen]
            bout = outer_metric_map[outer_id][chosen]["macro_f1"]
            expected_budget_domains.append({"outer_id": outer_id, "axis": old_outer["axis"],
                "held_out_id": old_outer["held_out_id"], "scheme": scheme,
                "selected_candidate": chosen, "inner_macro_f1": inner,
                "budget_outer_macro_f1": bout, "signed_error_outer_minus_inner": bout-inner,
                "absolute_error": abs(bout-inner), "budget_per_class": domains[outer_id]["budget_per_class"]})
            for candidate in CANDIDATES:
                hist = historical[outer_id][candidate]["macro_f1"]
                expected_historical_candidates.append({"outer_id": outer_id, "axis": old_outer["axis"],
                    "held_out_id": old_outer["held_out_id"], "scheme": scheme,
                    "candidate_id": candidate, "is_selected": candidate == chosen,
                    "selected_candidate": chosen, "inner_macro_f1": candidate_scores[(outer_id, scheme)][candidate],
                    "historical_full_outer_macro_f1": hist,
                    "signed_error_outer_minus_inner": hist-candidate_scores[(outer_id, scheme)][candidate],
                    "absolute_error": abs(hist-candidate_scores[(outer_id, scheme)][candidate]),
                    "historical_prediction_sha256": historical_hashes[outer_id][candidate]})
    for axis in ("vehicle", "scenario"):
        for scheme in SCHEME_KEY:
            ds = [r for r in expected_budget_domains if r["axis"] == axis and r["scheme"] == scheme]
            expected_budget_axes.append({"axis": axis, "scheme": scheme, "domains": len(ds),
                "mean_inner_macro_f1": float(np.mean([r["inner_macro_f1"] for r in ds])),
                "mean_outer_macro_f1": float(np.mean([r["budget_outer_macro_f1"] for r in ds])),
                "mean_mae": float(np.mean([r["absolute_error"] for r in ds])),
                "mean_signed_error": float(np.mean([r["signed_error_outer_minus_inner"] for r in ds]))})
            sh = [r for r in expected_historical_candidates if r["axis"] == axis and r["scheme"] == scheme and r["is_selected"]]
            expected_historical_axes.append({"axis": axis, "scheme": scheme, "domains": len(sh),
                "mean_inner_macro_f1": float(np.mean([r["inner_macro_f1"] for r in sh])),
                "mean_outer_macro_f1": float(np.mean([r["historical_full_outer_macro_f1"] for r in sh])),
                "mean_mae": float(np.mean([r["absolute_error"] for r in sh])),
                "mean_signed_error": float(np.mean([r["signed_error_outer_minus_inner"] for r in sh]))})
    expected_contrasts = []
    for target, table in (("budget_matched", expected_budget_axes),
                          ("historical_full_budget", expected_historical_axes)):
        for axis in ("vehicle", "scenario"):
            rows = {r["scheme"]: r["mean_mae"] for r in table if r["axis"] == axis}
            expected_contrasts.append({"axis": axis, "target": target,
                "random_mae_minus_matched_mae": rows["row_random"]-rows["matched_factor"]})
    check_rows(saved_summary.get("budget_matched_domain_rows", []), expected_budget_domains,
        ("outer_id", "scheme"), ("held_out_id", "inner_macro_f1", "budget_outer_macro_f1", "signed_error_outer_minus_inner", "absolute_error"), "budget domain summary")
    check_rows(saved_summary.get("budget_matched_axis_rows", []), expected_budget_axes,
        ("axis", "scheme"), ("domains", "mean_inner_macro_f1", "mean_outer_macro_f1", "mean_mae", "mean_signed_error"), "budget axis summary")
    check_rows(saved_summary.get("historical_full_budget_candidate_rows", []), expected_historical_candidates,
        ("outer_id", "scheme", "candidate_id"), ("held_out_id", "inner_macro_f1", "historical_full_outer_macro_f1", "signed_error_outer_minus_inner", "absolute_error"), "historical candidate summary")
    check_rows(saved_summary.get("historical_full_budget_selected_domain_rows", []),
        [r for r in expected_historical_candidates if r["is_selected"]],
        ("outer_id", "scheme", "candidate_id"), ("held_out_id", "inner_macro_f1", "historical_full_outer_macro_f1", "signed_error_outer_minus_inner", "absolute_error"), "historical selected summary")
    check_rows(saved_summary.get("historical_full_budget_axis_rows", []), expected_historical_axes,
        ("axis", "scheme"), ("domains", "mean_inner_macro_f1", "mean_outer_macro_f1", "mean_mae", "mean_signed_error"), "historical axis summary")
    check_rows(saved_summary.get("mae_contrasts", []), expected_contrasts,
        ("axis", "target"), ("random_mae_minus_matched_mae",), "MAE contrasts")
    check_rows(read_csv_rows(run_dir / "budget_matched_domain_summary.csv"), expected_budget_domains,
        ("outer_id", "scheme"), ("held_out_id", "inner_macro_f1", "budget_outer_macro_f1", "signed_error_outer_minus_inner", "absolute_error"), "budget domain CSV")
    check_rows(read_csv_rows(run_dir / "budget_matched_axis_summary.csv"), expected_budget_axes,
        ("axis", "scheme"), ("domains", "mean_inner_macro_f1", "mean_outer_macro_f1", "mean_mae", "mean_signed_error"), "budget axis CSV")
    check_rows(read_csv_rows(run_dir / "historical_full_budget_candidate_rows.csv"), expected_historical_candidates,
        ("outer_id", "scheme", "candidate_id"), ("held_out_id", "inner_macro_f1", "historical_full_outer_macro_f1", "signed_error_outer_minus_inner", "absolute_error"), "historical candidate CSV")
    check_rows(read_csv_rows(run_dir / "historical_full_budget_axis_summary.csv"), expected_historical_axes,
        ("axis", "scheme"), ("domains", "mean_inner_macro_f1", "mean_outer_macro_f1", "mean_mae", "mean_signed_error"), "historical axis CSV")

    result = {
        "status": "passed",
        "verifier": "independent; no experiment driver import and no fitting",
        "fit_count_expected_reloaded_reproduced": model_checks,
        "summary_sha256": sha256(summary_path),
        "verifier_code_sha256": sha256(Path(__file__).resolve()),
        "domains": budget_info,
        "fit_level_fixed_label_metrics": fit_level_metrics,
        "checks": {
            "historical_hashes_match_pre_fit_manifest_and_current_inputs": True,
            "exact_training_subsets_recomputed_from_historical_fit_pools": True,
            "exact_original_validation_test_memberships": True,
            "common_class_counts_across_all_11_fits_per_domain": True,
            "raw_source_row_nonoverlap": True,
            "all_model_reloads_reproduced_predictions": True,
            "model_settings_and_train_only_medians": True,
            "oof_coverage_scores_and_candidate_tie_order": True,
            "selection_freeze_precedes_outer_model_events": True,
            "historical_full_budget_predictions_reused": True,
        },
        "hashes": {"input_sha256": input_hashes, "selected_indices_sha256": index_hash,
                   "frozen_selections_sha256": declared_frozen_sha,
                   "summary_sha256": sha256(summary_path)},
        "outcomes": outcomes,
        "completion_inventory": {"outer_domains": 6, "candidates": 4, "inner_fits": 240,
                                 "outer_budget_refits": 24, "total_model_files_reloaded": model_checks,
                                 "historical_full_budget_prediction_files_reused": 24,
                                 "partial_or_failed_fits": 0},
        "notes": ["Post hoc common-budget diagnostic; not independent confirmation.",
                  "Historical full-budget comparisons reuse archived predictions and retain the training-size gap.",
                  "No population-significance or causal fraction claim is implied."],
    }
    return result


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=here / "budget_nonoverlap_v1")
    parser.add_argument("--base-dir", type=Path, default=here.parent / "external_pvs_v1" / "nonoverlap")
    parser.add_argument("--protocol", type=Path, default=here / "budget_diagnostic_protocol_v1.md")
    args = parser.parse_args()
    report = run(args)
    out = args.run_dir / "independent_verification.json"
    require(not out.exists(), f"refusing to overwrite verifier output: {out}")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(out), "models_reloaded": report["fit_count_expected_reloaded_reproduced"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
