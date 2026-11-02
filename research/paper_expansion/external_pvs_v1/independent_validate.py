#!/usr/bin/env python
"""Independent, fit-free verification of a completed external PVS run.

This deliberately does not import run_nested_evaluation.py. It rebuilds the
fold plan and caps, reloads all models and stored predictions, verifies hashes,
preprocessing and metrics, then writes the canonical checked summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
CAP_LIMIT = 20_000
CAP_SEED_BASE = 42_000
SCHEMES = ("row_random", "leave_one_trip", "matched_factor")
AXES = ("vehicle", "scenario")
CANDIDATES = (
    ("rf_sensor", "rf", "sensor"),
    ("rf_sensor_context", "rf", "sensor_context"),
    ("hgb_sensor", "hgb", "sensor"),
    ("hgb_sensor_context", "hgb", "sensor_context"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    labels = np.unique(y)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "class_labels": labels.astype(int).tolist(),
        "class_support": np.bincount(np.searchsorted(labels, y), minlength=len(labels)).astype(int).tolist(),
        "class_recalls": recall_score(y, pred, labels=labels, average=None, zero_division=0).astype(float).tolist(),
        "confusion_matrix_rows_true_columns_pred": confusion_matrix(y, pred, labels=labels).astype(int).tolist(),
    }


def close_metrics(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-12) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, list):
            aa, bb = np.asarray(av), np.asarray(bv)
            if aa.shape != bb.shape:
                return False
            if np.issubdtype(aa.dtype, np.integer) and np.issubdtype(bb.dtype, np.integer):
                if not np.array_equal(aa, bb):
                    return False
            elif not np.allclose(aa.astype(float), bb.astype(float), atol=tol, rtol=0):
                return False
        elif not np.isclose(float(av), float(bv), atol=tol, rtol=0):
            return False
    return True


def capped_indices(rows: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    if len(rows) <= CAP_LIMIT:
        return np.sort(rows)
    labels, counts = np.unique(y[rows], return_counts=True)
    if len(labels) != 2:
        raise ValueError("cannot reconstruct cap without two classes")
    ideals = counts * (CAP_LIMIT / counts.sum())
    q = np.floor(ideals).astype(int)
    remainders = ideals - q
    for i in np.argsort(-remainders, kind="stable")[:CAP_LIMIT - int(q.sum())]:
        q[i] += 1
    q = np.minimum(q, counts)
    while q.sum() < CAP_LIMIT:
        for i in range(len(q)):
            if q.sum() >= CAP_LIMIT:
                break
            if q[i] < counts[i]:
                q[i] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(rows[y[rows] == lab], size=int(n), replace=False)
                                  for lab, n in zip(labels, q)]).astype(np.int64))


def expected_outer(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    folds = []
    for axis, col in (("vehicle", "vehicle"), ("scenario", "scenario")):
        for held in np.sort(np.unique(data[col])):
            folds.append({"outer_id": f"{axis}_{int(held)}", "axis": axis, "held_out_id": int(held),
                          "train": np.flatnonzero(data[col] != held),
                          "test": np.flatnonzero(data[col] == held)})
    return folds


def expected_inner(train: np.ndarray, data: dict[str, np.ndarray], axis: str, scheme: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    if scheme == "row_random":
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(train)), data["y"][train])):
            out.append((f"random_{k + 1}", train[tr], train[te]))
    elif scheme == "leave_one_trip":
        for trip in np.sort(np.unique(data["trip"][train])):
            out.append((f"trip_{int(trip)}", train[data["trip"][train] != trip], train[data["trip"][train] == trip]))
    else:
        factor = "vehicle" if axis == "vehicle" else "scenario"
        for value in np.sort(np.unique(data[factor][train])):
            out.append((f"{factor}_{int(value)}", train[data[factor][train] != value], train[data[factor][train] == value]))
    return out


def overlap_samples(train: np.ndarray, test: np.ndarray, d: dict[str, np.ndarray]) -> dict[str, int]:
    tr_hit, te_hit = set(), set()
    unique_rows = 0
    shared_trips = np.intersect1d(np.unique(d["trip"][train]), np.unique(d["trip"][test]))
    for trip in shared_trips:
        tr = train[d["trip"][train] == trip]
        te = test[d["trip"][test] == trip]
        a = sorted((int(d["source_start"][i]), int(d["source_end"][i])) for i in tr)
        b = sorted((int(d["source_start"][i]), int(d["source_end"][i])) for i in te)
        # Merge each side's intervals and count unique rows in their intersection.
        def merge(items: list[tuple[int, int]]) -> list[tuple[int, int]]:
            vals = items
            merged: list[list[int]] = []
            for lo, hi in vals:
                if not merged or lo > merged[-1][1] + 1:
                    merged.append([lo, hi])
                else:
                    merged[-1][1] = max(merged[-1][1], hi)
            return [(x, y) for x, y in merged]
        ma, mb = merge(a), merge(b)
        b_starts = np.asarray([x for x, _ in mb], dtype=np.int64)
        b_ends = np.asarray([y for _, y in mb], dtype=np.int64)
        for idx in tr:
            lo, hi = int(d["source_start"][idx]), int(d["source_end"][idx])
            upto = int(np.searchsorted(b_starts, hi, side="right"))
            if upto and int(np.max(b_ends[:upto])) >= lo:
                tr_hit.add(int(idx))
        a_starts = np.asarray([x for x, _ in ma], dtype=np.int64)
        a_ends = np.asarray([y for _, y in ma], dtype=np.int64)
        for idx in te:
            lo, hi = int(d["source_start"][idx]), int(d["source_end"][idx])
            upto = int(np.searchsorted(a_starts, hi, side="right"))
            if upto and int(np.max(a_ends[:upto])) >= lo:
                te_hit.add(int(idx))
        i = j = 0
        while i < len(ma) and j < len(mb):
            unique_rows += max(0, min(ma[i][1], mb[j][1]) - max(ma[i][0], mb[j][0]) + 1)
            if ma[i][1] < mb[j][1]:
                i += 1
            else:
                j += 1
    return {"train_samples_with_overlap": len(tr_hit), "test_samples_with_overlap": len(te_hit),
            "unique_overlapping_raw_rows": int(unique_rows)}


def matrix_for(candidate: tuple[str, str, str], idx: np.ndarray, d: dict[str, np.ndarray],
               manifest: dict[str, Any]) -> np.ndarray:
    fi = manifest.get("feature_indices", {})
    sx = np.asarray(manifest.get("sensor_feature_indices", fi.get("sensor_feature_indices", list(range(30))),), dtype=int)
    cx = np.asarray(manifest.get("context_feature_indices", fi.get("context_feature_indices", [0, 1, 2])), dtype=int)
    sxv = d["X_sensor"][idx][:, sx]
    if candidate[2] == "sensor":
        return np.asarray(sxv, dtype=np.float32)
    return np.asarray(np.column_stack((sxv, d["X_context"][idx][:, cx])), dtype=np.float32)


def expect_model_params(model: Any, candidate: tuple[str, str, str]) -> None:
    if not hasattr(model, "named_steps") or set(model.named_steps) != {"imputer", "model"}:
        raise AssertionError(f"{candidate[0]}: unexpected model pipeline")
    imp, est = model.named_steps["imputer"], model.named_steps["model"]
    if imp.strategy != "median" or not imp.keep_empty_features:
        raise AssertionError("median train-only imputation/keep-columns configuration mismatch")
    p = est.get_params()
    if candidate[1] == "rf":
        expected = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                    "max_features": "sqrt", "class_weight": "balanced", "random_state": SEED, "n_jobs": 4}
    else:
        expected = {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1,
                    "l2_regularization": 1.0, "class_weight": "balanced", "random_state": SEED,
                    "early_stopping": False}
    for key, value in expected.items():
        if p.get(key) != value:
            raise AssertionError(f"{candidate[0]} estimator parameter {key}: {p.get(key)!r} != {value!r}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    with np.load(args.input_npz, allow_pickle=False) as z:
        keys = ("X_sensor", "X_context", "y", "trip", "vehicle", "scenario", "source_start", "source_end", "anchor_time")
        d = {k: np.asarray(z[k]) for k in keys}
    if np.isinf(d["X_sensor"]).any() or np.isinf(d["X_context"]).any():
        raise AssertionError("input contains infinity")
    provenance = json.loads((args.run_dir / "input_provenance.json").read_text(encoding="utf-8"))
    if provenance["npz_sha256"] != sha256(args.input_npz) or provenance["manifest_sha256"] != sha256(args.feature_manifest):
        raise AssertionError("input/manifest hash mismatch")
    protocol_file = args.protocol.resolve()
    if provenance.get("protocol_sha256") != sha256(protocol_file):
        raise AssertionError("protocol hash mismatch")
    driver_file = Path(__file__).with_name("run_nested_evaluation.py").resolve()
    if provenance.get("script_sha256") != sha256(driver_file):
        raise AssertionError("frozen driver hash mismatch")
    original_plan = json.loads((args.run_dir / "preflight.json").read_text(encoding="utf-8"))
    records_obj = json.loads((args.run_dir / "fit_records.json").read_text(encoding="utf-8"))
    records = records_obj["records"]
    if len(records) != 264 or records_obj.get("completed_fits") != 264 or records_obj.get("failures"):
        raise AssertionError("run is not a complete zero-failure 264-fit bundle")
    expected_by_id: dict[str, tuple[str, str, np.ndarray, np.ndarray, int]] = {}
    outer_folds = expected_outer(d)
    for outer_i, outer in enumerate(outer_folds):
        if not np.array_equal(np.sort(original_plan["folds"][outer_i]["train_indices"]), np.sort(outer["train"])):
            raise AssertionError(f"preflight outer train membership mismatch: {outer['outer_id']}")
        if not np.array_equal(np.sort(original_plan["folds"][outer_i]["test_indices"]), np.sort(outer["test"])):
            raise AssertionError(f"preflight outer test membership mismatch: {outer['outer_id']}")
        pos = 0
        for scheme in SCHEMES:
            for fold_name, tr, te in expected_inner(outer["train"], d, outer["axis"], scheme):
                fit_index = outer_i * 11 + pos
                for cid, _, _ in CANDIDATES:
                    fid = f"{outer['outer_id']}__{scheme}__{fold_name}__{cid}"
                    expected_by_id[fid] = (scheme, cid, tr, te, fit_index)
                pos += 1
        for cid, _, _ in CANDIDATES:
            fid = f"{outer['outer_id']}__outer_refit__{cid}"
            expected_by_id[fid] = ("outer_oracle_diagnostic", cid, outer["train"], outer["test"], outer_i * 11 + 10)
    if len(expected_by_id) != 264:
        raise AssertionError("internal expected fit matrix is not 264")
    by_id = {r["fit_id"]: r for r in records}
    if set(by_id) != set(expected_by_id):
        raise AssertionError("actual fit identities do not match the fixed matrix")
    score_data: dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]] = {}
    outer_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    verification_counts = {"fits": 0, "models_reloaded": 0, "predictions_reproduced": 0,
                           "metrics_recomputed": 0, "grouped_splits_checked": 0}
    for fid, (scheme, cid, train, test, fit_index) in expected_by_id.items():
        rec = by_id[fid]
        expected_phase = "outer_oracle_diagnostic" if scheme == "outer_oracle_diagnostic" else "inner"
        if rec["candidate_id"] != cid or rec["phase"] != expected_phase or rec["fit_index"] != fit_index or rec["cap_seed"] != CAP_SEED_BASE + fit_index:
            raise AssertionError(f"fit metadata mismatch: {fid}")
        if not np.array_equal(np.asarray(rec["train_indices_uncapped"]), train):
            raise AssertionError(f"uncapped train indices mismatch: {fid}")
        cap = capped_indices(train, d["y"], CAP_SEED_BASE + fit_index)
        if not np.array_equal(np.asarray(rec["train_indices_fit"]), cap):
            raise AssertionError(f"capped fit indices mismatch: {fid}")
        if not np.array_equal(np.asarray(rec["test_indices"]), test):
            raise AssertionError(f"test indices mismatch: {fid}")
        if len(np.unique(d["y"][cap])) != 2 or len(np.unique(d["y"][test])) != 2:
            raise AssertionError(f"missing class in fit/test partition: {fid}")
        observed_overlap = overlap_samples(cap, test, d)
        if rec.get("raw_window_overlap") != observed_overlap:
            raise AssertionError(f"raw-window overlap diagnostic mismatch: {fid}")
        if scheme != "row_random":
            if np.intersect1d(d["trip"][cap], d["trip"][test]).size or observed_overlap["train_samples_with_overlap"]:
                raise AssertionError(f"group/raw-window leakage: {fid}")
            verification_counts["grouped_splits_checked"] += 1
        fit_dir = args.run_dir / "fits" / fid
        model_path, pred_path = fit_dir / "model.joblib", fit_dir / "predictions.npz"
        fit_json = json.loads((fit_dir / "fit.json").read_text(encoding="utf-8"))
        if fit_json.get("fit_id") != fid or fit_json.get("fit_index") != fit_index or fit_json.get("cap_seed") != CAP_SEED_BASE + fit_index:
            raise AssertionError(f"per-fit record mismatch: {fid}")
        if (fit_json.get("train_indices_uncapped") != rec["train_indices_uncapped"] or
                fit_json.get("train_indices_fit") != rec["train_indices_fit"] or
                fit_json.get("test_indices") != rec["test_indices"] or
                fit_json.get("model_sha256") != rec["model_sha256"] or
                fit_json.get("prediction_sha256") != rec["prediction_sha256"]):
            raise AssertionError(f"per-fit provenance/index mismatch: {fid}")
        if rec["model_sha256"] != sha256(model_path) or rec["prediction_sha256"] != sha256(pred_path):
            raise AssertionError(f"artifact hash mismatch: {fid}")
        model = joblib.load(model_path)
        candidate = next(c for c in CANDIDATES if c[0] == cid)
        expect_model_params(model, candidate)
        Xfit = matrix_for(candidate, cap, d, manifest)
        expected_width = 30 if candidate[2] == "sensor" else 33
        if model.named_steps["model"].n_features_in_ != expected_width:
            raise AssertionError(f"candidate feature dimension mismatch: {fid}")
        if not np.isfinite(Xfit).any(axis=0).all():
            raise AssertionError(f"empty feature in exact training cap: {fid}")
        expected_medians = np.nanmedian(Xfit, axis=0)
        if not np.allclose(model.named_steps["imputer"].statistics_, expected_medians, atol=1e-12, rtol=0):
            raise AssertionError(f"training median mismatch: {fid}")
        with np.load(pred_path, allow_pickle=False) as p:
            pred_idx, y_true, y_pred = p["test_indices"], p["y_true"], p["y_pred"]
            if not np.array_equal(p["train_indices_uncapped"], train) or not np.array_equal(p["train_indices_fit"], cap):
                raise AssertionError(f"prediction file training indices mismatch: {fid}")
        if not np.array_equal(pred_idx, test) or not np.array_equal(y_true, d["y"][test]):
            raise AssertionError(f"prediction file test identity/labels mismatch: {fid}")
        reloaded = model.predict(matrix_for(candidate, test, d, manifest)).astype(np.int64)
        if not np.array_equal(reloaded, y_pred):
            raise AssertionError(f"reloaded predictions differ: {fid}")
        recomputed = metrics(y_true, y_pred)
        stored_keys = set(recomputed) - {"class_support"}
        if set(rec["metrics"]) != stored_keys or not close_metrics(rec["metrics"], {k: recomputed[k] for k in stored_keys}):
            raise AssertionError(f"stored metrics differ from recomputation: {fid}")
        if not np.array_equal(model.classes_, np.unique(d["y"][cap])):
            raise AssertionError(f"model class support mismatch: {fid}")
        if scheme == "outer_oracle_diagnostic":
            outer_id = fid.split("__outer_refit__", 1)[0]
            outer_metrics.setdefault(outer_id, {})[cid] = recomputed
        else:
            outer_id = fid.split(f"__{scheme}__", 1)[0]
            score_data.setdefault(outer_id, {}).setdefault(scheme, {}).setdefault(cid, []).append((test.copy(), y_pred.copy()))
        verification_counts["fits"] += 1
        verification_counts["models_reloaded"] += 1
        verification_counts["predictions_reproduced"] += 1
        verification_counts["metrics_recomputed"] += 1
        if verification_counts["fits"] % 20 == 0:
            print(json.dumps({"models_independently_verified": verification_counts["fits"], "total": 264}), flush=True)

    candidate_rows, selector_rows, axis_scheme_acc, candidate_axis_acc = [], [], {}, {}
    frozen_hashes = {}
    for outer in outer_folds:
        outer_id = outer["outer_id"]
        odir = args.run_dir / "outer_folds" / outer_id
        frozen_path, results_path = odir / "selection.json", odir / "selection_results.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        frozen_hash = sha256(frozen_path)
        if frozen_hash != results.get("frozen_selection_sha256"):
            raise AssertionError(f"frozen selection hash mismatch: {outer_id}")
        if Path(results.get("frozen_selection_path", "")).resolve() != frozen_path.resolve():
            raise AssertionError(f"frozen selection path mismatch: {outer_id}")
        frozen_hashes[outer_id] = frozen_hash
        recomputed_inner = {}
        for scheme in SCHEMES:
            recomputed_inner[scheme] = {}
            for cid, _, _ in CANDIDATES:
                chunks = score_data[outer_id][scheme][cid]
                ix = np.concatenate([a for a, _ in chunks])
                yp = np.concatenate([b for _, b in chunks])
                order = np.argsort(ix, kind="stable")
                if not np.array_equal(ix[order], np.sort(outer["train"])):
                    raise AssertionError(f"inner OOF coverage error: {outer_id}/{scheme}/{cid}")
                score = float(f1_score(d["y"][ix[order]], yp[order], average="macro", zero_division=0))
                recomputed_inner[scheme][cid] = score
            selected = max((c[0] for c in CANDIDATES), key=lambda c: recomputed_inner[scheme][c])
            stored = frozen["selectors"][scheme]
            if stored["selected_candidate_id"] != selected or not close_metrics(
                    {k: float(v) for k, v in stored["pooled_inner_oof_macro_f1"].items()},
                    recomputed_inner[scheme]):
                raise AssertionError(f"frozen selector score/identity mismatch: {outer_id}/{scheme}")
        # The immutable file must contain no post-test results.
        if frozen.get("outer_predictions_not_yet_generated") is not True:
            raise AssertionError(f"selection was not frozen before outer predictions: {outer_id}")
        chosen_outer: dict[str, dict[str, Any]] = {}
        oracle_f1 = max(v["macro_f1"] for v in outer_metrics[outer_id].values())
        for scheme in SCHEMES:
            sel = results["selectors"][scheme]
            chosen = frozen["selectors"][scheme]["selected_candidate_id"]
            of = outer_metrics[outer_id][chosen]
            estimate = recomputed_inner[scheme][chosen]
            signed = of["macro_f1"] - estimate
            if sel.get("selected_candidate_id") != chosen or not np.isclose(sel["signed_estimation_error_macro_f1"], signed, atol=1e-12, rtol=0):
                raise AssertionError(f"selection-results join mismatch: {outer_id}/{scheme}")
            if (not np.isclose(sel["selected_candidate_inner_macro_f1"], estimate, atol=1e-12, rtol=0) or
                    set(sel["selected_candidate_outer_test_metrics"]) != set(of) - {"class_support"} or
                    not close_metrics(sel["selected_candidate_outer_test_metrics"], {k:v for k,v in of.items() if k != "class_support"})):
                raise AssertionError(f"selected candidate estimate/outer metrics mismatch: {outer_id}/{scheme}")
            if not np.isclose(sel["absolute_estimation_error_macro_f1"], abs(signed), atol=1e-12, rtol=0):
                raise AssertionError(f"selection absolute error mismatch: {outer_id}/{scheme}")
            chosen_outer[scheme] = {"candidate": chosen, "inner": estimate, "outer": of["macro_f1"],
                                    "error": signed, "abs_error": abs(signed),
                                    "regret": oracle_f1 - of["macro_f1"]}
            selector_rows.append({"axis": outer["axis"], "outer_id": outer_id, "scheme": scheme,
                "chosen_candidate_id": chosen, "inner_macro_f1": estimate, "outer_macro_f1": of["macro_f1"],
                "signed_error_outer_minus_inner": signed, "absolute_error": abs(signed),
                "oracle_regret_macro_f1": oracle_f1 - of["macro_f1"]})
        for cid, _, _ in CANDIDATES:
            candidate_rows.append({"axis": outer["axis"], "outer_id": outer_id, "candidate_id": cid,
                "inner_random_macro_f1": recomputed_inner["row_random"][cid],
                "inner_trip_macro_f1": recomputed_inner["leave_one_trip"][cid],
                "inner_matched_macro_f1": recomputed_inner["matched_factor"][cid],
                "outer_macro_f1": outer_metrics[outer_id][cid]["macro_f1"],
                "outer_balanced_accuracy": outer_metrics[outer_id][cid]["balanced_accuracy"],
                "outer_class_recalls": outer_metrics[outer_id][cid]["class_recalls"],
                "outer_class_support": outer_metrics[outer_id][cid]["class_support"],
                "outer_confusion_matrix": outer_metrics[outer_id][cid]["confusion_matrix_rows_true_columns_pred"]})
            candidate_axis_acc.setdefault((outer["axis"], cid), []).append(candidate_rows[-1])
        for scheme in SCHEMES:
            axis_scheme_acc.setdefault((outer["axis"], scheme), []).append(chosen_outer[scheme])

    axis_selector_rows = []
    for (axis, scheme), rows in sorted(axis_scheme_acc.items()):
        axis_selector_rows.append({"axis": axis, "scheme": scheme, "outer_domains": len(rows),
            "mean_inner_macro_f1": float(np.mean([r["inner"] for r in rows])),
            "mean_outer_macro_f1": float(np.mean([r["outer"] for r in rows])),
            "mae": float(np.mean([r["abs_error"] for r in rows])),
            "bias_outer_minus_inner": float(np.mean([r["error"] for r in rows])),
            "mean_oracle_regret_macro_f1": float(np.mean([r["regret"] for r in rows]))})
    candidate_axis_rows = []
    for (axis, cid), rows in sorted(candidate_axis_acc.items()):
        candidate_axis_rows.append({"axis": axis, "candidate_id": cid,
            "mean_inner_random_macro_f1": float(np.mean([r["inner_random_macro_f1"] for r in rows])),
            "mean_inner_trip_macro_f1": float(np.mean([r["inner_trip_macro_f1"] for r in rows])),
            "mean_inner_matched_macro_f1": float(np.mean([r["inner_matched_macro_f1"] for r in rows])),
            "mean_outer_macro_f1": float(np.mean([r["outer_macro_f1"] for r in rows])),
            "mean_outer_balanced_accuracy": float(np.mean([r["outer_balanced_accuracy"] for r in rows]))})
    if (len(candidate_rows), len(selector_rows), len(axis_selector_rows), len(candidate_axis_rows)) != (24, 18, 6, 8):
        raise AssertionError("aggregate row counts differ from the frozen design")
    criterion = {}
    for axis in AXES:
        ma = {row["scheme"]: row["mae"] for row in axis_selector_rows if row["axis"] == axis}
        delta = ma["row_random"] - ma["matched_factor"]
        criterion[axis] = {"row_random_MAE": ma["row_random"], "matched_factor_MAE": ma["matched_factor"],
                           "improvement_random_minus_matched": delta, "threshold": 0.02,
                           "supported": bool(delta >= 0.02)}
    return {"status": "independently_validated", "planned_and_validated_fits": 264,
        "verification_counts": verification_counts, "candidate_domain_rows": candidate_rows,
        "selector_domain_rows": selector_rows, "axis_selector_rows": axis_selector_rows,
        "candidate_axis_mean_rows": candidate_axis_rows, "primary_criterion": {
            "axes": criterion, "supported_both_axes": all(v["supported"] for v in criterion.values()),
            "interpretation": "supports both axes" if all(v["supported"] for v in criterion.values()) else "partial support/unsupported"},
        "frozen_selection_sha256": frozen_hashes,
        "provenance": {"features_sha256": sha256(args.input_npz), "feature_manifest_sha256": sha256(args.feature_manifest),
                       "protocol_sha256": sha256(protocol_file), "driver_sha256": sha256(driver_file),
                       "analyzer_sha256": sha256(Path(__file__).resolve()),
                       "fit_records_sha256": sha256(args.run_dir / "fit_records.json")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = validate(args)
    write_json(args.output, summary)
    print(json.dumps({"status": summary["status"], "validated_fits": 264,
                      "output": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
