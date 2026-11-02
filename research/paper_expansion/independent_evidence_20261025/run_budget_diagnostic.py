#!/usr/bin/env python
"""Post-hoc PVS common-training-budget diagnostic (protocol v1)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

# Must precede numerical imports.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "4"

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROTOCOL = HERE / "budget_diagnostic_protocol_v1.md"
LOCKED_PYTHON = Path(r"C:\Users\Administrator\miniconda3\python.exe")
BASE = HERE.parent / "external_pvs_v1" / "nonoverlap"
ORIG_PREFLIGHT = BASE / "run_v1" / "preflight.json"
ORIG_SUMMARY = BASE / "checked_summary.json"
FEATURES = BASE / "features.npz"
HELPER = HERE.parent / "external_pvs_v1" / "run_nested_evaluation.py"
CANDIDATE_ORDER = ("rf_sensor", "rf_sensor_context", "hgb_sensor", "hgb_sensor_context")
METHOD_KEYS = (("row_random", "random"), ("leave_one_trip", "trip"), ("matched_factor", "matched"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def helper_module():
    spec = importlib.util.spec_from_file_location("pvs_original_helper", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data():
    z = np.load(FEATURES, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    data["_sensor_indices"] = np.arange(30, dtype=np.int64)
    data["_context_indices"] = np.arange(3, dtype=np.int64)
    return data


def counts(y, idx):
    return {str(c): int(np.count_nonzero(y[np.asarray(idx, dtype=np.int64)] == c)) for c in (0, 1)}


def selected_rows(pool, y, budget, seed):
    pool = np.asarray(pool, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    chosen = []
    for c in (0, 1):
        cp = np.sort(pool[y[pool] == c])
        n = int(budget[str(c)])
        if len(cp) < n:
            raise ValueError(f"class {c} pool has {len(cp)} rows, below budget {n}")
        chosen.extend(rng.choice(cp, size=n, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


def verify_disjoint(a, b, helper, data, where):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if np.intersect1d(a, b).size:
        raise ValueError(f"fit/test membership overlap at {where}")
    overlap = helper.raw_overlap(a, b, data)
    def nonzero_leaf(x):
        if isinstance(x, dict):
            return any(nonzero_leaf(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(nonzero_leaf(v) for v in x)
        if isinstance(x, (int, float, np.integer, np.floating, bool, np.bool_)):
            return bool(x != 0)
        return False
    if nonzero_leaf(overlap):
        raise ValueError(f"raw source-window overlap at {where}: {overlap}")
    # Explicit interval check, grouped by trip because offsets reset between trips.
    starts, ends = data["source_start"], data["source_end"]
    trips = data["trip"]
    if len(a) and len(b):
        by_trip = {}
        for trip in np.unique(trips[b]):
            rows = b[trips[b] == trip]
            order = np.argsort(starts[rows], kind="stable")
            by_trip[trip] = (starts[rows][order], ends[rows][order])
        for i in a:
            if trips[i] not in by_trip:
                continue
            bs, be = by_trip[trips[i]]
            j = np.searchsorted(bs, ends[i], side="right") - 1
            if j >= 0 and be[j] >= starts[i]:
                raise ValueError(f"intersecting source-row intervals at {where}")


def folds_in_protocol_order(outer):
    inner = outer["inner"]
    out = []
    for key, prefix in METHOD_KEYS:
        arr = inner[key]
        if len(arr) != {"row_random": 2, "leave_one_trip": 6, "matched_factor": 2}[key]:
            raise ValueError(f"wrong number of folds for {key}")
        for f in arr:
            out.append((key, prefix, f))
    return out


def validate_outer_membership(outer, data):
    axis = outer["axis"]
    if axis not in ("vehicle", "scenario"):
        raise ValueError(f"unexpected outer axis {axis}")
    held = int(outer["held_out_id"])
    meta = data[axis]
    expected_test = np.flatnonzero(meta == held)
    test = np.asarray(outer["test_indices"], dtype=np.int64)
    train = np.asarray(outer["train_indices"], dtype=np.int64)
    if not np.array_equal(np.sort(test), expected_test):
        raise ValueError(f"outer test membership differs from grouped {axis}={held}")
    if np.intersect1d(train, test).size or not np.array_equal(np.sort(np.r_[train, test]), np.arange(len(meta))):
        raise ValueError("outer train/test do not form exact disjoint population partition")


def make_preflight(outdir):
    if sys.executable.lower() != str(LOCKED_PYTHON).lower():
        raise RuntimeError(f"Use locked research interpreter: {LOCKED_PYTHON}; got {sys.executable}")
    for p in (PROTOCOL, ORIG_PREFLIGHT, ORIG_SUMMARY, FEATURES, HELPER):
        if not p.is_file():
            raise FileNotFoundError(p)
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {outdir}")
    h = helper_module()
    candidates = {c["id"]: c for c in h.CANDIDATES}
    if tuple(candidates) != CANDIDATE_ORDER:
        raise ValueError(f"original candidate order mismatch: {tuple(candidates)}")
    data = load_data()
    original = read_json(ORIG_PREFLIGHT)
    if len(original.get("folds", [])) != 6:
        raise ValueError("expected six original outer folds")
    if len(data["y"]) != 10804 or int(np.sum(data["y"] == 1)) != 170:
        raise ValueError("feature population does not match frozen protocol")
    arrays = {}
    domains = []
    for oi, outer in enumerate(original["folds"]):
        validate_outer_membership(outer, data)
        oid = outer["outer_id"]
        plans = folds_in_protocol_order(outer)
        outercap = outer["outer_refit_cap"]
        pools = [(f["fold_id"], f["train_indices_fit"], f["test_indices"], key, f, pos)
                 for pos, (key, _prefix, f) in enumerate(plans)]
        pools.append((outercap["fold_id"], outercap["train_indices_fit"], outer["test_indices"], "outer", outercap, 10))
        per_counts = []
        for fid, fitpool, testidx, key, record, pos in pools:
            fitpool = np.asarray(fitpool, dtype=np.int64)
            testidx = np.asarray(testidx, dtype=np.int64)
            verify_disjoint(fitpool, testidx, h, data, fid)
            per_counts.append(counts(data["y"], fitpool))
        budget = {str(c): min(x[str(c)] for x in per_counts) for c in (0, 1)}
        if min(budget.values()) < 2:
            raise ValueError(f"class budget below two for {oid}: {budget}")
        fold_records = []
        for fid, fitpool, testidx, key, record, pos in pools:
            seed = 20261025 + 100 * oi + pos
            chosen = selected_rows(fitpool, data["y"], budget, seed)
            if counts(data["y"], chosen) != budget:
                raise AssertionError("selected class counts differ")
            verify_disjoint(chosen, testidx, h, data, fid + " selected")
            # Declared features must have at least one finite observation in each selected fit set.
            for candidate_id in CANDIDATE_ORDER:
                X = np.asarray(h.matrix_for(candidates[candidate_id], chosen, data))
                if X.ndim != 2 or X.shape[0] != len(chosen) or not np.isfinite(X).any(axis=0).all():
                    raise ValueError(f"feature without finite selected training value: {fid}/{candidate_id}")
            arrname = f"{oid}__{key}__{pos if key != 'outer' else 'refit'}"
            arrays[arrname] = chosen
            fold_records.append({"fold_id": fid, "kind": key, "inner_position": pos if pos < 10 else None,
                                 "seed": seed, "fit_pool_sha256": hashlib.sha256(np.sort(np.asarray(fitpool, dtype=np.int64)).tobytes()).hexdigest(),
                                 "test_indices_sha256": hashlib.sha256(np.sort(np.asarray(testidx, dtype=np.int64)).tobytes()).hexdigest(),
                                 "selected_array": arrname, "selected_indices": chosen.tolist(),
                                 "class_counts": counts(data["y"], chosen),
                                 "pool_class_counts": counts(data["y"], fitpool),
                                 "test_class_counts": counts(data["y"], testidx),
                                 "raw_window_overlap": False})
        domains.append({"outer_position": oi, "axis": outer["axis"], "held_out_id": outer["held_out_id"],
                        "outer_id": oid, "budget_per_class": budget, "folds": fold_records})
    outdir.mkdir(parents=True)
    np.savez_compressed(outdir / "selected_indices.npz", **arrays)
    hashes = {"protocol": sha256_file(PROTOCOL), "code": sha256_file(Path(__file__)),
              "features": sha256_file(FEATURES), "original_preflight": sha256_file(ORIG_PREFLIGHT),
              "original_summary": sha256_file(ORIG_SUMMARY), "helper": sha256_file(HELPER)}
    manifest = {"protocol": "budget_diagnostic_protocol_v1", "status": "preflight_passed_fits_not_started",
                "interpreter": sys.executable, "candidate_order": list(CANDIDATE_ORDER),
                "inputs": {"features": str(FEATURES), "original_preflight": str(ORIG_PREFLIGHT),
                           "original_summary": str(ORIG_SUMMARY), "protocol": str(PROTOCOL), "helper": str(HELPER)},
                "input_sha256": hashes, "selected_indices_sha256": sha256_file(outdir / "selected_indices.npz"),
                "population_windows": len(data["y"]), "positive_windows": int(np.sum(data["y"] == 1)),
                "planned_new_fits": 264, "domains": domains}
    atomic_json(outdir / "preflight_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_dir": str(outdir), "planned_new_fits": 264,
                      "selected_indices_sha256": manifest["selected_indices_sha256"]}, indent=2))


def load_preflight(outdir):
    mf = outdir / "preflight_manifest.json"
    idx = outdir / "selected_indices.npz"
    if not mf.is_file() or not idx.is_file():
        raise FileNotFoundError("Preflight manifest/arrays missing; run --preflight first")
    m = read_json(mf)
    expected = {"protocol": sha256_file(PROTOCOL), "code": sha256_file(Path(__file__)),
                "features": sha256_file(FEATURES), "original_preflight": sha256_file(ORIG_PREFLIGHT),
                "original_summary": sha256_file(ORIG_SUMMARY), "helper": sha256_file(HELPER)}
    if m["input_sha256"] != expected or m["selected_indices_sha256"] != sha256_file(idx):
        raise RuntimeError("Preflight provenance mismatch; refusing to fit")
    if m["status"] != "preflight_passed_fits_not_started":
        raise RuntimeError("Preflight manifest status is not runnable")
    return m


def save_npz(path, **arrays):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def run(outdir):
    if sys.executable.lower() != str(LOCKED_PYTHON).lower():
        raise RuntimeError(f"Use locked research interpreter: {LOCKED_PYTHON}; got {sys.executable}")
    pre = load_preflight(outdir)
    runmark = outdir / "run_started.json"
    if runmark.exists():
        raise FileExistsError("Run already started; refusing a silent rerun")
    atomic_json(runmark, {"status": "running", "started": __import__("datetime").datetime.now().isoformat()})
    h, data = helper_module(), load_data()
    candidates = {c["id"]: c for c in h.CANDIDATES}
    if tuple(candidates) != CANDIDATE_ORDER:
        raise ValueError(f"original candidate order mismatch: {tuple(candidates)}")
    original = read_json(ORIG_PREFLIGHT)
    selected = np.load(outdir / "selected_indices.npz", allow_pickle=False)
    fitsdir = outdir / "fits"
    fitsdir.mkdir()
    fit_records, events = [], []
    oof = {}
    # Inner fits and all OOF predictions are completed before selections are frozen.
    for oi, outer in enumerate(original["folds"]):
        oid = outer["outer_id"]
        for key, prefix in METHOD_KEYS:
            pred_parts = {c: [] for c in CANDIDATE_ORDER}
            for pos, fold in enumerate(outer["inner"][key]):
                arr = selected[f"{oid}__{key}__{sum(len(outer['inner'][k]) for k, _ in METHOD_KEYS[:METHOD_KEYS.index((key,prefix))]) + pos}"]
                test = np.asarray(fold["test_indices"], dtype=np.int64)
                y = data["y"][test]
                for ci, cand in enumerate(CANDIDATE_ORDER):
                    Xtr = h.matrix_for(candidates[cand], arr, data)
                    Xte = h.matrix_for(candidates[cand], test, data)
                    model = h.make_model(candidates[cand])
                    model.fit(Xtr, data["y"][arr])
                    yp = model.predict(Xte).astype(np.int8)
                    stem = f"{oid}__{prefix}_{pos+1}__{cand}"
                    fitdir = fitsdir / stem
                    fitdir.mkdir()
                    modelpath = fitdir / "model.joblib"
                    predpath = fitdir / "predictions.npz"
                    with open(modelpath, "wb") as f:
                        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                    save_npz(predpath, train_indices_pool=np.asarray(fold["train_indices_fit"], dtype=np.int64),
                             train_indices_fit=arr, test_indices=test, y_true=y, y_pred=yp, candidate=np.asarray(cand))
                    fit_records.append({"fit_id": stem, "outer_id": oid, "axis": outer["axis"],
                                        "scheme": key, "inner_position": pos, "phase": "inner",
                                        "candidate_id": cand, "train_indices_pool": np.asarray(fold["train_indices_fit"], dtype=np.int64).tolist(),
                                        "train_indices_fit": arr.tolist(), "test_indices": test.tolist(),
                                        "model_sha256": sha256_file(modelpath), "prediction_sha256": sha256_file(predpath)})
                    events.append({"sequence": len(events), "event": "inner_fit_complete", "fit_id": stem})
                    pred_parts[cand].append((test, y, yp))
            for cand, parts in pred_parts.items():
                alltest = np.concatenate([p[0] for p in parts])
                ally = np.concatenate([p[1] for p in parts])
                allpred = np.concatenate([p[2] for p in parts])
                order = np.argsort(alltest, kind="stable")
                alltest, ally, allpred = alltest[order], ally[order], allpred[order]
                if len(np.unique(alltest)) != len(alltest):
                    raise ValueError(f"duplicate OOF coverage for {oid}/{key}")
                score = float(f1_score(ally, allpred, labels=[0, 1], average="macro", zero_division=0))
                oof[(oid, key, cand)] = {"indices": alltest, "y_true": ally, "y_pred": allpred, "macro_f1": score}
                save_npz(fitsdir / f"{oid}__{prefix}__{cand}__oof.npz", test_indices=alltest,
                         y_true=ally, y_pred=allpred, macro_f1=np.asarray(score))
    selections = []
    for outer in original["folds"]:
        oid = outer["outer_id"]
        for key, prefix in METHOD_KEYS:
            scores = [(c, oof[(oid, key, c)]["macro_f1"]) for c in CANDIDATE_ORDER]
            best = max(range(len(scores)), key=lambda i: scores[i][1])
            selections.append({"outer_id": oid, "axis": outer["axis"], "held_out_id": outer["held_out_id"],
                               "scheme": key, "candidate_scores": dict(scores),
                               "selected_candidate": scores[best][0], "inner_macro_f1": scores[best][1]})
    selection_payload = {"selections": selections, "candidate_order": list(CANDIDATE_ORDER),
                         "input_sha256": pre["input_sha256"], "selected_indices_sha256": pre["selected_indices_sha256"]}
    selection_payload["sha256"] = digest_obj(selection_payload)
    atomic_json(outdir / "frozen_selections.json", selection_payload)
    for sel in selections:
        d = outdir / "outer_folds" / sel["outer_id"]
        d.mkdir(parents=True, exist_ok=True)
        atomic_json(d / "selection.json", {**sel, "frozen_selections_sha256": selection_payload["sha256"]})
    events.append({"sequence": len(events), "event": "selections_frozen", "frozen_selections_sha256": selection_payload["sha256"]})
    # Outer fits start only after frozen selectors and their hash have been durably written.
    rows = []
    for oi, outer in enumerate(original["folds"]):
        oid, test = outer["outer_id"], np.asarray(outer["test_indices"], dtype=np.int64)
        fit = selected[f"{oid}__outer__refit"]
        for cand in CANDIDATE_ORDER:
            model = h.make_model(candidates[cand])
            model.fit(h.matrix_for(candidates[cand], fit, data), data["y"][fit])
            yp = model.predict(h.matrix_for(candidates[cand], test, data)).astype(np.int8)
            stem = f"{oid}__budget_outer__{cand}"
            fitdir = fitsdir / stem
            fitdir.mkdir()
            modelpath = fitdir / "model.joblib"
            predpath = fitdir / "predictions.npz"
            with open(modelpath, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            save_npz(predpath, train_indices_pool=np.asarray(outer["outer_refit_cap"]["train_indices_fit"], dtype=np.int64),
                     train_indices_fit=fit, test_indices=test, y_true=data["y"][test], y_pred=yp, candidate=np.asarray(cand))
            fit_records.append({"fit_id": stem, "outer_id": oid, "axis": outer["axis"],
                                "scheme": "shared_outer_refit", "inner_position": None, "phase": "budget_outer",
                                "candidate_id": cand, "train_indices_pool": np.asarray(outer["outer_refit_cap"]["train_indices_fit"], dtype=np.int64).tolist(),
                                "train_indices_fit": fit.tolist(), "test_indices": test.tolist(),
                                "model_sha256": sha256_file(modelpath), "prediction_sha256": sha256_file(predpath)})
            events.append({"sequence": len(events), "event": "budget_outer_fit_complete", "fit_id": stem,
                           "frozen_selections_sha256": selection_payload["sha256"]})
        for key, prefix in METHOD_KEYS:
            sel = next(s for s in selections if s["outer_id"] == oid and s["scheme"] == key)
            cand = sel["selected_candidate"]
            z = np.load(fitsdir / f"{oid}__budget_outer__{cand}" / "predictions.npz", allow_pickle=False)
            outer_f1 = float(f1_score(z["y_true"], z["y_pred"], labels=[0, 1], average="macro", zero_division=0))
            inner_f1 = float(sel["inner_macro_f1"])
            rows.append({"outer_id": oid, "axis": outer["axis"], "held_out_id": outer["held_out_id"],
                         "scheme": key, "selected_candidate": cand, "inner_macro_f1": inner_f1,
                         "budget_outer_macro_f1": outer_f1, "signed_error_outer_minus_inner": outer_f1-inner_f1,
                         "absolute_error": abs(outer_f1-inner_f1), "budget_per_class": pre["domains"][oi]["budget_per_class"]})
    write_csv(outdir / "budget_matched_domain_summary.csv", rows)
    axisrows = []
    for axis in ("vehicle", "scenario"):
        for key, _ in METHOD_KEYS:
            subset = [r for r in rows if r["axis"] == axis and r["scheme"] == key]
            axisrows.append({"axis": axis, "scheme": key, "domains": len(subset),
                             "mean_inner_macro_f1": float(np.mean([r["inner_macro_f1"] for r in subset])),
                             "mean_outer_macro_f1": float(np.mean([r["budget_outer_macro_f1"] for r in subset])),
                             "mean_mae": float(np.mean([r["absolute_error"] for r in subset])),
                             "mean_signed_error": float(np.mean([r["signed_error_outer_minus_inner"] for r in subset]))})
    write_csv(outdir / "budget_matched_axis_summary.csv", axisrows)
    historical_rows = []
    historical_base = BASE / "run_v1" / "fits"
    for outer in original["folds"]:
        oid = outer["outer_id"]
        for key, _prefix in METHOD_KEYS:
            selection = next(s for s in selections if s["outer_id"] == oid and s["scheme"] == key)
            for cand in CANDIDATE_ORDER:
                predpath = historical_base / f"{oid}__outer_refit__{cand}" / "predictions.npz"
                z = np.load(predpath, allow_pickle=False)
                if not np.array_equal(np.asarray(z["test_indices"], dtype=np.int64), np.asarray(outer["test_indices"], dtype=np.int64)):
                    raise ValueError(f"historical outer test membership mismatch: {oid}/{cand}")
                outer_f1 = float(f1_score(z["y_true"], z["y_pred"], labels=[0, 1], average="macro", zero_division=0))
                inner_f1 = float(oof[(oid, key, cand)]["macro_f1"])
                historical_rows.append({"outer_id": oid, "axis": outer["axis"], "held_out_id": outer["held_out_id"],
                                        "scheme": key, "candidate_id": cand,
                                        "is_selected": cand == selection["selected_candidate"],
                                        "selected_candidate": selection["selected_candidate"], "inner_macro_f1": inner_f1,
                                        "historical_full_outer_macro_f1": outer_f1,
                                        "signed_error_outer_minus_inner": outer_f1-inner_f1,
                                        "absolute_error": abs(outer_f1-inner_f1),
                                        "historical_prediction_sha256": sha256_file(predpath)})
    write_csv(outdir / "historical_full_budget_candidate_rows.csv", historical_rows)
    historical_selected = [r for r in historical_rows if r["is_selected"]]
    historical_axisrows = []
    for axis in ("vehicle", "scenario"):
        for key, _ in METHOD_KEYS:
            subset = [r for r in historical_selected if r["axis"] == axis and r["scheme"] == key]
            historical_axisrows.append({"axis": axis, "scheme": key, "domains": len(subset),
                                        "mean_inner_macro_f1": float(np.mean([r["inner_macro_f1"] for r in subset])),
                                        "mean_outer_macro_f1": float(np.mean([r["historical_full_outer_macro_f1"] for r in subset])),
                                        "mean_mae": float(np.mean([r["absolute_error"] for r in subset])),
                                        "mean_signed_error": float(np.mean([r["signed_error_outer_minus_inner"] for r in subset]))})
    contrasts = []
    for target, table in (("budget_matched", axisrows), ("historical_full_budget", historical_axisrows)):
        for axis in ("vehicle", "scenario"):
            subset = [r for r in table if r["axis"] == axis]
            vals = {r["scheme"]: r["mean_mae"] for r in subset}
            if "row_random" in vals and "matched_factor" in vals:
                contrasts.append({"axis": axis, "target": target,
                              "random_mae_minus_matched_mae": vals["row_random"] - vals["matched_factor"]})
    write_csv(outdir / "historical_full_budget_axis_summary.csv", historical_axisrows)
    atomic_json(outdir / "fit_records.json", {"records": fit_records, "events": events,
                "record_count": len(fit_records), "expected_new_fit_count": 264,
                "frozen_selections_sha256": selection_payload["sha256"]})
    summary = {"status": "completed", "protocol": "budget_diagnostic_protocol_v1",
               "planned_new_fits": 264, "frozen_selections_sha256": selection_payload["sha256"],
               "budget_matched_domain_rows": rows, "budget_matched_axis_rows": axisrows,
               "historical_full_budget_candidate_rows": historical_rows,
               "historical_full_budget_selected_domain_rows": historical_selected,
               "historical_full_budget_axis_rows": historical_axisrows,
               "mae_contrasts": contrasts,
               "input_sha256": pre["input_sha256"]}
    atomic_json(outdir / "summary.json", summary)
    atomic_json(runmark, {"status": "completed", "planned_new_fits": 264,
                          "frozen_selections_sha256": selection_payload["sha256"]})


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=HERE / "budget_nonoverlap_v1")
    a = ap.parse_args()
    out = a.output_dir.resolve()
    make_preflight(out) if a.preflight else run(out)


if __name__ == "__main__":
    main()
