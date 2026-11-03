#!/usr/bin/env python
"""Execute only the frozen KIT splits from a reviewed preflight. Requires --run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

BASE = Path(r"D:\三区灌水专用\research_20261015\candidate_roadsens\paper_expansion")
HERE = BASE / "engineering_validation_20261103"
OUT = HERE / "run_kit_v1"
PROTOCOL = HERE / "kit_protocol_v1.json"
FEATURES = HERE / "features_kit_v1.npz"
RAW = HERE / "kit_audit" / "raw_arrays_verified.npz"
PREFLIGHT = OUT / "preflight.json"
SPLITS = OUT / "split_map.npz"
HISTORIC = BASE / "external_pvs_v1" / "run_nested_evaluation.py"
EXPECTED_PROTOCOL_SHA = "2782b0f43fd442a408f9acb8e6521cf7e64432abec45172a800c3983bb9d9105"
EXPECTED_CANDIDATES = ["rf_sensor", "rf_sensor_context", "hgb_sensor", "hgb_sensor_context"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_historical():
    spec = importlib.util.spec_from_file_location("kit_historical_nested", HISTORIC)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen historical evaluator: {HISTORIC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    candidates = list(mod.CANDIDATES)
    if [c["id"] for c in candidates] != EXPECTED_CANDIDATES:
        raise ValueError("historical candidate order differs from frozen KIT protocol")
    return mod, candidates


def validate_hashes(pre: dict) -> None:
    if sha(PROTOCOL) != EXPECTED_PROTOCOL_SHA:
        raise ValueError("protocol SHA256 mismatch")
    expected = {
        "protocol_sha256": sha(PROTOCOL),
        "features_sha256": sha(FEATURES),
        "raw_sha256": sha(RAW),
    }
    for key, actual in expected.items():
        if pre.get(key) != actual:
            raise ValueError(f"preflight {key} mismatch (expected {actual}, found {pre.get(key)})")
    pre_script_hash = pre.get("script_sha256")
    if not pre_script_hash:
        raise ValueError("preflight does not record its script_sha256")
    script_path = Path(pre.get("script_path", ""))
    if not script_path.is_file():
        script_path = HERE / "kit_nested_runner.py"
    if not script_path.is_file():
        # Older lead-generated preflight may place this in provenance instead.
        prov = OUT / "provenance.json"
        if prov.is_file():
            pobj = json.loads(prov.read_text(encoding="utf-8"))
            script_path = Path(pobj.get("script", ""))
            pre_script_hash = pobj.get("script_sha256", pre_script_hash)
    if not script_path.is_file() or sha(script_path) != pre_script_hash:
        raise ValueError("preflight script source/hash cannot be verified")


def load_inputs():
    pre = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if pre.get("status") != "passed":
        raise ValueError("preflight status is not passed")
    validate_hashes(pre)
    if sha(SPLITS) != pre.get("split_map_sha256"):
        raise ValueError("Frozen split-map hash differs from preflight")
    with np.load(FEATURES, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    data['_sensor_indices'] = np.arange(30)
    data['_context_indices'] = np.arange(3)
    required = ("X_sensor", "X_context", "y")
    if any(k not in data for k in required):
        raise ValueError("features file lacks model arrays")
    if data["X_sensor"].shape != (len(data["y"]), 30) or data["X_context"].shape != (len(data["y"]), 3):
        raise ValueError("feature matrix dimensions mismatch")
    with np.load(SPLITS, allow_pickle=False) as z:
        splits = {k: np.asarray(z[k], dtype=np.int64) for k in z.files}
    expected_keys = {"outer_train", "outer_test"}
    for i in range(4):
        expected_keys.update({f"inner_{i}_train", f"inner_{i}_fit", f"inner_{i}_validation"})
    if not expected_keys.issubset(splits):
        raise ValueError(f"split map missing fields: {sorted(expected_keys - set(splits))}")
    if np.any(splits["outer_train"] < 0) or np.any(splits["outer_test"] >= len(data["y"])):
        raise ValueError("split indices outside feature rows")
    return pre, data, splits


def fit_save(mod, candidate, train_fit, test, data, fit_id, phase, root):
    y = data["y"].astype(np.int64)
    model = mod.make_model(candidate)
    model.fit(mod.matrix_for(candidate, train_fit, data), y[train_fit])
    pred = model.predict(mod.matrix_for(candidate, test, data)).astype(np.int64)
    fitdir = root / "fits" / fit_id
    fitdir.mkdir(parents=True, exist_ok=False)
    model_path = fitdir / "model.joblib"
    pred_path = fitdir / "predictions.npz"
    joblib.dump(model, model_path, compress=3)
    np.savez_compressed(pred_path, test_indices=test.astype(np.int64), y_true=y[test],
                        y_pred=pred, train_indices_fit=train_fit.astype(np.int64))
    loaded = joblib.load(model_path)
    check = loaded.predict(mod.matrix_for(candidate, test, data)).astype(np.int64)
    if not np.array_equal(pred, check):
        raise AssertionError(f"reload prediction mismatch: {fit_id}")
    record = {
        "fit_id": fit_id, "phase": phase, "candidate": candidate,
        "train_rows_fit": int(len(train_fit)), "test_rows": int(len(test)),
        "train_indices_fit": train_fit.astype(int).tolist(), "test_indices": test.astype(int).tolist(),
        "metrics": mod.metrics(y[test], pred), "reload_predictions_exact": True,
        "model_path": str(model_path.resolve()), "prediction_path": str(pred_path.resolve()),
        "model_sha256": sha(model_path), "prediction_sha256": sha(pred_path),
    }
    write_json(fitdir / "fit.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="perform the frozen 24 fits")
    args = parser.parse_args()
    if not args.run:
        print("Fit-free. Supply --run only after reviewing the preflight and split map.")
        return
    if not PREFLIGHT.is_file() or not SPLITS.is_file():
        raise FileNotFoundError("lead-generated preflight.json and split_map.npz are required")
    # Refuse all partial or completed output to preserve append-only provenance.
    if (OUT / "selection.json").exists() or (OUT / "results.json").exists() or (OUT / "fits").exists():
        raise FileExistsError("refusing to run: selection.json, results.json, or fits/ already exists")
    pre, data, splits = load_inputs()
    mod, candidates = load_historical()
    root = OUT
    fit_rows = [splits[f"inner_{i}_fit"] for i in range(4)]
    vals = [splits[f"inner_{i}_validation"] for i in range(4)]
    trains = [splits[f"inner_{i}_train"] for i in range(4)]
    start = datetime.now(timezone.utc).isoformat()
    records = []
    oof = {i: {} for i in range(4)}
    with threadpool_limits(limits=4):
        for i in range(4):
            for cand in candidates:
                fid = f"inner_{i}_{cand['id']}"
                rec = fit_save(mod, cand, fit_rows[i], vals[i], data, fid, "inner", root)
                records.append(rec)
                with np.load(root / "fits" / fid / "predictions.npz", allow_pickle=False) as p:
                    oof[i][cand["id"]] = (p["test_indices"], p["y_pred"])
    selector_specs = (("row_random_two_fold", (0, 1)), ("date_buffered_two_fold", (2, 3)))
    selectors = {}
    y = data["y"].astype(np.int64)
    for name, positions in selector_specs:
        scores = {}
        for cand in candidates:
            ix = np.concatenate([oof[j][cand["id"]][0] for j in positions])
            pred = np.concatenate([oof[j][cand["id"]][1] for j in positions])
            order = np.argsort(ix, kind="stable")
            if len(np.unique(ix)) != len(ix):
                raise AssertionError(f"{name} OOF validation memberships overlap")
            scores[cand["id"]] = float(mod.metrics(y[ix[order]], pred[order])["macro_f1"])
        chosen = max((c["id"] for c in candidates), key=lambda cid: scores[cid])
        selectors[name] = {"pooled_inner_oof_macro_f1": scores, "selected_candidate_id": chosen}
    write_json(root / "selection.json", {"selectors": selectors, "candidate_order": EXPECTED_CANDIDATES,
        "selection_rule": "maximum pooled inner OOF macro-F1; fixed candidate order resolves ties",
        "outer_predictions_not_yet_generated": True})
    outer_train = splits["outer_train"]
    outer_test = splits["outer_test"]
    if len(outer_train) > 20_000:
        raise ValueError("full-development outer training pool exceeds frozen 20,000-row cap")
    budget = pre["common_inner_per_class_budget"]
    rng = np.random.default_rng(46)
    common_outer = np.sort(np.concatenate([rng.choice(outer_train[y[outer_train] == c], int(budget[c]), replace=False)
                                           for c in (0, 1)]).astype(np.int64))
    targets = {"full_development": outer_train, "common_inner_budget": common_outer}
    outer_records = {}
    with threadpool_limits(limits=4):
        for target, tr in targets.items():
            outer_records[target] = {}
            for cand in candidates:
                fid = f"outer_{target}_{cand['id']}"
                rec = fit_save(mod, cand, tr, outer_test, data, fid, target, root)
                records.append(rec)
                outer_records[target][cand["id"]] = rec
    results = {"selectors": {}, "fit_count": len(records), "expected_fit_count": 24,
        "started_utc": start, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {"protocol": sha(PROTOCOL), "features": sha(FEATURES), "raw": sha(RAW),
                          "preflight": sha(PREFLIGHT), "split_map": sha(SPLITS), "historical_script": sha(HISTORIC),
                          "fit_script": sha(Path(__file__))},
        "python": sys.version, "platform": platform.platform()}
    if len(records) != 24:
        raise AssertionError(f"expected 24 fits, observed {len(records)}")
    for selector, info in selectors.items():
        chosen = info["selected_candidate_id"]
        results["selectors"][selector] = {"candidate_id": chosen, "targets": {}}
        for target, score_key in (("full_development", "full_development"), ("common_inner_budget", "common_inner_budget")):
            metric = outer_records[score_key][chosen]["metrics"]
            inner = info["pooled_inner_oof_macro_f1"][chosen]
            results["selectors"][selector]["targets"][target] = {
                "inner_macro_f1": inner, "outer_test_metrics": metric,
                "absolute_error": abs(inner - metric["macro_f1"]),
                "candidate_id": chosen,
                "prediction_path": outer_records[score_key][chosen]["prediction_path"]}
    write_json(root / "fit_records.json", {"records": records, "fit_count": len(records)})
    write_json(root / "results.json", results)


if __name__ == "__main__":
    main()
