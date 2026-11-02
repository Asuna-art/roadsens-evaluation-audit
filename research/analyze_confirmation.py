"""Validate and summarize the RoadSens seed-confirmation runs without loading models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FIRST_PASS = RUNS / "pilot_v2"
CACHE_PATH = ROOT / "session_feature_cache_v1.joblib"
POLICIES = ("primary", "blank_normal")
SPLITS = ("random", "group")
METHODS = ("full_numeric", "missing_only", "context_only", "imu_only")
FIRST_PASS_METHODS = METHODS + ("physics_window", "physics_calibrated")
CLASSES = ("Normal", "Pothole", "Bump")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _same_rows(actual: list[dict], indices: np.ndarray, groups: np.ndarray,
               source_rows: np.ndarray) -> bool:
    if len(actual) != len(indices):
        return False
    expected = [(int(i), str(groups[i]), int(source_rows[i])) for i in indices]
    observed = [(int(r["row"]), str(r["session"]), int(r["source_row"])) for r in actual]
    return observed == expected


def _labels_for(sessions: list[dict], policy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, group_parts, source_parts, owner_parts = [], [], [], []
    for owner, s in enumerate(sessions):
        labels = s["labels"] if policy == "primary" else s["labels_blank_normal"]
        valid = np.isfinite(s["times"]) & np.isin(labels, CLASSES)
        ys.extend(labels[valid].tolist())
        group_parts.extend([s["session_key"]] * int(valid.sum()))
        source_parts.extend(s["source_row"][valid].astype(int).tolist())
        owner_parts.extend([owner] * int(valid.sum()))
    return (np.asarray(ys, dtype=object), np.asarray(group_parts, dtype=object),
            np.asarray(source_parts, dtype=np.int64), np.asarray(owner_parts, dtype=np.int64))


def _rows_for(indices: np.ndarray, groups: np.ndarray, source_rows: np.ndarray) -> list[dict]:
    return [{"row": int(i), "session": str(groups[i]), "source_row": int(source_rows[i])}
            for i in indices]


def _cap(train: np.ndarray, y: np.ndarray, fold: int, seed: int, cap: int = 20000) -> np.ndarray:
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


def _folds(y: np.ndarray, groups: np.ndarray, split: str, seed: int,
           group_map: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    if split == "random":
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    if set(np.unique(groups)) - set(group_map):
        raise ValueError("fold map misses one or more scored session keys")
    result = []
    for fold in range(1, 6):
        test = np.flatnonzero(np.fromiter((group_map[g] == fold for g in groups),
                                         dtype=bool, count=len(groups)))
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        result.append((train, test))
    return result


def _read_result(run_dir: Path, policy: str, split: str) -> dict:
    path = run_dir / f"{policy}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("policy") != policy or payload.get("split") != split:
        raise ValueError(f"result header mismatch: {path}")
    return payload


def _validate_predictions(fold: dict, y: np.ndarray, groups: np.ndarray,
                          source_rows: np.ndarray, expected_test: np.ndarray,
                          label: str) -> None:
    preds = fold.get("predictions")
    if not isinstance(preds, list) or len(preds) != len(expected_test):
        raise ValueError(f"{label}: predictions do not cover the test rows")
    expected = _rows_for(expected_test, groups, source_rows)
    observed = [{"row": int(p["row"]), "session": str(p["session"]),
                 "source_row": int(p["source_row"])} for p in preds]
    if observed != expected:
        raise ValueError(f"{label}: prediction row/session/source_row sequence differs from split")
    y_true = []
    y_pred = []
    for p, idx in zip(preds, expected_test):
        if str(p.get("true")) != str(y[idx]):
            raise ValueError(f"{label}: prediction truth differs from cached source label at row {idx}")
        if str(p.get("pred")) not in CLASSES:
            raise ValueError(f"{label}: invalid predicted class")
        y_true.append(str(p["true"]))
        y_pred.append(str(p["pred"]))
    recomputed = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_by_class": dict(zip(CLASSES, map(float, recall_score(
            y_true, y_pred, labels=CLASSES, average=None, zero_division=0)))),
    }
    for key in ("macro_f1", "balanced_accuracy"):
        if not math.isclose(float(fold[key]), recomputed[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{label}: recorded {key} differs from predictions")
    for klass in CLASSES:
        if not math.isclose(float(fold["recall_by_class"][klass]),
                            recomputed["recall_by_class"][klass], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{label}: recorded {klass} recall differs from predictions")


def _validate_result_set(seed: int, run_dir: Path, sessions: list[dict], map_payload: dict,
                         required_methods: tuple[str, ...]) -> tuple[dict, list[str]]:
    fmap = map_payload["session_fold"]
    if set(fmap) != {s["session_key"] for s in sessions}:
        raise ValueError(f"seed {seed}: map sessions differ from feature cache")
    for s in sessions:
        if map_payload["source_sha256"].get(s["session_key"]) != s["sha256"]:
            raise ValueError(f"seed {seed}: source hash mismatch in map for {s['session_key']}")
        source_path = ROOT / "data" / "inner" / Path(s["session_key"])
        if not source_path.is_file() or sha256(source_path) != s["sha256"]:
            raise ValueError(f"seed {seed}: live source file changed for {s['session_key']}")
    results = {}
    issues = []
    for policy in POLICIES:
        y, groups, source_rows, _ = _labels_for(sessions, policy)
        folds_by_split = {split: _folds(y, groups, split, seed, fmap) for split in SPLITS}
        for split in SPLITS:
            payload = _read_result(run_dir, policy, split)
            methods = payload.get("methods", {})
            if not set(required_methods).issubset(methods):
                raise ValueError(f"seed {seed} {policy}/{split}: required methods missing")
            for method in required_methods:
                method_folds = sorted(methods[method], key=lambda f: int(f["fold"]))
                if len(method_folds) != 5 or [int(f["fold"]) for f in method_folds] != list(range(5)):
                    raise ValueError(f"seed {seed} {policy}/{split}/{method}: expected folds 0..4")
                for fold_idx, (record, (train, test)) in enumerate(zip(method_folds, folds_by_split[split])):
                    tag = f"seed{seed}/{policy}/{split}/{method}/fold{fold_idx + 1}"
                    if not _same_rows(record.get("split_train_rows", []), train, groups, source_rows):
                        raise ValueError(f"{tag}: split training row/session/source mapping mismatch")
                    if not _same_rows(record.get("test_rows", []), test, groups, source_rows):
                        raise ValueError(f"{tag}: test row/session/source mapping mismatch")
                    fit = _cap(train, y, fold_idx, seed)
                    if not _same_rows(record.get("fit_train_rows", []), fit, groups, source_rows):
                        raise ValueError(f"{tag}: fitted training rows differ from frozen seed sampler")
                    if any(int(record.get("test_class_counts", {}).get(c, 0)) != int(np.sum(y[test] == c))
                           for c in CLASSES):
                        raise ValueError(f"{tag}: test class counts mismatch cached labels")
                    if any(int(record.get("train_class_counts", {}).get(c, 0)) != int(np.sum(y[fit] == c))
                           for c in CLASSES):
                        raise ValueError(f"{tag}: fitted train class counts mismatch sampler")
                    if split == "group":
                        tr_sessions = set(groups[train]); te_sessions = set(groups[test])
                        if tr_sessions & te_sessions:
                            raise ValueError(f"{tag}: group train/test session overlap")
                        if any(fmap[g] != fold_idx + 1 for g in te_sessions):
                            raise ValueError(f"{tag}: test session violates seed map")
                    if set(np.unique(y[train])) != set(CLASSES) or set(np.unique(y[test])) != set(CLASSES):
                        raise ValueError(f"{tag}: train or test lacks three-class support")
                    _validate_predictions(record, y, groups, source_rows, test, tag)
                # Discard predictions and row lists after validation; retain only compact metrics.
                results[(policy, split, method)] = [
                    {"fold": int(f["fold"]), "macro_f1": float(f["macro_f1"]),
                     "balanced_accuracy": float(f["balanced_accuracy"]),
                     "recall_by_class": {c: float(f["recall_by_class"][c]) for c in CLASSES}}
                    for f in method_folds]
            # Verify every method used exactly the same partition and fit subset.
            baseline = methods[required_methods[0]]
            for method in required_methods[1:]:
                other = methods[method]
                for a, b in zip(baseline, other):
                    for key in ("split_train_rows", "fit_train_rows", "test_rows"):
                        if a[key] != b[key]:
                            raise ValueError(f"seed {seed} {policy}/{split}/fold{a['fold']}: {key} differ across methods")
    return results, issues


def _validate_run_metadata(seed: int, run_dir: Path, cache: dict, cache_path: Path,
                           expected_map: dict | None = None) -> tuple[dict, dict]:
    if seed == 42:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = json.loads((ROOT / "session_fold_map_v1.json").read_text(encoding="utf-8"))
        if manifest.get("session_fold_map_sha256") != sha256(ROOT / "session_fold_map_v1.json"):
            raise ValueError("seed42 manifest map hash mismatch")
        if manifest.get("frozen_session_fold") != frozen.get("session_fold"):
            raise ValueError("seed42 manifest mapping mismatch")
        file_hashes = {Path(s["path"]).name: s["sha256"] for s in manifest.get("source_files", [])}
        for name in ("pilot_protocol.md", "pre_run_clarifications.md", "fold_feasibility_amendment.md",
                     "session_fold_map_v1.json", "run_pilot.py"):
            if file_hashes.get(name) != sha256(ROOT / name):
                raise ValueError(f"seed42 manifest source hash mismatch: {name}")
        map_payload = frozen
    else:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("seed") != seed:
            raise ValueError(f"seed{seed} confirmation manifest is not complete")
        if manifest.get("feature_cache_sha256") != sha256(cache_path):
            raise ValueError(f"seed{seed} feature cache hash mismatch")
        if manifest.get("parent_runner_sha256") != cache.get("runner_sha256"):
            raise ValueError(f"seed{seed} parent runner hash mismatch")
        if manifest.get("confirmation_driver_sha256") != sha256(ROOT / "confirm_seeds.py"):
            raise ValueError(f"seed{seed} confirmation driver hash mismatch")
        if manifest.get("confirmation_plan_sha256") != sha256(ROOT / "confirmation_plan.md"):
            raise ValueError(f"seed{seed} confirmation plan hash mismatch")
        map_path = run_dir / manifest["session_fold_map_file"]
        if manifest.get("session_fold_map_sha256") != sha256(map_path):
            raise ValueError(f"seed{seed} seed-map file hash mismatch")
        map_payload = json.loads(map_path.read_text(encoding="utf-8"))
        if map_payload.get("seed") != seed or map_payload.get("session_fold") != expected_map.get("session_fold"):
            raise ValueError(f"seed{seed} confirmation map disagrees with deterministic generator")
        if manifest.get("source_sha256") != cache.get("source_sha256"):
            raise ValueError(f"seed{seed} manifest source hashes differ from feature cache")
    return map_payload, manifest


def _fold_metric(folds: list[dict], key: str, klass: str | None = None) -> list[float]:
    return [float(f[key] if klass is None else f[key][klass]) for f in folds]


def _gate(results: dict) -> dict:
    by_policy = {}
    for policy in POLICIES:
        by_seed = {}
        for seed in SEEDS:
            rand_full = results[(seed, policy, "random", "full_numeric")]
            group_full = results[(seed, policy, "group", "full_numeric")]
            deltas = [a - b for a, b in zip(_fold_metric(rand_full, "macro_f1"),
                                            _fold_metric(group_full, "macro_f1"))]
            contexts = results[(seed, policy, "random", "context_only")]
            missings = results[(seed, policy, "random", "missing_only")]
            context_mean = mean(_fold_metric(contexts, "macro_f1"))
            missing_mean = mean(_fold_metric(missings, "macro_f1"))
            passed = mean(deltas) >= 0.10 and sum(d > 0 for d in deltas) >= 4 and max(context_mean, missing_mean) >= 0.65
            by_seed[str(seed)] = {"status": "pass" if passed else "fail",
                                  "random_full_minus_group_full": {"per_fold": deltas,
                                                                   "mean": mean(deltas),
                                                                   "positive_folds": sum(d > 0 for d in deltas)},
                                  "random_context_mean_macro_f1": context_mean,
                                  "random_missing_mean_macro_f1": missing_mean,
                                  "non_inertial_threshold_met": max(context_mean, missing_mean) >= 0.65}
        pass_seeds = [s for s, item in by_seed.items() if item["status"] == "pass"]
        directions = {s: by_seed[s]["random_full_minus_group_full"]["mean"] > 0 for s in by_seed}
        by_policy[policy] = {"by_seed": by_seed, "passing_seeds": pass_seeds,
                             "two_of_three_met": len(pass_seeds) >= 2,
                             "all_three_mean_directions_positive": all(directions.values()),
                             "direction_by_seed": directions}
    both_policy_seeds = [str(seed) for seed in SEEDS if all(
        by_policy[p]["by_seed"][str(seed)]["status"] == "pass" for p in POLICIES)]
    overall_pass = len(both_policy_seeds) >= 2 and all(
        by_policy[p]["all_three_mean_directions_positive"] for p in POLICIES)
    return {"status": "confirmation_rule_met" if overall_pass else "confirmation_rule_not_met",
            "per_policy": by_policy,
            "seeds_passing_both_policies": both_policy_seeds,
            "two_of_three_seed_rule_met_in_both_policies": all(by_policy[p]["two_of_three_met"] for p in POLICIES),
            "all_three_seed_directions_positive_in_both_policies": all(
                by_policy[p]["all_three_mean_directions_positive"] for p in POLICIES),
            "confirmation_rule_met": overall_pass,
            "note": "Seed 42 is the initial run; seeds 43/44 are stability repetitions, not independent external validation. Fold-number differences are screening comparisons, not paired significance tests."}


def analyze(first_pass_dir: Path = FIRST_PASS, cache_path: Path = CACHE_PATH,
            confirmation_dirs: dict[int, Path] | None = None) -> dict:
    sys.path.insert(0, str(ROOT))
    import joblib
    import confirm_seeds
    cache = joblib.load(cache_path)
    if cache.get("runner_sha256") != sha256(ROOT / "run_pilot.py"):
        raise ValueError("feature cache parent code hash mismatch")
    seed_dirs = confirmation_dirs or {43: RUNS / "confirmation_seed43_v1",
                                      44: RUNS / "confirmation_seed44_v1"}
    run_dirs = {42: first_pass_dir, **seed_dirs}
    results = {}
    validation = {}
    coverage = {}
    for seed in SEEDS:
        run_dir = run_dirs[seed]
        if not run_dir.exists():
            coverage[str(seed)] = {"status": "missing", "validated_folds": 0, "expected_folds": 80 if seed != 42 else 120}
            continue
        if seed == 42:
            map_payload, manifest = _validate_run_metadata(seed, run_dir, cache, cache_path)
            required_methods = FIRST_PASS_METHODS
        else:
            generated = confirm_seeds.build_seed_fold_map(cache["sessions"], cache["source_sha256"],
                                                          seed, json.loads((ROOT / "session_fold_map_v1.json").read_text(encoding="utf-8"))["session_fold"])
            map_payload, manifest = _validate_run_metadata(seed, run_dir, cache, cache_path, generated)
            required_methods = METHODS
        validated, issues = _validate_result_set(seed, run_dir, cache["sessions"], map_payload, required_methods)
        expected_count = 120 if seed == 42 else 80
        actual_count = len(validated) * 5
        coverage[str(seed)] = {"status": "complete" if actual_count == expected_count else "partial",
                               "validated_method_folds": actual_count,
                               "expected_method_folds": expected_count,
                               "issues": issues}
        for (policy, split, method), folds in validated.items():
            results[(seed, policy, split, method)] = folds
        validation[str(seed)] = {"manifest_status": manifest.get("status", "first_pass"),
                                 "map_sessions": len(map_payload["session_fold"]),
                                 "source_hashes_verified": len(map_payload["source_sha256"]),
                                 "row_maps_and_predictions_validated": actual_count}

    complete = all(coverage.get(str(s), {}).get("status") == "complete" for s in SEEDS)
    gate = _gate(results) if complete else {"status": "not_evaluable_incomplete",
                                            "confirmation_rule_met": False,
                                            "note": "All 120 seed-42 and 80-fold seed-43/44 result records are required."}
    return {"status": "complete" if complete else "partial",
            "coverage": coverage, "validation": validation,
            "seed_metrics": _summarize(results), "confirmation_gate": gate,
            "interpretation": "A group fold holds out a whole source CSV recording. A CSV is the operational session unit; it is not independently verified as a distinct trip, drive, road, or route. Confirmation is a stability check across resplits of one dataset and does not establish new-road or external generalization, or paper readiness."}


def _summarize(results: dict) -> dict:
    out = {}
    for seed in SEEDS:
        for policy in POLICIES:
            for split in SPLITS:
                for method in METHODS:
                    folds = results.get((seed, policy, split, method))
                    if not folds:
                        continue
                    out.setdefault(str(seed), {}).setdefault(policy, {}).setdefault(split, {})[method] = {
                        "mean_macro_f1": mean(_fold_metric(folds, "macro_f1")),
                        "mean_balanced_accuracy": mean(_fold_metric(folds, "balanced_accuracy")),
                        "mean_recall_by_class": {c: mean(_fold_metric(folds, "recall_by_class", c)) for c in CLASSES},
                        "fold_macro_f1": _fold_metric(folds, "macro_f1"),
                    }
    return out


def render_markdown(payload: dict) -> str:
    lines = ["# RoadSens seed confirmation audit", "", f"Status: **{payload['status']}**.", "",
             "## Coverage and validation", "", "| Seed | Status | Validated folds | Expected folds |", "|---:|---|---:|---:|"]
    for seed in SEEDS:
        item = payload["coverage"].get(str(seed), {"status": "missing", "validated_method_folds": 0,
                                                  "expected_method_folds": 120 if seed == 42 else 80})
        lines.append(f"| {seed} | {item['status']} | {item.get('validated_method_folds', 0)} | {item.get('expected_method_folds', 120 if seed == 42 else 80)} |")
    if payload["status"] != "complete":
        lines += ["", "Gate adjudication is withheld until all seed 42/43/44 results are complete and validated."]
    lines += ["", "## Track A per policy and seed", "",
              "| Policy | Seed | Mean random−group full macro-F1 | Positive folds | Context random mean | Missing random mean | Gate |",
              "|---|---:|---:|---:|---:|---:|---|"]
    gate = payload["confirmation_gate"]
    for policy, row in gate.get("per_policy", {}).items():
        for seed, item in row["by_seed"].items():
            d = item["random_full_minus_group_full"]
            lines.append(f"| {policy} | {seed} | {d['mean']:+.3f} | {d['positive_folds']}/5 | "
                         f"{item['random_context_mean_macro_f1']:.3f} | {item['random_missing_mean_macro_f1']:.3f} | {item['status']} |")
    lines += ["", "## Confirmation rule", ""]
    if gate.get("status") == "not_evaluable_incomplete":
        lines.append("Not evaluable because the required seed results are incomplete.")
    else:
        lines.append(f"Overall: **{gate['status']}**. Seeds passing both policies: "
                     f"{', '.join(gate['seeds_passing_both_policies']) or 'none'}.")
        for policy, row in gate["per_policy"].items():
            lines.append(f"- {policy}: passing seeds {', '.join(row['passing_seeds']) or 'none'}; "
                         f"at least two of three: {row['two_of_three_met']}; all three mean differences positive: "
                         f"{row['all_three_mean_directions_positive']}.")
    lines += ["", payload["interpretation"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass-dir", type=Path, default=FIRST_PASS)
    parser.add_argument("--seed43-dir", type=Path, default=RUNS / "confirmation_seed43_v1")
    parser.add_argument("--seed44-dir", type=Path, default=RUNS / "confirmation_seed44_v1")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    payload = analyze(args.first_pass_dir, args.cache, {43: args.seed43_dir, 44: args.seed44_dir})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "confirmation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                                               encoding="utf-8")
    (args.out_dir / "confirmation_summary.md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote confirmation_summary.json/md; status={payload['status']}; "
          f"gate={payload['confirmation_gate']['status']}.")


if __name__ == "__main__":
    main()
