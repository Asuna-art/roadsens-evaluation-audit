"""Read-only validation and compact screening summary for RoadSens pilot outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent
POLICIES = ("primary", "blank_normal")
SPLITS = ("random", "group")
METHODS = ("full_numeric", "missing_only", "context_only", "imu_only",
           "physics_window", "physics_calibrated")
CLASSES = ("Normal", "Pothole", "Bump")
REQUIRED_HASH_FILES = ("pilot_protocol.md", "pre_run_clarifications.md",
                       "fold_feasibility_amendment.md", "session_fold_map_v1.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _close(a: Any, b: Any) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(
        float(a), float(b), rel_tol=0, abs_tol=1e-12)


def _canon_rows(rows: list[dict]) -> list[tuple]:
    return sorted((int(r["row"]), str(r["session"]), int(r.get("source_row", -1))) for r in rows)


def _metrics(fold: dict) -> dict:
    needed = ("macro_f1", "balanced_accuracy", "recall_by_class")
    if any(k not in fold for k in needed):
        raise ValueError("fold result missing required metrics")
    if any(not isinstance(fold["recall_by_class"].get(c), (int, float)) or
           not math.isfinite(float(fold["recall_by_class"][c])) for c in CLASSES):
        raise ValueError("fold result missing finite class recalls")
    values = [fold["macro_f1"], fold["balanced_accuracy"],
              *(fold["recall_by_class"][c] for c in CLASSES)]
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values):
        raise ValueError("non-finite metric in fold result")
    if any(float(v) < 0 or float(v) > 1 for v in values):
        raise ValueError("metric outside [0,1]")
    return {"macro_f1": float(fold["macro_f1"]),
            "balanced_accuracy": float(fold["balanced_accuracy"]),
            "recall_by_class": {c: float(fold["recall_by_class"][c]) for c in CLASSES}}


def _validate_manifest(run_dir: Path, pilot_dir: Path, map_payload: dict) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise ValueError("manifest.json is required for protocol/source hash validation")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source_files = {Path(item["path"]).name: item["sha256"]
                    for item in manifest.get("source_files", [])}
    checked = {}
    for name in REQUIRED_HASH_FILES:
        current = sha256(pilot_dir / name)
        declared = source_files.get(name)
        if not declared or declared != current:
            raise ValueError(f"protocol/map hash mismatch or missing in manifest: {name}")
        checked[name] = current
    map_hash = sha256(pilot_dir / "session_fold_map_v1.json")
    if manifest.get("session_fold_map_sha256") != map_hash:
        raise ValueError("manifest frozen session fold map hash mismatch")
    if manifest.get("frozen_session_fold") != map_payload.get("session_fold"):
        raise ValueError("manifest session fold mapping differs from frozen map")
    sessions = manifest.get("sessions", [])
    session_hashes = {s.get("session_key"): s.get("sha256") for s in sessions}
    if session_hashes != map_payload.get("source_sha256"):
        raise ValueError("manifest session source hashes/keys differ from frozen map")
    data_root = pilot_dir / "data" / "inner"
    for key, expected in map_payload["source_sha256"].items():
        source = data_root / Path(key)
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"live source hash mismatch: {key}")
    return {"passed": True, "protocol_hashes": checked, "session_count": len(session_hashes),
            "source_hashes_verified": len(session_hashes), "fold_map_sha256": map_hash}


def _load_inputs(run_dir: Path) -> tuple[dict, bool, list[str]]:
    """Return keyed method-folds, whether all four final files exist, and input names."""
    records: dict[tuple[str, str, str], list[dict]] = {}
    complete_files = True
    input_names = []
    for policy in POLICIES:
        for split in SPLITS:
            stem = f"{policy}_{split}"
            final_path = run_dir / f"{stem}.json"
            if final_path.exists():
                input_names.append(final_path.name)
                payload = json.loads(final_path.read_text(encoding="utf-8"))
                if payload.get("policy") != policy or payload.get("split") != split:
                    raise ValueError(f"policy/split header mismatch: {final_path.name}")
                methods_obj = payload.get("methods", {})
                for method, folds in methods_obj.items():
                    if method not in METHODS:
                        raise ValueError(f"unexpected method: {method}")
                    records[(policy, split, method)] = folds
            else:
                complete_files = False
    # Method checkpoints can contribute partial method results; final JSON wins when present.
    for checkpoint in sorted(run_dir.glob("*.checkpoint.json")):
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        policy, split, method = payload.get("policy"), payload.get("split"), payload.get("method")
        if policy not in POLICIES or split not in SPLITS or method not in METHODS:
            continue
        input_names.append(checkpoint.name)
        key = (policy, split, method)
        if key not in records:
            records[key] = payload.get("folds", [])
    return records, complete_files, input_names


def _validate_records(records: dict, map_payload: dict) -> tuple[dict, list[str]]:
    fmap = map_payload["session_fold"]
    issues: list[str] = []
    checked = {}
    for (policy, split, method), folds in records.items():
        if len(folds) != 5:
            issues.append(f"{policy}/{split}/{method}: expected 5 folds, found {len(folds)}")
            continue
        ordered = sorted(folds, key=lambda f: int(f["fold"]))
        if [int(f["fold"]) for f in ordered] != list(range(5)):
            issues.append(f"{policy}/{split}/{method}: fold IDs must be 0..4")
            continue
        checked[(policy, split, method)] = ordered
        for fold in ordered:
            _metrics(fold)
            if any(int(fold.get("test_class_counts", {}).get(c, 0)) <= 0 for c in CLASSES):
                issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: test lacks class support")
            if any(int(fold.get("train_class_counts", {}).get(c, 0)) <= 0 for c in CLASSES):
                issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: fit train lacks class support")
            train_rows, fit_rows, test_rows = (fold.get("split_train_rows"), fold.get("fit_train_rows"),
                                                fold.get("test_rows"))
            if not isinstance(train_rows, list) or not isinstance(fit_rows, list) or not isinstance(test_rows, list):
                issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: split row audit absent")
                continue
            train_ids = {int(r["row"]) for r in train_rows}
            fit_ids = {int(r["row"]) for r in fit_rows}
            test_ids = {int(r["row"]) for r in test_rows}
            if len(train_ids) != len(train_rows) or len(fit_ids) != len(fit_rows) or len(test_ids) != len(test_rows):
                issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: duplicate row IDs")
            if train_ids & test_ids or not fit_ids <= train_ids:
                issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: train/fit/test row mismatch")
            train_sessions = {str(r["session"]) for r in train_rows}
            fit_sessions = {str(r["session"]) for r in fit_rows}
            test_sessions = {str(r["session"]) for r in test_rows}
            if not fit_sessions <= train_sessions:
                issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: fit session outside train")
            if split == "group":
                if train_sessions & test_sessions:
                    issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: train/test session overlap")
                for session in test_sessions:
                    if session not in fmap or fmap[session] != int(fold["fold"]) + 1:
                        issues.append(f"{policy}/{split}/{method}/fold{fold['fold'] + 1}: test session violates frozen map")
        # metrics and row consistency across methods are checked below.
    for policy in POLICIES:
        for split in SPLITS:
            present = [(m, checked[(policy, split, m)]) for m in METHODS
                       if (policy, split, m) in checked]
            if not present:
                continue
            base_method, base_folds = present[0]
            for method, folds in present[1:]:
                for index, (base, other) in enumerate(zip(base_folds, folds), 1):
                    if _canon_rows(base["split_train_rows"]) != _canon_rows(other["split_train_rows"]):
                        issues.append(f"{policy}/{split}/fold{index}: split train rows differ {base_method}/{method}")
                    if _canon_rows(base["fit_train_rows"]) != _canon_rows(other["fit_train_rows"]):
                        issues.append(f"{policy}/{split}/fold{index}: fit train rows differ {base_method}/{method}")
                    if _canon_rows(base["test_rows"]) != _canon_rows(other["test_rows"]):
                        issues.append(f"{policy}/{split}/fold{index}: test rows differ {base_method}/{method}")
    return checked, issues


def _mean_metric(folds: list[dict], key: str, klass: str | None = None) -> float:
    if klass is None:
        return mean(float(f[key]) for f in folds)
    return mean(float(f[key][klass]) for f in folds)


def _difference(left: list[dict], right: list[dict], field: str,
                klass: str | None = None) -> dict:
    vals = []
    for a, b in zip(left, right):
        av = a[field] if klass is None else a[field][klass]
        bv = b[field] if klass is None else b[field][klass]
        vals.append(float(av) - float(bv))
    return {"per_fold": vals, "mean": mean(vals), "positive_folds": sum(v > 0 for v in vals)}


def _screen_tracks(checked: dict, all_full_files: bool, all_120: bool) -> dict:
    result = {"track_A": {"status": "not_evaluable_incomplete"},
              "track_B": {"status": "not_evaluable_incomplete", "by_policy": {}}}
    if not all_full_files or not all_120:
        return result
    a_by_policy = {}
    for policy in POLICIES:
        rand = checked[(policy, "random", "full_numeric")]
        group = checked[(policy, "group", "full_numeric")]
        delta = _difference(rand, group, "macro_f1")
        context = checked[(policy, "random", "context_only")]
        missing = checked[(policy, "random", "missing_only")]
        context_mean = _mean_metric(context, "macro_f1")
        missing_mean = _mean_metric(missing, "macro_f1")
        audit_signal = max(context_mean, missing_mean) >= 0.65
        passed = delta["mean"] >= 0.10 and delta["positive_folds"] >= 4 and audit_signal
        a_by_policy[policy] = {"status": "meets_screening_thresholds" if passed else "does_not_meet_thresholds",
                               "full_random_minus_group_macro_f1": delta,
                               "context_random_mean_macro_f1": context_mean,
                               "missing_random_mean_macro_f1": missing_mean,
                               "non_inertial_signal_threshold_met": audit_signal,
                               "thresholds": {"mean_delta_at_least": 0.10,
                                              "positive_folds_at_least": 4,
                                              "context_or_missing_random_mean_at_least": 0.65},
                               "fold_pairing_note": "Fold-number differences are a predeclared screening rule; random/group test rows differ and this is not a paired significance test."}
    universal = all(v["status"] == "meets_screening_thresholds" for v in a_by_policy.values())
    result["track_A"] = {"status": "meets_in_both_label_policies" if universal else
                         ("label_policy_dependent" if any(v["status"] == "meets_screening_thresholds" for v in a_by_policy.values())
                          else "does_not_meet_thresholds"),
                         "general_claim_allowed_by_screen": universal, "by_policy": a_by_policy,
                         "scope_note": "Screening result only; follow-up time-block/spatial checks and multiple seeds remain required."}
    b_by_policy = {}
    for policy in POLICIES:
        imu = checked[(policy, "group", "imu_only")]
        candidates = {}
        for method in ("physics_window", "physics_calibrated"):
            folds = checked[(policy, "group", method)]
            delta = _difference(folds, imu, "macro_f1")
            pothole_delta = _difference(folds, imu, "recall_by_class", "Pothole")
            passed = delta["mean"] >= 0.05 and delta["positive_folds"] >= 3 and pothole_delta["mean"] >= -0.05
            candidates[method] = {"status": "meets_numeric_thresholds" if passed else "does_not_meet_numeric_thresholds",
                                  "macro_f1_vs_imu_only": delta,
                                  "pothole_recall_vs_imu_only": pothole_delta,
                                  "thresholds": {"mean_macro_f1_gain_at_least": 0.05,
                                                 "positive_folds_at_least": 3,
                                                 "mean_pothole_recall_loss_no_more_than": 0.05}}
        passed_methods = [m for m, v in candidates.items() if v["status"] == "meets_numeric_thresholds"]
        b_by_policy[policy] = {"status": "meets_numeric_thresholds" if passed_methods else "does_not_meet_numeric_thresholds",
                               "qualifying_methods": passed_methods,
                               "method_results": candidates,
                               "full_numeric_comparison_and_latency_interpretation_still_required": bool(passed_methods)}
    result["track_B"] = {"status": "numeric_thresholds_met_in_both_policies" if all(
        v["status"] == "meets_numeric_thresholds" for v in b_by_policy.values()) else
        ("label_policy_dependent" if any(v["status"] == "meets_numeric_thresholds" for v in b_by_policy.values())
         else "does_not_meet_thresholds"), "by_policy": b_by_policy,
        "scope_note": "Numeric screening only; compare with full_numeric and interpret latency, trade-offs, and remaining failures before any research claim."}
    return result


def analyze(run_dir: Path, pilot_dir: Path = ROOT) -> dict:
    run_dir = run_dir.resolve()
    map_payload = json.loads((pilot_dir / "session_fold_map_v1.json").read_text(encoding="utf-8"))
    metadata = _validate_manifest(run_dir, pilot_dir, map_payload)
    records, all_final_files, input_names = _load_inputs(run_dir)
    checked, issues = _validate_records(records, map_payload)
    expected = {(p, s, m) for p in POLICIES for s in SPLITS for m in METHODS}
    all_120 = set(checked) == expected and all(len(checked[k]) == 5 for k in expected)
    complete = all_final_files and all_120 and not issues
    summary_metrics = {}
    for key, folds in sorted(checked.items()):
        p, s, m = key
        summary_metrics.setdefault(p, {}).setdefault(s, {})[m] = {
            "folds": [{"fold": int(f["fold"]) + 1,
                       "macro_f1": float(f["macro_f1"]),
                       "balanced_accuracy": float(f["balanced_accuracy"]),
                       "recall_by_class": {c: float(f["recall_by_class"][c]) for c in CLASSES}}
                      for f in folds],
            "mean": {"macro_f1": _mean_metric(folds, "macro_f1"),
                     "balanced_accuracy": _mean_metric(folds, "balanced_accuracy"),
                     "recall_by_class": {c: _mean_metric(folds, "recall_by_class", c)
                                          for c in CLASSES}},
        }
    tracks = _screen_tracks(checked, all_final_files, all_120 and not issues)
    return {"status": "complete" if complete else "partial",
            "complete_four_policy_split_files": all_final_files,
            "complete_120_method_folds": all_120,
            "validated_method_fold_count": sum(len(v) for v in checked.values()),
            "expected_method_fold_count": 120,
            "inputs": sorted(set(input_names)), "metadata_validation": metadata,
            "validation_issues": issues, "metrics": summary_metrics,
            "screening": tracks,
            "interpretation": "Threshold screening does not establish a paper-ready contribution or a journal outcome."}


def render_markdown(summary: dict) -> str:
    lines = ["# RoadSens pilot summary", "", f"Status: **{summary['status']}**. "
             f"Validated {summary['validated_method_fold_count']}/{summary['expected_method_fold_count']} method-fold results.", ""]
    if summary["status"] != "complete":
        lines.append("This is a partial result set. Threshold gates are not adjudicated until all four policy/split files and 120 method-fold results are validated.")
        lines.append("")
    lines += ["## Method means by label policy and split", "",
              "| Policy | Split | Method | Macro-F1 | Balanced accuracy | Normal recall | Pothole recall | Bump recall |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        for split in SPLITS:
            for method in METHODS:
                item = summary["metrics"].get(policy, {}).get(split, {}).get(method)
                if not item:
                    continue
                m = item["mean"]
                lines.append(f"| {policy} | {split} | {method} | {m['macro_f1']:.3f} | {m['balanced_accuracy']:.3f} | "
                             f"{m['recall_by_class']['Normal']:.3f} | {m['recall_by_class']['Pothole']:.3f} | {m['recall_by_class']['Bump']:.3f} |")
    lines += ["", "## Track A", "", f"Status: **{summary['screening']['track_A']['status']}**.", ""]
    for policy, item in summary["screening"]["track_A"].get("by_policy", {}).items():
        d = item["full_random_minus_group_macro_f1"]
        lines.append(f"- {policy}: mean random−group macro-F1 {d['mean']:+.3f}; positive folds {d['positive_folds']}/5; "
                     f"context random mean {item['context_random_mean_macro_f1']:.3f}; "
                     f"missing-only random mean {item['missing_random_mean_macro_f1']:.3f}; {item['status']}.")
    lines += ["", "Fold-number differences are the frozen screening rule; random and group test rows differ, so these are not paired significance tests.", "",
              "## Track B", "", f"Status: **{summary['screening']['track_B']['status']}**.", ""]
    for policy, item in summary["screening"]["track_B"].get("by_policy", {}).items():
        lines.append(f"- {policy}: numeric gate {item['status']}; qualifying methods: {', '.join(item['qualifying_methods']) or 'none'}.")
        for method, result in item["method_results"].items():
            d = result["macro_f1_vs_imu_only"]
            r = result["pothole_recall_vs_imu_only"]
            lines.append(f"  - {method}: macro-F1 gain {d['mean']:+.3f} ({d['positive_folds']}/5 positive); "
                         f"Pothole recall change {r['mean']:+.3f}; {result['status']}.")
    lines += ["", "Track B remains a numeric screen. Any qualifying method still needs a full_numeric comparison and an interpretation of latency, trade-offs, and failures.", "",
              "## Validation", "", f"Metadata/source hashes: {'passed' if summary['metadata_validation']['passed'] else 'failed'}; "
              f"session files verified: {summary['metadata_validation']['source_hashes_verified']}."]
    if summary["validation_issues"]:
        lines.append("")
        lines.append("Validation issues:")
        lines.extend(f"- {issue}" for issue in summary["validation_issues"][:30])
        if len(summary["validation_issues"]) > 30:
            lines.append(f"- … plus {len(summary['validation_issues']) - 30} additional issues (see summary.json).")
    lines += ["", summary["interpretation"], ""]
    return "\n".join(lines)


def synthetic_qa() -> dict:
    """Small deterministic gate/partial-state QA without loading any pilot output."""
    checked = {}
    for p in POLICIES:
        for s in SPLITS:
            for m in METHODS:
                folds = []
                for i in range(5):
                    f1 = 0.75 if m == "full_numeric" and s == "random" else 0.60
                    if m in ("context_only", "missing_only") and s == "random":
                        f1 = 0.70
                    if m == "physics_window" and s == "group":
                        f1 = 0.68
                    if m == "imu_only" and s == "group":
                        f1 = 0.60
                    folds.append({"fold": i, "macro_f1": f1, "balanced_accuracy": f1,
                                  "recall_by_class": {c: 0.60 for c in CLASSES}})
                checked[(p, s, m)] = folds
    tracks = _screen_tracks(checked, True, True)
    assert tracks["track_A"]["status"] == "meets_in_both_label_policies"
    assert tracks["track_B"]["status"] == "numeric_thresholds_met_in_both_policies"
    assert _screen_tracks(checked, False, False)["track_A"]["status"] == "not_evaluable_incomplete"
    assert tracks["track_A"]["by_policy"]["primary"]["full_random_minus_group_macro_f1"]["positive_folds"] == 5
    return {"passed": True, "checks": ["A gates and cross-policy rule", "B numeric gates by policy",
                                        "partial run never adjudicates gates", "fold deltas are summarized"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "pilot_v2")
    parser.add_argument("--pilot-dir", type=Path, default=ROOT)
    parser.add_argument("--synthetic-qa", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write a clearly marked partial snapshot when the four final result files are not complete")
    args = parser.parse_args()
    if args.synthetic_qa:
        print(json.dumps(synthetic_qa(), ensure_ascii=False, indent=2))
        return
    summary = analyze(args.run_dir, args.pilot_dir)
    if summary["status"] != "complete" and not args.allow_partial:
        raise SystemExit("Partial run detected. Re-run with --allow-partial to write an explicitly partial snapshot.")
    (args.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    (args.run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote summary.json and summary.md; status={summary['status']}; "
          f"validated={summary['validated_method_fold_count']}/120 method-folds.")


if __name__ == "__main__":
    main()
