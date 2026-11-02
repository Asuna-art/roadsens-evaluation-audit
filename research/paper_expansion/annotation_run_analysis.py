from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "session_feature_cache_v1.joblib"
RUNS = ROOT / "runs"
OUT_JSON = HERE / "annotated_runs_v1.json"
OUT_CSV = HERE / "annotated_run_detail_v1.csv"
PROTOCOL = HERE / "annotation_run_analysis_v1.md"
POLICIES = ("primary", "blank_normal")
SPLITS = ("random", "group", "timeblock")
METHODS = ("full_numeric", "context_only", "imu_only")
CLASSES = ("Normal", "Pothole", "Bump")
EXPECTED_COUNTS = {"Pothole": 149, "Bump": 199}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def runs_for_file(path: Path, key: str) -> list[dict]:
    frame = pd.read_csv(path, usecols=lambda c: c == "annotation_text", dtype={"annotation_text": "string"},
                        keep_default_na=False)
    if "annotation_text" not in frame:
        return []
    labels = frame["annotation_text"].astype(str).str.strip().to_numpy()
    found, label, start = [], None, None
    for i, raw in enumerate(labels):
        current = raw if raw in ("Pothole", "Bump") else None
        if current != label:
            if label is not None:
                found.append({"session": key, "start_source_row": int(start), "end_source_row": int(i - 1), "label": label})
            label, start = current, (i if current is not None else None)
    if label is not None:
        found.append({"session": key, "start_source_row": int(start), "end_source_row": int(len(labels) - 1), "label": label})
    return found


def load_prediction(path: Path) -> dict[tuple[str, int], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for fold in data["folds"]:
        for p in fold["predictions"]:
            key = (str(p["session"]), int(p["source_row"]))
            if key in result:
                raise RuntimeError(f"Duplicate OOF prediction key {key} in {path}")
            result[key] = str(p["pred"])
    return result


def score_run(run: dict, prediction: dict) -> dict:
    # Cache source rows are 0,10,20,... even when a label run starts between samples.
    first = ((run["start_source_row"] + 9) // 10) * 10
    expected = list(range(first, run["end_source_row"] + 1, 10))
    pairs = [(r, prediction[(run["session"], r)]) for r in expected if (run["session"], r) in prediction]
    missing = [r for r in expected if (run["session"], r) not in prediction]
    if not pairs:
        return {**run, "sampled_scored_rows": 0, "excluded_reason": "no sampled scored rows", "missing_sampled_rows": missing}
    pred_counts = Counter(p for _, p in pairs)
    majority = max(CLASSES, key=lambda c: pred_counts[c])
    correct = sum(p == run["label"] for _, p in pairs)
    return {**run, "sampled_scored_rows": len(pairs), "missing_sampled_rows": missing,
            "row_fraction_predicted_as_explicit_label": correct / len(pairs), "row_correct_count": correct,
            "majority_predicted_class": majority, "majority_predicted_correct": majority == run["label"],
            "predicted_class_counts": {c: int(pred_counts[c]) for c in CLASSES},
            "scored_source_rows": [int(r) for r, _ in pairs]}


def summarize(scored: list[dict]) -> dict:
    valid = [x for x in scored if x["sampled_scored_rows"] > 0]
    result = {"all_runs": len(scored), "runs_with_sampled_scored_rows": len(valid),
              "runs_without_sampled_rows": sum(x["sampled_scored_rows"] == 0 for x in scored), "by_explicit_class": {}}
    for cls in ("Pothole", "Bump"):
        rows = [x for x in valid if x["label"] == cls]
        vals = sorted(x["row_fraction_predicted_as_explicit_label"] for x in rows)
        n = len(vals)
        median = (vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2) if n else None
        result["by_explicit_class"][cls] = {
            "runs_with_scored_rows": n,
            "mean_run_row_fraction_predicted_as_label": sum(vals) / n if n else None,
            "median_run_row_fraction_predicted_as_label": median,
            "fraction_runs_with_correct_majority_prediction": sum(x["majority_predicted_correct"] for x in rows) / n if n else None,
            "correct_majority_runs": int(sum(x["majority_predicted_correct"] for x in rows)),
        }
    return result


def main() -> None:
    if OUT_JSON.exists() or OUT_CSV.exists():
        raise FileExistsError("Refusing to overwrite annotated-run results")
    started = time.perf_counter()
    cache = joblib.load(CACHE)
    all_runs, source_hashes = [], {}
    for s in cache["sessions"]:
        key, path = s["session_key"], Path(s["path"])
        actual = sha(path)
        if actual != s["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {key}")
        source_hashes[key] = actual
        all_runs.extend(runs_for_file(path, key))
    observed = Counter(r["label"] for r in all_runs)
    if dict(observed) != EXPECTED_COUNTS:
        raise RuntimeError(f"Expected {EXPECTED_COUNTS}, extracted {dict(observed)}")

    results, detail_rows, pred_hashes = {}, [], {}
    row_sets = {}
    for policy in POLICIES:
        results[policy] = {}
        for split in SPLITS:
            results[policy][split] = {}
            for method in METHODS:
                path = (RUNS / "timeblock_v1" / f"{policy}_timeblock.{method}.checkpoint.json" if split == "timeblock"
                        else RUNS / "pilot_v2" / f"{policy}_{split}.{method}.checkpoint.json")
                pred_hashes[str(path.relative_to(ROOT))] = sha(path)
                preds = load_prediction(path)
                scored = [score_run(run, preds) for run in all_runs]
                if any(x["missing_sampled_rows"] for x in scored):
                    raise RuntimeError(f"Missing sampled explicit-run rows for {policy}/{split}/{method}")
                signature = {(x["session"], tuple(x.get("scored_source_rows", []))) for x in scored}
                key = (policy, split)
                if key in row_sets and row_sets[key] != signature:
                    raise RuntimeError(f"Methods do not use the same annotated-run rows: {key}")
                row_sets[key] = signature
                results[policy][split][method] = summarize(scored)
                for x in scored:
                    detail_rows.append({"policy": policy, "split": split, "method": method,
                                        **{k: v for k, v in x.items() if k != "scored_source_rows"},
                                        "scored_source_rows": json.dumps(x.get("scored_source_rows", []))})
                print(f"scored {policy}/{split}/{method}: {results[policy][split][method]['runs_with_sampled_scored_rows']} runs", flush=True)

    result = {"analysis": "annotated contiguous-run summaries from fixed seed-42 OOF predictions", "model_fit_or_scores": False,
              "run_definition": "Consecutive identical explicit annotation_text Pothole/Bump full-cadence source rows; blank/unrecognized labels break a run; no duration column used.",
              "verified_original_explicit_run_counts": dict(observed), "source_recordings": len(cache["sessions"]),
              "source_hashes": source_hashes, "prediction_checkpoint_hashes": pred_hashes,
              "protocol_sha256": sha(PROTOCOL), "results": results,
              "limitations": [
                  "These summaries describe explicit annotation runs, not verified physical road events or event-boundary accuracy.",
                  "Row-random OOF may include other rows from the same run in training; it is not unseen-event performance.",
                  "Normal-directory periods and blank-as-Normal background are excluded.",
                  "No false alarms per hour, event localization, GPS/route analysis, or physical event metric is computed.",
                  "Runs from one CSV recording or content component are not independent; no run bootstrap or significance test is used."],
              "elapsed_seconds": time.perf_counter() - started}
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader(); writer.writerows(detail_rows)
    print(json.dumps({"runs": dict(observed), "summary": str(OUT_JSON), "details": str(OUT_CSV),
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
