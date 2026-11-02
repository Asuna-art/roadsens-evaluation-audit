from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
AUDIT = ROOT / "paper_expansion" / "data_audit"
FREEZE_JSON = AUDIT / "content_component_freeze_v4.json"
MEMBERSHIP = AUDIT / "content_component_membership_v4.npz"
OUT = ROOT / "paper_expansion" / "cluster_uncertainty"
OUT_JSON = OUT / "component_bootstrap_v1.json"
OUT_NPZ = OUT / "component_confusions_v1.npz"
POLICIES = ["primary", "blank_normal"]
SPLITS = ["random", "group", "timeblock"]
METHODS = ["full_numeric", "context_only", "imu_only", "missing_only"]
CLASSES = ["Normal", "Pothole", "Bump"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
BOOTSTRAPS = 2000
BOOT_SEED = 20261016


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_f1(cm: np.ndarray) -> float:
    support = cm.sum(axis=1)
    if np.any(support == 0):
        return float("nan")
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + cm.sum(axis=0) - tp + support - tp
    f = np.divide(2 * tp, denom, out=np.zeros(3, dtype=float), where=denom != 0)
    return float(f.mean())


def metric(cm: np.ndarray) -> float:
    return macro_f1(cm)


def pct_ci(values: np.ndarray) -> list[float] | None:
    v = values[np.isfinite(values)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if OUT_JSON.exists() or OUT_NPZ.exists():
        raise FileExistsError("Refusing to overwrite component bootstrap outputs")
    start = time.perf_counter()
    freeze = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    if freeze["artifact_version"] != 4:
        raise RuntimeError("Need policy-scoreable-row-aligned component freeze v4")
    with np.load(MEMBERSHIP, allow_pickle=False) as d:
        cache_ids = d["cache_row_id"].astype(np.int64)
        source = d["session_key"].astype(str)
        source_rows = d["source_row"].astype(np.int64)
        labels = {p: d[f"label_{p}"].astype(str) for p in POLICIES}
        component = d["component_id"].astype(str)
        score_maps = {
            "primary": d["scoreable_global_row_primary"].astype(np.int64),
            "blank_normal": d["scoreable_global_row_blank_normal"].astype(np.int64),
        }
    if not np.array_equal(cache_ids, np.arange(len(cache_ids))):
        raise RuntimeError("Membership cache rows are not contiguous immutable cache order")
    comp_ids = [x["component_id"] for x in freeze["components"]]
    comp_index = {c: i for i, c in enumerate(comp_ids)}
    comp_strata = [x["stratum"] for x in freeze["components"]]
    normal_ci = np.flatnonzero(np.asarray(comp_strata) == "normal_only")
    mixed_ci = np.flatnonzero(np.asarray(comp_strata) == "mixed")
    if len(normal_ci) != 5 or len(mixed_ci) != 91:
        raise RuntimeError("Unexpected frozen component strata")
    row_ci = np.asarray([comp_index[x] for x in component], dtype=np.int16)
    ncomp = len(comp_ids)
    cms = np.zeros((2, 3, 4, 5, ncomp, 3, 3), dtype=np.int64)
    source_ids = sorted(set(source.tolist()))
    source_index = {s: i for i, s in enumerate(source_ids)}
    per_record = np.zeros((2, 3, 4, 5, len(source_ids), 3, 3), dtype=np.int64)
    observed_rows: dict[str, list[int]] = {}
    label_mismatches = []
    checkpoint_hashes = {}
    for pi, policy in enumerate(POLICIES):
        expected = labels[policy]
        score_to_cache = np.empty(int(np.max(score_maps[policy])) + 1, dtype=np.int64)
        keep = score_maps[policy] >= 0
        score_to_cache[score_maps[policy][keep]] = np.flatnonzero(keep)
        expected_rows = np.arange(len(score_to_cache), dtype=np.int64)
        for si, split in enumerate(SPLITS):
            seen_by_method = {}
            for mi, method in enumerate(METHODS):
                if split == "timeblock":
                    path = RUNS / "timeblock_v1" / f"{policy}_timeblock.{method}.checkpoint.json"
                else:
                    path = RUNS / "pilot_v2" / f"{policy}_{split}.{method}.checkpoint.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                checkpoint_hashes[str(path.relative_to(ROOT))] = sha(path)
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("policy") != policy or data.get("method") != method:
                    raise RuntimeError(f"Checkpoint identity mismatch: {path}")
                if len(data.get("folds", [])) != 5:
                    raise RuntimeError(f"Incomplete folds in {path}")
                got_rows = []
                for fold_obj in data["folds"]:
                    fold = int(fold_obj["fold"])
                    if fold not in range(5):
                        raise RuntimeError(f"Unexpected fold numbering {fold} in {path}")
                    for pred in fold_obj["predictions"]:
                        rid = int(pred["row"])
                        yt, yp = str(pred["true"]), str(pred["pred"])
                        if not 0 <= rid < len(score_to_cache):
                            raise RuntimeError(f"Out of range row id {rid} in {path}")
                        cache_row = int(score_to_cache[rid])
                        if yt not in CLASS_ID or yp not in CLASS_ID:
                            raise RuntimeError(f"Unexpected labels {yt!r}/{yp!r}")
                        if expected[cache_row] != yt:
                            label_mismatches.append({"file": str(path), "row": rid, "checkpoint": yt,
                                                     "cache_row": cache_row, "membership": expected[cache_row]})
                            if len(label_mismatches) >= 3:
                                raise RuntimeError(f"Prediction labels disagree with membership: {label_mismatches[:3]}")
                        got_rows.append(rid)
                        if str(pred["session"]) != source[cache_row] or int(pred["source_row"]) != source_rows[cache_row]:
                            raise RuntimeError(f"Prediction source identity mismatch in {path}: row={rid}, cache_row={cache_row}")
                        cms[pi, si, mi, fold, row_ci[cache_row], CLASS_ID[yt], CLASS_ID[yp]] += 1
                        per_record[pi, si, mi, fold, source_index[source[cache_row]], CLASS_ID[yt], CLASS_ID[yp]] += 1
                got_rows = np.asarray(got_rows, dtype=np.int64)
                if len(np.unique(got_rows)) != len(got_rows):
                    raise RuntimeError(f"Duplicate test row within {path}")
                if not np.array_equal(np.sort(got_rows), expected_rows):
                    raise RuntimeError(f"OOF row population mismatch in {path}: got {len(got_rows)}, expected {len(expected_rows)}")
                seen_by_method[method] = got_rows
            base = seen_by_method[METHODS[0]]
            for method in METHODS[1:]:
                if not np.array_equal(base, seen_by_method[method]):
                    raise RuntimeError(f"Method test rows differ: {policy}/{split}/{method}")
            observed_rows[f"{policy}/{split}"] = base.tolist()
            print(f"validated {policy}/{split}: {len(base)} common OOF rows", flush=True)

    # Confirm component sums equal source-record sums.
    component_from_records = np.zeros_like(cms)
    for ri, src in enumerate(source_ids):
        ci = comp_index[component[np.flatnonzero(source == src)[0]]]
        component_from_records[:, :, :, :, ci] += per_record[:, :, :, :, ri]
    if not np.array_equal(component_from_records, cms):
        raise RuntimeError("Component confusion matrices do not reconstruct source-level matrices")

    # Bootstrap shared component weights; no redraw for degenerate replicates.
    rng = np.random.RandomState(BOOT_SEED)
    contrasts = {
        "full_random_minus_record": ((0, 0), (0, 1), 0),
        "full_random_minus_timeblock": ((0, 0), (0, 2), 0),
        "context_minus_full_record": ((1, 1), (0, 1), 1),
        "imu_minus_full_record": ((2, 1), (0, 1), 2),
    }
    # Use a clearer internal tuple: (method index A, split A, method index B, split B, policy index).
    contrast_defs = {
        "full_random_minus_record": (0, 0, 0, 1),
        "full_random_minus_timeblock": (0, 0, 0, 2),
        "context_minus_full_record": (1, 1, 0, 1),
        "imu_minus_full_record": (2, 1, 0, 1),
    }
    point = {}
    for pi, policy in enumerate(POLICIES):
        point[policy] = {}
        for si, split in enumerate(SPLITS):
            point[policy][split] = {}
            for mi, method in enumerate(METHODS):
                fold_cms = cms[pi, si, mi].sum(axis=1)
                pooled = fold_cms.sum(axis=0)
                point[policy][split][method] = {
                    "pooled_oof_macro_f1": metric(pooled),
                    "fold_mean_macro_f1": float(np.mean([metric(x) for x in fold_cms])),
                    "pooled_oof_confusion_matrix": pooled.tolist(),
                    "fold_confusion_matrices": fold_cms.tolist(),
                }
    boot = {policy: {} for policy in POLICIES}
    invalid = {policy: {} for policy in POLICIES}
    draws = np.zeros((BOOTSTRAPS, ncomp), dtype=np.int16)
    for b in range(BOOTSTRAPS):
        draws[b, normal_ci] = np.bincount(rng.choice(normal_ci, size=len(normal_ci), replace=True), minlength=ncomp)[normal_ci]
        draws[b, mixed_ci] = np.bincount(rng.choice(mixed_ci, size=len(mixed_ci), replace=True), minlength=ncomp)[mixed_ci]
    for policy_i, policy in enumerate(POLICIES):
        boot[policy], invalid[policy] = {}, {}
        for name, (ma, sa, mb, sb) in contrast_defs.items():
            d_pool, d_fold = [], []
            invalid_pool = invalid_fold = 0
            for b in range(BOOTSTRAPS):
                weights = draws[b]
                cma = np.einsum("c,fcij->fij", weights, cms[policy_i, sa, ma])
                cmb = np.einsum("c,fcij->fij", weights, cms[policy_i, sb, mb])
                pa, pb = metric(cma.sum(axis=0)), metric(cmb.sum(axis=0))
                fa = np.asarray([metric(x) for x in cma]); fb = np.asarray([metric(x) for x in cmb])
                if np.isfinite(pa) and np.isfinite(pb):
                    d_pool.append(pa - pb)
                else:
                    invalid_pool += 1
                if np.all(np.isfinite(fa)) and np.all(np.isfinite(fb)):
                    d_fold.append(float(np.mean(fa) - np.mean(fb)))
                else:
                    invalid_fold += 1
            vals_pool, vals_fold = np.asarray(d_pool), np.asarray(d_fold)
            boot[policy][name] = {
                "pooled_oof_macro_f1_difference_ci95": pct_ci(vals_pool),
                "pooled_oof_valid_replicates": int(len(vals_pool)),
                "fold_mean_macro_f1_difference_ci95": pct_ci(vals_fold),
                "fold_mean_valid_replicates": int(len(vals_fold)),
            }
            invalid[policy][name] = {"pooled_degenerate_replicates_not_redrawn": invalid_pool,
                                     "fold_mean_degenerate_replicates_not_redrawn": invalid_fold}
    # Per-record output has both source-aggregated counts and observed-support macro recall.
    record_summaries = {}
    for pi, policy in enumerate(POLICIES):
        record_summaries[policy] = {}
        for si, split in enumerate(SPLITS):
            record_summaries[policy][split] = {}
            for mi, method in enumerate(METHODS):
                rows = []
                support_records = {c: 0 for c in CLASSES}
                recalls = {c: [] for c in CLASSES}
                for ri, src in enumerate(source_ids):
                    cm = per_record[pi, si, mi, :, ri].sum(axis=0)
                    if cm.sum() == 0:
                        continue
                    row = {"source_recording": src, "confusion_matrix": cm.tolist(),
                           "support": {c: int(cm[i].sum()) for i, c in enumerate(CLASSES)}}
                    for i, c in enumerate(CLASSES):
                        if cm[i].sum() > 0:
                            support_records[c] += 1
                            recalls[c].append(float(cm[i, i] / cm[i].sum()))
                    observed = [recalls[c][-1] for i, c in enumerate(CLASSES) if cm[i].sum() > 0]
                    row["observed_class_macro_recall"] = float(np.mean(observed))
                    rows.append(row)
                record_summaries[policy][split][method] = {
                    "record_count_with_test_rows": len(rows),
                    "recordings_supporting_each_class": support_records,
                    "mean_record_recall_by_class_over_supporting_recordings": {
                        c: (float(np.mean(recalls[c])) if recalls[c] else None) for c in CLASSES},
                    "records": rows,
                }

    np.savez_compressed(OUT_NPZ, component_fold_confusions=cms,
                        per_record_fold_confusions=per_record,
                        component_ids=np.asarray(comp_ids, dtype="U96"),
                        component_stratum=np.asarray(comp_strata, dtype="U16"),
                        source_recordings=np.asarray(source_ids, dtype="U64"),
                        bootstrap_component_draw_counts=draws)
    result = {
        "analysis": "shared-stratum content-component bootstrap of fixed OOF predictions",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOT_SEED,
        "resampling_unit": "frozen content component; 5 Normal-only and 91 mixed components sampled separately with replacement; identical component weights shared across methods, splits, and policies",
        "component_counts": {"all": ncomp, "normal_only": len(normal_ci), "mixed": len(mixed_ci),
                             "source_recordings": len(source_ids), "normal_source_recordings": 5,
                             "mixed_source_recordings": len(source_ids) - 5},
        "checkpoint_hashes": checkpoint_hashes,
        "component_freeze_sha256": sha(FREEZE_JSON), "membership_npz_sha256": sha(MEMBERSHIP),
        "point_estimates": point, "pooled_oof_contrast_bootstrap_cis": boot,
        "degenerate_replicates": invalid, "record_level_confusion_and_recall": record_summaries,
        "limitations": [
            "Intervals condition on the already fitted models and their fixed OOF predictions; they do not include training uncertainty.",
            "Bootstrap resampling does not repair train/test component overlap in the historical filename-heldout and row-random evaluations.",
            "CSV recordings are operational source units; they are not verified trips, routes, or independent acquisition sessions.",
            "No p-values or row-level independence claims are made; fold-mean and pooled OOF metrics are reported separately."
        ],
        "elapsed_seconds": time.perf_counter() - start,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(OUT_JSON), "output_npz": str(OUT_NPZ),
                      "elapsed_seconds": result["elapsed_seconds"], "component_counts": result["component_counts"],
                      "ci_summary": boot}, ensure_ascii=False))


if __name__ == "__main__":
    main()
