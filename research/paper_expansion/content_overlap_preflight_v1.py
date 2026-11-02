from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "session_feature_cache_v1.joblib"
COMPONENT_FREEZE = HERE / "data_audit" / "content_component_freeze_v4.json"
COMPONENT_ROWS = HERE / "data_audit" / "content_component_membership_v4.npz"
TIME_QA = ROOT / "timeblock_split_support_v3.json"
OUT_JSON = HERE / "content_overlap_preflight_v1.json"
OUT_NPZ = HERE / "content_overlap_preflight_rows_v1.npz"
CLASSES = ("Normal", "Pothole", "Bump")
POLICIES = ("primary", "blank_normal")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def session_rows(sessions: list[dict], policy: str, omit: set[str] | None = None) -> dict:
    labkey = "labels" if policy == "primary" else "labels_blank_normal"
    out = {k: [] for k in ("y", "group", "source_row", "cache_row")}
    offset = 0
    for s in sessions:
        key = s["session_key"]
        n = len(s["source_row"])
        cache_rows = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
        if omit and key in omit:
            continue
        labels = np.asarray(s[labkey], dtype=object)
        keep = np.isfinite(np.asarray(s["times"], dtype=float)) & np.isin(labels, CLASSES)
        out["y"].extend(labels[keep].tolist())
        out["group"].extend([key] * int(keep.sum()))
        out["source_row"].extend(np.asarray(s["source_row"], dtype=np.int64)[keep].tolist())
        out["cache_row"].extend(cache_rows[keep].tolist())
    return {k: np.asarray(v, dtype=object if k in ("y", "group") else np.int64) for k, v in out.items()}


def norm_map(keys: list[str], seed: int) -> dict[str, int]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(sorted(keys))
    return {str(k): int(i + 1) for i, k in enumerate(order)}


def grouped_record_map(rows: dict, seed: int, normal_keys: list[str], group_component: dict | None = None) -> dict[str, int]:
    # The anomaly SGKF is driven only by explicit primary Pothole/Bump rows.
    y = rows["y"].astype(str); groups = rows["group"].astype(str)
    anomaly = np.flatnonzero(np.isin(y, ["Pothole", "Bump"]) & np.asarray([not k.startswith("combined_normal/") for k in groups]))
    sgkf_groups = np.asarray([group_component.get(groups[i], groups[i]) if group_component else groups[i] for i in anomaly], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    mapping = norm_map(normal_keys, seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(anomaly)), y[anomaly], sgkf_groups), start=1):
        for g in np.unique(sgkf_groups[test]):
            names = sorted(k for k in np.unique(groups[anomaly])
                           if (group_component.get(k, k) if group_component else k) == g)
            for key in names:
                if key in mapping:
                    raise RuntimeError(f"Recording/component assigned twice: {key}")
                mapping[key] = fold
    return mapping


def folds_from_map(rows: dict, fmap: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = rows["group"].astype(str)
    if set(groups) - set(fmap):
        raise RuntimeError(f"Recording fold map misses rows: {set(groups)-set(fmap)}")
    f = np.asarray([fmap[k] for k in groups], dtype=np.int8)
    return [(np.flatnonzero(f != fold), np.flatnonzero(f == fold)) for fold in range(1, 6)]


def counts(y: np.ndarray, ix: np.ndarray) -> dict[str, int]:
    return {c: int(np.sum(y[ix] == c)) for c in CLASSES}


def row_membership_sha(rows: dict, ix: np.ndarray) -> str:
    values = [(str(rows["group"][i]), int(rows["source_row"][i]), str(rows["y"][i])) for i in ix]
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capped_counts(y: np.ndarray, ix: np.ndarray, cap: int = 20000) -> dict[str, int]:
    c = counts(y, ix)
    if len(ix) <= cap:
        return c
    quotas = {k: min(v, max(1, int(cap * v / len(ix)))) for k, v in c.items()}
    while sum(quotas.values()) > cap:
        k = max(quotas, key=quotas.get); quotas[k] -= 1
    return quotas


def describe_folds(rows: dict, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    y = rows["y"].astype(str)
    out = []
    for fold, (tr, te) in enumerate(folds, start=1):
        train, test = counts(y, tr), counts(y, te)
        fit = capped_counts(y, tr)
        if min(train.values()) == 0 or min(test.values()) == 0 or min(fit.values()) == 0:
            raise RuntimeError(f"Missing three-class support at fold {fold}: train={train}, capped={fit}, test={test}")
        out.append({"fold": fold, "train_rows_before_cap": len(tr), "train_class_rows_before_cap": train,
                    "capped_train_rows": min(20000, len(tr)), "capped_train_class_rows_expected": fit,
                    "test_rows": len(te), "test_class_rows": test,
                    "train_group_count": len(set(rows["group"][tr].astype(str))),
                    "test_group_count": len(set(rows["group"][te].astype(str))),
                    "train_group_keys": sorted(set(rows["group"][tr].astype(str))),
                    "test_group_keys": sorted(set(rows["group"][te].astype(str))),
                    "train_row_membership_sha256": row_membership_sha(rows, tr),
                    "test_row_membership_sha256": row_membership_sha(rows, te)})
    return out


def timeblock_folds(rows: dict, support: dict, policy: str) -> list[tuple[np.ndarray, np.ndarray]]:
    policy_key = "primary" if policy == "primary" else "blank_normal_sensitivity"
    group = rows["group"].astype(str); sr = rows["source_row"].astype(np.int64)
    folds = []
    for fold in range(1, 6):
        qa = support["folds"][policy_key][str(fold)]["session_membership"]
        test_lookup, train_lookup = {}, {}
        for sess in qa:
            key = sess["session_key"]
            test_lookup[key] = set(map(int, sess["test_source_rows"]))
            train_lookup[key] = set(map(int, sess["train_source_rows"]))
        te = np.asarray([i for i, (g, r) in enumerate(zip(group, sr)) if int(r) in test_lookup.get(g, set())], dtype=np.int64)
        tr = np.asarray([i for i, (g, r) in enumerate(zip(group, sr)) if int(r) in train_lookup.get(g, set())], dtype=np.int64)
        if set(tr) & set(te):
            raise RuntimeError("Timeblock train/test source-row overlap")
        folds.append((tr, te))
    return folds


def main() -> None:
    if OUT_JSON.exists() or OUT_NPZ.exists():
        raise FileExistsError("Refusing to overwrite content-overlap preflight")
    cache = joblib.load(CACHE)
    frozen = json.loads(COMPONENT_FREEZE.read_text(encoding="utf-8"))
    if frozen["artifact_version"] != 4 or frozen["software"]["scikit_learn"] != sklearn.__version__:
        raise RuntimeError("Frozen content component graph/runtime mismatch")
    if sklearn.__version__ != "1.9.0":
        raise RuntimeError(f"Use frozen sklearn 1.9.0, got {sklearn.__version__}")
    if sha(CACHE) != frozen["cache_sha256"]:
        raise RuntimeError("Live cache hash differs from component freeze")
    with np.load(COMPONENT_ROWS, allow_pickle=False) as d:
        cache_rows = d["cache_row_id"].astype(np.int64)
        component_ids = d["component_id"].astype(str)
    if len(cache_rows) != sum(len(s["source_row"]) for s in cache["sessions"]):
        raise RuntimeError("Component membership row count differs from cache")
    compmap = {k: x["component_id"] for x in frozen["components"] for k in x["members"]}
    normal_keys = [s["session_key"] for s in cache["sessions"] if s["session_key"].startswith("combined_normal/")]
    sessions = cache["sessions"]
    component_folds_npz = {}
    results = {"H1": {}, "H2": {}}

    # H1: content-component heldout; same fold map for both label policies.
    for seed in (42, 43, 44):
        comp_fold = frozen["fold_seeds"][str(seed)]["component_fold"]
        record_fold = {k: comp_fold[compmap[k]] for k in compmap}
        seed_result = {}
        for policy in POLICIES:
            rows = session_rows(sessions, policy)
            folds = folds_from_map(rows, record_fold)
            seed_result[policy] = describe_folds(rows, folds)
            if seed == 42:
                component_folds_npz[f"H1_{policy}_cache_row_id"] = rows["cache_row"].astype(np.int32)
                component_folds_npz[f"H1_{policy}_source_row"] = rows["source_row"].astype(np.int32)
                component_folds_npz[f"H1_{policy}_source_key"] = rows["group"].astype("U64")
                component_folds_npz[f"H1_{policy}_label"] = rows["y"].astype("U16")
            component_folds_npz[f"H1_{seed}_{policy}_fold"] = np.asarray([record_fold[g] for g in rows["group"].astype(str)], dtype=np.int8)
        results["H1"][str(seed)] = seed_result

    # H2: byte-identical dedup cohort; lexicographically retain 46.csv, exclude 47.csv.
    dedup = {"combined_anomalies/47.csv"}
    if not set(dedup).issubset(set(s["session_key"] for s in sessions)):
        raise RuntimeError("Expected byte-identical 47.csv not found in cache")
    dedup_sessions = [s for s in sessions if s["session_key"] not in dedup]
    h2 = {"dedup_removed": sorted(dedup), "representative_retained": "combined_anomalies/46.csv", "policies": {}}
    time_qa = json.loads(TIME_QA.read_text(encoding="utf-8"))
    # Recording group map rebuilt on the 100-file cohort using the frozen recipe.
    primary_rows_h2 = session_rows(dedup_sessions, "primary")
    h2_record_map = grouped_record_map(primary_rows_h2, 42, normal_keys)
    if len(h2_record_map) != 100:
        raise RuntimeError(f"H2 group map has {len(h2_record_map)} sources, expected 100")
    h2["record_component_fold"] = h2_record_map
    for policy in POLICIES:
        rows = session_rows(dedup_sessions, policy)
        component_folds_npz[f"H2_{policy}_cache_row_id"] = rows["cache_row"].astype(np.int32)
        component_folds_npz[f"H2_{policy}_source_row"] = rows["source_row"].astype(np.int32)
        component_folds_npz[f"H2_{policy}_source_key"] = rows["group"].astype("U64")
        component_folds_npz[f"H2_{policy}_label"] = rows["y"].astype("U16")
        rf = [(np.asarray(tr), np.asarray(te)) for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(rows["y"])), rows["y"])]
        gf = folds_from_map(rows, h2_record_map)
        tf = timeblock_folds(rows, time_qa, policy)
        h2["policies"][policy] = {"n_scoreable_rows": len(rows["y"]), "scoreable_source_recordings": len(set(rows["group"])),
                                   "random42": describe_folds(rows, rf), "record42": describe_folds(rows, gf),
                                   "timeblock42": describe_folds(rows, tf)}
        for name, folds in (("random42", rf), ("record42", gf), ("timeblock42", tf)):
            fold_assignment = np.full(len(rows["y"]), 0, dtype=np.int8)
            for fold, (_, te) in enumerate(folds, start=1):
                if np.any(fold_assignment[te] != 0):
                    raise RuntimeError("Duplicate test assignment in H2 split")
                fold_assignment[te] = fold
            if np.any(fold_assignment == 0):
                raise RuntimeError(f"Unassigned H2 scoreable rows {policy}/{name}")
            component_folds_npz[f"H2_{policy}_{name}_fold"] = fold_assignment
    results["H2"] = h2

    np.savez_compressed(OUT_NPZ, cache_component_id=component_ids,
                        cache_row_id=cache_rows, **component_folds_npz)
    manifest = {"type": "fit_free_content_overlap_fold_preflight", "model_fit_or_scores": False,
                "runtime": {"python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
                "cache_sha256": sha(CACHE), "component_freeze_sha256": sha(COMPONENT_FREEZE),
                "component_membership_sha256": sha(COMPONENT_ROWS), "timeblock_support_sha256": sha(TIME_QA),
                "source_recordings_h1": 101, "source_recordings_h2": 100,
                "H1_expected_fits": 100, "H2_expected_fits": 90,
                "H1": results["H1"], "H2": results["H2"],
                "interpretation": "Preflight only: class supports, source-row memberships, and fold integrity. No model fits or score-based choices.",
                "timeblock_definition": "Reuses the frozen original per-recording source-row bins and inclusive one-second embargo, restricted to the deduplicated cohort.",
                "H2_cohort_definition": "Remove only combined_anomalies/47.csv as the byte-identical duplicate of 46.csv; lexicographically smaller 46.csv retained.",
                "npz_path": OUT_NPZ.name}
    OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "H1_seeds": [42,43,44], "H2_sources": 100,
                      "expected_fits": [100,90], "output": str(OUT_JSON)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
