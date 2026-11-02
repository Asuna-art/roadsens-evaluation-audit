from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = Path(__file__).resolve().parent
CACHE = ROOT / "session_feature_cache_v1.joblib"
V2 = AUDIT_DIR / "cached_data_audit_v2.json"
AMENDMENT = ROOT / "paper_expansion" / "content_overlap_amendment_v1.md"
OUT_JSON = AUDIT_DIR / "content_component_freeze_v4.json"
OUT_NPZ = AUDIT_DIR / "content_component_membership_v4.npz"
CLASSES = ["Normal", "Pothole", "Bump"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if OUT_JSON.exists() or OUT_NPZ.exists():
        raise FileExistsError("Refusing to overwrite a freeze artifact")
    cache_hash = sha(CACHE)
    cache = joblib.load(CACHE)
    audit = json.loads(V2.read_text(encoding="utf-8"))
    # Preserve the immutable cache / run_pilot._concat order (five Normal recordings,
    # then lexically sorted anomaly recordings), so saved global row IDs align exactly.
    sessions = cache["sessions"]
    keys = [s["session_key"] for s in sessions]
    if len(keys) != 101 or len(set(keys)) != 101:
        raise RuntimeError(f"Expected 101 unique recordings, got {len(keys)}")
    live_hashes = {}
    for s in sessions:
        p = Path(s["path"])
        live = sha(p)
        if live != s["sha256"] or live != cache["source_sha256"][s["session_key"]]:
            raise RuntimeError(f"Live source hash mismatch: {s['session_key']}")
        live_hashes[s["session_key"]] = live

    # Content graph is defined only by frozen finite-only exact-chain and byte-hash evidence.
    parent = {k: k for k in keys}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    edges = []
    for pair in audit["12_axis_imu"]["pair_results"]:
        if pair["longest_consecutive_exact_run_retained_rows"] >= 20:
            a, b = pair["recording_a"], pair["recording_b"]
            union(a, b)
            sa, sb = next(s for s in sessions if s["session_key"] == a), next(s for s in sessions if s["session_key"] == b)
            ra = pair["longest_run_original_source_rows_a"]
            rb = pair["longest_run_original_source_rows_b"]
            ia = int(ra[0] // 10); ib = int(rb[0] // 10)
            n = int(pair["longest_consecutive_exact_run_retained_rows"])
            ta = np.asarray(sa["times"])[ia:ia+n]
            tb = np.asarray(sb["times"])[ib:ib+n]
            edges.append({
                "a": a, "b": b, "criterion": "consecutive_exact_fully_finite_12axis_imu",
                "matched_rows": n, "source_row_bounds_a_inclusive": [int(ra[0]), int(ra[1])],
                "source_row_bounds_b_inclusive": [int(rb[0]), int(rb[1])],
                "elapsed_span_a_seconds": [float(ta[0]), float(ta[-1])],
                "elapsed_span_b_seconds": [float(tb[0]), float(tb[-1])],
                "elapsed_duration_a_seconds": float(ta[-1] - ta[0]),
                "elapsed_duration_b_seconds": float(tb[-1] - tb[0]),
                "shared_fraction_smaller_finite_record": float(pair["shared_fraction_of_smaller_record_finite_rows"]),
                "label_time_agreement_diagnostic_only": pair["longest_run_diagnostics"],
            })
    for group in audit["source_csv_duplicate_hash_groups"]:
        a, b = group
        union(a, b)
        if not any(e["a"] == a and e["b"] == b for e in edges):
            edges.append({"a": a, "b": b, "criterion": "byte_identical_source_sha256", "sha256": live_hashes[a]})

    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    components = []
    component_of = {}
    for members in sorted((sorted(v) for v in groups.values()), key=lambda x: x[0]):
        cid = members[0]
        typ = "normal_only" if all(k.startswith("combined_normal/") for k in members) else "mixed"
        for k in members:
            component_of[k] = cid
        components.append({"component_id": cid, "members": members, "stratum": typ,
                           "source_count": len(members)})
    normal_components = [c for c in components if c["stratum"] == "normal_only"]
    mixed_components = [c for c in components if c["stratum"] == "mixed"]
    if len(normal_components) != 5 or any(len(c["members"]) != 1 for c in normal_components):
        raise RuntimeError("Frozen constrained component design requires five singleton Normal components")

    # Explicit row table matches run_pilot._concat's labeled-row indexing.
    row_id, row_session, row_source, row_label_primary, row_label_blank, row_component = [], [], [], [], [], []
    for s in sessions:
        key = s["session_key"]
        for i, sr in enumerate(s["source_row"]):
            p = str(s["labels"][i]) if s["labels"][i] is not None else ""
            b = str(s["labels_blank_normal"][i]) if s["labels_blank_normal"][i] is not None else ""
            row_id.append(len(row_id)); row_session.append(key); row_source.append(int(sr)); row_label_primary.append(p)
            row_label_blank.append(b); row_component.append(component_of[key])
    row_id = np.asarray(row_id, dtype=np.int32)
    row_session = np.asarray(row_session, dtype="U64")
    row_source = np.asarray(row_source, dtype=np.int32)
    row_label_primary = np.asarray(row_label_primary, dtype="U16")
    row_label_blank = np.asarray(row_label_blank, dtype="U16")
    row_component = np.asarray(row_component, dtype="U64")
    if len(row_session) != audit["retained_sample_rows_total"]:
        raise RuntimeError("Retained row membership count differs from frozen audit")

    fold_metadata = {}
    map_npz = {}
    for seed in (42, 43, 44):
        rng = np.random.RandomState(seed)
        normal_order = rng.permutation([c["component_id"] for c in normal_components])
        normal_fold = {cid: int(i + 1) for i, cid in enumerate(normal_order)}
        # Stratify explicitly labeled P/B sampled rows by connected mixed component.
        idx = np.flatnonzero((row_label_primary == "Pothole") | (row_label_primary == "Bump"))
        anomaly_rows = idx[np.asarray([not k.startswith("combined_normal/") for k in row_session[idx]])]
        y = row_label_primary[anomaly_rows]
        g = row_component[anomaly_rows]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        anomaly_fold = {}
        for fold, (_, test_index) in enumerate(splitter.split(np.zeros(len(y)), y, g), start=1):
            for cid in np.unique(g[test_index]):
                if cid in anomaly_fold:
                    raise RuntimeError("Mixed component assigned to multiple folds")
                anomaly_fold[str(cid)] = fold
        component_fold = {**normal_fold, **anomaly_fold}
        if set(component_fold) != set(c["component_id"] for c in components):
            raise RuntimeError(f"Component assignment incomplete for seed {seed}")
        row_fold = np.asarray([component_fold[c] for c in row_component], dtype=np.int8)
        map_npz[f"seed{seed}_fold"] = row_fold
        by_seed = {}
        for policy, ylabs in (("primary", row_label_primary), ("blank_normal", row_label_blank)):
            rows_meta = {}
            for fold in range(1, 6):
                tr = row_fold != fold; te = row_fold == fold
                train_counts = {cl: int(np.count_nonzero(tr & (ylabs == cl))) for cl in CLASSES}
                test_counts = {cl: int(np.count_nonzero(te & (ylabs == cl))) for cl in CLASSES}
                if min(train_counts.values()) == 0 or min(test_counts.values()) == 0:
                    raise RuntimeError(f"Missing class support seed{seed} {policy} fold{fold}: {train_counts} {test_counts}")
                rows_meta[str(fold)] = {
                    "train_components": sorted(set(row_component[tr])),
                    "test_components": sorted(set(row_component[te])),
                    "train_source_recordings": int(len(set(row_session[tr]))),
                    "test_source_recordings": int(len(set(row_session[te]))),
                    "train_class_rows_before_cap": train_counts, "test_class_rows": test_counts,
                    "train_rows_before_cap": int(tr.sum()), "test_rows": int(te.sum()),
                }
            by_seed[policy] = rows_meta
        fold_metadata[str(seed)] = {"normal_component_order": list(map(str, normal_order)),
                                    "component_fold": component_fold, "policies": by_seed}

    scoreable_primary = np.full(len(row_id), -1, dtype=np.int32)
    scoreable_blank = np.full(len(row_id), -1, dtype=np.int32)
    primary_ix = np.flatnonzero(np.isin(row_label_primary, CLASSES))
    blank_ix = np.flatnonzero(np.isin(row_label_blank, CLASSES))
    scoreable_primary[primary_ix] = np.arange(len(primary_ix), dtype=np.int32)
    scoreable_blank[blank_ix] = np.arange(len(blank_ix), dtype=np.int32)
    np.savez_compressed(OUT_NPZ, cache_row_id=row_id,
                        scoreable_global_row_primary=scoreable_primary,
                        scoreable_global_row_blank_normal=scoreable_blank,
                        session_key=row_session, source_row=row_source,
                        label_primary=row_label_primary, label_blank_normal=row_label_blank,
                        component_id=row_component, **map_npz)
    freeze = {
        "freeze_type": "label_free_content_component_membership_and_constrained_fold_support",
        "artifact_version": 4,
        "supersedes": "content_component_freeze_v1/v2.json; v4 preserves policy-specific scoreable row IDs and records the exact scikit-learn version used for SGKF",
        "component_rule": "byte-identical source SHA256 OR >=20 consecutive retained fully finite exact 12-axis IMU vectors; edges are label/model/GPS/weather independent",
        "cache_path": str(CACHE), "cache_sha256": cache_hash,
        "cache_runner_sha256": cache["runner_sha256"], "source_hashes_live_verified": live_hashes,
        "source_count": len(sessions), "retained_rows": len(row_session),
        "component_membership_npz": OUT_NPZ.name, "component_membership_npz_sha256": sha(OUT_NPZ),
        "global_row_order": "cache_row_id follows immutable session_feature_cache_v1.joblib order. Policy-specific scoreable row IDs concatenate only labels in Normal/Pothole/Bump class order set and finite times, matching run_pilot._concat; retained cache timestamps are all finite.",
        "scoreable_rows_by_policy": {"primary": int(len(primary_ix)), "blank_normal": int(len(blank_ix))},
        "component_count": len(components), "normal_only_component_count": len(normal_components),
        "mixed_component_count": len(mixed_components), "components": components,
        "edges": edges, "fold_design": "5 shuffled singleton Normal components, plus StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed) on explicit primary Pothole/Bump rows grouped by component; same row fold map reused for both policies",
        "fold_seeds": fold_metadata,
        "inputs_sha256": {"finite_only_audit_v2": sha(V2), "content_overlap_amendment_v1": sha(AMENDMENT)},
        "software": {"python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "model_fits_performed": 0,
    }
    OUT_JSON.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"freeze_json": str(OUT_JSON), "freeze_npz": str(OUT_NPZ),
                      "components": components, "edge_count": len(edges),
                      "normal_components": len(normal_components), "mixed_components": len(mixed_components),
                      "fold_support_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
