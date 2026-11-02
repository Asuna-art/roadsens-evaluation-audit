"""Finite-only exact cross-record row/contiguous-chain audit; read-only and fit-free."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
CACHE = BASE / "session_feature_cache_v1.joblib"
CLASSES = ("Normal", "Pothole", "Bump")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def signature(row: np.ndarray) -> bytes:
    # Caller guarantees all components finite. Canonicalize -0/+0 as numeric equality.
    v = np.asarray(row, dtype="<f8").copy()
    v[v == 0] = 0.0
    return hashlib.sha256(v.tobytes()).digest()


def chain_runs(offset_to_i: dict[int, set[int]]) -> list[dict]:
    runs = []
    for offset, indices in offset_to_i.items():
        ordered = sorted(indices)
        if not ordered:
            continue
        start = prev = ordered[0]
        for i in ordered[1:] + [None]:
            if i is not None and i == prev + 1:
                prev = i
                continue
            n = prev - start + 1
            runs.append({"start_a_retained_index": start, "end_a_retained_index": prev,
                         "start_b_retained_index": start + offset,
                         "end_b_retained_index": prev + offset,
                         "length_retained_rows": n, "offset_b_minus_a_retained_index": offset})
            if i is not None:
                start = prev = i
    return sorted(runs, key=lambda x: (-x["length_retained_rows"], x["start_a_retained_index"],
                                        x["start_b_retained_index"]))


def labels_and_times(s_a: dict, s_b: dict, chain: dict) -> dict:
    ia = np.arange(chain["start_a_retained_index"], chain["end_a_retained_index"] + 1)
    ib = np.arange(chain["start_b_retained_index"], chain["end_b_retained_index"] + 1)
    out = {}
    for policy, key in (("primary", "labels"), ("blank_normal_sensitivity", "labels_blank_normal")):
        ya = np.asarray(s_a[key], dtype=object)[ia]
        yb = np.asarray(s_b[key], dtype=object)[ib]
        # Unknown is a state for agreement diagnostics; these values never define matching or components.
        ya = np.asarray([str(x) if x is not None else "<unknown>" for x in ya], dtype=object)
        yb = np.asarray([str(x) if x is not None else "<unknown>" for x in yb], dtype=object)
        out[policy] = {"state_agreement_n": int(np.sum(ya == yb)),
                       "state_agreement_fraction": float(np.mean(ya == yb)) if len(ya) else None,
                       "known_both_n": int(np.sum(np.isin(ya, CLASSES) & np.isin(yb, CLASSES))),
                       "known_both_equal_n": int(np.sum((ya == yb) & np.isin(ya, CLASSES) & np.isin(yb, CLASSES)))}
        nknown = out[policy]["known_both_n"]
        out[policy]["known_label_agreement_fraction"] = (out[policy]["known_both_equal_n"] / nknown) if nknown else None
    ta = np.asarray(s_a["times"], dtype=np.float64)[ia]
    tb = np.asarray(s_b["times"], dtype=np.float64)[ib]
    dt = np.abs(ta - tb)
    return {"labels_diagnostic_only_not_used_for_matching": out,
            "elapsed_time_abs_difference_diagnostic_only_seconds": {
                "exact_equal_count": int(np.sum(dt == 0)), "count": int(len(dt)),
                "min": float(np.min(dt)) if len(dt) else None,
                "median": float(np.median(dt)) if len(dt) else None,
                "max": float(np.max(dt)) if len(dt) else None}}


def collect_type(sessions: list[dict], kind: str, arrays: list[np.ndarray], threshold: int) -> dict:
    # hash -> session index -> retained row positions
    index = defaultdict(lambda: defaultdict(list))
    valid_counts = {}
    for si, (s, x) in enumerate(zip(sessions, arrays)):
        valid = np.isfinite(x).all(axis=1)
        valid_counts[si] = int(valid.sum())
        for row_i in np.flatnonzero(valid):
            index[signature(x[row_i])][si].append(int(row_i))
    pairs = {}
    shared_sig_sets = defaultdict(set)
    for sig, by_session in index.items():
        sis = sorted(by_session)
        if len(sis) < 2:
            continue
        for a, b in itertools.combinations(sis, 2):
            pair = pairs.setdefault((a, b), {"shared_unique_signatures": 0,
                "matched_row_pairs": 0, "matched_rows_a": set(), "matched_rows_b": set(),
                "offset_to_a_indices": defaultdict(set)})
            pair["shared_unique_signatures"] += 1
            shared_sig_sets[(a, b)].add(sig)
            for ia in by_session[a]:
                for ib in by_session[b]:
                    pair["matched_row_pairs"] += 1
                    pair["matched_rows_a"].add(ia); pair["matched_rows_b"].add(ib)
                    pair["offset_to_a_indices"][ib - ia].add(ia)
    result_pairs = []
    edge_set = set()
    for (a, b), value in sorted(pairs.items()):
        runs = chain_runs(value["offset_to_a_indices"])
        longest = runs[0] if runs else None
        rec_a, rec_b = sessions[a], sessions[b]
        matched_a, matched_b = len(value["matched_rows_a"]), len(value["matched_rows_b"])
        smaller = min(valid_counts[a], valid_counts[b])
        row = {
            "recording_a": rec_a["session_key"], "recording_b": rec_b["session_key"],
            "finite_rows_a": valid_counts[a], "finite_rows_b": valid_counts[b],
            "shared_unique_signatures": value["shared_unique_signatures"],
            "matched_index_pair_count": value["matched_row_pairs"],
            "matched_unique_rows_a": matched_a, "matched_unique_rows_b": matched_b,
            "shared_fraction_of_a_finite_rows": matched_a / valid_counts[a] if valid_counts[a] else None,
            "shared_fraction_of_b_finite_rows": matched_b / valid_counts[b] if valid_counts[b] else None,
            "shared_fraction_of_smaller_record_finite_rows": min(matched_a, matched_b) / smaller if smaller else None,
            "longest_consecutive_exact_run_retained_rows": int(longest["length_retained_rows"]) if longest else 0,
            "longest_run_original_source_rows_a": ([int(rec_a["source_row"][longest["start_a_retained_index"]]),
                                                      int(rec_a["source_row"][longest["end_a_retained_index"]])] if longest else None),
            "longest_run_original_source_rows_b": ([int(rec_b["source_row"][longest["start_b_retained_index"]]),
                                                      int(rec_b["source_row"][longest["end_b_retained_index"]])] if longest else None),
            "longest_run_diagnostics": labels_and_times(rec_a, rec_b, longest) if longest else None,
            "chains_at_least_threshold": [r for r in runs if r["length_retained_rows"] >= threshold],
        }
        if longest and longest["length_retained_rows"] >= threshold:
            edge_set.add((a, b))
        result_pairs.append(row)
    return {"feature_kind": kind, "finite_row_count_by_recording": {
                sessions[i]["session_key"]: n for i, n in valid_counts.items()},
            "cross_record_pair_count": len(result_pairs),
            "pairs_with_any_chain_at_least_threshold": len(edge_set),
            "longest_run_threshold_retained_rows": threshold,
            "pair_results": result_pairs, "component_edges": [list(p) for p in sorted(edge_set)]}


def connected_components(sessions: list[dict], edges: list[list[int]]) -> list[list[str]]:
    graph = {i: set() for i in range(len(sessions))}
    for a, b in edges:
        graph[a].add(b); graph[b].add(a)
    seen, comps = set(), []
    for i in range(len(sessions)):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            v = stack.pop(); comp.append(v)
            for n in graph[v]:
                if n not in seen:
                    seen.add(n); stack.append(n)
        if len(comp) > 1:
            comps.append(sorted(sessions[x]["session_key"] for x in comp))
    return sorted(comps)


def main() -> None:
    cache = joblib.load(CACHE)
    sessions = sorted(cache["sessions"], key=lambda s: s["session_key"])
    for s in sessions:
        if sha256_file(Path(s["path"])) != s["sha256"]:
            raise RuntimeError(f"Live source hash changed: {s['session_key']}")
    imu_columns = list(sessions[0]["imu_columns"])
    full_columns = list(sessions[0]["numeric_columns"])
    imu_arrays = [np.asarray(s["numeric"][:, [full_columns.index(c) for c in imu_columns]], dtype=np.float64) for s in sessions]
    full_arrays = [np.asarray(s["numeric"], dtype=np.float64) for s in sessions]
    threshold = 20
    imu = collect_type(sessions, "12_axis_imu_fully_finite_exact", imu_arrays, threshold)
    full = collect_type(sessions, "42_union_numeric_fully_finite_exact", full_arrays, threshold)
    result = {
        "audit_type": "finite_only_cross_record_exact_row_and_consecutive_chain_audit",
        "cache_sha256": sha256_file(CACHE), "feature_cache_runner_sha256": cache["runner_sha256"],
        "source_record_count": len(sessions), "retained_sample_rows_total": int(sum(len(s["source_row"]) for s in sessions)),
        "retained_row_sampling": "original source rows 0,10,20,... only",
        "fully_finite_vector_requirement": {"imu": "all 12 IMU axes finite", "full": "all 42 union-schema numeric features finite"},
        "matching": "Exact float64 values after -0/+0 normalization; no NaN rows; SHA-256 row signatures; no raw vectors emitted.",
        "consecutive_run_definition": "The exact row signatures match at a constant retained-index offset in both recordings; chain length >=20 is about 2 seconds at the observed 100 Hz raw cadence after 1-in-10 sampling.",
        "threshold_retained_rows": threshold,
        "source_csv_duplicate_hash_groups": [sorted([s["session_key"] for s in sessions if s["sha256"] == digest])
                                                for digest in {s["sha256"] for s in sessions}
                                                if sum(s["sha256"] == digest for s in sessions) > 1],
        "label_time_agreement_role": "Diagnostic summaries for longest exact chains only; never used to discover matches or build graph components.",
        "12_axis_imu": imu, "full_42_numeric": full,
        "components_by_at_least_20_row_12_axis_run": connected_components(sessions, imu["component_edges"]),
        "components_by_at_least_20_row_full42_run": connected_components(sessions, full["component_edges"]),
        "model_fit_or_score_performed": False,
    }
    path = HERE / "cached_data_audit_v2.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(path), "pairs_12axis": imu["cross_record_pair_count"],
                      "pairs_12axis_run_ge20": imu["pairs_with_any_chain_at_least_threshold"],
                      "pairs_full42": full["cross_record_pair_count"],
                      "pairs_full42_run_ge20": full["pairs_with_any_chain_at_least_threshold"],
                      "components_12axis": result["components_by_at_least_20_row_12_axis_run"],
                      "components_full42": result["components_by_at_least_20_row_full42_run"],
                      "model_fit_or_score_performed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
