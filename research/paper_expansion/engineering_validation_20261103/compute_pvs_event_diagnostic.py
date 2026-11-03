"""Post-hoc diagnostic using only saved PVS outer predictions and frozen arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
EV = Path(__file__).resolve().parent
PROTO = EV / "pvs_event_protocol.json"
AMENDMENT = EV / "pvs_event_reporting_amendment.json"
FEATURES = BASE / "external_pvs_v1" / "features.npz"
MANIFEST = BASE / "external_pvs_v1" / "feature_manifest.json"
ROOT = BASE / "external_pvs_v1" / "run_primary_v1"
SCHEMA = EV / "pvs_schema_audit.json"
OUT = EV / "pvs_event_results.json"
TOLS = [0.0, 0.5]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def assignment(adj: list[list[int]], n_alarms: int) -> list[tuple[int, int]]:
    """Maximum matching via deterministic DFS augmenting paths; events/alarm indices ordered."""
    owner = [-1] * n_alarms

    def visit(event: int, seen: set[int]) -> bool:
        for alarm in adj[event]:
            if alarm in seen:
                continue
            seen.add(alarm)
            if owner[alarm] < 0 or visit(owner[alarm], seen):
                owner[alarm] = event
                return True
        return False

    for event in range(len(adj)):
        visit(event, set())
    return sorted([(event, alarm) for alarm, event in enumerate(owner) if event >= 0])


def main() -> None:
    protocol = json.loads(PROTO.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if sha(PROTO) != "2e56a31b564da2d1cb2ba669c23b4c753daf86fb613d2813ab21c63796ad6d05":
        raise RuntimeError("Protocol hash mismatch; refusing to proceed")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    trips = {int(x["trip"]): x["positive_annotation_segments_gap_le_0.1s"] for x in schema["trips"]}
    with np.load(FEATURES, allow_pickle=False) as z:
        y = z["y"].astype(int)
        trip = z["trip"].astype(int)
        anchor = z["anchor_row"].astype(int)
        atime = z["anchor_time"].astype(float)
    n = len(y)
    raw_labels = {}
    raw_label_paths = {}
    for tid in range(1, 10):
        path = BASE / "external_pvs_v1" / "data" / f"PVS_{tid}" / "dataset_labels.csv"
        a = pd.read_csv(path, usecols=["no_speed_bump", "speed_bump_asphalt", "speed_bump_cobblestone"])
        a = a[["no_speed_bump", "speed_bump_asphalt", "speed_bump_cobblestone"]].to_numpy()
        valid = np.isin(a, [0, 1]).all(axis=1) & (a.sum(axis=1) == 1)
        raw_labels[tid] = (valid, valid & (a[:, 0] == 1))
        raw_label_paths[tid] = path
    covered: dict[tuple[str, str], list[int]] = {}
    outputs = []
    input_hashes = {str(PROTO.relative_to(BASE)): sha(PROTO), str(AMENDMENT.relative_to(BASE)): sha(AMENDMENT), str(FEATURES.relative_to(BASE)): sha(FEATURES),
                    str(MANIFEST.relative_to(BASE)): sha(MANIFEST), str(SCHEMA.relative_to(BASE)): sha(SCHEMA)}
    input_hashes[str(Path(__file__).relative_to(BASE))] = sha(Path(__file__))
    for path in raw_label_paths.values():
        input_hashes[str(path.relative_to(BASE))] = sha(path)
    for axis in protocol["axes_order"]:
        for held in range(1, 4):
            outer = f"{axis}_{held}"
            selpath = ROOT / "outer_folds" / outer / "selection.json"
            selection = json.loads(selpath.read_text(encoding="utf-8"))
            input_hashes[str(selpath.relative_to(BASE))] = sha(selpath)
            test_trip = ((trip - 1) // 3 + 1 == held if axis == "vehicle" else
                         ((trip - 1) % 3 + 1) == held)
            test_ix = np.flatnonzero(test_trip)
            keybase = (axis, outer)
            covered[keybase] = test_ix.tolist()
            for selector in protocol["selectors_order"]:
                candidate = selection["selectors"][selector]["selected_candidate_id"]
                predpath = ROOT / "fits" / f"{outer}__outer_refit__{candidate}" / "predictions.npz"
                with np.load(predpath, allow_pickle=False) as z:
                    pred = {k: z[k] for k in z.files}
                input_hashes[str(predpath.relative_to(BASE))] = sha(predpath)
                # Saved predictions carry feature-row test_indices; verify exact identity and target integrity.
                ix = pred.get("test_indices", pred.get("indices"))
                if ix is None:
                    raise ValueError(f"Missing test indices in {predpath}")
                ix = np.asarray(ix, dtype=int).reshape(-1)
                yt = np.asarray(pred["y_true"]).reshape(-1).astype(int)
                yp = np.asarray(pred["y_pred"]).reshape(-1).astype(int)
                if len(ix) != len(yt) or len(ix) != len(yp) or not np.array_equal(np.sort(ix), test_ix):
                    raise AssertionError(f"Test index mismatch: {predpath}")
                if not np.array_equal(yt, y[ix]):
                    raise AssertionError(f"Saved y_true differs from frozen labels: {predpath}")
                # Work in feature index order to preserve deterministic chronological cluster construction.
                order = np.argsort(ix, kind="stable")
                ix, yt, yp = ix[order], yt[order], yp[order]
                domain_records = []
                for tid in sorted(set(trip[ix].tolist())):
                    local = np.flatnonzero(trip[ix] == tid)
                    li = ix[local]
                    rows, times, labels, hats = anchor[li], atime[li], yt[local], yp[local]
                    if not np.all(np.diff(rows) > 0):
                        raise AssertionError("Decision anchors not strictly increasing")
                    # Cluster consecutive positive alarms only under all three protocol criteria.
                    alarms: list[np.ndarray] = []
                    active: list[int] = []
                    for j in range(len(hats)):
                        if hats[j] == 1:
                            if active and not (hats[j-1] == 1 and rows[j] - rows[j-1] == 25 and times[j] - times[j-1] <= .251):
                                alarms.append(np.asarray(active, dtype=int)); active = []
                            active.append(j)
                        elif active:
                            alarms.append(np.asarray(active, dtype=int)); active = []
                    if active:
                        alarms.append(np.asarray(active, dtype=int))
                    raw_segments = trips.get(int(tid), [])
                    for tol in TOLS:
                        events, excluded = [], 0
                        for s in raw_segments:
                            arow, brow = int(s["start_row_zero_based"]), int(s["end_row_zero_based"])
                            eligible = np.flatnonzero((rows >= arow) & (rows <= brow))
                            if not len(eligible):
                                excluded += 1
                                continue
                            events.append({"start": float(s["start_timestamp"]), "end": float(s["end_timestamp"]),
                                           "start_row": arow, "end_row": brow, "eligible": eligible})
                        adj = []
                        for ev in events:
                            edges = []
                            for ai, cluster in enumerate(alarms):
                                if np.any((hats[cluster] == 1) & (times[cluster] >= ev["start"] - tol) & (times[cluster] <= ev["end"] + tol)):
                                    edges.append(ai)
                            adj.append(edges)
                        pairs = assignment(adj, len(alarms))
                        alarms_with_event_edge = {ai for edge_list in adj for ai in edge_list}
                        matched_alarm_ids = {ai for _, ai in pairs}
                        duplicate_event_alarms = len(alarms_with_event_edge - matched_alarm_ids)
                        nonbump_alarms = len(set(range(len(alarms))) - alarms_with_event_edge)
                        unmatched = len(alarms) - len(pairs)
                        if unmatched != nonbump_alarms + duplicate_event_alarms:
                            raise AssertionError("Unmatched alarm categorization identity failed")
                        pairs_by_event = {e: a for e, a in pairs}
                        delays = []
                        for ei, ai in pairs:
                            c = alarms[ai]
                            eligible_alarm = c[(hats[c] == 1) & (times[c] >= events[ei]["start"] - tol) &
                                               (times[c] <= events[ei]["end"] + tol)]
                            if len(eligible_alarm):
                                delays.append(float(times[eligible_alarm[0]] - events[ei]["start"]))
                        raw_valid, raw_nonbump = raw_labels[int(tid)]
                        exposed = slice(int(rows[0]), int(rows[-1]) + 1)
                        nonbumprows = int(raw_nonbump[exposed].sum())
                        assessedrows = int(raw_valid[exposed].sum())
                        domain_records.append({"trip": int(tid), "axis": axis, "outer_id": outer, "selector": selector,
                            "candidate": candidate, "tolerance_seconds": tol, "evaluated_intervals": len(events),
                            "excluded_intervals_no_exact_anchor": excluded, "matched_intervals": len(pairs),
                            "interval_recall": len(pairs)/len(events) if events else None,
                            "alarm_clusters": len(alarms), "matched_alarms": len(pairs), "unmatched_alarm_clusters": unmatched,
                            "nonbump_alarm_clusters": nonbump_alarms,
                            "duplicate_event_alarm_clusters": duplicate_event_alarms,
                            "matched_alarm_fraction": len(pairs)/len(alarms) if alarms else None,
                            "nonbump_exposure_hours": nonbumprows*.01/3600,
                            "assessed_exposure_hours": assessedrows*.01/3600,
                            "unmatched_alarms_per_nonbump_hour": unmatched/(nonbumprows*.01/3600) if nonbumprows else None,
                            "rate_nonbump_alarms_per_nonbump_hour": nonbump_alarms/(nonbumprows*.01/3600) if nonbumprows else None,
                            "matched_first_alarm_offsets_seconds": delays})
                outputs.extend(domain_records)
    # Coverage must partition every frozen feature index exactly once per axis/selector combination.
    for axis in protocol["axes_order"]:
        domains = [np.asarray(covered[(axis, f"{axis}_{k}")], dtype=int) for k in range(1, 4)]
        all_ix = np.concatenate(domains)
        if len(all_ix) != n or not np.array_equal(np.sort(all_ix), np.arange(n)):
            raise AssertionError(f"Outer domains do not cover features exactly once for {axis}")
    aggregates = []
    for axis in protocol["axes_order"]:
        for selector in protocol["selectors_order"]:
            for tol in TOLS:
                rs = [r for r in outputs if r["axis"] == axis and r["selector"] == selector and r["tolerance_seconds"] == tol]
                evaluated = sum(r["evaluated_intervals"] for r in rs)
                matched = sum(r["matched_intervals"] for r in rs)
                alarms = sum(r["alarm_clusters"] for r in rs)
                unmatched = sum(r["unmatched_alarm_clusters"] for r in rs)
                nonbump_alarms = sum(r["nonbump_alarm_clusters"] for r in rs)
                duplicate_alarms = sum(r["duplicate_event_alarm_clusters"] for r in rs)
                if unmatched != nonbump_alarms + duplicate_alarms:
                    raise AssertionError("Pooled unmatched alarm categorization identity failed")
                nonbump = sum(r["nonbump_exposure_hours"] for r in rs)
                delays = [d for r in rs for d in r["matched_first_alarm_offsets_seconds"]]
                aggregates.append({"axis": axis, "selector": selector, "tolerance_seconds": tol,
                    "evaluated_intervals": evaluated, "matched_intervals": matched,
                    "interval_recall": matched/evaluated if evaluated else None,
                    "alarm_clusters": alarms, "matched_alarm_fraction": matched/alarms if alarms else None,
                    "unmatched_alarm_clusters": unmatched, "nonbump_exposure_hours": nonbump,
                    "nonbump_alarm_clusters": nonbump_alarms,
                    "duplicate_event_alarm_clusters": duplicate_alarms,
                    "unmatched_alarms_per_nonbump_hour": unmatched/nonbump if nonbump else None,
                    "rate_nonbump_alarms_per_nonbump_hour": nonbump_alarms/nonbump if nonbump else None,
                    "matched_first_alarm_offsets_seconds": delays,
                    "delay_median_seconds": float(np.median(delays)) if delays else None,
                    "delay_p90_seconds": float(np.quantile(delays, .9)) if delays else None})
    result = {"protocol_id": protocol["protocol_id"], "protocol_sha256": sha(PROTO),
              "reporting_amendment_id": amendment["amendment_id"], "reporting_amendment_sha256": sha(AMENDMENT),
              "interpretation": protocol["role"], "integrity": {"all_checks_passed": True,
              "feature_label_exact_match": True, "outer_partition_exact_once_by_axis": True,
              "source_artifact_sha256": input_hashes}, "by_trip_domain_selector_tolerance": outputs,
              "pooled_by_axis_selector_tolerance": aggregates}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
