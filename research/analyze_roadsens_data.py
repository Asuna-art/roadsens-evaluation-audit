"""Schema/session/annotation audit only; performs no model fitting or scoring."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_JSON = ROOT / "data_feasibility_manifest.json"
OUT_CSV = ROOT / "session_inventory.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_csv(path: Path, group: str) -> dict:
    header = None
    rows = 0
    field_finite = Counter()
    field_blank = Counter()
    field_nonfinite = Counter()
    annotation_counts = Counter()
    duration_counts = Counter()
    observed = defaultdict(set)
    time_min = time_max = None
    malformed_rows = 0
    # Degree sign in the source CSV header is Windows-1252; all values are ASCII.
    with path.open("r", encoding="cp1252", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or (len(row) == 1 and not row[0]):
                continue
            rows += 1
            if len(row) != len(header):
                malformed_rows += 1
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                else:
                    row = row[:len(header)]
            values = dict(zip(header, row))
            for key, raw in values.items():
                v = raw.strip()
                if not v:
                    field_blank[key] += 1
                    continue
                try:
                    x = float(v)
                except ValueError:
                    if key == "annotation_text":
                        annotation_counts[v] += 1
                    elif key == "annotation_millisecond_press_duration":
                        duration_counts[v] += 1
                    continue
                if math.isfinite(x):
                    field_finite[key] += 1
                    if key.startswith("Temperature") or key in {"Humidity (%)", "Cloud Cover (%)", "Wind Speed (km/h)", "barometer_pressure", "barometer_relativeAltitude", "location_latitude", "location_longitude", "location_verticalAccuracy"}:
                        if len(observed[key]) < 100000:
                            observed[key].add(x)
                    if key == "seconds_elapsed":
                        time_min = x if time_min is None else min(time_min, x)
                        time_max = x if time_max is None else max(time_max, x)
                else:
                    field_nonfinite[key] += 1
                    if key == "annotation_text":
                        annotation_counts[v] += 1
                    elif key == "annotation_millisecond_press_duration":
                        duration_counts[v] += 1
    assert header is not None
    return {
        "session_file": path.name,
        "session_id": path.stem.split(" ")[0],
        "group": group,
        "file_bytes": path.stat().st_size,
        "rows": rows,
        "header": header,
        "n_fields": len(header),
        "malformed_row_count": malformed_rows,
        "seconds_elapsed_min": time_min,
        "seconds_elapsed_max": time_max,
        "annotation_text_counts": dict(annotation_counts),
        "annotation_duration_counts": dict(duration_counts),
        "field_finite": dict(field_finite),
        "field_blank": dict(field_blank),
        "field_nonfinite": dict(field_nonfinite),
        "selected_field_unique_counts": {k: len(v) for k, v in observed.items()},
        "sha256": sha256(path),
    }


def main() -> None:
    combined_root = DATA / "inner"
    groups = [
        ("normal", combined_root / "combined_normal"),
        ("anomaly_recording", combined_root / "combined_anomalies"),
    ]
    rows = []
    for group, folder in groups:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = sorted(folder.glob("*.csv"), key=lambda p: (int(p.stem.split()[0]) if p.stem.split()[0].isdigit() else 10**9, p.name))
        if not files:
            raise RuntimeError(f"no session CSVs in {folder}")
        for i, path in enumerate(files, 1):
            print(f"Scanning {group}: {i}/{len(files)} {path.name}", flush=True)
            rows.append(count_csv(path, group))

    field_presence = defaultdict(Counter)
    group_rows = defaultdict(int)
    group_sessions = Counter()
    all_annotation = defaultdict(Counter)
    for rec in rows:
        g = rec["group"]
        group_sessions[g] += 1
        group_rows[g] += rec["rows"]
        all_annotation[g].update(rec["annotation_text_counts"])
        for field in rec["header"]:
            field_presence[field][g] += 1

    groups_order = ["normal", "anomaly_recording"]
    schema = {}
    all_fields = sorted(field_presence)
    for field in all_fields:
        vals = {}
        for group in groups_order:
            files_present = field_presence[field][group]
            den = group_sessions[group]
            denom_rows = sum(r["rows"] for r in rows if r["group"] == group and field in r["header"])
            finite = sum(r["field_finite"].get(field, 0) for r in rows if r["group"] == group and field in r["header"])
            vals[group] = {
                "files_with_field": files_present,
                "file_presence_rate": files_present / den if den else None,
                "rows_where_field_is_present": denom_rows,
                "finite_values": finite,
                "finite_rate_among_present_rows": finite / denom_rows if denom_rows else None,
            }
        schema[field] = vals

    summaries = {}
    for group in groups_order:
        selected = [r for r in rows if r["group"] == group]
        summaries[group] = {
            "sessions": len(selected),
            "rows": sum(r["rows"] for r in selected),
            "total_bytes": sum(r["file_bytes"] for r in selected),
            "field_count_distribution": dict(Counter(r["n_fields"] for r in selected)),
            "annotation_text_counts": dict(all_annotation[group]),
            "sessions_with_nonempty_annotation_text": sum(bool(r["annotation_text_counts"]) for r in selected),
            "sessions_with_annotation_columns": sum("annotation_text" in r["header"] for r in selected),
            "sessions_with_GIS_location_columns": sum({"location_longitude", "location_latitude"}.issubset(r["header"]) for r in selected),
        }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Combined ZIP session CSV inventory only; no model/score; videos excluded",
        "source_data_directory": str(DATA / "inner"),
        "group_summaries": summaries,
        "schema_by_field": schema,
        "sessions": rows,
    }
    for p in (OUT_JSON, OUT_CSV):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with OUT_CSV.open("x", newline="", encoding="utf-8-sig") as f:
        fields = ["session_id", "session_file", "group", "file_bytes", "rows", "n_fields", "seconds_elapsed_min", "seconds_elapsed_max", "malformed_row_count", "annotation_text_counts", "annotation_duration_counts", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            item = {k: r.get(k) for k in fields}
            item["annotation_text_counts"] = json.dumps(item["annotation_text_counts"], ensure_ascii=False)
            item["annotation_duration_counts"] = json.dumps(item["annotation_duration_counts"], ensure_ascii=False)
            writer.writerow(item)
    print(json.dumps({"groups": summaries, "n_fields": len(all_fields), "field_names": all_fields, "nonempty_annotation_counts": {k: dict(v) for k, v in all_annotation.items()}}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
