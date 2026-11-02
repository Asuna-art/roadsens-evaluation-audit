#!/usr/bin/env python3
"""Independently verify context-ablation Markdown tables against checked results.

Does not import or execute write_tables.py. It rebuilds expected rows directly
from checked_summary.json, then checks visible table cells and provenance cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "run_v1"
GRIDS = ("primary", "nonoverlap")
AXES = ("vehicle", "scenario")
MODELS = ("rf", "hgb")
FEATURES = ("imu", "imu_speed", "imu_gps", "imu_all")
CONTRASTS = ("speed_minus_imu", "gps_minus_imu", "all_minus_imu",
             "gps_conditional_speed", "speed_conditional_gps")
DOMAINS = {"vehicle": ("vehicle_1", "vehicle_2", "vehicle_3"),
           "scenario": ("scenario_1", "scenario_2", "scenario_3")}
GRID_LABEL = {"primary": "Primary", "nonoverlap": "Nonoverlap"}
AXIS_LABEL = {"vehicle": "Vehicle", "scenario": "Scenario"}
MODEL_LABEL = {"rf": "RF100", "hgb": "HGB"}
FEATURE_LABEL = {"imu": "IMU", "imu_speed": "+speed", "imu_gps": "+GPS",
                 "imu_all": "+GPS+speed"}
CONTRAST_LABEL = {"speed_minus_imu": "Speed − IMU", "gps_minus_imu": "GPS − IMU",
                  "all_minus_imu": "GPS+speed − IMU", "gps_conditional_speed": "GPS given speed",
                  "speed_conditional_gps": "Speed given GPS"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(x: Any, signed: bool = False) -> str:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"not numeric: {x!r}")
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError(f"nonfinite numeric value: {x!r}")
    if isinstance(x, int):
        return f"{x:+d}" if signed else str(x)
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def indexed(rows, keys, label):
    out = {}
    for row in rows:
        k = tuple(row[x] for x in keys)
        if k in out:
            raise ValueError(f"duplicate {label} key: {k}")
        out[k] = row
    return out


def table_rows(path: Path) -> dict[str, list[list[str]]]:
    """Read pipe tables keyed by the immediately preceding Table caption."""
    lines = path.read_text(encoding="utf-8").splitlines()
    result = {}
    caption = None
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        match = re.match(r"\*\*(Table (?:S\d+|\d+)\.)", s)
        if match:
            caption = match.group(1)[:-1]
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("|"):
                raise ValueError(f"caption {caption} is not followed by a Markdown table")
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                # The Markdown separator line is structural, not displayed data.
                if not row or not all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in row):
                    rows.append(row)
                i += 1
            if caption in result:
                raise ValueError(f"duplicate table caption {caption}")
            result[caption] = rows
            caption = None
            continue
        i += 1
    return result


def expected_tables(s):
    d = indexed(s["domain_rows"], ("grid", "axis", "outer_id", "model", "features"), "domain")
    m = indexed(s["axis_mean_rows"], ("grid", "axis", "model", "features"), "means")
    cd = indexed(s["contrast_domain_rows"], ("grid", "axis", "outer_id", "model", "contrast"), "contrast domain")
    cm = indexed(s["contrast_mean_rows"], ("grid", "axis", "model", "contrast"), "contrast means")
    it = indexed(s["interaction_rows"], ("grid", "model", "contrast"), "interaction")
    expected_keys = {
        "domain": {(g, a, outer, model, feat) for g in GRIDS for a in AXES
                   for outer in DOMAINS[a] for model in MODELS for feat in FEATURES},
        "means": {(g, a, model, feat) for g in GRIDS for a in AXES
                  for model in MODELS for feat in FEATURES},
        "contrast_domain": {(g, a, outer, model, contrast) for g in GRIDS for a in AXES
                            for outer in DOMAINS[a] for model in MODELS for contrast in CONTRASTS},
        "contrast_means": {(g, a, model, contrast) for g in GRIDS for a in AXES
                           for model in MODELS for contrast in CONTRASTS},
        "interaction": {(g, model, contrast) for g in GRIDS for model in MODELS for contrast in CONTRASTS},
    }
    for name, actual in (("domain", set(d)), ("means", set(m)),
                         ("contrast_domain", set(cd)), ("contrast_means", set(cm)),
                         ("interaction", set(it))):
        if actual != expected_keys[name]:
            raise ValueError(f"summary {name} key matrix differs from the frozen table design")
    expected = {}

    rows = []
    for g in GRIDS:
        for a in AXES:
            for model in MODELS:
                rows.append([GRID_LABEL[g], AXIS_LABEL[a], MODEL_LABEL[model]] +
                            [fmt(m[(g, a, model, feat)]["mean_macro_f1"]) for feat in FEATURES])
    expected["Table 10"] = (["Grid", "Axis", "Model", "IMU", "+speed", "+GPS", "+GPS+speed"], rows)

    for grid, tid in (("primary", "Table S17"), ("nonoverlap", "Table S18")):
        rows = []
        for a in AXES:
            for model in MODELS:
                for feat in FEATURES:
                    r = m[(grid, a, model, feat)]
                    rows.append([AXIS_LABEL[a], MODEL_LABEL[model], FEATURE_LABEL[feat],
                                 fmt(r["mean_macro_f1"]), fmt(r["mean_balanced_accuracy"])])
        expected[tid] = (["Axis", "Model", "Features", "Macro-F1", "Balanced accuracy"], rows)

    tidnum = 19
    for grid in GRIDS:
        for axis in AXES:
            rows = []
            for model in MODELS:
                for contrast in CONTRASTS:
                    mean = cm[(grid, axis, model, contrast)]["mean_delta_macro_f1"]
                    gains = [cd[(grid, axis, domain, model, contrast)]["delta_macro_f1"]
                             for domain in DOMAINS[axis]]
                    rows.append([MODEL_LABEL[model], CONTRAST_LABEL[contrast], fmt(mean, True)] +
                                [fmt(x, True) for x in gains])
            expected[f"Table S{tidnum}"] = (["Model", "Contrast", "Mean gain", "Domain 1", "Domain 2", "Domain 3"], rows)
            tidnum += 1

    rows = []
    for grid in GRIDS:
        for model in MODELS:
            vals = [it[(grid, model, contrast)]["vehicle_gain_minus_scenario_gain"] for contrast in CONTRASTS]
            rows.append([GRID_LABEL[grid], MODEL_LABEL[model]] + [fmt(x, True) for x in vals])
    expected["Table S23"] = (["Grid", "Model", "Speed − IMU", "GPS − IMU", "All − IMU",
                              "GPS given speed", "Speed given GPS"], rows)
    return expected


def expected_raw_rows(s, table_id: str):
    """Independently reconstruct manifest raw cells from source summary rows."""
    means = indexed(s["axis_mean_rows"], ("grid", "axis", "model", "features"), "means")
    cdomains = indexed(s["contrast_domain_rows"], ("grid", "axis", "outer_id", "model", "contrast"), "contrast domain")
    cmeans = indexed(s["contrast_mean_rows"], ("grid", "axis", "model", "contrast"), "contrast means")
    interactions = indexed(s["interaction_rows"], ("grid", "model", "contrast"), "interaction")
    if table_id == "Table 10":
        rows = []
        for g in GRIDS:
            for a in AXES:
                for model in MODELS:
                    rows.append([GRID_LABEL[g], AXIS_LABEL[a], MODEL_LABEL[model]] +
                                 [means[(g, a, model, f)]["mean_macro_f1"] for f in FEATURES])
        return rows
    if table_id in ("Table S17", "Table S18"):
        grid = "primary" if table_id == "Table S17" else "nonoverlap"
        return [[AXIS_LABEL[a], MODEL_LABEL[model], FEATURE_LABEL[f],
                 means[(grid, a, model, f)]["mean_macro_f1"],
                 means[(grid, a, model, f)]["mean_balanced_accuracy"]]
                for a in AXES for model in MODELS for f in FEATURES]
    if table_id in ("Table S19", "Table S20", "Table S21", "Table S22"):
        num = int(table_id.removeprefix("Table S"))
        grid = "primary" if num <= 20 else "nonoverlap"
        axis = "vehicle" if num in (19, 21) else "scenario"
        rows = []
        for model in MODELS:
            for contrast in CONTRASTS:
                rows.append([MODEL_LABEL[model], CONTRAST_LABEL[contrast],
                             cmeans[(grid, axis, model, contrast)]["mean_delta_macro_f1"]] +
                            [cdomains[(grid, axis, domain, model, contrast)]["delta_macro_f1"]
                             for domain in DOMAINS[axis]])
        return rows
    if table_id == "Table S23":
        return [[GRID_LABEL[grid], MODEL_LABEL[model]] +
                [interactions[(grid, model, contrast)]["vehicle_gain_minus_scenario_gain"]
                 for contrast in CONTRASTS]
                for grid in GRIDS for model in MODELS]
    raise ValueError(f"unknown table ID: {table_id}")


def raw_label_for_display(table_id: str, column: int, display: str):
    """Map display labels back to their exact source enum/string values."""
    reverse = None
    if table_id in ("Table 10", "Table S23") and column == 1:
        reverse = {v: k for k, v in GRID_LABEL.items()}
    if table_id == "Table 10" and column == 2 or table_id in ("Table S17", "Table S18") and column == 1:
        reverse = {v: k for k, v in AXIS_LABEL.items()}
    if table_id in ("Table 10", "Table S23") and column == 3 or table_id in ("Table S17", "Table S18") and column == 2 or table_id == "Table S23" and column == 2:
        reverse = {v: k for k, v in MODEL_LABEL.items()}
    if table_id in ("Table S17", "Table S18") and column == 3:
        reverse = {v: k for k, v in FEATURE_LABEL.items()}
    if table_id in ("Table S19", "Table S20", "Table S21", "Table S22") and column == 1:
        reverse = {v: k for k, v in MODEL_LABEL.items()}
    if table_id in ("Table S19", "Table S20", "Table S21", "Table S22") and column == 2:
        reverse = {v: k for k, v in CONTRAST_LABEL.items()}
    if reverse is None:
        return display
    try:
        return reverse[display]
    except KeyError as e:
        raise ValueError(f"unexpected display label {display!r} at {table_id} column {column}") from e


def check_table_file(path: Path, expected, manifest_cells, table_cell_counts) -> int:
    actual = table_rows(path)
    expected_ids = set(expected)
    if set(actual) != expected_ids:
        raise ValueError(f"{path.name} table IDs differ; expected={sorted(expected_ids)}, actual={sorted(actual)}")
    count = 0
    for tid, (headers, rows) in expected.items():
        got = actual[tid]
        wanted = [headers] + rows
        if got != wanted:
            limit = min(len(got), len(wanted))
            mismatch = next((i for i in range(limit) if got[i] != wanted[i]), limit)
            raise ValueError(f"{tid} mismatch in {path.name} at displayed row {mismatch+1}; expected {wanted[mismatch:mismatch+1]}, got {got[mismatch:mismatch+1]}")
        n = sum(map(len, rows))
        count += n
        table_cell_counts[tid] = n
        manifest_cells[tid] = rows
    return count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=HERE / "table_validation_v1.json")
    a = p.parse_args()
    summary = a.run_dir / "checked_summary.json"
    main_md = HERE / "tables_main.md"
    supp_md = HERE / "tables_supplement.md"
    manifest_path = HERE / "table_numeric_manifest.json"
    for path in (summary, main_md, supp_md, manifest_path):
        if not path.is_file():
            raise SystemExit(f"required input not found: {path}")

    data = json.loads(summary.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_sha = digest(summary)
    if manifest.get("source_summary_sha256") != summary_sha:
        raise ValueError("numeric manifest source SHA does not match checked_summary.json")
    expected = expected_tables(data)
    manifest_rows = manifest.get("tables")
    if not isinstance(manifest_rows, list):
        raise ValueError("numeric manifest must contain a tables list")
    manifest_index = {}
    for entry in manifest_rows:
        key = (entry["table"], entry["row"], entry["column"])
        if key in manifest_index:
            raise ValueError(f"duplicate numeric manifest cell {key}")
        manifest_index[key] = entry

    cell_counts = {}
    check_rows_by_table = {}
    total = 0
    for path, subset in ((main_md, {"Table 10": expected["Table 10"]}),
                         (supp_md, {k: v for k, v in expected.items() if k != "Table 10"})):
        total += check_table_file(path, subset, check_rows_by_table, cell_counts)

    expected_manifest = {}
    for tid, rows in check_rows_by_table.items():
        raw_rows = expected_raw_rows(data, tid)
        if len(raw_rows) != len(rows):
            raise ValueError(f"internal expected raw-row mismatch for {tid}")
        for ri, row in enumerate(rows, 1):
            for ci, formatted in enumerate(row, 1):
                expected_manifest[(tid, ri, ci)] = (formatted, raw_rows[ri - 1][ci - 1])
    if set(manifest_index) != set(expected_manifest):
        raise ValueError(f"manifest cell coverage mismatch: expected {len(expected_manifest)}, got {len(manifest_index)}")
    for key, (formatted, raw_expected) in expected_manifest.items():
        entry = manifest_index[key]
        if entry.get("formatted") != formatted:
            raise ValueError(f"manifest formatted value mismatch at {key}")
        # Re-check numeric raw values independently from the summary-derived
        # display text. Labels must be raw/display-identical.
        raw_manifest = entry.get("raw")
        if isinstance(raw_expected, str):
            raw_expected = raw_label_for_display(key[0], key[2], raw_expected)
        if raw_manifest != raw_expected:
            raise ValueError(f"manifest raw value differs from checked summary at {key}")
        if isinstance(raw_manifest, (int, float)) and not isinstance(raw_manifest, bool):
            if fmt(raw_manifest, signed=(formatted.startswith("+") or formatted.startswith("-"))) != formatted:
                raise ValueError(f"manifest raw number does not format to display at {key}")
        else:
            if raw_label_for_display(key[0], key[2], formatted) != raw_manifest:
                raise ValueError(f"manifest raw label does not map to displayed label at {key}")

    out = {"status": "passed", "checked_summary": str(summary.resolve()),
           "checked_summary_sha256": summary_sha,
           "tables_main_sha256": digest(main_md), "tables_supplement_sha256": digest(supp_md),
           "numeric_manifest_sha256": digest(manifest_path), "table_count": len(expected),
           "data_cell_count": total, "table_cell_counts": cell_counts,
           "checks": ["table IDs and row order", "all displayed values from checked summary",
                      "feature/model/grid/axis combinations", "signed contrast direction and formatting",
                      "manifest cell coverage and displayed values", "manifest source-summary SHA"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
