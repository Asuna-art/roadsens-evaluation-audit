#!/usr/bin/env python3
"""Format frozen context-ablation checked results as manuscript tables.

This is an output-only formatter: it does not fit models or alter the source
summary. Run after run_v1/checked_summary.json has been produced and checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE / "run_v1" / "checked_summary.json"
GRIDS = ("primary", "nonoverlap")
AXES = ("vehicle", "scenario")
MODELS = ("rf", "hgb")
FEATURES = ("imu", "imu_speed", "imu_gps", "imu_all")
CONTRASTS = (
    "speed_minus_imu",
    "gps_minus_imu",
    "all_minus_imu",
    "gps_conditional_speed",
    "speed_conditional_gps",
)
DOMAIN_IDS = {"vehicle": ("vehicle_1", "vehicle_2", "vehicle_3"),
              "scenario": ("scenario_1", "scenario_2", "scenario_3")}

GRID_LABEL = {"primary": "Primary", "nonoverlap": "Nonoverlap"}
AXIS_LABEL = {"vehicle": "Vehicle", "scenario": "Scenario"}
MODEL_LABEL = {"rf": "RF100", "hgb": "HGB"}
FEATURE_LABEL = {"imu": "IMU", "imu_speed": "+speed", "imu_gps": "+GPS",
                 "imu_all": "+GPS+speed"}
CONTRAST_LABEL = {
    "speed_minus_imu": "Speed − IMU",
    "gps_minus_imu": "GPS − IMU",
    "all_minus_imu": "GPS+speed − IMU",
    "gps_conditional_speed": "GPS | speed",
    "speed_conditional_gps": "Speed | GPS",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value: Any, *, signed: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric value, got {value!r}")
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    result = f"{value:+.4f}" if signed else f"{value:.4f}"
    return result


def rows_by_key(rows: Iterable[dict], keys: tuple[str, ...], name: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for row in rows:
        try:
            key = tuple(row[k] for k in keys)
        except KeyError as e:
            raise ValueError(f"{name} missing required field {e.args[0]!r}") from e
        if key in out:
            raise ValueError(f"duplicate {name} key: {key}")
        out[key] = row
    return out


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


class TableWriter:
    def __init__(self) -> None:
        self.manifest: list[dict[str, Any]] = []

    def table(self, table_id: str, title: str, headers: list[str],
              body: list[list[tuple[Any, str]]]) -> str:
        rendered: list[list[str]] = []
        for ri, row in enumerate(body, 1):
            if len(row) != len(headers):
                raise ValueError(f"{table_id} row {ri}: expected {len(headers)} cells, got {len(row)}")
            rendered_row = []
            for ci, (raw, fmt) in enumerate(row, 1):
                rendered_row.append(fmt)
                self.manifest.append({"table": table_id, "row": ri, "column": ci,
                                      "header": headers[ci - 1], "raw": raw,
                                      "formatted": fmt})
            rendered.append(rendered_row)
        return f"**{title}**\n\n{md_table(headers, rendered)}\n"


def cell(value: Any, label: str | None = None, signed: bool = False) -> tuple[Any, str]:
    if label is not None:
        return value, label
    return value, number(value, signed=signed)


def build(summary: dict[str, Any], summary_path: Path) -> tuple[str, str, dict]:
    required = ("domain_rows", "axis_mean_rows", "contrast_domain_rows",
                "contrast_mean_rows", "interaction_rows")
    for field in required:
        if field not in summary or not isinstance(summary[field], list):
            raise ValueError(f"summary must contain list field {field!r}")
    domains = rows_by_key(summary["domain_rows"],
                          ("grid", "axis", "outer_id", "model", "features"), "domain_rows")
    means = rows_by_key(summary["axis_mean_rows"],
                        ("grid", "axis", "model", "features"), "axis_mean_rows")
    cdomains = rows_by_key(summary["contrast_domain_rows"],
                           ("grid", "axis", "outer_id", "model", "contrast"), "contrast_domain_rows")
    cmeans = rows_by_key(summary["contrast_mean_rows"],
                         ("grid", "axis", "model", "contrast"), "contrast_mean_rows")
    interactions = rows_by_key(summary["interaction_rows"],
                               ("grid", "model", "contrast"), "interaction_rows")

    expected = {"domain_rows": 96, "axis_mean_rows": 32, "contrast_domain_rows": 120,
                "contrast_mean_rows": 40, "interaction_rows": 20}
    for name, count in expected.items():
        if len(summary[name]) != count:
            raise ValueError(f"{name}: expected {count} rows, got {len(summary[name])}")
    expected_domains = {(g, a, d, m, f) for g in GRIDS for a in AXES
                        for d in DOMAIN_IDS[a] for m in MODELS for f in FEATURES}
    if set(domains) != expected_domains:
        raise ValueError("domain_rows keys do not match the frozen grid/axis/domain/model/feature matrix")
    expected_means = {(g, a, m, f) for g in GRIDS for a in AXES for m in MODELS for f in FEATURES}
    if set(means) != expected_means:
        raise ValueError("axis_mean_rows keys do not match the frozen matrix")
    expected_cd = {(g, a, d, m, c) for g in GRIDS for a in AXES for d in DOMAIN_IDS[a]
                   for m in MODELS for c in CONTRASTS}
    if set(cdomains) != expected_cd:
        raise ValueError("contrast_domain_rows keys do not match the frozen matrix")
    expected_cm = {(g, a, m, c) for g in GRIDS for a in AXES for m in MODELS for c in CONTRASTS}
    if set(cmeans) != expected_cm:
        raise ValueError("contrast_mean_rows keys do not match the frozen matrix")
    expected_i = {(g, m, c) for g in GRIDS for m in MODELS for c in CONTRASTS}
    if set(interactions) != expected_i:
        raise ValueError("interaction_rows keys do not match the frozen matrix")

    w = TableWriter()
    main_sections: list[str] = ["# Context-ablation results\n"]
    body = []
    for g in GRIDS:
        for a in AXES:
            for m in MODELS:
                vals = []
                for f in FEATURES:
                    r = means[(g, a, m, f)]
                    vals.append(cell(r["mean_macro_f1"]))
                body.append([cell(g, GRID_LABEL[g]), cell(a, AXIS_LABEL[a]),
                             cell(m, MODEL_LABEL[m]), *vals])
    main_sections.append(w.table("Table 10", "Table 10. Macro-F1 by context feature set and outer grouping.",
                                 ["Grid", "Axis", "Model", "IMU", "+speed", "+GPS", "+GPS+speed"], body))
    main_sections.append("GPS features use the protocol-defined coordinates. Values are macro-F1 means over the three outer domains.\n")

    supp: list[str] = ["# Supplementary context-ablation tables\n"]
    for grid, tid in (("primary", "Table S17"), ("nonoverlap", "Table S18")):
        body = []
        for a in AXES:
            for m in MODELS:
                for f in FEATURES:
                    r = means[(grid, a, m, f)]
                    body.append([cell(a, AXIS_LABEL[a]), cell(m, MODEL_LABEL[m]),
                                 cell(f, FEATURE_LABEL[f]), cell(r["mean_macro_f1"]),
                                 cell(r["mean_balanced_accuracy"])])
        supp.append(w.table(tid, f"{tid}. Domain-mean performance for the {GRID_LABEL[grid].lower()} grid.",
                            ["Axis", "Model", "Features", "Macro-F1", "Balanced accuracy"], body))

    table_num = 19
    for grid in GRIDS:
        for axis in AXES:
            tid = f"Table S{table_num}"
            body = []
            for m in MODELS:
                for c in CONTRASTS:
                    mean = cmeans[(grid, axis, m, c)]["mean_delta_macro_f1"]
                    dvals = [cdomains[(grid, axis, d, m, c)]["delta_macro_f1"]
                             for d in DOMAIN_IDS[axis]]
                    body.append([cell(m, MODEL_LABEL[m]), cell(c, CONTRAST_LABEL[c]),
                                 cell(mean, signed=True), *(cell(x, signed=True) for x in dvals)])
            supp.append(w.table(tid, f"{tid}. {GRID_LABEL[grid]} {AXIS_LABEL[axis].lower()}-domain macro-F1 contrasts.",
                                ["Model", "Contrast", "Mean gain", "Domain 1", "Domain 2", "Domain 3"], body))
            table_num += 1

    body = []
    for g in GRIDS:
        for m in MODELS:
            vals = [interactions[(g, m, c)]["vehicle_gain_minus_scenario_gain"] for c in CONTRASTS]
            body.append([cell(g, GRID_LABEL[g]), cell(m, MODEL_LABEL[m]),
                         *(cell(v, signed=True) for v in vals)])
    supp.append(w.table("Table S23", "Table S23. Vehicle-minus-scenario differences in domain-mean gains.",
                        ["Grid", "Model", "Speed − IMU", "GPS − IMU", "All − IMU",
                         "GPS | speed", "Speed | GPS"], body))
    supp.append("All contrast values are signed macro-F1 differences. Domain-level repeats and axes are not independent observations; no inferential test is implied.\n")

    manifest = {"source_summary": str(summary_path), "source_summary_sha256": sha256(summary_path),
                "format": "four decimal places for noninteger numeric values; integer values remain integers",
                "tables": w.manifest}
    return "\n".join(main_sections), "\n".join(supp), manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    source = args.summary.resolve()
    if not source.is_file():
        raise SystemExit(f"checked summary not found: {source}")
    summary = json.loads(source.read_text(encoding="utf-8"))
    main_md, supp_md, manifest = build(summary, source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables_main.md").write_text(main_md, encoding="utf-8")
    (args.out_dir / "tables_supplement.md").write_text(supp_md, encoding="utf-8")
    (args.out_dir / "table_numeric_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
