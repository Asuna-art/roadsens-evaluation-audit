"""Inventory the bounded public road-sensor artifacts collected for feasibility review.

This is metadata-only: it hashes local downloads and inspects archive headers and
member sizes without extracting the 305 MB uncompressed STRIDE archive.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_summary(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return {"bytes": path.stat().st_size, "sha256": sha256(path), "header": header, "rows": rows}


def main() -> None:
    result = {"generated_by": Path(__file__).name, "artifacts": {}}
    zip_path = ROOT / "stride_road_data.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            anomaly = [m for m in members if m.filename.startswith("Road Data/Road Anomalies/")]
            grouped = {}
            matrix = []
            for member in anomaly:
                parts = member.filename.split("/")
                if len(parts) >= 4:
                    folder = parts[2]
                    row = grouped.setdefault(folder, {"files": 0, "uncompressed_bytes": 0, "members": []})
                    row["files"] += 1
                    row["uncompressed_bytes"] += member.file_size
                    row["members"].append(parts[-1])
            for folder in sorted(grouped, key=lambda s: int(s.split(".", 1)[0])):
                pref = f"Road Data/Road Anomalies/{folder}/"
                md_member = z.getinfo(pref + "Metadata.csv")
                with z.open(md_member) as f:
                    meta = next(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")))
                row = {"folder": folder, "label": folder.split(".", 1)[1].strip(),
                       "device_id": meta.get("device id"), "device_name": meta.get("device name"),
                       "recording_epoch_time": meta.get("recording epoch time"),
                       "recording_time": meta.get("recording time"),
                       "timezone": meta.get("recording timezone"), "app_version": meta.get("appVersion")}
                for sensor in ("Accelerometer.csv", "Gyroscope.csv"):
                    m = z.getinfo(pref + sensor)
                    with z.open(m) as f:
                        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig"))
                        header = next(reader, [])
                        count = sum(1 for _ in reader)
                    base = sensor.removesuffix(".csv").lower()
                    row[f"{base}_bytes"] = m.file_size
                    row[f"{base}_rows"] = count
                    row[f"{base}_axes"] = ",".join(x for x in header if x.lower() in {"x", "y", "z"})
                matrix.append(row)
            matrix_path = ROOT / "stride_recording_matrix.csv"
            if matrix:
                with matrix_path.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=list(matrix[0]))
                    writer.writeheader()
                    writer.writerows(matrix)
            # Hash every local RoadSens CSV and compare against source CSV payloads in STRIDE.
            local_csv_hashes = {}
            roadsens_data = ROOT.parent / "data"
            if roadsens_data.exists():
                for local in roadsens_data.rglob("*.csv"):
                    local_csv_hashes.setdefault(sha256(local), []).append(str(local.relative_to(roadsens_data)))
            stride_matches = []
            for m in members:
                if m.filename.lower().endswith(".csv"):
                    with z.open(m) as f:
                        h = hashlib.sha256()
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h.update(chunk)
                    if h.hexdigest() in local_csv_hashes:
                        stride_matches.append({"stride_member": m.filename,
                                               "roadsens_files": local_csv_hashes[h.hexdigest()]})
            result["artifacts"]["stride_figshare"] = {
                "path": zip_path.name,
                "bytes": zip_path.stat().st_size,
                "sha256": sha256(zip_path),
                "zip_file_count": len(members),
                "uncompressed_bytes": sum(m.file_size for m in members),
                "anomaly_recordings": grouped,
                "recording_matrix_csv": matrix_path.name,
                "roadsens_exact_csv_hash_matches": stride_matches,
            }
    csvs = {}
    for name in ("bachesf.csv", "normalf.csv", "topef.csv"):
        path = ROOT / name
        if path.exists():
            csvs[name] = csv_summary(path)
    if csvs:
        result["artifacts"]["mzareei_github_csvs"] = csvs
    out = ROOT / "external_data_inventory.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
