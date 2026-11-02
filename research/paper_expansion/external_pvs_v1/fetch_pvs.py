from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import requests

OUT = pathlib.Path(__file__).resolve().parent
OWNER = "jefmenegazzo"
SLUG = "pvs-passive-vehicular-sensors-datasets"
LIST_URL = f"https://www.kaggle.com/api/v1/datasets/list/{OWNER}/{SLUG}"
DOWNLOAD_BASE = f"https://www.kaggle.com/api/v1/datasets/download/{OWNER}/{SLUG}/"
WANTED_SUFFIXES = ("dataset_labels.csv", "dataset_gps_mpu_left.csv")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def catalogue(session: requests.Session):
    files = []
    token = None
    pages = []
    for _ in range(20):
        resp = session.get(LIST_URL, params={"pageToken": token} if token else None, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        pages.append({"url": resp.url, "status": resp.status_code, "headers": dict(resp.headers), "body_sha256": sha256_bytes(resp.content)})
        files.extend(body.get("datasetFiles", []))
        token = body.get("nextPageTokenNullable")
        if not token:
            break
    else:
        raise RuntimeError("Kaggle listing pagination exceeded 20 pages")
    targets = [x for x in files if pathlib.PurePosixPath(x["name"]).name in WANTED_SUFFIXES]
    trips = sorted({x["name"].split("/")[0] for x in targets})
    return files, targets, pages, trips


def read_csv_schema(path: pathlib.Path):
    # Read all rows for accurate counts; preserve headers and report probable time columns.
    df = pd.read_csv(path, low_memory=False)
    label_cols = [c for c in df.columns if any(t in str(c).lower() for t in ("label", "class", "event", "type", "surface", "bump"))]
    counts = {}
    for col in label_cols:
        vals = df[col].dropna().astype(str).value_counts().to_dict()
        if vals:
            counts[str(col)] = {str(k): int(v) for k, v in vals.items()}
    time_cols = [str(c) for c in df.columns if any(t in str(c).lower() for t in ("time", "date", "timestamp", "epoch"))]
    return {
        "rows": int(len(df)), "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "time_columns": time_cols, "label_like_columns": label_cols,
        "label_counts": counts,
        "time_ranges": {c: [str(df[c].dropna().iloc[0]) if df[c].notna().any() else None,
                             str(df[c].dropna().iloc[-1]) if df[c].notna().any() else None] for c in time_cols},
        "preview": df.head(3).astype(object).where(pd.notna(df.head(3)), None).to_dict(orient="records"),
    }


def download_one(session: requests.Session, entry: dict):
    rel = entry["name"]
    quoted = requests.utils.quote(rel, safe="")
    url = DOWNLOAD_BASE + quoted
    existing = (OUT / "data" / rel.split("/")[0].replace(" ", "_") / (pathlib.PurePosixPath(rel).name + ".zip"))
    trip, filename = rel.split("/", 1)
    trip_dir = OUT / "data" / trip.replace(" ", "_")
    trip_dir.mkdir(parents=True, exist_ok=True)
    archive_path = trip_dir / (filename + ".zip")
    csv_path = trip_dir / filename
    if archive_path.exists() and zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            members = [m for m in zf.namelist() if pathlib.PurePosixPath(m).name == filename]
            if len(members) == 1:
                csv_bytes = zf.read(members[0])
                csv_path.write_bytes(csv_bytes)
                return {
                    "dataset_path": rel, "requested_url": url,
                    "final_url_without_query": "reused previously completed archive",
                    "request_status": 200, "response_headers": {"reused_archive": True},
                    "archive_bytes": archive_path.stat().st_size,
                    "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    "csv_path": str(csv_path.relative_to(OUT)), "csv_bytes": len(csv_bytes),
                    "csv_sha256": sha256_bytes(csv_bytes), "schema": read_csv_schema(csv_path),
                }
    offset = existing.stat().st_size if existing.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = None
    last_error = None
    for attempt in range(5):
        try:
            response = session.get(url, headers=headers, stream=True, timeout=(30, 180))
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if response is not None:
                response.close()
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError(f"Download request did not produce a response: {last_error}")
    append = offset > 0 and response.status_code == 206
    if offset and not append:
        offset = 0
    with archive_path.open("ab" if append else "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                f.flush()
                if f.tell() % (16 * 1024 * 1024) < len(chunk):
                    print(f"  received {f.tell():,} bytes for {rel}", flush=True)
    size = archive_path.stat().st_size
    digest = hashlib.sha256()
    with archive_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(archive_path) as zf:
        members = [m for m in zf.namelist() if pathlib.PurePosixPath(m).name == filename]
        if len(members) != 1:
            raise RuntimeError(f"Unexpected zip members for {rel}: {zf.namelist()}")
        csv_bytes = zf.read(members[0])
    csv_path.write_bytes(csv_bytes)
    return {
        "dataset_path": rel, "requested_url": url,
        "final_url_without_query": response.url.split("?", 1)[0],
        "request_status": response.status_code,
        "response_headers": {k: v for k, v in response.headers.items() if k.lower() not in ("set-cookie",)},
        "archive_bytes": size, "archive_sha256": digest.hexdigest(),
        "csv_path": str(csv_path.relative_to(OUT)), "csv_bytes": len(csv_bytes),
        "csv_sha256": sha256_bytes(csv_bytes), "schema": read_csv_schema(csv_path),
    }


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "RoadSens-PVS-read-only-audit/1.0"})
    all_files, targets, pages, trips = catalogue(session)
    selected = []
    for trip in trips:
        for suffix in WANTED_SUFFIXES:
            matched = [x for x in targets if x["name"] == f"{trip}/{suffix}"]
            if len(matched) != 1:
                raise RuntimeError(f"Expected exactly one {trip}/{suffix}, found {len(matched)}")
            selected.append(matched[0])
    totals = {"metadata_uncompressed_bytes": sum(int(x.get("totalBytes") or 0) for x in selected)}
    report = {
        "dataset_owner": OWNER, "dataset_slug": SLUG,
        "catalogue_pages": pages,
        "all_file_count": len(all_files),
        "video_files_excluded": [x["name"] for x in all_files if pathlib.PurePosixPath(x["name"]).suffix.lower() in (".mp4", ".mov", ".avi")],
        "trip_names": trips,
        "selected_inventory": [{"name": x["name"], "metadata_bytes": x.get("totalBytes")} for x in selected],
        "totals": totals,
        "downloaded": [],
    }
    (OUT / "kaggle_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_lock = threading.Lock()
    def one(entry):
        local_session = requests.Session()
        local_session.headers.update({"User-Agent": "RoadSens-PVS-read-only-audit/1.0"})
        return download_one(local_session, entry)
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {pool.submit(one, entry): entry for entry in selected}
        for i, future in enumerate(as_completed(future_map), 1):
            entry = future_map[future]
            print(f"[{i}/{len(selected)}] completed {entry['name']}", flush=True)
            item = future.result()
            with write_lock:
                report["downloaded"].append(item)
                (OUT / "kaggle_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "kaggle_inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(report["downloaded"]), "files", flush=True)


if __name__ == "__main__":
    main()
