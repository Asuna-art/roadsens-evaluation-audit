"""Bounded, resumable Figshare v3 RoadSens-4M ZIP downloader.

Final files are never overwritten. Partial files are kept after network errors.
Only the two requested CSV archives are downloaded; GIS.zip is excluded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
API_URL = "https://api.figshare.com/v2/articles/30341143/versions/3"
FILES = [
    {
        "name": "Combined CSV with GIS and Weather Data.zip",
        "size": 113710926,
        "md5": "371536cc260e2a590c972ce0b1fe43bd",
        "url": "https://ndownloader.figshare.com/files/58678312",
    },
    {
        "name": "Isolated Data.zip",
        "size": 141188237,
        "md5": "4d0d3a60f08e431019e8130ab43f1fdb",
        "url": "https://ndownloader.figshare.com/files/58678315",
    },
]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_one(spec: dict) -> dict:
    final = DATA / spec["name"]
    part = DATA / (spec["name"] + ".part")
    if final.exists():
        if final.stat().st_size != spec["size"] or md5_file(final) != spec["md5"]:
            raise RuntimeError(f"Refusing to overwrite existing unverified file: {final}")
        return {**spec, "state": "already_present_verified", "sha256": sha256_file(final)}

    if part.exists():
        offset = part.stat().st_size
        if offset > spec["size"]:
            raise RuntimeError(f"Partial file is larger than expected; preserving it: {part}")
        if offset == spec["size"]:
            if md5_file(part) != spec["md5"]:
                raise RuntimeError(f"Complete partial file fails official MD5; preserving it: {part}")
            if final.exists():
                raise RuntimeError(f"Destination appeared during run; preserving partial: {part}")
            part.rename(final)
            return {**spec, "state": "resumed_complete_verified", "sha256": sha256_file(final)}
    else:
        offset = 0

    headers = {"Accept-Encoding": "identity", "User-Agent": "RoadSens-4M-feasibility/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    else:
        headers["Range"] = "bytes=0-"
    req = urllib.request.Request(spec["url"], headers=headers)
    response = urllib.request.urlopen(req, timeout=60)
    try:
        status = response.status
        content_range = response.headers.get("Content-Range", "")
        expected_start = offset
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if status != 206 or not match:
            raise RuntimeError(
                f"Refusing response without HTTP 206 and Content-Range; "
                f"status={status}, Content-Range={content_range!r}; partial retained"
            )
        start, end, total = map(int, match.groups())
        if start != expected_start or total != spec["size"] or end != total - 1:
            raise RuntimeError(
                f"Unexpected Content-Range {content_range!r}; expected start={expected_start}, "
                f"total={spec['size']}; partial retained"
            )
        remaining = spec["size"] - offset
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != remaining:
            raise RuntimeError(f"Unexpected Content-Length={content_length}; partial retained")
        mode = "ab" if offset or part.exists() else "xb"
        received = 0
        with part.open(mode) as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                received += len(block)
                if received > remaining:
                    raise RuntimeError("Server sent more bytes than requested; partial retained")
                out.write(block)
                if received and received % (16 * 1024 * 1024) < len(block):
                    print(f"{spec['name']}: {offset + received}/{spec['size']} bytes", flush=True)
        if received != remaining or part.stat().st_size != spec["size"]:
            raise RuntimeError(
                f"Short response: received {received}, expected {remaining}; partial retained"
            )
    finally:
        response.close()

    actual_md5 = md5_file(part)
    if actual_md5 != spec["md5"]:
        raise RuntimeError(
            f"Official MD5 mismatch for partial {part}: got {actual_md5}; preserving it"
        )
    if final.exists():
        raise RuntimeError(f"Destination appeared during download; preserving partial: {part}")
    part.rename(final)
    return {**spec, "state": "downloaded_verified", "sha256": sha256_file(final)}


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    source = {
        "api_url": API_URL,
        "doi": "10.6084/m9.figshare.30341143.v3",
        "version": 3,
        "license": "CC BY 4.0",
        "published_date": "2026-02-24T16:23:16Z",
        "files": FILES,
        "excluded": [
            {"name": "GIS.zip", "size": 42455081, "md5": "b59044ee2e7b99a90ef8651050f0fdf5"}
        ],
    }
    source_path = DATA / "figshare_v3_source_metadata.json"
    if source_path.exists():
        existing = json.loads(source_path.read_text(encoding="utf-8"))
        if existing != source:
            raise RuntimeError(f"Refusing to overwrite different metadata file: {source_path}")
    else:
        source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    results = []
    for spec in FILES:
        print(f"Starting: {spec['name']}", flush=True)
        results.append(download_one(spec))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata_sha256": sha256_file(source_path),
        "files": results,
    }
    manifest_path = DATA / "download_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in prior.get("files", []):
            if item.get("state") in {"downloaded_verified", "already_present_verified", "resumed_complete_verified"}:
                if not any(x["name"] == item["name"] and x["sha256"] == item.get("sha256") for x in results):
                    raise RuntimeError(f"Refusing to replace conflicting download manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
