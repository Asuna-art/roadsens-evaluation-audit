"""Validate all RAR member paths before extracting inside the data workspace."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath

import rarfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASE = DATA.resolve()
ARCHIVES = [
    (DATA / "outer_combined/Combined CSV with GIS and Weather Data/Normal Road (No Annotation)/Normal Road (No Annotation).rar", DATA / "inner/combined_normal"),
    (DATA / "outer_combined/Combined CSV with GIS and Weather Data/Road Anomalies/Road Anomalies.rar", DATA / "inner/combined_anomalies"),
    (DATA / "outer_isolated/Isolated Data/Normal Road/Normal Road.rar", DATA / "inner/isolated_normal"),
    (DATA / "outer_isolated/Isolated Data/Road Anomalies/Road Anomalies (1-86).rar", DATA / "inner/isolated_anomalies_1_86"),
    (DATA / "outer_isolated/Isolated Data/Road Anomalies/87-103-GIS-integrated.rar", DATA / "inner/isolated_anomalies_87_103"),
]
MAX_UNCOMPRESSED = 1_000_000_000


def safe_name(name: str) -> tuple[str, ...]:
    if "\\" in name or "\x00" in name or name.startswith("/") or ":" in name:
        raise ValueError(f"unsafe RAR member path: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        raise ValueError(f"unsafe RAR member path: {name!r}")
    return p.parts


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def validate(archive: Path, target: Path) -> list[dict]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    target_abs = target.resolve()
    if not target_abs.is_relative_to(BASE):
        raise ValueError(f"target escapes workspace data directory: {target_abs}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite RAR extraction target: {target}")
    rows = []
    seen: set[str] = set()
    with rarfile.RarFile(archive) as rf:
        for info in rf.infolist():
            parts = safe_name(info.filename.rstrip("/"))
            key = "/".join(parts).casefold()
            if key in seen:
                raise ValueError(f"duplicate/case-colliding RAR path: {info.filename!r}")
            seen.add(key)
            if info.is_symlink():
                raise ValueError(f"RAR link rejected: {info.filename!r}")
            resolved = target_abs.joinpath(*parts).resolve()
            if not resolved.is_relative_to(target_abs):
                raise ValueError(f"RAR member escapes extraction target: {info.filename!r}")
            rows.append({"name": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size, "is_dir": info.isdir()})
    if sum(x["bytes"] for x in rows) > MAX_UNCOMPRESSED:
        raise ValueError(f"archive exceeds uncompressed safety ceiling: {archive}")

    listing = subprocess.run(["bsdtar", "-tf", str(archive)], capture_output=True, text=True, encoding="utf-8", errors="strict", check=True)
    listed = [name.rstrip("/") for name in listing.stdout.splitlines()]
    expected_names = [x["name"].rstrip("/") for x in rows]
    if len(listed) != len(expected_names) or set(listed) != set(expected_names):
        raise ValueError(f"bsdtar and rarfile listings differ for {archive}")
    for name in listed:
        safe_name(name)
    return rows


def extract(archive: Path, target: Path, expected: list[dict]) -> list[dict]:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    p = subprocess.run(
        ["bsdtar", "-x", "-f", str(archive), "-C", str(target), "--no-same-owner", "--no-same-permissions"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode:
        raise RuntimeError(f"bsdtar extraction failed for {archive}: {p.stderr[-2000:]}")
    result = []
    for item in expected:
        parts = safe_name(item["name"].rstrip("/"))
        path = target.joinpath(*parts)
        if item["is_dir"]:
            if not path.is_dir() or path.is_symlink():
                raise RuntimeError(f"missing or unsafe extracted directory: {path}")
            continue
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe extracted file: {path}")
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"size mismatch for {path}: {path.stat().st_size} != {item['bytes']}")
        if not path.resolve().is_relative_to(target.resolve()):
            raise RuntimeError(f"extracted path escaped target: {path}")
        result.append({**item, "extracted_path": str(path.relative_to(DATA)), "sha256": sha256(path)})
    return result


def main() -> None:
    manifest_path = DATA / "inner_extraction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
    validated = []
    for archive, target in ARCHIVES:
        print(f"Validating paths: {archive.name}", flush=True)
        entries = validate(archive, target)
        print(f"Safe listing: {len(entries)} entries / {sum(x['bytes'] for x in entries):,} bytes", flush=True)
        validated.append((archive, target, entries))
    output = []
    for archive, target, entries in validated:
        print(f"Extracting: {archive.name}", flush=True)
        files = extract(archive, target, entries)
        output.append({"rar_archive": str(archive.relative_to(DATA)), "rar_sha256": sha256(archive), "members": files})
    manifest_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
