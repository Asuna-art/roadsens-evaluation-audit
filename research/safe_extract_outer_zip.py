"""Inspect and safely extract Figshare outer ZIPs (nested RARs only)."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARCHIVES = [
    (DATA / "Combined CSV with GIS and Weather Data.zip", DATA / "outer_combined"),
    (DATA / "Isolated Data.zip", DATA / "outer_isolated"),
]
BASE = DATA.resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def safe_parts(name: str) -> tuple[str, ...]:
    if "\\" in name or "\x00" in name or name.startswith("/") or ":" in name:
        raise ValueError(f"unsafe ZIP path: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("..", "") for part in p.parts):
        raise ValueError(f"unsafe ZIP path: {name!r}")
    return p.parts


def inspect_and_extract(archive: Path, target: Path) -> list[dict]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    target_abs = target.resolve()
    if not target_abs.is_relative_to(BASE):
        raise ValueError(f"extraction target escapes data directory: {target_abs}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite extraction directory: {target}")

    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        folded: set[str] = set()
        for info in infos:
            parts = safe_parts(info.filename.rstrip("/")) if not info.is_dir() else safe_parts(info.filename.rstrip("/"))
            key = "/".join(parts).casefold()
            if key in folded:
                raise ValueError(f"duplicate/case-colliding archive path: {info.filename!r}")
            folded.add(key)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"link entry rejected: {info.filename!r}")
            resolved = target_abs.joinpath(*parts).resolve()
            if not resolved.is_relative_to(target_abs):
                raise ValueError(f"entry escapes target: {info.filename!r}")

        # Validate CRCs through the ZIP reader before extraction, then write each
        # file exclusively under a newly-created destination directory.
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in ZIP member: {bad}")
        target.mkdir(parents=True, exist_ok=False)
        result = []
        for info in infos:
            parts = safe_parts(info.filename.rstrip("/"))
            dest = target.joinpath(*parts)
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(f"Refusing to overwrite extracted file: {dest}")
            h = hashlib.sha256()
            n = 0
            with z.open(info, "r") as src, dest.open("xb") as out:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    h.update(block)
                    n += len(block)
            if n != info.file_size:
                raise ValueError(f"size mismatch for {info.filename}: {n} vs {info.file_size}")
            result.append({
                "archive_member": info.filename,
                "extracted_path": str(dest.relative_to(DATA)),
                "bytes": n,
                "zip_crc32": f"{info.CRC:08x}",
                "sha256": h.hexdigest(),
            })
    return result


def main() -> None:
    output = []
    for archive, target in ARCHIVES:
        print(f"Inspect/extract {archive.name}", flush=True)
        rows = inspect_and_extract(archive, target)
        output.append({"archive": archive.name, "archive_sha256": sha256(archive), "extracted": rows})
    manifest = DATA / "outer_extraction_manifest.json"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest}")
    manifest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
