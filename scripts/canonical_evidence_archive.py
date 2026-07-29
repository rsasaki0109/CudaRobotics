#!/usr/bin/env python3
"""Canonical ZIP primitives shared by release evidence bundles."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any
import zipfile


ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
REGULAR_FILE_MODE = stat.S_IFREG | 0o644
MAX_TOTAL_BYTES = 16 * 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive_root(archive_root: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", archive_root):
        raise ValueError("archive root must be one safe path component")


def directory_files(root: Path) -> list[Path]:
    resolved = root.resolve()
    return sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )


def create_archive(
    source_root: Path,
    archive_path: Path,
    archive_root: str,
) -> dict[str, Any]:
    validate_archive_root(archive_root)
    root = source_root.resolve()
    destination = archive_path.resolve()
    if not root.is_dir():
        raise ValueError(f"source directory is missing: {root}")
    if destination.is_relative_to(root):
        raise ValueError("archive must be outside the bundle directory")
    if destination.exists():
        raise ValueError(f"refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = directory_files(root)
    if not files:
        raise ValueError("source directory contains no files")

    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(
                f"{archive_root}/{relative}",
                date_time=ARCHIVE_TIMESTAMP,
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.external_attr = REGULAR_FILE_MODE << 16
            info.file_size = path.stat().st_size
            with path.open("rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)

    return {
        "archive": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "members": len(files),
    }


def write_checksum(archive_path: Path, checksum_path: Path) -> dict[str, Any]:
    archive = archive_path.resolve()
    checksum = checksum_path.resolve()
    if not archive.is_file():
        raise ValueError(f"archive is missing: {archive}")
    if checksum.exists():
        raise ValueError(f"refusing to overwrite checksum: {checksum}")
    checksum.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(archive)
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return {
        "checksum": str(checksum),
        "sha256": digest,
    }


def read_checksum(checksum_path: Path, archive_name: str) -> str:
    line = checksum_path.read_text(encoding="ascii")
    suffix = f"  {archive_name}\n"
    digest = line[:64]
    if (
        len(line) != 64 + len(suffix)
        or not line.endswith(suffix)
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("checksum file is not canonical")
    return digest


def safe_member(info: zipfile.ZipInfo, archive_root: str) -> bool:
    path = PurePosixPath(info.filename)
    parts = path.parts
    mode = info.external_attr >> 16
    return (
        len(parts) >= 2
        and parts[0] == archive_root
        and info.filename == path.as_posix()
        and all(part not in ("", ".", "..") for part in parts)
        and "\\" not in info.filename
        and not info.is_dir()
        and info.flag_bits == 0
        and info.compress_type == zipfile.ZIP_STORED
        and info.compress_size == info.file_size
        and info.date_time == ARCHIVE_TIMESTAMP
        and info.create_system == 3
        and info.create_version == 20
        and info.extract_version == 20
        and stat.S_IFMT(mode) == stat.S_IFREG
        and stat.S_IMODE(mode) == 0o644
        and not info.extra
        and not info.comment
    )


def evaluate_archive(
    archive_path: Path,
    *,
    archive_root: str,
    manifest_relative: str,
    validate_manifest: Callable[[Path], dict[str, Any]],
    manifest_passes: Callable[[dict[str, Any]], bool],
    expected_sha256: str | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    validate_archive_root(archive_root)
    manifest_path = PurePosixPath(manifest_relative)
    if (
        manifest_path.is_absolute()
        or not manifest_path.parts
        or any(part in ("", ".", "..") for part in manifest_path.parts)
        or manifest_relative != manifest_path.as_posix()
    ):
        raise ValueError("manifest path must be a safe relative POSIX path")

    path = archive_path.resolve()
    checks = {
        "archive_readable": False,
        "archive_sha256": False,
        "canonical_comment": False,
        "member_table": False,
        "safe_canonical_members": False,
        "size_limit": False,
        "payload_crc": False,
        "bundle_present": False,
        "bundle_valid": False,
    }
    result: dict[str, Any] = {
        "valid": False,
        "ready": False,
        "checks": checks,
        "archive": str(path),
        "sha256": "",
        "members": 0,
        "uncompressed_bytes": 0,
        "bundle": {},
    }
    if not path.is_file() or max_total_bytes <= 0:
        return result
    if path.stat().st_size > max_total_bytes + 64 * 1024 * 1024:
        return result

    digest = sha256_file(path)
    result["sha256"] = digest
    checks["archive_sha256"] = (
        expected_sha256 is None or digest == expected_sha256
    )
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            total = sum(info.file_size for info in infos)
            result["members"] = len(infos)
            result["uncompressed_bytes"] = total
            checks["archive_readable"] = True
            checks["canonical_comment"] = not archive.comment
            checks["member_table"] = (
                bool(infos)
                and len(names) == len(set(names))
                and names == sorted(names)
            )
            checks["safe_canonical_members"] = all(
                safe_member(info, archive_root) for info in infos
            )
            checks["size_limit"] = total <= max_total_bytes
            bundle_member = f"{archive_root}/{manifest_relative}"
            checks["bundle_present"] = bundle_member in names
            safe_to_read = all(
                checks[name]
                for name in (
                    "archive_readable",
                    "canonical_comment",
                    "member_table",
                    "safe_canonical_members",
                    "size_limit",
                    "bundle_present",
                )
            )
            if safe_to_read:
                checks["payload_crc"] = archive.testzip() is None
            structural = safe_to_read and checks["payload_crc"]
            if structural:
                with tempfile.TemporaryDirectory() as directory:
                    extraction_root = Path(directory)
                    for info in infos:
                        relative = PurePosixPath(info.filename)
                        destination = extraction_root.joinpath(*relative.parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as source, destination.open(
                            "xb"
                        ) as target:
                            shutil.copyfileobj(
                                source, target, length=1024 * 1024
                            )
                    bundle_gate = validate_manifest(
                        extraction_root / bundle_member
                    )
                    result["bundle"] = bundle_gate
                    checks["bundle_valid"] = manifest_passes(bundle_gate)
    except (OSError, TypeError, ValueError, zipfile.BadZipFile):
        return result

    result["valid"] = all(checks.values())
    result["ready"] = result["valid"]
    return result


def load_archive(
    archive_path: Path,
    *,
    archive_root: str,
    manifest_relative: str,
    validate_manifest: Callable[[Path], dict[str, Any]],
    manifest_passes: Callable[[dict[str, Any]], bool],
    checksum_path: Path | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    expected_sha256 = None
    if checksum_path is not None:
        try:
            expected_sha256 = read_checksum(
                checksum_path.resolve(),
                archive_path.name,
            )
        except (OSError, UnicodeError, ValueError):
            return {
                "valid": False,
                "ready": False,
                "checks": {"checksum_readable": False},
                "archive": str(archive_path.resolve()),
                "bundle": {},
            }
    result = evaluate_archive(
        archive_path,
        archive_root=archive_root,
        manifest_relative=manifest_relative,
        validate_manifest=validate_manifest,
        manifest_passes=manifest_passes,
        expected_sha256=expected_sha256,
        max_total_bytes=max_total_bytes,
    )
    if checksum_path is not None:
        result["checks"]["checksum_readable"] = True
        result["valid"] = all(result["checks"].values())
        result["ready"] = result["valid"]
    return result
