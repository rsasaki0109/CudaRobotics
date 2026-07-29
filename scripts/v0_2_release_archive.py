#!/usr/bin/env python3
"""Create and validate canonical ZIPs of v0.2.0 release evidence bundles."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any
import zipfile

from v0_2_release_bundle import load_bundle, sha256_file


ARCHIVE_ROOT = "cudarobotics-v0.2.0-evidence"
ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
REGULAR_FILE_MODE = stat.S_IFREG | 0o644
MAX_TOTAL_BYTES = 16 * 1024 * 1024 * 1024


def _bundle_files(bundle_path: Path) -> list[Path]:
    root = bundle_path.resolve().parent
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def create_archive(
    bundle_path: Path,
    archive_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    bundle = bundle_path.resolve()
    gate = load_bundle(bundle, expected_commit)
    if not gate["valid"] or not gate["ready"]:
        failed = [
            name for name, passed in gate.get("checks", {}).items() if not passed
        ]
        raise ValueError("bundle is not ready: " + ", ".join(failed))
    if bundle.name != "bundle.json":
        raise ValueError("bundle manifest must be named bundle.json")

    root = bundle.parent
    destination = archive_path.resolve()
    if destination.is_relative_to(root):
        raise ValueError("archive must be outside the bundle directory")
    if destination.exists():
        raise ValueError(f"refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in _bundle_files(bundle):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(
                f"{ARCHIVE_ROOT}/{relative}",
                date_time=ARCHIVE_TIMESTAMP,
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = REGULAR_FILE_MODE << 16
            with path.open("rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)

    validation = evaluate_archive(destination, expected_commit)
    if not validation["valid"]:
        failed = [
            name
            for name, passed in validation["checks"].items()
            if not passed
        ]
        raise ValueError("created archive is invalid: " + ", ".join(failed))
    return {
        "archive": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "members": len(_bundle_files(bundle)),
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


def _safe_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    parts = path.parts
    mode = info.external_attr >> 16
    return (
        len(parts) >= 2
        and parts[0] == ARCHIVE_ROOT
        and info.filename == path.as_posix()
        and all(part not in ("", ".", "..") for part in parts)
        and "\\" not in info.filename
        and not info.is_dir()
        and info.flag_bits == 0
        and info.compress_type == zipfile.ZIP_STORED
        and info.compress_size == info.file_size
        and info.date_time == ARCHIVE_TIMESTAMP
        and info.create_system == 3
        and stat.S_IFMT(mode) == stat.S_IFREG
        and stat.S_IMODE(mode) == 0o644
        and not info.extra
        and not info.comment
    )


def evaluate_archive(
    archive_path: Path,
    expected_commit: str,
    *,
    expected_sha256: str | None = None,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
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
        "bundle": {"valid": False, "ready": False},
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
                _safe_member(info) for info in infos
            )
            checks["size_limit"] = total <= max_total_bytes
            bundle_member = f"{ARCHIVE_ROOT}/bundle.json"
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
            structural = all(
                checks[name]
                for name in (
                    "archive_readable",
                    "canonical_comment",
                    "member_table",
                    "safe_canonical_members",
                    "size_limit",
                    "payload_crc",
                    "bundle_present",
                )
            )
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
                    bundle_gate = load_bundle(
                        extraction_root / bundle_member,
                        expected_commit,
                    )
                    result["bundle"] = bundle_gate
                    checks["bundle_valid"] = bool(
                        bundle_gate["valid"] and bundle_gate["ready"]
                    )
    except (OSError, ValueError, zipfile.BadZipFile):
        return result

    result["valid"] = all(checks.values())
    result["ready"] = result["valid"]
    return result


def load_archive(
    archive_path: Path,
    expected_commit: str,
    *,
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
                "bundle": {"valid": False, "ready": False},
            }
    result = evaluate_archive(
        archive_path,
        expected_commit,
        expected_sha256=expected_sha256,
        max_total_bytes=max_total_bytes,
    )
    if checksum_path is not None:
        result["checks"]["checksum_readable"] = True
        result["valid"] = all(result["checks"].values())
        result["ready"] = result["valid"]
    return result
