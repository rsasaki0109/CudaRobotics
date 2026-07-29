#!/usr/bin/env python3
"""Fetch and retain the deployed v1 documentation for release validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from v1_documentation_evidence import (
    REQUIRED_ARTIFACTS,
    SITE,
    describe_artifacts,
    evaluate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def fetch(url: str, commit: str, timeout: float) -> tuple[int, bytes]:
    separator = "&" if "?" in url else "?"
    request = Request(
        f"{url}{separator}release_commit={commit}",
        headers={"User-Agent": "CudaRobotics-v1-release-validator"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except HTTPError as error:
        return int(error.code), error.read()
    except URLError:
        return 0, b""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--commit")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.tag != "v1.0.0":
        raise SystemExit("documentation release tag must be v1.0.0")
    if args.attempts <= 0 or not 0 <= args.retry_seconds <= 60:
        raise SystemExit("invalid retry policy")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    site_root = output / "site"
    site_root.mkdir()
    commit = args.commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
    )
    urls = {
        "index": SITE,
        "install": SITE + "install.html",
        "nav2": SITE + "nav2.html",
        "release": SITE + "release.json",
    }
    filenames = {
        "index": "index.html",
        "install": "install.html",
        "nav2": "nav2.html",
        "release": "release.json",
    }
    statuses: dict[str, int] = {}
    started = datetime.now(timezone.utc).isoformat()
    for attempt in range(args.attempts):
        statuses = {}
        for key, url in urls.items():
            status, content = fetch(url, commit, args.http_timeout)
            statuses[key] = status
            (site_root / filenames[key]).write_bytes(content)
        release: dict[str, Any] = {}
        try:
            candidate = json.loads(
                (site_root / "release.json").read_text(encoding="utf-8")
            )
            if isinstance(candidate, dict):
                release = candidate
        except (OSError, json.JSONDecodeError):
            pass
        if (
            set(statuses.values()) == {200}
            and release.get("source_commit") == commit
            and release.get("target_tag") == args.tag
        ):
            break
        if attempt + 1 < args.attempts:
            time.sleep(args.retry_seconds)
    manifest = {
        "schema_version": 1,
        "evidence_mode": "v1_documentation_http_deployment",
        "status": "passed",
        "version": "1.0.0",
        "target_tag": args.tag,
        "git_commit": commit,
        "git_dirty": dirty,
        "site": SITE,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "urls": urls,
        "http_status": statuses,
        "artifacts": describe_artifacts(output, set(REQUIRED_ARTIFACTS)),
    }
    gate = evaluate_manifest(
        manifest, output, expected_commit=commit
    )
    manifest["gate"] = gate
    manifest["status"] = "passed" if gate["passed"] else "failed"
    write_json(output / "manifest.json", manifest)
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(output / "manifest.json")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
