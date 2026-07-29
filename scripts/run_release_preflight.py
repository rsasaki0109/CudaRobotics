#!/usr/bin/env python3
"""Run local v0.2 release gates and write auditable JSON/Markdown evidence."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from release_preflight_evidence import (
    collect_evidence_files,
    evaluate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/release_preflight"),
    )
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def check_specs(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    python = sys.executable
    build_dir = repo_path(args.build_dir)
    specs: list[dict[str, Any]] = [
        {
            "name": "version_consistency",
            "command": [python, "scripts/test_version_consistency.py"],
        },
        {
            "name": "python_core_sync",
            "command": [python, "scripts/test_python_core_sync.py"],
        },
        {
            "name": "artifact_verifier_tests",
            "command": [
                python,
                "scripts/test_verify_python_release_artifacts.py",
            ],
        },
        {
            "name": "python_labelled_ctest",
            "command": [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-C",
                "Release",
                "-L",
                "python",
                "--output-on-failure",
            ],
        },
        {
            "name": "python_package_tests",
            "command": [python, "-m", "pytest", "python/tests", "-q"],
        },
        {
            "name": "whitespace",
            "command": ["git", "diff", "--check"],
        },
    ]
    if args.dist_dir is not None:
        artifact_command = [
            python,
            "scripts/verify_python_release_artifacts.py",
            "--dist-dir",
            str(repo_path(args.dist_dir)),
            "--json",
            str(output_dir / "python_artifacts.json"),
        ]
        if args.require_clean:
            artifact_command.append("--require-clean")
        specs.append(
            {
                "name": "python_release_artifacts",
                "command": artifact_command,
            }
        )
    if args.profile == "gpu":
        registration_csv = output_dir / "registration_smoke.csv"
        registration_md = output_dir / "registration_smoke.md"
        specs.extend(
            [
                {
                    "name": "registration_gpu_consistency",
                    "command": [
                        python,
                        "-m",
                        "pytest",
                        "python/tests/test_registration_consistency.py",
                        "-q",
                    ],
                },
                {
                    "name": "registration_gpu_smoke",
                    "command": [
                        python,
                        "scripts/run_registration_suite.py",
                        "--algorithms",
                        "cudarobotics_filterreg_gpu",
                        "--scenarios",
                        "lumpy_partial",
                        "--sizes",
                        "512",
                        "--trials",
                        "1",
                        "--strict",
                        "--csv",
                        str(registration_csv),
                        "--markdown",
                        str(registration_md),
                    ],
                },
            ]
        )
    return specs


def run_check(spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    command = spec["command"]
    log_path = output_dir / "logs" / f"{spec['name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(command)}\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        **spec,
        "display": command_text(command),
        "log": str(log_path.relative_to(ROOT)),
        "report_log": f"logs/{spec['name']}.log",
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "elapsed_seconds": round(time.perf_counter() - begin, 3),
    }


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# CudaRobotics Release Preflight",
        "",
        f"- Status: **{manifest['status']}**",
        f"- Profile: `{manifest['profile']}`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Dirty checkout: `{str(manifest['git_dirty']).lower()}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Python: `{manifest['python']}`",
        f"- Content-bound evidence files: `{len(manifest.get('evidence_files', []))}`",
        "",
        "| Local gate | Status | Seconds | Log |",
        "|---|---:|---:|---|",
    ]
    for check in manifest["checks"]:
        log = check.get("report_log", check.get("log", ""))
        log_cell = f"[log]({log})" if log else "not run"
        lines.append(
            f"| `{check['name']}` | {check['status']} | "
            f"{check.get('elapsed_seconds', 0):.3f} | {log_cell} |"
        )
    lines.extend(
        [
            "",
            "## External gates",
            "",
            "These are deliberately not inferred from local checks:",
            "",
        ]
    )
    for gate in manifest["external_gates"]:
        lines.append(f"- `{gate}`: verify on the final release-candidate commit")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = check_specs(args, output_dir)
    dirty = bool(git("status", "--porcelain"))
    started_at = datetime.now(timezone.utc).isoformat()

    if args.dry_run:
        checks = [
            {
                **spec,
                "display": command_text(spec["command"]),
                "status": "planned",
                "returncode": None,
                "elapsed_seconds": 0.0,
            }
            for spec in specs
        ]
        if args.require_clean:
            checks.append(
                {
                    "name": "clean_checkout",
                    "command": [],
                    "display": "git status --porcelain",
                    "status": "planned",
                    "returncode": None,
                    "elapsed_seconds": 0.0,
                }
            )
        status = "planned"
    else:
        checks = [run_check(spec, output_dir) for spec in specs]
        if args.require_clean:
            checks.append(
                {
                    "name": "clean_checkout",
                    "command": ["git", "status", "--porcelain"],
                    "display": "git status --porcelain",
                    "status": "failed" if dirty else "passed",
                    "returncode": 1 if dirty else 0,
                    "elapsed_seconds": 0.0,
                }
            )
        status = (
            "passed"
            if all(check["status"] == "passed" for check in checks)
            else "failed"
        )

    manifest = {
        "schema_version": 1,
        "status": status,
        "profile": args.profile,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "checks": checks,
        "external_gates": [
            "github_build",
            "python_manylinux_wheels",
            "ros2_cuda_mppi",
            "closed_loop_rosbag_or_explicit_negative_result",
        ],
        "evidence_files": collect_evidence_files(output_dir, checks)
        if not args.dry_run
        else [],
    }
    if not args.dry_run:
        evidence_gate = evaluate_manifest(
            manifest,
            output_dir,
            expected_profile=args.profile,
            expected_commit=manifest["git_commit"],
        )
        manifest["evidence_gate"] = evidence_gate
        if not evidence_gate["passed"]:
            status = "failed"
            manifest["status"] = status
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.md"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(manifest), encoding="utf-8")
    print(f"release preflight: {status}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    return 0 if status in {"passed", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
